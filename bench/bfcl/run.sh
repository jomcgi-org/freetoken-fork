#!/usr/bin/env bash
# BFCL v4 tool-calling subset against a running FreeToken server (~1 hour).
#
# Runs a fixed slice of the Berkeley Function Calling Leaderboard through the
# server's own OpenAI tool-call parser, one request at a time, and scores it
# offline (the multi-turn backends are simulated Python; no keys or network
# beyond the first install). Compare two builds with compare.py.
#
# Usage:
#   OUT=results/bfcl-base bash bench/bfcl/run.sh
#   OUT=results/bfcl-cand bash bench/bfcl/run.sh
#   python3 bench/bfcl/compare.py results/bfcl-base results/bfcl-cand
#
# Env:
#   BASE_URL    server base URL (default http://127.0.0.1:8090/v1)
#   OUT         BFCL project root for this run; must not already hold results
#   SUBSET      test-id JSON (default bench/bfcl/subset.json)
#   MAX_TOKENS  per-call output cap (default 4096)
#   THINKING    "off" disables the chat template's thinking (default: server default)
#   BFCL_VENV   venv location (default bench/bfcl/.venv; keep it off a full root disk)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BASE_URL="${BASE_URL:-http://127.0.0.1:8090/v1}"
OUT="${OUT:?set OUT to a fresh results directory}"
SUBSET="${SUBSET:-$HERE/subset.json}"
BFCL_VENV="${BFCL_VENV:-$HERE/.venv}"
BFCL_VERSION=2026.3.23

if [ ! -x "$BFCL_VENV/bin/python" ]; then
  uv venv --python 3.10 "$BFCL_VENV"
  uv pip install --python "$BFCL_VENV/bin/python" --no-deps "bfcl-eval==$BFCL_VERSION"
  # Everything bfcl-eval needs except sentence-transformers (only memory_vector uses
  # it, and it drags in torch), plus soundfile, which qwen-agent imports undeclared.
  "$BFCL_VENV/bin/python" - <<'EOF' > "$BFCL_VENV/bfcl-deps.txt"
import importlib.metadata as m
for r in m.requires("bfcl-eval"):
    if "extra ==" not in r and not r.startswith("sentence-transformers"):
        print(r)
print("soundfile")
EOF
  uv pip install --python "$BFCL_VENV/bin/python" -r "$BFCL_VENV/bfcl-deps.txt"
fi

if [ -e "$OUT/result" ]; then
  echo "$OUT already holds results; BFCL would silently reuse them. Use a fresh OUT." >&2
  exit 1
fi
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
cp "$SUBSET" "$OUT/test_case_ids_to_generate.json"
: > "$OUT/.env"  # BFCL loads $OUT/.env with override=True; keep it empty

export BFCL_PROJECT_ROOT="$OUT"
export OPENAI_BASE_URL="$BASE_URL"
export OPENAI_API_KEY="${OPENAI_API_KEY:-unused}"
export BFCL_MODEL_NAME="${BFCL_MODEL_NAME:-$(curl -sf "$BASE_URL/models" \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])')}"
export BFCL_MAX_TOKENS="${MAX_TOKENS:-4096}"
if [ "${THINKING:-}" = off ]; then
  export BFCL_EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":false}}'
fi
CATEGORIES="$(python3 -c 'import json,sys;print(",".join(k for k,v in json.load(open(sys.argv[1])).items() if v))' "$SUBSET")"

git -C "$HERE" rev-parse --short HEAD > "$OUT/client-rev.txt" 2>/dev/null || true
PY="$BFCL_VENV/bin/python"
start=$(date +%s)
"$PY" "$HERE/bfcl_selfhosted.py" generate --model local-qwen-FC --run-ids \
  --num-threads 1 --temperature 0
"$PY" "$HERE/bfcl_selfhosted.py" evaluate --model local-qwen-FC \
  --test-category "$CATEGORIES" --partial-eval
echo "wall $(( $(date +%s) - start ))s" | tee "$OUT/wall.txt"
python3 "$HERE/compare.py" "$OUT"
