#!/usr/bin/env bash
# Round-2 bench: prefill vs decode, short vs long prompts, per config.
# Usage: CAP=64G LAYERS=auto BUDGET=52 PREFILL=cpu TAG=n4-cpuprefill MODEL=/mnt/nvme/flash.ftw OUT=/mnt/data/results2 bash bench2.sh
set -uo pipefail
CAP="${CAP:-}"; LAYERS="${LAYERS:-auto}"; BUDGET="${BUDGET:-52}"
PREFILL="${PREFILL:-cpu}"; TAG="${TAG:-bench2}"
MODEL="${MODEL:-/mnt/nvme/flash.ftw}"
OUT="${OUT:-/mnt/data/results2}"; PORT="${PORT:-8100}"; mkdir -p "$OUT"

SHORT="$OUT/p-short.txt"; LONG="$OUT/p-long.txt"
printf 'Count from one to ten.' > "$SHORT"
python3 -c "print('Summarize the history of the Linux page cache. ' * 40, end='')" > "$LONG"

payload() { # $1=prompt_file $2=max_tokens (MODEL_ID resolved after serve)
  python3 - "$1" "$2" "$MODEL_ID" <<'EOF'
import json, sys
print(json.dumps({"model": sys.argv[3], "prompt": open(sys.argv[1]).read(),
                  "max_tokens": int(sys.argv[2]), "temperature": 0}))
EOF
}

log() { echo "$@" | tee -a "$OUT/summary.txt"; }

# ---- serve ----
args=(--model "$MODEL" --moe-backend offload --moe-cache-auto \
      --max-running-requests 1 --port "$PORT" --moe-disk-prefill "$PREFILL")
[ "${PLE:-pinned}" != "pinned" ] && args+=(--ple-backend "${PLE}")
[ -n "${PROFILE:-}" ] && args+=(--moe-disk-layer-profile "${PROFILE}")
[ -n "${PLECACHE:-}" ] && args+=(--ple-cache-gib "${PLECACHE}")
[ "$LAYERS" != "auto" ] && [ "$LAYERS" != "0" ] && args+=(--moe-disk-layers "$LAYERS")
[ "$LAYERS" = "auto" ] && export FREETOKEN_PIN_BUDGET_GB="$BUDGET"
launch() {
  source /mnt/data/FreeToken/.venv/bin/activate
  export CUDA_HOME=/usr/local/cuda-13.0
  export PATH="$CUDA_HOME/bin:$PATH"
  local argstr; argstr=$(printf '%q ' "${args[@]}")
  local budget_env=""
  [ -n "${FREETOKEN_PIN_BUDGET_GB:-}" ] && budget_env="export FREETOKEN_PIN_BUDGET_GB=$FREETOKEN_PIN_BUDGET_GB && "
  local inner="source /mnt/data/FreeToken/.venv/bin/activate && export CUDA_HOME=/usr/local/cuda-13.0 && export PATH=\$CUDA_HOME/bin:\$PATH && ${budget_env}exec ft serve $argstr"
  if [ -n "$CAP" ]; then
    # activation must happen INSIDE the scope: sudo secure_path wipes the venv
    sudo systemd-run --scope --uid="$USER" -p MemoryMax="$CAP" -p MemorySwapMax=0 \
      --unit="ftb2-$$" bash -c "$inner" &> "$OUT/serve-$TAG.log" &
  else
    bash -c "$inner" &> "$OUT/serve-$TAG.log" &
  fi
}
launch
cleanup() { pkill -9 -f "ft serve --model $MODEL" 2>/dev/null; }
trap cleanup EXIT
until curl -sf --max-time 5 "localhost:$PORT/v1/models" >/dev/null; do
  pgrep -f "ft serve --model $MODEL" >/dev/null || { log "$TAG: SERVER DIED DURING BOOT"; exit 1; }
  sleep 5
done
MODEL_ID=$(curl -s --max-time 5 "localhost:$PORT/v1/models" | python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])")
SPID=$(pgrep -f "ft serve --model $MODEL" | head -1)
log "=== $TAG (layers=$LAYERS budget=$BUDGET cap=${CAP:-none} prefill=$PREFILL) ==="
# The API binds long before the engine is ready: retry the tiny completion until
# it succeeds, but bail if the server process dies (never stack orphans).
t0=$(date +%s)
until curl -sf --max-time 2400 "localhost:$PORT/v1/completions" \
    -H 'content-type: application/json' -d "$(payload "$SHORT" 2)" >/dev/null; do
  kill -0 "$SPID" 2>/dev/null || { log "$TAG: SERVER DIED IN WARMUP"; exit 1; }
  [ $(( $(date +%s) - t0 )) -gt 2700 ] && { log "$TAG: WARMUP TIMEOUT"; exit 1; }
  sleep 20
done
log "$TAG-warmup: $(( $(date +%s) - t0 ))s"

measure() { # $1=name $2=prompt_file $3=max_tokens
  local mf0 mf1 t0 t1
  mf0=$(awk '{print $12}' "/proc/$SPID/stat"); t0=$(date +%s.%N)
  curl -sf --max-time 2400 "localhost:$PORT/v1/completions" -H 'content-type: application/json' \
    -d "$(payload "$2" "$3")" > "$OUT/resp-$TAG-$1.json" || { log "$TAG-$1: FAILED"; return 1; }
  t1=$(date +%s.%N); mf1=$(awk '{print $12}' "/proc/$SPID/stat")
  python3 -c "
import json
r=json.load(open('$OUT/resp-$TAG-$1.json')); u=r.get('usage',{})
dt=$t1-$t0; ct=u.get('completion_tokens',0); pt=u.get('prompt_tokens',0)
print(f'$TAG-$1: prompt={pt} completion={ct} in {dt:.1f}s'
      + (f' = {ct/dt:.2f} decode tok/s' if ct>=64 else '')
      + (f' = {pt/dt:.1f} prefill tok/s' if ct<=2 and pt>=64 else '')
      + f', majflt_delta={$mf1-$mf0}')" | tee -a "$OUT/summary.txt"
}

sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
measure prefill-long-cold "$LONG" 1
measure prefill-long-warm "$LONG" 1
measure decode-short "$SHORT" 512
measure decode-short2 "$SHORT" 512
measure full-long "$LONG" 256
grep -o 'disk prefetch calls.*' "$OUT/serve-$TAG.log" | tail -2 >> "$OUT/summary.txt"
kill "$SPID" 2>/dev/null; sleep 5; pkill -f "ft serv[e]" 2>/dev/null
