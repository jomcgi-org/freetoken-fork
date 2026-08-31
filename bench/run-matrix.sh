#!/usr/bin/env bash
# Bench matrix: all-pinned baseline vs N disk layers, cold vs warm.
# Run from the FreeToken checkout with the venv active.
set -uo pipefail   # deliberately no -e: one failed measure must not kill the matrix

MODEL="${MODEL:-/mnt/nvme/model.ftw}"
OUT="${OUT:-/mnt/data/results}"; mkdir -p "$OUT"
PORT=8100
DECODE_TOKENS=512
PROMPT_FILE="$OUT/prompt.txt"
[ -f "$PROMPT_FILE" ] || python3 -c "print('Summarize the history of the Linux page cache. ' * 40)" > "$PROMPT_FILE"

payload() { # $1=max_tokens
  python3 - "$1" <<EOF
import json, sys
print(json.dumps({"model": "$MODEL_ID", "prompt": open("$PROMPT_FILE").read(),
                  "max_tokens": int(sys.argv[1]), "temperature": 0}))
EOF
}

serve() { # $1=disk_layers $2=pin_budget_gb (empty = uncapped) $3=tag
  local args=(--model "$MODEL" --moe-backend offload --moe-cache-auto \
              --max-running-requests 1 --port "$PORT")
  [ -n "$1" ] && [ "$1" != "0" ] && args+=(--moe-disk-layers "$1")
  [ -n "${2:-}" ] && export FREETOKEN_PIN_BUDGET_GB="$2" || unset FREETOKEN_PIN_BUDGET_GB
  ft serve "${args[@]}" &> "$OUT/serve-$3.log" & echo $! > "$OUT/serve.pid"
  # ready = the engine actually decodes, not just the port being bound
  until curl -sf --max-time 5 "localhost:$PORT/v1/models" >/dev/null; do sleep 5; done
  MODEL_ID=$(curl -s --max-time 5 "localhost:$PORT/v1/models" | python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])")
  until curl -sf --max-time 120 "localhost:$PORT/v1/completions" -H 'content-type: application/json' \
        -d "$(payload 2)" >/dev/null; do sleep 10; done
}

measure() { # $1=tag  — one decode pass; tok/s + major-fault delta
  local pid; pid=$(cat "$OUT/serve.pid")
  local mf0 mf1 t0 t1
  mf0=$(awk '{print $12}' "/proc/$pid/stat")
  t0=$(date +%s.%N)
  curl -sf --max-time 900 "localhost:$PORT/v1/completions" -H 'content-type: application/json' \
    -d "$(payload $DECODE_TOKENS)" > "$OUT/resp-$1.json" || { echo "$1: MEASURE FAILED" | tee -a "$OUT/summary.txt"; return 1; }
  t1=$(date +%s.%N)
  mf1=$(awk '{print $12}' "/proc/$pid/stat")
  python3 -c "
import json
r=json.load(open('$OUT/resp-$1.json')); u=r.get('usage',{})
dt=$t1-$t0; toks=u.get('completion_tokens',$DECODE_TOKENS)
print(f'$1: {toks} tok in {dt:.1f}s = {toks/dt:.2f} tok/s, majflt_delta={$mf1-$mf0}')" | tee -a "$OUT/summary.txt"
}

run_config() { # $1=disk_layers $2=pin_budget $3=tag
  echo "=== $3 (disk_layers=$1 pin_budget=${2:-none}) ===" | tee -a "$OUT/summary.txt"
  serve "$1" "${2:-}" "$3"
  sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null   # cold
  measure "$3-cold"
  measure "$3-warm1"; measure "$3-warm2"
  grep -o 'disk prefetch calls.*' "$OUT/serve-$3.log" | tail -3 >> "$OUT/summary.txt"
  kill "$(cat "$OUT/serve.pid")" 2>/dev/null; sleep 5
  pkill -f "ft serve" 2>/dev/null; sleep 3
}

run_config 0  ""  baseline-pinned
run_config 4  ""  disk4
run_config 8  ""  disk8
run_config 16 ""  disk16
echo "MATRIX COMPLETE" | tee -a "$OUT/summary.txt"
cat "$OUT/summary.txt"
