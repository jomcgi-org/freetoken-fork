#!/usr/bin/env bash
# One memory-capped bench config: forces real page-cache pressure on DISK layers.
# Usage: CAP=20G LAYERS=8 TAG=disk8-capped20 bash capped-run.sh
set -uo pipefail
CAP="${CAP:-20G}"; LAYERS="${LAYERS:-8}"; TAG="${TAG:-disk${LAYERS}-capped}"
MODEL="${MODEL:-/mnt/nvme/model.ftw}"
OUT="${OUT:-/mnt/data/results}"; PORT=8100
DECODE_TOKENS=512
PROMPT_FILE="$OUT/prompt.txt"

payload() {
  python3 - "$1" <<EOF
import json, sys
print(json.dumps({"model": "$MODEL_ID", "prompt": open("$PROMPT_FILE").read(),
                  "max_tokens": int(sys.argv[1]), "temperature": 0}))
EOF
}

echo "=== $TAG (layers=$LAYERS cap=$CAP) ===" | tee -a "$OUT/summary.txt"
DISKFLAG="--moe-disk-layers $LAYERS"; BUDGET_ENV=""
if [ "$LAYERS" = "auto" ]; then DISKFLAG=""; BUDGET_ENV="export FREETOKEN_PIN_BUDGET_GB=${BUDGET:-52} &&"; fi
sudo systemd-run --scope --uid="$USER" -p MemoryMax="$CAP" -p MemorySwapMax=0 --unit="ftbench-$$" \
  bash -c "source /mnt/data/FreeToken/.venv/bin/activate && export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:\$PATH && $BUDGET_ENV ft serve --model '$MODEL' --moe-backend offload --moe-cache-auto --max-running-requests 1 --port $PORT $DISKFLAG" \
  &> "$OUT/serve-$TAG.log" &
until curl -sf --max-time 5 "localhost:$PORT/v1/models" >/dev/null; do sleep 5; done
MODEL_ID=$(curl -s --max-time 5 "localhost:$PORT/v1/models" | python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])")
until curl -sf --max-time 120 "localhost:$PORT/v1/completions" -H 'content-type: application/json' -d "$(payload 2)" >/dev/null; do sleep 10; done
SPID=$(pgrep -f "[f]t serve --model $MODEL" | head -1)

sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
for run in cold warm1 warm2; do
  mf0=$(awk '{print $12}' "/proc/$SPID/stat"); t0=$(date +%s.%N)
  curl -sf --max-time 900 "localhost:$PORT/v1/completions" -H 'content-type: application/json' \
    -d "$(payload $DECODE_TOKENS)" > "$OUT/resp-$TAG-$run.json" || { echo "$TAG-$run: MEASURE FAILED" | tee -a "$OUT/summary.txt"; continue; }
  t1=$(date +%s.%N); mf1=$(awk '{print $12}' "/proc/$SPID/stat")
  python3 -c "
import json
r=json.load(open('$OUT/resp-$TAG-$run.json')); u=r.get('usage',{})
dt=$t1-$t0; toks=u.get('completion_tokens',$DECODE_TOKENS)
print(f'$TAG-$run: {toks} tok in {dt:.1f}s = {toks/dt:.2f} tok/s, majflt_delta={$mf1-$mf0}')" | tee -a "$OUT/summary.txt"
done
grep -o 'disk prefetch calls.*' "$OUT/serve-$TAG.log" | tail -3 >> "$OUT/summary.txt"
kill "$SPID" 2>/dev/null; sleep 5; pkill -f "[f]t serve" 2>/dev/null
