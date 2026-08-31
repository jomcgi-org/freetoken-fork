#!/usr/bin/env bash
# Concurrent aggregate-throughput probe against an already-running server.
# Usage: PORT=8300 TAG=cached12 STREAMS=4 TOKENS=192 OUT=/mnt/data/results2 bash load.sh
set -uo pipefail
PORT="${PORT:-8100}"; TAG="${TAG:-load}"; STREAMS="${STREAMS:-4}"
TOKENS="${TOKENS:-192}"; OUT="${OUT:-/mnt/data/results2}"
MODEL_ID=$(curl -s --max-time 5 "localhost:$PORT/v1/models" | python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])")

one() { # $1=stream index
  curl -sf --max-time 2400 "localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' \
    -d "$(python3 -c "import json,sys;print(json.dumps({'model':'$MODEL_ID','messages':[{'role':'user','content':f'Stream {sys.argv[1]}: write a numbered list of facts about the Linux kernel, keep going.'}],'max_tokens':int(sys.argv[2]),'temperature':0}))" "$1" "$TOKENS")" \
    | python3 -c "import sys,json;print(json.load(sys.stdin).get('usage',{}).get('completion_tokens',0))"
}

t0=$(date +%s.%N)
pids=(); outs=()
for i in $(seq 1 "$STREAMS"); do o=$(mktemp); outs+=("$o"); one "$i" > "$o" & pids+=($!); done
for p in "${pids[@]}"; do wait "$p"; done
t1=$(date +%s.%N)
total=0; for o in "${outs[@]}"; do total=$((total + $(cat "$o" 2>/dev/null || echo 0))); rm -f "$o"; done
python3 -c "print(f'$TAG-load: {$total} tok across $STREAMS streams in {$t1-$t0:.1f}s = {$total/($t1-$t0):.2f} aggregate tok/s')" | tee -a "$OUT/summary.txt"
