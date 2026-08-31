#!/usr/bin/env bash
# Quality harness: fixed greedy prompts against a running server.
# The disk tier must be OUTPUT-NEUTRAL: temperature-0 completions from a
# disk-tier config must match the pinned baseline token-for-token.
# Usage: PORT=8180 TAG=n4-fast OUT=/mnt/data/results2 bash quality.sh
set -uo pipefail
PORT="${PORT:-8100}"; TAG="${TAG:-quality}"; OUT="${OUT:-/mnt/data/results2}"
QD="$OUT/quality"; mkdir -p "$QD"
MODEL_ID=$(curl -s --max-time 5 "localhost:$PORT/v1/models" | python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])")

run() { # $1=name $2=prompt $3=max_tokens
  curl -sf --max-time 1200 "localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' \
    -d "$(python3 -c "import json,sys;print(json.dumps({'model':'$MODEL_ID','messages':[{'role':'user','content':sys.argv[1]}],'max_tokens':int(sys.argv[2]),'temperature':0}))" "$2" "$3")" \
    | python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['message']['content'])" > "$QD/$TAG-$1.txt" \
    || echo "FAILED" > "$QD/$TAG-$1.txt"
}

run arith "Compute step by step, showing each partial sum: 17 + 28 + 45 + 96 + 133 = " 256
run recall "Remember this codeword: ZEPHYR-9142. Now count from one to twenty, then repeat the codeword exactly. " 256
run reason "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left? Think step by step. " 256
run longgen "Write a detailed technical explanation of how an operating system page cache interacts with mmap, covering read faults, writeback, and eviction. " 512

python3 - "$QD" "$TAG" <<'EOF'
import sys, os, re
qd, tag = sys.argv[1], sys.argv[2]
report = []
def read(name):
    with open(os.path.join(qd, f"{tag}-{name}.txt")) as f: return f.read()
a = read("arith");   report.append(("arith-319",   "319" in a))
r = read("recall");  report.append(("recall-codeword", "ZEPHYR-9142" in r))
s = read("reason");  report.append(("reason-9", "9" in s.split("=")[-1] if "=" in s else "9" in s[-200:]))
g = read("longgen"); words = g.split()
grams = [tuple(words[i:i+4]) for i in range(max(0, len(words)-3))]
ratio = (len(set(grams)) / len(grams)) if grams else 0
report.append(("longgen-distinct4>0.7", ratio > 0.7))
report.append(("longgen-length>300w", len(words) > 300))
for name, ok in report: print(f"{tag} {name}: {'PASS' if ok else 'FAIL'}")
EOF
