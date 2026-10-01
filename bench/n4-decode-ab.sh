#!/bin/bash
# Decode A/B on node-4: serve REF (a git ref fetched into wt-decode) with the production flags plus EXTRA,
# measure single-lane decode with the same prompts, restore the production unit. Usage: n4-decode-ab.sh REF TAG "EXTRA FLAGS"
#
# Hot-set A/B hygiene (issue #23). The production unit runs `--moe-hot-adapt-interval-steps auto`,
# which sits at 1000 routed tokens once a persisted plan seeds every layer, longer than an arm. So by default:
#   ADAPT_INTERVAL=150    appended to EXTRA (unless EXTRA sets its own interval); "keep" leaves the unit's value
#   PLAN_MODE=backup|cold backup keeps the production-adapted plan before the arm (default); cold removes it
#                         aside so the arm derives and adapts its own plan (restored after the arm either way)
#   HOTSET_KNOB=1|0       1 (default) marks the arm INVALID unless >= MIN_ADAPT_TICKS non-idle adapt ticks fired;
#                         set 0 for executor-mechanic knobs (empty-skip, barrier, threads) that do not need the adapter
#   MIN_ADAPT_TICKS=3     threshold for the check above
# Per-arm tick counts are always logged (adapt-ticks-check.py, summed from the arm journal).
REF=$1; TAG=$2; EXTRA=${3:-}
ADAPT_INTERVAL=${ADAPT_INTERVAL:-150}; PLAN_MODE=${PLAN_MODE:-backup}; HOTSET_KNOB=${HOTSET_KNOB:-1}; MIN_ADAPT_TICKS=${MIN_ADAPT_TICKS:-3}
case "$EXTRA" in *moe-hot-adapt-interval-steps*) ;; *) [ "$ADAPT_INTERVAL" = keep ] || EXTRA="$EXTRA --moe-hot-adapt-interval-steps $ADAPT_INTERVAL";; esac
CHECK=$(dirname "$(readlink -f "$0")")/adapt-ticks-check.py
R=/var/lib/longhorn/nvme-02/freetoken; WT=$R/wt-decode; SRC=$R/wt-plegather; M=$R/models; OUT=$R/results; mkdir -p $OUT
export CUDA_HOME=/usr/local/cuda-13.0 PATH=$SRC/.venv/bin:/usr/local/cuda-13.0/bin:/usr/bin:/bin TMPDIR=$R/tmp PYTHONPATH=$WT/python
log(){ echo "[$TAG $(date +%H:%M:%S)] $1" | tee -a $OUT/decode-ab.log; }
cd $WT && git fetch -q https://github.com/jomcgi/FreeToken.git "$REF:ab-$TAG" 2>/dev/null || git fetch -q https://github.com/jomcgi/FreeToken.git "$REF" && git checkout -q "ab-$TAG" 2>/dev/null || git checkout -q FETCH_HEAD
log "wt-decode at $(git log --oneline -1 | cut -c1-60)"
python setup.py build_ext --inplace > /tmp/ab-build-$TAG.log 2>&1 || { log "BUILD FAILED"; tail -c 400 /tmp/ab-build-$TAG.log >> $OUT/decode-ab.log; exit 1; }
CMD=$(sed -n "/^ExecStart=/,/[^\\\\]$/p" /etc/systemd/system/freetoken-serve.service | sed "s/^ExecStart=//" | tr -d "\\\\\n" | tr -s " " | sed "s|$SRC/.venv/bin/ft|ft|")
UNIT=decode-ab-$TAG
restore() { sudo -n systemctl stop $UNIT 2>/dev/null; for i in $(seq 1 30); do ss -ltn | grep -qE ":(8090|8091) " || break; sleep 2; done; sudo -n fuser -k 8090/tcp 8091/tcp 2>/dev/null; restore_plan; sudo -n systemctl start freetoken-serve; log "production unit restored: $(systemctl is-active freetoken-serve)"; }
PLAN=$M/flash-e2m1.ftw/freetoken_hot_plan.json; PLANBAK=/tmp/hot-plan-backup-$TAG.json; cp -p $PLAN $PLANBAK 2>/dev/null
restore_plan() { [ -f $PLANBAK ] && cp -p $PLANBAK $PLAN && log "hot plan restored from pre-arm backup"; }
[ "$PLAN_MODE" = cold ] && [ -f $PLANBAK ] && { rm -f $PLAN; log "PLAN_MODE=cold: hot plan removed (backup kept), arm starts from startup derivation"; }
trap restore EXIT
( sleep 900; log "WATCHDOG: 15 min elapsed, restoring"; kill -TERM $$ 2>/dev/null ) & WD=$!
sudo -n systemctl stop freetoken-serve; for i in $(seq 1 30); do ss -ltn | grep -qE ":(8090|8091) " || break; sleep 2; done
sudo -n systemd-run --uid=jomcgi --working-directory=$WT --setenv=CUDA_HOME=$CUDA_HOME --setenv=PATH=$PATH --setenv=TMPDIR=$TMPDIR --setenv=PYTHONPATH=$PYTHONPATH --setenv=HOME=/home/jomcgi ${AB_ENV:+--setenv=$AB_ENV} --unit=$UNIT $SRC/.venv/bin/$CMD $EXTRA
log "adapt interval=$ADAPT_INTERVAL plan_mode=$PLAN_MODE hotset_knob=$HOTSET_KNOB"
log "serve started extra='$EXTRA' env='${AB_ENV:-}'"
J=/tmp/decode-journal-$TAG.log; sudo -n journalctl -u $UNIT -o cat -f -n 0 > $J 2>&1 & JP=$!
READY=0; for i in $(seq 1 40); do RSP=$(curl -s -m 5 localhost:8090/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"qwen3.6-27b","messages":[{"role":"user","content":"hi"}],"max_tokens":1}' 2>/dev/null); if [ -n "$RSP" ] && ! echo "$RSP" | grep -qE "still loading|unavailable"; then READY=1; log "READY after $((i*10))s"; break; fi; systemctl is-active -q $UNIT || break; sleep 10; done
[ $READY = 1 ] || { log "NOT READY"; grep -iE "error|Traceback|refus" $J | grep -v starlette | tail -4 | cut -c1-220 | tee -a $OUT/decode-ab.log; sudo -n kill $JP; exit 1; }
log "startup: $(grep -oE "cold_fetch[^,]{0,60}|step timing[^\n]{0,80}|moe_cache_size=[0-9]+|hot_budget_gib=[0-9.]+|HOT plan source[^\n]{0,80}" $J | sort -u | head -5 | paste -sd'|' | cut -c1-400)"
gen() { T0=$(date +%s.%N); RSP=$(curl -s -m 300 localhost:8090/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"qwen3.6-27b\",\"messages\":[{\"role\":\"user\",\"content\":\"Run $RANDOM. $1\"}],\"max_tokens\":$2,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}"); N=$(echo "$RSP" | grep -oE '"completion_tokens":[0-9]+' | grep -oE '[0-9]+'); W=$(echo "$(date +%s.%N) - $T0" | bc); printf "%s: %s tok in %.1fs = %.2f tok/s\n" "$3" "$N" "$W" "$(echo "${N:-0} / $W" | bc -l)"; }
for i in 1 2 3; do gen "Warm-up $i. Write a detailed 400-word essay on the history of the Roman Republic." 300 warm-$i >/dev/null; done
for i in 1 2 3; do log "$(gen "Write a detailed 400-word essay on the history of the Roman Republic, part $i." 300 x1-$i)"; done
log "decode tok/s x1 (last 12 batches): $(grep -oE 'gen throughput \(token/s\): [0-9.]+' $J | grep -oE '[0-9.]+$' | tail -12 | paste -sd' ')"
log "stats: $(grep -oE 'hot_pair_rate: [0-9.]+%|cpu_experts: [0-9]+|disk major faults/decode step: [0-9.]+|minor faults/decode step: [0-9.]+|cpu_(head|wake|compute|signal|tail|groups|gil|precb|notify|coord|h2d|d2h)_us[^,]{0,14}|gpu_mid_us[^,]{0,12}|cpu_layers_per_step[^,]{0,10}|cold_fetched[^,]{0,22}|cold_cpu[^,]{0,22}|cold_fetch_bytes[^,]{0,22}|willneed_[a-z]+[^,]{0,14}|cpu_gpu_(in|out)_us[^,]{0,10}|pinned_(hot_pair_rate|missing/step|h2d_mb/step)[^,]{0,12}|all[-_]hot layers/step: [0-9.]+|hot_capacity[a-z_]*: [^,]{0,12}' $J | tail -26 | paste -sd' ' | cut -c1-1100)"
[ "$HOTSET_KNOB" = 1 ] && REQ="--require --min-ticks $MIN_ADAPT_TICKS" || REQ=""
log "$(python3 $CHECK $J $REQ)"
log "crash markers: $(grep -cE 'illegal memory|scheduler exited|Traceback \(most' $J)  dmesg uvm/nvrm: $(sudo -n dmesg -T 2>/dev/null | grep -ciE 'uvm|NVRM: Xid')"
sudo -n kill $JP 2>/dev/null
kill $WD 2>/dev/null
