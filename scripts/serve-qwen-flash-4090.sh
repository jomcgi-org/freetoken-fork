#!/bin/bash
# Qualified capacity-one Qwen Flash profile for the RTX 4090 + CPU + disk tier.
set -euo pipefail
if (( $# < 2 )); then
  printf 'Usage: %s MODEL_PATH LAYER_PROFILE_JSON [extra ft serve arguments]\n' "$0" >&2
  exit 2
fi
model_path=$1
layer_profile=$2
shift 2
export FREETOKEN_DECODE_PREFIX_SNAPSHOT=0
export FREETOKEN_CONTINUATION_TRACE_DIR=
export FREETOKEN_HOT_HOST_CACHE_CENSUS_DIR=
export FREETOKEN_PREFILL_SELECTIVE_MAX_TOKENS=128
export FREETOKEN_PREFILL_HOT_OVERLAP=0
exec "${FREETOKEN_BIN:-ft}" serve \
  --model "$model_path" \
  --moe-backend offload --moe-cache-auto \
  --max-running-requests 1 --linear-state-cache-ratio 4.0 \
  --max-extend-length 2048 --max-seq-len-override 100352 \
  --host 127.0.0.1 --port 8090 \
  --moe-disk-prefill staged --moe-prefill-hot-split on \
  --moe-prefill-split-kernel grouped --moe-bank-hugepages off \
  --moe-cpu-threads 14 --ple-backend uring --enable-cache-report \
  --moe-disk-layer-profile "$layer_profile" \
  --moe-hot-expert-budget-gib 6 --moe-hot-adapt-interval-steps auto \
  --served-model-name qwen3.6-27b --moe-cpu-willneed recent \
  --moe-hot-adapt-prefill-weight 0.1 --moe-hot-adapt-histories split \
  --moe-hot-adapt-aim phase \
  --kv-disk-cache-dir "${FREETOKEN_PREFIX_CACHE_DIR:-/tmp/freetoken-prefix-cache}" \
  --kv-disk-cache-gib 0 --moe-hot-plan-persist off --cache-type radix \
  --kv-ladder off --kv-reserve-tokens 65536 --cuda-graph-max-bs 1 \
  --moe-disk-prefill-io buffered --moe-hot-staging-io mmap \
  --moe-hot-host-cache reclaim "$@"
