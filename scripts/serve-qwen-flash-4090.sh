#!/bin/bash
# Qualified capacity-one Qwen Flash profile for the RTX 4090 + CPU + disk tier.
# The disk prefix cache is on by default (500 GiB LRU under FREETOKEN_PREFIX_CACHE_DIR);
# set FREETOKEN_PREFIX_CACHE_GIB=0 to disable it. Keep the directory on local NVMe.
# Prefill chunk: 8192 tokens by default (FREETOKEN_PREFILL_CHUNK overrides). The
# 2026-09-22 matched curve on the deferred-workspace runtime cut 100k cold TTFT
# from 222 s to 89 s with exact parity and no cached-repeat or continuation
# regression. See docs/prefill-depth-benchmark.md for the data.
# HOT adaptation runs no idle ticks and one bounded decode tick after each prefill.
# Idle ticks re-aimed the HOT set at decode between requests, so the next prefill
# (every request starts with one) began at a 26% hot rate: live 100k cold ~108 s
# against ~92 s without them. The post-prefill tick keeps decode after a long
# prefill at the idle-tick level (2026-09-27 idlegap/livelike sweeps).
# Long prompts prefill layer by layer in 64k-token groups of 4096-token chunks
# (a prompt that fits one 8192-token chunk stays whole), DISK experts stage through
# 8 parallel readers with predicted rows, cold rows by direct I/O and resident rows
# from the page cache (docs/prefill-depth-benchmark.md). A group holds its residual
# (20 KiB per token) on the GPU: 64k groups need --memory-ratio 0.87, which leaves
# ~0.7 GB more activation headroom than the 0.9 default at the cost of that many
# expert slots; they OOM at 0.9. 100k cold ~22 s. Raise both on larger GPUs.
# Pinned (non-DISK) layers decode in hybrid mode: each layer fetches at most 2 missing
# experts per step over PCIe into the GPU LRU and the CPU W4A8 executor computes the
# rest in place from the pinned host bank, concurrently with the GPU's cached experts.
# Prefill is unchanged. PCIe gather fell from 7.4 to 3.5 ms per token and decode rose
# ~9% (2026-09-30 cpuinplace screens; docs/4090-performance.md). Caps 0/1 starve the
# LRU and 3/4 fetch too much.
# Non-expert projections and lm_head run as per-row FP8 (E4M3): 7.8 -> 3.9 GiB read per
# decode step, decode +14-24% and BFCL subset 38/54 vs 35/54 for BF16 (2026-10-01, #117).
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
export FREETOKEN_DISK_STAGING_WORKERS="${FREETOKEN_DISK_STAGING_WORKERS:-8}"
exec "${FREETOKEN_BIN:-ft}" serve \
  --model "$model_path" \
  --moe-backend hybrid --moe-hybrid-max-fetch 2 --moe-cache-auto \
  --max-running-requests 1 --linear-state-cache-ratio 4.0 \
  --max-extend-length "${FREETOKEN_PREFILL_CHUNK:-8192}" --max-seq-len-override 100352 \
  --host 127.0.0.1 --port 8090 \
  --moe-disk-prefill staged --moe-prefill-hot-split on \
  --moe-prefill-split-kernel grouped --moe-bank-hugepages off \
  --moe-cpu-threads 8 --ple-backend uring --enable-cache-report \
  --moe-disk-layer-profile "$layer_profile" \
  --moe-hot-expert-budget-gib 6 --moe-hot-adapt-interval-steps auto \
  --served-model-name qwen3.6-27b --moe-cpu-willneed recent \
  --moe-hot-adapt-prefill-weight 0.1 --moe-hot-adapt-histories split \
  --moe-hot-adapt-aim phase \
  --kv-disk-cache-dir "${FREETOKEN_PREFIX_CACHE_DIR:-/tmp/freetoken-prefix-cache}" \
  --kv-disk-cache-gib "${FREETOKEN_PREFIX_CACHE_GIB:-500}" --moe-hot-plan-persist off --cache-type radix \
  --kv-ladder off --kv-reserve-tokens 65536 --cuda-graph-max-bs 1 \
  --moe-disk-prefill-io cached --moe-hot-staging-io mmap \
  --moe-hot-host-cache reclaim \
  --moe-hot-adapt-idle-ms 0 --moe-hot-adapt-post-prefill-tick on \
  --prefill-layer-major-tokens 65536 --prefill-layer-major-chunk 4096 \
  --dense-weight-dtype "${FREETOKEN_DENSE_WEIGHT_DTYPE:-fp8}" \
  --memory-ratio 0.87 "$@"
