# Selective prefill transfers and diagnostic overhead

Short prefills can transfer only their selected experts from pinned RAM into
the existing GPU prefill buffers:

```sh
FREETOKEN_PREFILL_SELECTIVE_MAX_TOKENS=512 ft serve <normal serving arguments>
```

The default is `0`, which preserves full-layer streaming. The limit applies to
the actual prefill chunk, including chat-template tokens. Larger chunks retain
the existing lookahead copies. A small remainder after a large chunk can use
selective copies.

Each eligible layer computes the actual routed expert union. Unions occupying
at least 75% of the layer use a full copy. Otherwise, contiguous expert runs
share DMA descriptors. Small scale banks still copy in full to avoid the CUDA
driver's synchronous fallback for batches mixing tiny and large transfers.

This requires pinned overlap buffers, an aligned bank copy plan, and CUDA 13
batch memcpy. An unavailable batch API falls back to full copies. Existing
`--moe-prefill-hit-d2d` takes precedence. Selective chunks wait for the actual
route decision, so they forgo speculative next-layer copies; this tradeoff must
be measured on the target machine and prompt distribution.

Expert IDs, routing weights, packed weights, scales, activation precision, and
grouped GEMM execution stay the same. Buffer release/ready events retain their
existing ownership rules. Unselected rows are never substituted for selected
experts.

## Diagnostics

`--moe-collect-stats` enables diagnostic route counters, HOT/COLD prefill
summaries, transfer counters, and periodic MoE/PLE statistics. These diagnostics
can add GPU reductions, host reads, and CPU work. Omit the flag for performance
acceptance runs. `--moe-step-timing` remains a separate opt-in for detailed
CPU/GPU timing.

Disabling diagnostics preserves the histories used by HOT adaptation, session
prefetch, and the CPU WILLNEED fault guard. Basic request progress and client
usage reporting remain available. `--enable-cache-report` adds cached-token
usage fields; it does not enable the expensive MoE diagnostics. An explicit
`--ple-cache-profile-out` still requests PLE profile collection.

`decode_miss_stats()` exposes cumulative pinned-overlap transfer bytes and
selective layer/row counts when collection is enabled. These counters exclude
pageable copies, DISK CPU compute, and decode transfers; they reset with the
cache statistics or a cache rebuild.

## Reproducing the wall-time comparison

Use fresh servers in A/B/B/A order, with identical model, RAM/VRAM budgets,
expert placement, prompt order, and concurrency. Set the environment limit to
`0` for A and `512` for B. Disable diagnostics in both arms. For a fixed-work
comparison, use `--moe-hot-adapt-interval-steps 0 --moe-hot-plan-persist off` with
the same static expert profile, and `--kv-disk-cache-gib 0` to prevent disk
prefix reuse. Those are benchmark controls, not production recommendations.

Run the same client in every arm:

```sh
python bench/selective-prefill.py \
  --base-url http://127.0.0.1:18090 \
  --tokenizer /path/to/flash-e2m1.ftw \
  --output /tmp/arm.jsonl --repeats 2
```

The client uses a fixed source excerpt and deterministic per-request nonces.
Preserve the benchmark source revision across arms. Exclude warmup requests,
verify prompt/completion counts and generated text, and compare client TTFT.
Decode throughput uses generated tokens after the first token divided by the
remaining stream wall time. Inspect server logs to confirm zero cached prompt
tokens and matching pinned-layer, HOT-slot, KV, and CPU-thread geometry.

The synthetic NVFP4 tests use Qwen's H=2560/I=640 expert dimensions and compare
outputs bit for bit, including poisoned unused rows, sparse/dense unions,
buffer reuse, and unavailable-API fallback. Separate CPU and CUDA tests verify
that disabling diagnostic counters preserves both routes and adaptation
histories.
