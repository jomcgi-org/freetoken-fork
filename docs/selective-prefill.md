# Selective prefill transfers and diagnostic overhead

Short prefills can transfer only their selected experts from pinned RAM into
the existing GPU prefill buffers:

```sh
FREETOKEN_PREFILL_SELECTIVE_MAX_TOKENS=128 ft serve <normal serving arguments>
```

The default is `0`, which preserves full-layer streaming. Start with `128` on
the 4090: the measured gain is concentrated in very short prompts; `512` did
not consistently improve larger prompts. The limit applies to
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

The per-layer all-HOT decode classification requires either
`--moe-collect-stats` or `--moe-step-timing`. With both flags off, decode graph
capture omits its route reduction, scalar cast, and counter update. This count
only feeds diagnostics; native empty-task skipping and expert execution retain
their own functional checks. No separate wall-time gain is claimed for this
additional gate.

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
`0` for A and `512` for B to reproduce the exploratory measurements below.
Use `128` for a conservative deployment trial. Disable diagnostics in both
arms. For a fixed-work
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

## RTX 4090 results, 2026-09-05

The acceptance run used `--moe-collect-stats` off, the existing Qwen NVFP4
checkpoint, 20 pinned layers (26.44 GiB), 28 DISK layers, 2,296 protected rows,
3,753 expert slots, and 14 CPU threads. Each of four fresh servers ran the
same ten measured prefill requests in A/B/B/A order. All reported cached
prompt counts were zero. A is full copying; B uses the exploratory 512-token
limit. Times below are mean client TTFT, excluding warmups.

| Prompt tokens, including template | Fresh servers A / B (s) | Same server A / B (s) |
| ---: | ---: | ---: |
| 76 | 2.685 / 2.281 | 2.644 / 1.975 |
| 140 | 3.508 / 3.522 | 3.114 / 2.904 |
| 268 | 4.299 / 4.964 | 4.229 / 4.405 |
| 524 | 7.525 / 7.790 | 6.169 / 6.213 |
| 2,060 | 23.367 / 24.855 | 20.335 / 19.866 |

The robust result is the 76-token case: 15.0% lower TTFT across fresh servers.
The 524-token control uses the unchanged full-copy policy and illustrates the
noise between starts. Larger prompts did not show a consistent benefit; the
268-token case regressed. The conservative 128-token limit preserves the
measured 76-token execution path and excludes the uncertain 140/268-token
cases. It was not separately benchmarked as a complete serving configuration.

The additional same-server experiment ran three adjacent pairs per size,
with identical prompts and alternating policy order. It used the naive cache
manager to prevent prefix reuse, which changed the GPU allocation to 4,045
expert slots. Both policies shared that allocation. A benchmark-only wrapper
read an eight-byte shared mmap at each prefill boundary to set the limit;
there were no diagnostic counters, GPU timers, or profile polls. This measured
a 25.3% reduction for the 76-token case and confirmed that broader gains are
small or absent. Compare A with B within each column, not absolute times
between the two cache configurations.

All matched generated text and usage counts were identical. Decode measured
17.56 versus 18.32 tokens/s after the first token in the restart experiment;
the decode kernel is unchanged and this spread is insufficient to claim a
decode improvement. Telemetry overhead was not isolated as an independent
factor, so these results do not claim a separate speedup from its removal.

The initial diagnostic pair, all non-debug requests, experiment configuration,
and output hashes are retained in
[the measurement record](../bench/results/4090-selective-prefill-20260905.json).
The primary A/B/B/A run used revision `6ed99a4`; the paired run used `bd00a4a`.
The intervening runtime change only gates diagnostics during KV pool growth,
which did not occur in these runs.

Validation on node-4: 245 focused CUDA/CPU regression tests passed. After the
final KV growth diagnostic gate, all 31 KV ladder tests passed, including an
assertion that disabled diagnostics never read the GPU/PLE statistics. The
original serving checkout and configuration were restored and a real
completion verified before delivery.
