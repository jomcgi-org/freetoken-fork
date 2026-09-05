# Buffered reads for HOT adaptation staging

The sustained reader-isolation run exposed a separate staging cost. In the
first cached start, the first full-response warmup triggers a single
193-expert HOT swap batch. Its log reports 5,513.7 ms to stage 0.50 GiB and
backs the adaptation interval off from 1,000 to 2,000 routed tokens. This is
one observation under concurrent serving, not a storage-bandwidth benchmark.
The matching buffered start backs off later in its sequence. The source
records are `astra-sustained-reader-wall-r1-journal.log` and
`astra-sustained-reader-wall-r2-journal.log` on node-4.

`OffloadMoeCache._stage_hot_rows()` starts its timer after the initial GPU
readiness event. For a single batch, it copies each incoming expert's banks
from file-mapped tensors into existing pinned staging rows. GPU installation
occurs later on the publication stream. Those host copies can incur file
faults while the CPU decoder runs. The observed timer does not by itself
separate faults, CPU scheduling, and memory copying.

The experimental `--moe-hot-staging-io buffered` option reads native NVFP4
ordinary-file rows directly into those same staging tensors with `preadv`.
It reuses descriptors within a batch, handles partial reads, and raises on
EOF. It preserves source-view offsets, row order, every packed weight and
scale byte, and per-expert scalar scales. RAM, UFFD, and tmpfs sources retain
their tensor-copy path. Unsupported bank layouts are rejected. The default
remains `mmap`.

The option adds no pinned weight allocation. Cancellation still occurs only
between whole experts, after every bank for the completed prefix has landed.
Existing readiness waits, H2D completion fences, retired HOT ownership, and
publication are unchanged. The router, HOT swap planner, cadence controller,
swap bounds, and arithmetic are unchanged. Faster staging may naturally
change the controller's measured backoff and the time at which a valid cache
assignment becomes available; the benchmark must retain those observations.

No per-row diagnostic timers or GPU readbacks are added. The existing
functional staging timer still feeds automatic backoff, and one startup log
identifies the configured reader. A throughput claim requires complete client
wall time with diagnostics off and both matched orders, including warmup and
later requests. The current live reader-isolation run stays frozen and does
not contain this change.

The new tests cover exact byte transport for weight and scale dtypes, scalar
rows, source views, short reads, EOF, bounds, non-file fallback, cancellation
after a weight read but before its scales, and CUDA consumption before a
staging buffer is reused. Syntax checks pass. Linux and CUDA validation,
memcheck, and model wall-time qualification are pending. No performance gain
is claimed yet.
