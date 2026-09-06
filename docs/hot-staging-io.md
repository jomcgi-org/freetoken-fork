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
later requests. The reader-isolation run completed on frozen `132b629` and
did not contain this change.

The new tests cover exact byte transport for weight and scale dtypes, scalar
rows, source views, short reads, EOF, bounds, non-file fallback, cancellation
after a weight read but before its scales, and CUDA consumption before a
staging buffer is reused. At `878d723`, all 164 focused Linux/CUDA HOT staging,
adaptation, materialization, and DISK policy checks pass. All 25 new transport
checks also pass under CUDA memcheck with zero errors. The fifteen client
diagnostic protocol checks pass on Linux. The
[validation record](../bench/results/4090-hot-staging-io-validation-20260906.json)
includes exact commands, driver, output, successful unit completion, and a
real `OK` response from the restored original service. Model wall-time
qualification remains pending. No performance gain is claimed yet.

The next model comparison uses the checked-in
[`sustained-hot-staging-wall-driver.py`](../bench/sustained-hot-staging-wall-driver.py)
on the frozen, validated `878d723` runtime for both policies. Only
`--moe-hot-staging-io` changes, in mmap/buffered/buffered/mmap start order.
Both modes use buffered staged GPU DISK prefill, which won the sustained
DISK reader isolation. Each start retains automatic HOT adaptation through
four warmups, twelve measured complete JSON/prose responses, and eight
fidelity cases. Cache geometry, native binaries, prompts, and other settings
must match. Actual adaptation transitions remain observations under the
same controller settings.

Diagnostic stats, GPU timing, the client's `--phase-io` diagnostic, HOT plan
persistence, and KV reuse are off. Report all responses, both matched start
orders, equal-output subsets, and source-balanced early and late halves.
Review complete prose against the reference separately from formatting.
Retain storage counters and functional staging logs without interpreting
either as isolated disk bandwidth. A staging-time improvement alone does
not qualify a response wall-time gain.

The driver supports a read-only `--preflight` before service interruption.
Its two-hour systemd limit must be paired with
[`sustained-hot-staging-restore.sh`](../bench/sustained-hot-staging-restore.sh)
as `ExecStopPost`. The normal finalizer stops the benchmark service, restores
the original system service, and verifies a real completion. Each model
start has a 45-minute limit. Never restart a live driver after a session
interruption. This experiment does not qualify very long contexts,
concurrency, production prefix reuse, or broad model-quality equivalence.

The driver and recovery script pass syntax checks. The
[Linux preflight](../bench/results/4090-hot-staging-wall-preflight-20260906.json)
passes with identical frozen revisions and native binaries, buffered GPU
DISK prefill, and diagnostic stats disabled. Its driver SHA-256 matches the
committed script. Preflight does not measure model performance.

The tested reader also changes implicit readahead behavior. `HostBank`
requests `MADV_RANDOM` for ordinary-file mappings, whereas `HotRowFileReader`
at `878d723` opens ordinary descriptors without `posix_fadvise`. Linux 6.8
stores [madvise random behavior in VMA flags](https://github.com/torvalds/linux/blob/v6.8/mm/madvise.c#L984),
which the [mmap fault path](https://github.com/torvalds/linux/blob/v6.8/mm/filemap.c#L2908)
uses to suppress ordinary synchronous and asynchronous readahead. A new
descriptor does not inherit that mapping flag.

By contrast, [`POSIX_FADV_RANDOM`](https://github.com/torvalds/linux/blob/v6.8/mm/fadvise.c#L80)
sets state on the open file. Its
[synchronous read path](https://github.com/torvalds/linux/blob/v6.8/mm/readahead.c#L633)
uses forced reads for the requested range instead of normal readahead
prediction. This is not a cache-insertion ban or proof of zero extra I/O.
The distinction is a candidate explanation to investigate if the buffered
HOT reader loses. Whole-worker counters cannot establish readahead as the
cause, and the frozen wall-time experiment does not contain an advice change.
Any follow-up must preserve the same bytes and validate complete response
wall time with diagnostics off.
