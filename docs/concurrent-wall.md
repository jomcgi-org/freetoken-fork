# Concurrent complete-response qualification

The sustained reader gates measure one request at a time. The
[`concurrent-prefill-wall.py`](../bench/concurrent-prefill-wall.py) client
uses the same deterministic JSON and reference-based prose prompts to
measure a fixed workload with bounded client concurrency. It continuously
refills completed requests in manifest submission order. All warmup responses
finish before any measured request is submitted.

The primary metric is the elapsed time for the entire measured workload,
from first submission through final response recording and worker cleanup.
Summed request wall time is retained separately and is not aggregate elapsed
time. Each response retains its full text, usage, finish reason, streaming
latencies, client queue time, and offsets within the group. A transport error
is retained as a failed record; remaining requests still run. Checked rates
are omitted when any response fails completion or output checks. Prose
formatting is separate from semantic review and broad quality equivalence.

Whole-worker I/O is sampled only before and after each group, outside the
timer. It cannot be attributed to concurrent individual requests. There are
no per-request I/O snapshots or first-text diagnostics. The CLI refuses to
overwrite an earlier output, prompt manifest, or summary.

A model experiment must keep the runtime, reader policy, precision, prompts,
cache geometry, and server admission limit fixed while varying offered
concurrency. Record actual server arguments, native identities, startup
geometry, adaptation transitions, and retained state. Use both start orders,
disable diagnostic stats and GPU timing, and restore the original system
service after the comparison. Report aggregate wall time and individual
latency together. A throughput gain with slower individual responses is a
tradeoff, not a single-request speedup. This client introduces no serving
policy or model changes, and has not yet qualified concurrent serving on
the 4090. The HOT staging reader gate is separate and unchanged.

All eighteen targeted pure Python checks pass locally and on Linux without
a model or network service. They cover actual overlapping workers, the concurrency
bound, refilling while an earlier response remains pending, error retention,
warmup separation, exact JSON checks, prose format versus semantics, and
workload timing that includes recording but excludes outer I/O snapshots.
The [Linux validation record](../bench/results/4090-concurrent-client-validation-20260906.json)
retains the exact source hashes and command at `c0775ea`. It verifies that the
HOT reader wall-time driver finished successfully before testing and that
the original service remained active with the same PID. Model wall-time
comparison remains pending; these checks qualify the client protocol only.

The checked-in [model driver](../bench/concurrent-wall-driver.py) freezes
`c0775ea` and runs four starts with offered concurrency 1/4/4/1. Every start
uses the same four-request admission limit, CUDA graph capacity four, mmap
HOT staging, buffered GPU DISK prefill, and automatic HOT adaptation settings.
The existing KV ladder only supports admission capacity one, so this gate
explicitly disables it and reserves 65,536 KV tokens in both modes. This
matches the earlier pool floor, but the four-request state allocation may
change other startup budgets. Verify identical geometry within this gate;
do not combine its percentages with the earlier single-capacity results.

Each start has four complete warmups, twelve measured complete responses,
and eight fidelity cases run at that start's offered concurrency. Fidelity
backgrounds come from manifest order, independent of response completion
order. Whole-worker snapshots bracket each client group. Failed responses
remain in the record; a client output-check failure does not skip the
opposite mode when all scheduled records and group summaries are present.
Report every failure and semantic-review limitation before qualifying a gain.

Concurrent response windows overlap. Source-balanced early/late and per-kind
latencies may describe behavior, but their sums are not separate elapsed
workload times. Pair JSON and prose by prompt identity, compare both start
orders, and retain per-request latency alongside the full measured-group
duration. This does not qualify production prefix reuse, idle convergence,
very long contexts, or a single-request speedup from batching.

The driver checks actual CPU executor and graph capacity at startup, native
worker mappings, transport settings, and clean frozen sources. `--preflight`
is read-only. Launch its detached unit with a two-hour runtime limit and
the [recovery script](../bench/concurrent-wall-restore.sh) as `ExecStopPost`.
Each server start has a 45-minute limit. The normal finalizer restores the
original service and verifies a real completion. Observe the live driver
after a session interruption; never restart it merely because observation
timed out.

The driver and recovery script pass syntax checks. The
[Linux preflight](../bench/results/4090-concurrent-wall-preflight-20260906.json)
confirms identical frozen revisions and native identities, the fixed
four-request settings, and a driver hash matching the committed script.
It does not measure concurrent model performance.
