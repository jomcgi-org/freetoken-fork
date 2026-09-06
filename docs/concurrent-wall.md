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
the 4090. The running HOT staging reader gate is separate and unchanged.

All eighteen targeted pure Python checks pass locally without a model or
network service. They cover actual overlapping workers, the concurrency
bound, refilling while an earlier response remains pending, error retention,
warmup separation, exact JSON checks, prose format versus semantics, and
workload timing that includes recording but excludes outer I/O snapshots.
This local pass is advisory. Linux validation and model wall-time comparison
remain pending.
