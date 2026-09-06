# HOT host file-cache reclaim wall comparison

Run `bench/pi-decode-prefix-wall-driver.py --hot-host-cache-reclaim
--fixed-continuation --run-id astra-pi-agentic-<unique-name>
--output-dir /private/tmp/<unique-name>` from a clean linked worktree.
Add `--preflight` to check source identities and configuration without pausing
serving. The runtime comes from the source-only experiment in #53.

Both arms use the same pinned runtime revision and four native extensions.
Only `--moe-hot-host-cache retain|reclaim` changes. The reclaim policy advises
away complete host file pages of published GPU HOT experts. The weight files,
GPU values, routing and compute placement remain the same. Later reads can
fault the immutable source pages back.

The controller runs retain, reclaim, reclaim, retain, each with a fresh server,
one warmup and two measured three-turn conversations. The existing geometry
gate requires identical expert slots, HOT capacity, CPU threads, KV capacity,
activation precision and graph size. The host page cache is retained between
starts. Both arms disable decode snapshots and invasive diagnostics. The
explicit HOT census environment variable is empty, and PYTHONPATH contains
only the runtime Python directory, excluding the diagnostic import hook.
Do not use mincore or GPU weight hashing during the wall comparison.

Worker and system memory snapshots are sampled outside the client timer and
include warmup. Worker I/O also includes warmup. File RSS omits unmapped file
cache; snapshots cannot establish peak memory savings or attribute system
memory changes solely to this policy. Reclaimed bytes alone do not establish
better throughput.

Summarize with `bench/pi-agentic-runtime-summary.py <results-dir>
--hot-host-cache-reclaim`. Reporting a wall reduction requires identical
request bodies, answer bytes and token counts, including warmups. Retain both
orders, failures, first-request and continuation times. Passing this workload
does not establish broad quality equivalence.

The remote lease and restoration hook recover original serving on cancellation
or heartbeat loss. Check exclusive GPU ownership and recovery before timing;
review service errors and original serving after completion. Keep measured
artifacts private.
