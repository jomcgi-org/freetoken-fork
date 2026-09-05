# Sustained reader isolation on RTX 4090

The [combined sustained run](sustained-prefill.md) reduced total wall time by
17.8%, but generation after the first token slowed and storage reads stayed
higher than the original CPU server late in each sequence. This experiment
isolates the file reader before changing which cold rows enter the RAM cache.

The [driver](../bench/sustained-reader-wall-driver.py) runs on node-4 using
the frozen optimized runtime `132b629`. Both policies use staged GPU prefill,
published HOT reuse, CPU input reuse, parallel HOT routing, the same native
binary, and the same 64 MiB staging ring. Only `--moe-disk-prefill-io` differs:
`buffered` is recorded as `baseline`, `cached` as `optimized`. The labels
refer to the two file readers in this experiment. Serving defaults remain
unchanged.

Start order is buffered/cached/cached/buffered. Each start has four complete
warmup responses, twelve measured responses, and eight subsequent fidelity
questions. The unchanged sustained client prepares all prompts before
warmup, alternating JSON and prose within six measured blocks. Both halves
contain the same three source excerpts in equal proportions. JSON uses a
512-token ceiling and strict value/type/order/multiplicity checks. Prose
uses a 1,024-token ceiling and the same reference specification for review.

Automatic adaptation, phase aim, split histories, swap bounds, and memory
geometry remain the same. The process retains HOT assignments and histories
throughout its sequence. Host page cache is retained between starts. HOT
persistence and KV reuse are off. Diagnostics and GPU timing are off; client
wall time includes residency inspection and range planning. Whole-worker
I/O snapshots run outside the timer and cannot isolate prefill or decode.

Report every measured request, both matched start orders, JSON and prose
separately, equal-output and equal-usage subsets, and source-balanced halves.
Retain the narrower two-block early/middle/late windows with their source-mix
limitation. Compare first-token and subsequent generation time, and storage
reads. Complete prose needs reference-based review separate from formatting.
Reuse a previous review only for an identical response hash and identical
reference specification. Keep omissions, contradictions, and uncertain
explanations visible in both modes.

The driver checks clean source trees, identical native binary identity, worker
mappings, actual transport startup logs, and one serving process throughout
each start. It refuses to overwrite previous outputs. It has a two-hour
systemd limit and an exit recovery command that stops its own benchmark
server and starts the original service. Each model start has a 45-minute
limit. Normal completion also verifies a real response from the restored
original service. Observe the detached unit before continuing an interrupted
session; never restart it merely because observation timed out.

This is a finite continuous workload. It does not qualify idle convergence,
production prefix reuse, very long contexts, concurrency, or broad model
quality. No routing preferences, contributions, checkpoint bytes, or
quantization change. If buffered staging removes the generation penalty,
the next candidate should admit recurring useful weights without sacrificing
the short-run benefit of selective direct reads. If it does not, investigate
other differences in the combined stack before changing cache admission.

The four-start comparison is running in the detached systemd unit
`astra-sustained-reader-wall-driver`. Its first start has verified the actual
buffered transport and entered measured requests. Complete comparisons and
the final restoration check remain pending. The previous 99 focused
CUDA tests and zero-error memcheck cover both readers on this unchanged
inference runtime. The sustained client's ten protocol checks also remain
unchanged. The driver and recovery script pass syntax checks. A read-only
preflight on node-4 verifies the identical frozen revisions and native
binaries, diagnostics off, staged prefill in both modes, and the single
argument difference. Its SHA-256 matches the committed driver:
`b84a67d5b3c7e5025ebc16ae80f975048c72c31dbae9dedcc216b438a0d82852`.

The live comparison stays on the frozen `132b629` client. A later optional
client diagnostic, described in [the sustained protocol](sustained-prefill.md),
can split whole-worker I/O around the first observed generated text. It is
not enabled in this wall-time gate. The current row-based reader sends an
entire partially resident expert row through direct I/O; after the isolated
comparison, inspect whether storage traffic comes from ongoing generation,
prefill rereading resident portions of those rows, or both. Existing whole
request counters cannot distinguish those cases.
