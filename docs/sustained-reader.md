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

The four-start comparison completed successfully. Its driver and benchmark
server are inactive, and the original service was restored and verified with
a real `OK` completion. The [complete record](../bench/results/4090-sustained-reader-wall-20260905.json)
retains final systemd state, all outputs, sources, journals, and analysis.
The previous 99 focused
CUDA tests and zero-error memcheck cover both readers on this unchanged
inference runtime. The sustained client's ten protocol checks also remain
unchanged. The driver and recovery script pass syntax checks. A read-only
preflight on node-4 verifies the identical frozen revisions and native
binaries, diagnostics off, staged prefill in both modes, and the single
argument difference. Its SHA-256 matches the committed driver:
`b84a67d5b3c7e5025ebc16ae80f975048c72c31dbae9dedcc216b438a0d82852`.

The completed comparison used the frozen `132b629` client. A later optional
client diagnostic, described in [the sustained protocol](sustained-prefill.md),
can split whole-worker I/O around the first observed generated text. It is
not enabled in this wall-time gate. The current row-based reader sends an
entire partially resident expert row through direct I/O; after the isolated
comparison, inspect whether storage traffic comes from ongoing generation,
prefill rereading resident portions of those rows, or both. Existing whole
request counters cannot distinguish those cases.

The cache-aware reader loses this sustained comparison:

| Client response | Buffered staging | Cached staging | Cached wall-time increase |
| --- | ---: | ---: | ---: |
| Complete JSON, mean | 24.320 s | 24.701 s | 1.6% |
| Complete prose, mean | 17.221 s | 19.647 s | 14.1% |
| Fixed 24-request mix, total | 498.493 s | 532.174 s | 6.8% |

Cached staging is slower in both matched start orders, by 5.6% and 7.9%.
Eighteen of 24 matched requests regress. The source-balanced first halves
are effectively tied (284.677 versus 284.719 seconds); cached staging is
15.7% slower in the second halves (213.816 versus 247.455 seconds). The
narrower final two-block window is 21.0% slower, with its source-mix caveat.
The earlier short reader comparison's 18.3% gain does not extend to this
longer workload. Keep buffered staging for this sustained workload.

In the second halves, JSON first-token time averages 5.000 versus 6.415
seconds and prose 4.938 versus 6.365 seconds. Subsequent generation also
slows, from 16.179 to 17.246 seconds for JSON and from 9.520 to 11.217 seconds
for prose. Mean whole-worker storage reads fall from 3.857 to 1.055 GiB per
request across the buffered halves, compared with 7.569 to 5.679 GiB for
cached staging. The counters still do not isolate prefill, decode, or
partially resident rows. The first cached start also backs adaptation off
during its initial warmup after a 5.5-second, 0.50-GiB host-staging batch;
the first buffered start backs off later. These are downstream observations
under identical controller settings, not independently controlled causes.

All twelve measured JSON pairs have identical text and token usage. One
prose pair also has identical text. Those thirteen equal-text pairs still
take 2.3% longer with cached reads, and the fourteen equal-usage pairs take
2.8% longer. Mean prose output is 250.667 versus 249.417 tokens, so the
aggregate prose regression is not explained by longer cached responses.
All responses, failures, and length differences remain in the artifact.

All 32 JSON responses including warmups finish normally and pass strict
checks. All 32 prose responses finish normally in three paragraphs, and all
four starts score 7/8 on fidelity with the identical `108` code-trace error
(expected `68`). The reference-based prose review covers all 32 responses,
reusing earlier reviews only for verified identical text and reference
specifications. Measured prose explicitly covers 82 of 84 constraint instances
with buffered staging and 80 with cached staging. Both modes have omissions
and unsupported explanatory connections, with no identified direct core
contradictions. This small assistant audit is not a model-quality score or
statistical noninferiority evidence.

Reader isolation is complete for this workload. Cache admission, redundant
reads of resident bytes, and slow HOT staging remain candidates for further
investigation. None justifies a general cache-aware speedup claim or adding
percentages from different comparisons. The serving defaults are unchanged.
