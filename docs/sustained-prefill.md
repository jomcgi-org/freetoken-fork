# Sustained prefill and decode qualification

The previous four-start combined comparison improved the fixed response mix
by 37.0%, but measured only four requests after two warmups per start. This
follow-up keeps a serving process and its HOT adaptation state alive through
four warmups and twelve measured responses before running fidelity checks.
It compares original CPU serving with the combined cached-staging configuration
in original/optimized/optimized/original order.

`bench/sustained-prefill-wall.py` prepares an identical prompt manifest before
each start's warmups. Every block contains one complete-JSON task and one
prose task; their order alternates between blocks. Six measured blocks make
it possible to compare early, middle, and late response times. Source excerpts
rotate among cache, adaptation, and staging code. JSON record values and
prompt nonces vary within a run but match between serving policies.

The same process retains its page cache, protected HOT assignments, and
decayed routing histories throughout the sequence. Automatic adaptation,
phase aim, split histories, and swap bounds remain unchanged. HOT plan
persistence is off to protect the production plan. KV reuse is off to keep
prefix hits from hiding prefill costs. There are no intentional idle gaps or
per-request policy toggles. This exercises continuous request traffic, not
idle-driven convergence or production prefix-cache reuse.

Diagnostic stats and GPU timing remain off. Client wall time includes the
residency hint and transfer-planning work. Optional whole-worker I/O snapshots
run outside the client timer. Cumulative counts in each record cover this
client's warmups and requests, excluding the server's startup completion.
Startup source revisions, actual native mappings, memory geometry, and the
real `file_io=cached` log are retained by the driver.

JSON has a 512-token budget and strict value, integer-type, key-order, and
multiplicity checks. Prose has a 1,024-token budget and must finish normally;
the client separately records compliance with the requested three-paragraph
format. Every prose prompt includes a reference specification covering routing,
checkpoint bytes, sparse ownership, transfer completion, and CPU/GPU numerical
differences. Formatting is not a semantic score. A subsequent review must
inspect factual coverage and contradictions against that specification, and
report uncertainty rather than inferring quality from response length.

Report whole-task wall time for all responses and separately for pairs meeting
the output checks. Retain failures, token-count differences, and every start's
outputs. Compare the early and late blocks, not just a grand mean. A gain in
this finite continuous workload does not establish optimal throughput for all
context lengths, concurrency levels, or production cache states.

Also compare blocks 0-2 with blocks 3-5. Each half contains all three source
files in the same proportions, unlike the narrower two-block early/middle/late
windows. This reduces source-mix confounding when inspecting time within a
run. Nonces and JSON values still differ, and all narrower windows remain in
the record. The analysis does not attribute the entire time trend to HOT
adaptation alone.

At `132b629`, the client's ten pure Python protocol checks pass locally and
on node-4's Linux environment. The [validation record](../bench/results/4090-sustained-client-validation-20260905.txt)
also confirms no inference-code changes from the previously validated
`78848ce` runtime. The model run and reference-based prose review are complete;
the results below retain the limits of both measurements.

The detached driver has a two-hour runtime limit and a recovery command that
stops its benchmark server and starts the original system service on exit.
The normal driver finalizer also waits for readiness and verifies a real
completion. Each model start has a separate 45-minute limit. The driver is
observed through its systemd unit and must not be restarted while active.

The driver finished successfully and is inactive. Its benchmark server is
inactive, the original service is active, and the normal finalizer verified
a real `OK` completion. Final systemd state and the completion are retained
in the [complete record](../bench/results/4090-sustained-cached-wall-20260905.json).

The sustained timing sequence at `132b629` is complete. Each policy has two
starts, four warmups per start, and twelve measured responses per start.
The aggregate below covers 24 responses per policy and excludes warmups.
It includes every measured response, including individual regressions.

| Client response | Original CPU | Combined cached staging | Less wall time |
| --- | ---: | ---: | ---: |
| Complete JSON, mean | 30.062 s | 24.475 s | 18.6% |
| Complete prose, mean | 22.812 s | 19.003 s | 16.7% |
| Fixed 24-request mix, total | 634.481 s | 521.738 s | 17.8% |

The fixed workload finishes about 1.22 times as quickly overall. The first
matched start pair improves by 16.8%; the reversed pair improves by 18.7%.
This longer result qualifies the earlier 37.0% short-sequence observation:
it does not support extrapolating that larger percentage to sustained traffic.

First-token time falls from 12.837 to 6.276 seconds for JSON and from 11.651
to 6.123 seconds for prose. Time after the first token rises from 17.225 to
18.199 seconds for JSON and from 11.161 to 12.879 seconds for prose. The
prefill improvement outweighs the generation penalty on this workload.
All measured JSON responses use 448 output tokens on average in both modes;
mean prose output is 253.417 versus 253.500 tokens. The raw pairs retain
individual answer-length and text differences.

The source-balanced first half improves by 23.1%, the second half by 11.4%.
The narrower final two-block window improves by 5.7%, but uses a different
source subset than the first two blocks. Whole-worker mean storage reads
fall from 3.126 to 1.145 GiB per response between the original's balanced
halves, versus 5.465 to 5.071 GiB for cached staging. These counters cover
all worker I/O; they cannot attribute reads to prefill, decode, or particular
expert rows. The cache-aware reader bypasses insertion of cold selected rows,
so insufficient admission of recurring useful weights is a candidate cause,
not an established explanation of the generation penalty. A sustained
comparison between buffered and cached GPU staging, holding the inference
runtime fixed, should isolate the reader before changing admission policy.
Any admission change must preserve the router's choices and contributions.

Twenty of 24 matched responses improve. All twelve JSON pairs have identical
text and token usage, and their combined wall time improves by 18.6%. Every
prose pair differs in text and token usage; all raw pairs are retained. All
32 JSON responses, including warmups, finish normally and pass strict value,
type, order, and multiplicity checks. All 32 prose responses finish normally
and have three paragraphs. All four starts score 7/8 on the eight fidelity
questions, with identical answers including the same code-trace failure
(`108`, expected `68`).

The assistant reviewed all 32 prose responses against the seven constraints
in each prompt's reference specification. Among measured responses, the
original explicitly covers 78 of 84 constraint instances and the optimized
mode covers 81 of 84. Five original responses and two optimized responses
have omissions. Review notes flag omissions or unsupported explanatory
connections in six original and eight optimized responses; neither group
contains an identified direct contradiction of a core constraint. These
coverage counts are an audit of this small reference-based task, not a
model-quality score or evidence of statistical equivalence. All eight warmup
prose responses cover all seven constraints. The record retains each review,
notes, coverage decisions, and the hash of its response. Formatting checks
are kept separate from semantic review.

The artifact validates all four prompt manifests, 64 main responses, native
identities and worker mappings, identical startup memory geometry, final
service restoration, and hash-matched coverage of all prose responses. It
also retains automatic adaptation's actual fill-completion and bandwidth
back-off logs. These transitions are observations under unchanged settings;
they are not fixed or replayed across the original and optimized processes.
The previous 99 focused CUDA checks and zero-error memcheck still cover the
unchanged inference runtime. No new inference code is introduced by this
results update. The PR remains draft because reader isolation, the generation
penalty, and broader serving and quality qualification remain open.

The client now supports an optional `--phase-io` diagnostic for follow-up
investigation. It snapshots the worker's I/O counters at the first nonempty
generated text, in addition to the before/after snapshots. Empty role events
and keepalives do not trigger it. Records set `diagnostic_phase_io=true` and
retain `first_text`, `before_first_text_delta`, and `after_first_text_delta`
inside `process_io`. The flag defaults off. Neither the completed comparison
above nor the running reader-isolation gate uses this newer diagnostic.

TTFT is recorded before the diagnostic snapshot, and its cost remains in total
client wall time and time after first text. The boundary includes network
delay, the first generated token, and possibly queued generation. Counters
cover all worker activity and are not atomic across reads. This is an
approximate client-observed split, not exact prefill/decode or expert-only
attribution. Use it to locate a cause, then verify any candidate with the flag
off. Reject diagnostic records from the non-debug wall-time summary.

All fifteen targeted pure Python client checks pass locally. They verify
default-off sampling, a single observation at first content or reasoning,
empty-event handling, visible observer cost, and identity changes at the
intermediate snapshot even when the outer identities match. Linux validation
of these added client checks is pending until the active model gate finishes.
No serving runtime or source excerpt is changed by the client diagnostic.
