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

The client's ten pure Python protocol checks pass locally. Linux validation,
the model run, and full prose review remain pending. The runner must restore
the original service and verify a real completion after the model gate.
