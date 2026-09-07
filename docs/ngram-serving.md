# Causal ngram target verification

`--speculative-ngram on` enables an experimental serving path for one greedy
Qwen Flash request. It requires TP size one, native MTP off, NVFP4 CPU MoE,
staged PLE, a page size of at least eight, and ordinary CUDA graph size one.
The default is off. Initial serving correctness checks pass, but the first
non-debug comparison regressed wall time on both repetition and multi-turn JSON
continuation workloads. This path is not selected for normal serving.

The proposer finds the most recent complete occurrence of the current eight-token
suffix within the last 8192 known tokens. It copies the four following known
tokens as candidates. A single target graph evaluates the seed and candidates
with ordinary per-token dense, GDN, PLE and QSA arithmetic. The target's greedy
predictions determine the accepted prefix and bonus token. Every rejected suffix
is discarded, and mutable state is restored to the retained prefix. Expert
selection and model precision are unchanged.

When fewer than two draft tokens match, the request waits 16 ordinary decoded
tokens before probing again. Consecutive weak proposals double that delay up to
256 tokens; a productive proposal resets it. The pause is local to the request,
and fallback still uses ordinary target decoding. EOS and tool cuts do not score
an otherwise matching proposal as weak. HOT adaptation counts every evaluated
target row, including candidates that verification later rejects.

The previously qualified graph, address bindings and checkpoints now live in
`freetoken.verification`; the diagnostic entry points import the same code.
Serving captures the target graph against allocated padding storage at startup,
restores borrowed storage afterward, and stages real request indices and addresses
for each replay. Ordinary fallback work retains scheduling overlap. A causal
lookahead uses the previous sampled token only after its host-copy fence, without
committing request state. A possible draft makes the scheduler drain prior output
before allocating the next batch, then recheck eligibility. EOS, stop strings and
new tool anchors can therefore prevent speculation. Each speculative batch drains
immediately after its forward and owns the graph until host output processing
completes. Only the ordinary scheduler allocates real request pages. The explicit
environment override for serial scheduling remains supported.

Before waiting for a pending host token, the scheduler checks whether the seven
known preceding tokens have a possible earlier continuation. With no such
occurrence, ordinary work can launch without that extra fence. This precheck
requires four known following tokens, since the pending token may itself be the
final draft. A possible match still requires the full eight-token lookup after
the copy fence; the proposal policy and target arithmetic are unchanged.

Requests without a full causal draft use ordinary decoding. Sampling, guided
decoding, multimodal inputs, incomplete lazy restores, insufficient output budget
and stale host history also fall back. EOS and new tool openers end a verification
window. Ordinary decoding then processes a tool opener until its prefix anchor
has been captured. If a stop string ends inside an accepted chunk, host output
processing restores the corresponding prefix before caching or freeing the
request and returns any unused pages.

Detokenization commits each request's token offsets before processing its next
token, including when several accepted tokens arrive together. Distinct requests
remain batched. This preserves serial text assembly, Unicode fragments, EOS and
stop-string holdback.

`--ngram-debug` logs per-window acceptance only when explicitly enabled. It adds
no activation copies or timing events. Wall-time qualification must run without
this flag and without the invasive diagnostic probes. Startup capture and scheduling
can affect practical latency, so component measurements do not predict
the serving result by themselves.

Runtime cache resizing currently rejects while the target graph is present;
restart with the desired geometry. Keep automatic KV growth disabled during the
initial serving experiments. Wider serving concurrency, runtime cache resizing,
and stronger non-debug wall evidence remain required before selecting this path
for normal use. Detailed measured records stay private.

The real-model and wall records below precede the known-prefix precheck. Its
renewed model and wall qualification remain pending.

Validation: 364 focused Linux checks passed, with the three exclusive CUDA checks
passing separately. Twelve serving fixtures matched ordinary decoding exactly in
content, reasoning output, finish reason and completion-token count. They exercise
repeated text, seven stop strings, three output budgets and a follow-up turn, with
real speculative windows and host stop rollback observed under the debug flag.
These fixtures establish the tested behavior, not broad quality equivalence.
The selective-overlap scheduler passed this renewed model qualification and
original serving recovered with a verified completion. Its separate non-debug
comparison also matched complete answers and token counts, with every
conversation passing independent checks. Repetition favored speculation, while
multi-turn continuation still favored ordinary decoding. This latest comparison
has one execution order, so it does not establish a general wall improvement.
Original serving recovered with a verified completion afterward.

The non-debug comparison used the same runtime source, native extensions, cache
geometry and request bodies with the ngram flag off and on. Each mode completed
three repetition requests and three three-turn conversations; the first repetition
and conversation were warm-ups. All complete answers and token counts matched.
The first comparison ran off before on and regressed on both workloads. After
the backoff and routed-token accounting changes passed renewed model checks,
the non-debug comparison was repeated in both execution orders on unchanged
source. All request bodies, complete answers and token counts matched across
both runs, and every conversation passed its independent checks.

Both orders with the preceding serial scheduler favor ngram verification on
repetition, but still favor ordinary
decode on the multi-turn workload. The larger remaining difference is on initial
turns. Blanket serial scheduling also affected ordinary fallback work, motivating
the selective-overlap change. This timing pattern does not by itself demonstrate
the cause of the difference or qualify the new scheduler.

Original serving recovered with a verified completion after both runs. The
counterbalanced comparison does not qualify a general serving improvement;
this mode remains off by default and unselected.

Code editing is a useful workload because full-file writes can preserve
long stretches of previously read source. Earlier successful leaderboard edits
contain substantial unchanged text, but final-file overlap is not a measured
draft acceptance rate. The prior task reports do not retain request transcripts
or original generated token IDs. Reasoning, tool serialization, intervening
context and the bounded lookup window can reduce usable matches. Any diagnostic follow-up must keep debug telemetry separate from normal wall
time. Final-file reuse alone does not establish a programming throughput gain.

A separate coding comparison ran the same two frozen leaderboard tasks in both
execution orders, with the original prompts, tools, budgets and graders. All
eight attempts passed the grader and permitted-file checks, with no failed model
calls. Speculation completed the chosen tasks sooner in both orders. It also
produced substantially fewer tokens, with fewer model calls overall and different
final edits. The reduced task wall time therefore does not isolate a general
engine throughput improvement. The smaller matched-work repetition gain remains
a separate result. This small task sample does not establish broad quality
equivalence. Debug telemetry was off, and original serving recovered with a
verified completion. Detailed records and independent audit scripts remain private.
