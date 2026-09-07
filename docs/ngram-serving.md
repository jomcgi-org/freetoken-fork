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
for each replay. The scheduler runs without overlap while this mode is enabled
and owns the graph until host output processing completes. Only the ordinary
scheduler allocates real request pages.

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
this flag and without the invasive diagnostic probes. Startup capture and serial
scheduling can affect practical latency, so component measurements do not predict
the serving result by themselves.

Runtime cache resizing currently rejects while the target graph is present;
restart with the desired geometry. Keep automatic KV growth disabled during the
initial serving experiments. Wider serving concurrency, runtime cache resizing,
actual task completion and improved non-debug wall measurements remain required before
selecting this path for normal use. Detailed measured records stay private.

Initial implementation validation: 328 focused Linux checks passed, with the three exclusive CUDA checks
passing separately. Twelve serving fixtures matched ordinary decoding exactly in
content, reasoning output, finish reason and completion-token count. They exercise
repeated text, seven stop strings, three output budgets and a follow-up turn, with
real speculative windows and host stop rollback observed under the debug flag.
These fixtures establish the tested behavior, not broad quality equivalence.

The non-debug comparison used the same runtime source, native extensions, cache
geometry and request bodies with the ngram flag off and on. Each mode completed
three repetition requests and three three-turn conversations; the first repetition
and conversation were warm-ups. All complete answers and token counts matched.
The first comparison ran off before on. Scheduling overhead and discarded draft
work still need investigation; component gains did not translate into serving gains.
The subsequent backoff and routed-token accounting changes require renewed model
and non-debug wall qualification before making any performance claim.
