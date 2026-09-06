# Causal ngram target verification

`--speculative-ngram on` enables an experimental serving path for one greedy
Qwen Flash request. It requires TP size one, native MTP off, NVFP4 CPU MoE,
staged PLE, a page size of at least eight, and ordinary CUDA graph size one.
The default is off. This is awaiting full serving qualification and makes no
new wall-time claim.

The proposer finds the most recent complete occurrence of the current eight-token
suffix within the last 8192 known tokens. It copies the four following known
tokens as candidates. A single target graph evaluates the seed and candidates
with ordinary per-token dense, GDN, PLE and QSA arithmetic. The target's greedy
predictions determine the accepted prefix and bonus token. Every rejected suffix
is discarded, and mutable state is restored to the retained prefix. Expert
selection and model precision are unchanged.

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

`--ngram-debug` logs per-window acceptance only when explicitly enabled. It adds
no activation copies or timing events. Wall-time qualification must run without
this flag and without the invasive diagnostic probes. Startup capture and serial
scheduling can affect practical latency, so component measurements do not predict
the serving result by themselves.

Runtime cache resizing currently rejects while the target graph is present;
restart with the desired geometry. Keep automatic KV growth disabled during the
initial serving experiments. Wider serving concurrency, runtime cache resizing,
actual task completion and non-debug wall measurements remain required before
selecting this path for normal use. Detailed measured records stay private.
