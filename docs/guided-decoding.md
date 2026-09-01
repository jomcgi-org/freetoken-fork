# Guided decoding

Install the optional backend with:

```bash
pip install 'freetoken[guided]'
```

FreeToken uses XGrammar for token-level constrained decoding. The dependency is
optional and is not imported or initialized for ordinary requests.

`/v1/chat/completions` supports:

- `response_format: {"type": "json_object"}`
- `response_format: {"type": "json_schema", "json_schema": {"name": "...", "strict": true, "schema": {...}}}`
- function `tools` with `tool_choice: "required"`

JSON response formats compile through XGrammar's JSON Schema compiler. Required
tool calls compile through its model-specific structural-tag grammars, including
the declared function parameter schemas. The supported FreeToken parser styles
are `qwen3_coder`, `qwen`/`qwen25`, `llama3`, `gpt_oss`, `deepseekv32`, `glm47`,
and `minimax`. A required-tool request using another post-hoc parser is rejected
before generation.

For Qwen3-style thinking, required tool grammars include the reasoning segment
and constrain only the structured tool portion. Plain JSON response grammars are
activated after the reasoning closer (`</think>`, or `</mm:think>` for enabled
MiniMax-M3 thinking). Adaptive thinking formats without a deterministic closer
are constrained from the first generated token. Plain JSON response formats on
other reasoning parsers, including channel-based formats such as GPT-OSS, are
also constrained from the first token. Their reasoning/content transition is
not yet exposed to the engine as a deterministic token sequence.

The mask is applied to the logits after eager forward or CUDA graph replay and
before sampling. Constrained requests synchronize sampled token ids to advance
their host matcher and can therefore be slower. Unconstrained requests do not
load XGrammar, allocate masks, apply kernels, or add synchronization. Speculative
MTP verification is disabled per constrained request because its multi-token
verification path does not yet expose a mask at each draft position.

Client stop strings are rejected for constrained chat requests because they can
cut a JSON value before the grammar completes. As with other guided-decoding
engines, a `max_tokens` limit reached before grammar completion still returns a
length-truncated result and can therefore be incomplete; size the output budget
for the requested schema.

Scheduler decode interval logs include `constrained_requests` and `mask_us`.
`mask_us` is the sum of host mask construction and device mask transfer/application
time for the interval.
