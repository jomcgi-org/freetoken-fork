# Spec: guided decoding for structured output (FreeToken, patch 19)

## Workspace

A worktree on a branch off `feat/moe-disk-tier` (current tip). Commit on
the branch; do NOT push. Study first: the sampling path (core.py /
engine sampling, is_greedy handling), the OpenAI adapter's request
parsing (server/api_server.py, adapters), and the existing tool-call
parser (qwen3_coder) which parses AFTER generation today.

## Problem

Agent workloads need valid JSON tool calls every time. FreeToken parses
tool calls post-hoc: a malformed generation fails the request. vLLM-class
engines constrain sampling so invalid tokens are unpickable.

## Task

1. OpenAI-compatible surface: support `response_format: {"type":
   "json_object"}` and `{"type": "json_schema", "json_schema": {...}}`
   on /v1/chat/completions, plus `tools` + `tool_choice: "required"`
   constraining the tool-call payload to the declared parameter schemas.
2. Engine: token-level constrained sampling via a grammar/FSM mask
   applied to logits before sampling. Use an existing library if it fits
   the dependency policy (llguidance or xgrammar or outlines-core;
   prefer one with prebuilt manylinux wheels and no torch version
   pinning; add as an OPTIONAL extra so base installs are unaffected),
   else a minimal JSON-mode FSM implemented directly (json_object mode
   only) is an acceptable reduced scope - state which you shipped.
3. The mask must compose with the CUDA-graph decode step: apply on the
   logits tensor the graph produces, host-side between replay and
   sampling if that is where sampling lives; do not break graph capture.
4. Constrained requests may be slower per token; unconstrained requests
   must be COMPLETELY unaffected (flag-gated per request, zero overhead
   when absent).
5. Thinking models: the constraint applies AFTER the reasoning segment
   (the qwen3 parser already splits reasoning from content; constrain
   content/tool segments only - if that split cannot be enforced
   mid-generation, document the limitation and constrain from the first
   token as reduced scope).
6. Stats: constrained_requests count and mask_us per interval.

## Tests

GPU-free: schema-to-FSM/grammar construction, mask correctness on
synthetic logits (invalid tokens masked), adapter parsing of
response_format/tool_choice, optional-dependency gating. CUDA-gated:
end-to-end json_object generation is valid JSON on the test model.
Platform note: the Mac cannot run the package's tests; write them, state
that plainly, never fake a pytest line.

## Deliverable

Commits + report: library choice + rationale, scope shipped vs reduced,
files, deviations.
