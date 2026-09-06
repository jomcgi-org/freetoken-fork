# Aligned decode prefix snapshots

`FREETOKEN_DECODE_PREFIX_SNAPSHOT=1` enables an experimental hybrid-cache
optimization. The default is off. It applies to ordinary single-token decode
with cache pages larger than one token; speculative MTP leaves it disabled.

An unaligned response ending cannot donate its live recurrent state to an
earlier page boundary. The state includes the later tokens. Without an earlier
snapshot, the next exact continuation may have to replay generated tokens from
the preceding prefill checkpoint.

After a decode forward reaches a page boundary, this option copies the live
linear state into an existing request-owned ping-pong slot. A later boundary
replaces that pending snapshot. At completion, the normal hybrid-cache path
donates the frozen state and its matching KV pages. No extra state slots or KV
reservation are allocated. Exact token matching still decides whether a later
request can use that state.

Copies run on the engine stream after the forward and before the next step.
They include recurrent, convolution and every declared slot state, including
Qwen PLE convolution and integer token history. QSA compression groups close at cache-page boundaries and
their index rows follow KV-page ownership. The table-local partial-group ring
is not needed when starting at a complete group boundary.

When special-token checkpoints are enabled, snapshot advancement stops at the
tool-call opener. This retains a valid earlier boundary if the client rewrites
the call body. Multimodal requests do not publish reusable token-only prefixes.
Failed requests still use the existing discard path. Cache snapshots remain
evictable under the existing memory budget.

Publication also requires CPU token IDs through the full snapshot boundary.
An overlapped abort can leave those IDs behind consumed device state. Such a
state is released instead of being attached to an earlier, truncated cache key.

This is a state-retention change, with no routing or precision changes. Snapshot
copies consume GPU bandwidth and can affect scheduling. Qualification requires
exact state and continuation checks, then complete multi-turn task wall time
with token tracing and invasive telemetry disabled. No speedup is established
by the implementation alone.

The focused CPU tests exercise snapshot replacement, no additional slot use,
concurrent deduplication, eviction, edited histories, aborts and the first
decode's interaction with the previous prefill drain. Five CUDA checks pass:
the real GDN kernel resumes with identical output bits, QSA resumes on a new
request slot in both dense and sparse selection regimes, and QSA/PLE graph
replays retain their existing parity. Full-model integration and non-debug
wall-time qualification remain pending.
