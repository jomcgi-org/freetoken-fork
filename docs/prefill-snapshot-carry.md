# Frozen state across short prefill chunks

Hybrid prefix caching stores a frozen recurrent state alongside its KV prefix.
FLA prefill can snapshot an internal 64-token boundary into a request's
ping-pong slots. Intermediate chunks do not insert into radix under overlap
scheduling: doing so would change ownership after the next chunk had already
inherited the prior cache handle.

Creating a continuation previously carried the state slots and next-write index
but reset the frozen boundary marker. If the final extend contained at most 64
tokens, it created no new internal snapshot. The final commit therefore had no
marked state to donate, even though the earlier frozen tensors still existed.

The continuation now carries that marker with the existing slots and index.
A later prefill snapshot replaces it normally. Otherwise, final prefill or
finish donates the earlier frozen state through the existing ownership path.
No intermediate radix insertion, extra slot, GPU copy, model computation,
quantization, routing change or telemetry is added. Fresh admissions still
start without a pending marker.

For example, a 128-token chunk followed by a seven-token tail can retain the
snapshot at token 64. It does not claim to retain token 128: that chunk's exact
end state lives in the mutable slot, which the tail advances.

Focused tests drive the real CPU prefill scheduler, FLA tracking metadata,
state pool and radix cache. They cover short tails, replacement by a later
snapshot, multiple chunks without a new boundary, finish before normal commit,
prefix-page ownership and bit-identical state restoration including PLE state.
Full-model timing and quality qualification remain separate requirements.
