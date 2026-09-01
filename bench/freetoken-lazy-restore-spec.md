# Spec: lazy session restore (FreeToken, patch 22) - QUEUED behind patch 21 + OOM guard + bench

Firecracker-style resume-before-load for the disk prefix store.

## Problem (measured)

A session restore loads its full KV from NVMe before the first token
(~13.8s at ~100k context). But the model's attention is QSA (sparse,
block top-k): a decode step touches only a few KV blocks. Loading
everything up front is Firecracker's pre-snapshot-restore world; the fix
is theirs too.

## Task

1. On restore: install GDN/conv states eagerly (small, always needed),
   but map the KV pages lazily - either UFFD-backed (reuse the
   moe/uffd_pager machinery, repointed at KV page granularity) or an
   explicit block-presence bitmap checked by the QSA gather with an
   on-miss synchronous pread (choose whichever fits the attention
   backend's read path; state the choice).
2. Decode starts as soon as states + the newest KV blocks (the attention
   sink / recent window QSA always touches) are resident. Background
   reader streams the remaining blocks in priority order (most recent
   first); the top-k selection naturally hides most of the tail.
3. Correctness: a block must never be read before its bytes are fully
   installed (same no-torn rule as the hot-swap machinery); outputs
   bit-identical to eager restore.
4. Stats: restore_eager_ms (states + hot blocks), blocks_faulted vs
   blocks_streamed, first-token-after-restore ms (extend the scorecard's
   resume leg).
5. Flag --lazy-restore {on,off}, default on; degrades to eager when the
   store entry predates the block index.

## Follow-on (patch 23, only if agent fleets materialize): CoW store -
common system-prompt prefixes stored once as read-only base entries,
sessions store deltas; spawning the Nth agent with a standard context
becomes metadata. Radix-on-disk.

## Tests

GPU-free: block bitmap/index round-trip, priority-order streaming plan,
no-torn bookkeeping, flag/versioning. CUDA-gated: bit-identical outputs
lazy vs eager. Mac cannot run package tests; write them, say so, never
fake a pytest line.
