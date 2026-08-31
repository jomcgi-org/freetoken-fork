# Spec: prefill overlap under split residency (FreeToken, patch 12)

## Workspace

Given as a worktree on a branch off `feat/moe-disk-tier`. Commit on the
branch; do NOT push.

## Problem (measured)

With split bank residency (some layers pinned, some DISK) the engine logs
`split MoE bank residency: disabling prefill overlap (non-pinned layers use
synchronous pageable copies)` and turns prefill overlap off for EVERY
layer. On node-4 (4090, 30 pinned + 18 DISK layers) warm 441-token prefill
is ~54 tok/s; the pinned layers' prefill misses are copied synchronously
even though the double-buffer overlap machinery exists and works when all
layers are pinned.

## Task

Re-enable prefill overlap for the PINNED layers when residency is split:

1. The overlap double-buffer path (`prefill_bank_buffers`,
   `_init_prefill_overlap_buffers`, the copy stream) applies only to layers
   whose banks are pinned; DISK layers keep `--moe-disk-prefill cpu` (CPU
   executor prefill) and LOCKED/PAGEABLE keep the synchronous whole-layer
   pageable branch.
2. The incompatibility being guarded (`prefill_overlap` asserting no
   unpinned layers in set_bank_sources) becomes a per-layer decision, not a
   global one. Keep the hard error for configurations that would route an
   unpinned layer through the overlap path.
3. No behavior change when residency is uniform (all pinned: overlap as
   today; overlap explicitly disabled: unchanged).
4. Stats: log which layer count got overlap vs sync at boot.

## Correctness bar

Prefill outputs must be bit-identical to the current synchronous path
(same kernels, same data, different scheduling). The existing prefill
parity tests must pass unchanged; add a split-residency prefill test
(GPU-free where the machinery allows, CUDA-gated parity otherwise).

## Tests

GPU-free subset: zero new failures, exact pytest line reported.

## Deliverable

Commits on the branch + report: files touched, pytest line, expected
prefill gain rationale, deviations.
