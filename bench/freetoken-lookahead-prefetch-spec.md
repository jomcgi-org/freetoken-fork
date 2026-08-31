# Spec: one-step-ahead WILLNEED prefetch for DISK layers (FreeToken, patch 14)

## Workspace

Given as a worktree on a branch off `feat/moe-disk-tier` (HEAD 52d737a).
Commit on the branch; do NOT push.

## Problem (measured)

DISK-layer prefetch today is reactive: `prefetch_experts` fires after layer
L's routing lands in the pinned D2H buffer, inside the step, so the NVMe
read latency is only partially overlapped with that layer's compute.
Expert routing is temporally sticky across consecutive decode steps
(same-expert reuse is what makes the GPU LRU and page cache work at all).

## Task

1. Keep, per DISK layer, the previous decode step's deduped routed-expert
   set (the list `prefetch_experts` already computes). At the START of a
   decode step (before any layer runs; the seam that kicks the step's
   CUDA graph / first CPU submit is the right hook), issue the coalesced
   WILLNEED sweep for EVERY DISK layer using its previous-step set. When
   layer L's real routing arrives mid-step, prefetch only the DELTA
   (routed experts not in the predicted set) - the common case should be
   an empty or tiny delta.
2. Stats: extend the disk stats line with lookahead_hit_rate (predicted
   set coverage of actual routes) and delta_pages/step.
3. Flag: --moe-disk-lookahead {on,off}, default on for the madvise pager
   (no-op under uffd, whose prefetch path already covers this
   differently; state in the report if wiring it for uffd is trivial,
   but do not scope-creep).
4. First step / prefill boundaries: no previous set means fall back to
   the current reactive behavior. Prefill is untouched.

## Tests

GPU-free: predicted-set bookkeeping, delta computation, stats, flag
gating, fallback-on-first-step. Platform note: the Mac cannot run the
package's tests (linux-only deps); write them, state that plainly, never
fake a pytest line.

## Deliverable

Commits on the branch + report: files, expected latency-hiding rationale,
deviations.
