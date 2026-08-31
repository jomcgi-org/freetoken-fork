# Spec: batch-aware routed-expert dedup for CPU decode (FreeToken, patch 13)

## Workspace

Given as a worktree on a branch off `feat/moe-disk-tier` (now includes
prefill-overlap and gpufetch merges). Commit on the branch; do NOT push.

## Problem (measured)

At x8 concurrency on node-4 (4090 + 64GB, 18 DISK layers on the CPU
executor), aggregate is 33.8 tok/s. Each decode step routes
8 tokens x top_k 10 = up to 80 (token, expert) pairs per DISK layer. The
CPU executor computes each pair independently; when multiple tokens in the
batch route to the SAME expert, the expert's weights are re-read (and the
WILLNEED sweep re-requests the same pages) once per token instead of once
per step. With 512 experts and 80 draws, expected distinct experts is
~74 at uniform routing, but real routing is Zipf-ish: popular experts
recur, and the recurrence grows with batch size, which is exactly where
we want aggregate throughput.

## Task

1. In the CPU MoE executor's decode path (`cpu_moe_ext.cpp` MoeTask work
   loop and its Python driver in `moe/cpu_executor.py`): group the step's
   (token, expert) pairs by expert so each distinct expert's weight rows
   are read once and applied to all tokens routed to it (a small GEMM per
   expert over its token group, instead of per-token GEMV). Preserve
   output semantics exactly: per-token accumulation order may change only
   within the existing floating-point tolerance the current worker tiling
   already accepts; greedy decode outputs must be bit-identical if the
   current path is bit-deterministic, otherwise match its existing
   determinism contract. State which contract holds in the report.
2. The WILLNEED prefetch sweep already page-dedupes; confirm the routing
   D2H it consumes can carry the deduped expert list, and dedupe there too
   if it does not already.
3. Extend the disk stats line with distinct_experts/step and
   dedup_ratio (pairs / distinct).
4. No change to GPU-layer decode, prefill, or the gpufetch path.

## Tests

GPU-free: dedup grouping unit tests (synthetic routing with heavy
recurrence: outputs equal the ungrouped reference within the stated
contract; stats correct). The Mac cannot run the package's tests (linux
deps): write them, do not fake a pytest line. CUDA-gated parity: existing
CPU-executor parity tests must still pass unchanged.

## Deliverable

Commits on the branch + report: files, expected win rationale (bytes
read/step before vs after at x8), determinism contract statement,
deviations.
