# Spec: expert-granular residency for DISK layers (FreeToken, patch 15)

## Workspace

Given as a worktree on a branch off `feat/moe-disk-tier` (HEAD 52d737a).
Commit on the branch; do NOT push. This is the largest remaining patch;
read the residency machinery end to end before writing code
(moe/host_banks.py, moe/offload_cache.py set_bank_sources + copy plans,
moe/cpu_executor.py, engine/engine.py residency planning, and the
--moe-disk-layer-profile flow).

## Problem (measured, node-4: 4090 24GB + 64GB RAM)

Residency is per-LAYER: a layer is either pinned (GPU slot-cache decode)
or DISK (CPU-executor decode). At the interactive optimum 14 layers are
DISK and their CPU compute dominates the 51ms step. But routing is
Zipf-ish: ~58 distinct experts/step from 512 per layer, with a stable hot
head (the layer profile and dedup stats prove recurrence). Pinning whole
layers wastes pin budget on cold experts while hot experts of DISK layers
pay CPU price every step.

## Task

Split residency WITHIN a DISK layer by expert:

1. New profile-driven partition: from a decode profile (extend the
   existing GET /v1/moe-layer-profile capture to per-expert hit counts;
   new file format section, versioned), mark the top-N experts of each
   DISK layer as HOT. N derives from a byte budget:
   --moe-hot-expert-budget-gib (default 0 = feature off, pure layer
   residency as today).
2. HOT experts' bank rows are copied into pinned host memory at load
   (they join the normal GPU movement paths: slot cache, fused copy
   plan). COLD experts stay file-backed. Routing for a DISK layer then
   splits per token-expert pair: HOT pairs go through the GPU slot-cache
   decode path, COLD pairs to the CPU executor, and the layer's output is
   the sum of both partial results (both paths already produce
   weighted partial outputs; follow how decode_target "hybrid" merges
   GPU and CPU partials - that machinery is the closest prior art and
   may be reusable wholesale).
3. The CUDA-graph handshake must stay sound: the CPU executor's doorbell
   task for the layer now covers only COLD pairs; the GPU side computes
   HOT pairs in-graph. If per-pair splitting inside the graph is not
   achievable with the existing hybrid machinery, implement the largest
   correct subset (e.g. whole-layer HOT promotion when a layer's hot set
   covers >X% of profiled routes) and report precisely what blocked the
   full split. Correctness over speed; greedy outputs bit-identical to
   the current path.
4. Stats: hot_pair_rate (share of routed pairs served by pinned HOT
   experts), per the disk stats line.
5. Pin-budget accounting: HOT expert bytes count against
   FREETOKEN_PIN_BUDGET_GB alongside whole-layer pins.

## Tests

GPU-free: partition planner (profile -> hot sets under budget), residency
bookkeeping, config validation, stats. CUDA-gated: parity test hot/cold
split output == pure CPU path. Platform note: the Mac cannot run the
package's tests; write them, state that plainly, never fake a pytest
line.

## Deliverable

Commits on the branch + report: files, design decisions (especially what
was reused from hybrid), the achievable-split statement, expected win
math from the measured Zipf (hot_pair_rate at 4/8/12 GiB budgets if
derivable from the profile format), deviations.
