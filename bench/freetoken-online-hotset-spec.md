# Spec: online hot-set adaptation (FreeToken, patch 16)

## Workspace

Given as a worktree on a branch off `feat/moe-disk-tier` (HEAD = tip with
expert-granular residency merged). Commit on the branch; do NOT push.
Study first: the hot-expert machinery just merged (engine
_resolve_hot_expert_sets/_plan_hot_experts/_load_hot_expert_profile,
offload_cache hot_sources/hot_row_for_expert, offload_kernels
ensure_experts_hot), and the live per-expert stats the collector keeps.

## Problem (measured, node-4)

Static profile-driven hot sets fit the capture workload: hot_pair_rate
72.5% on matching traffic falls to ~62% on diverse traffic, costing ~15%
throughput. Real serving drifts, so the hot partition should follow.

## Task

1. Track per-(layer, expert) route counts online with exponential decay
   (the collector already counts per-expert hits when
   --moe-collect-stats is on; add a decayed accumulator so old traffic
   ages out; decay half-life configurable,
   --moe-hot-adapt-halflife-steps, default ~2000 decode steps).
2. Periodically (every --moe-hot-adapt-interval-steps, default 1000,
   0 = feature off) recompute the hot partition from the decayed counts
   under the SAME byte budget, and swap: newly-hot experts' rows are
   copied from the file mapping into the pinned hot bank slots vacated by
   newly-cold experts. The swap must not stall decode:
   - Perform copies on a background thread/stream into a staging area,
     then flip the hot_row_for_expert mapping at a step boundary (the
     device-side table update must be graph-safe: fixed tensor updated
     in place, never reallocated).
   - Bound work per interval (--moe-hot-adapt-max-swap-gib per interval,
     default 0.5) so a drifting workload converges over a few intervals
     instead of thrashing.
   - An expert mid-swap serves from its OLD residency until the flip.
3. Correctness: outputs must be identical regardless of swap timing (an
   expert is served either as hot or cold, both paths are already
   parity-tested; the invariant is no torn mapping - a route must never
   see a hot row whose bytes are not fully installed).
4. Stats: extend the disk stats line with hot_swaps/interval and the
   CURRENT decayed hot_pair_rate. Log a rank0 line per adaptation tick.
5. Startup: with no profile file, start all-cold and let adaptation warm
   up (this removes the profile-capture step entirely); with a profile,
   seed from it as today.

## Tests

GPU-free: decayed-counter math, partition recompute under budget, swap
planner bounds, torn-mapping guard bookkeeping, flag validation.
CUDA-gated: parity under forced adaptation ticks. Platform note: the Mac
cannot run the package's tests (linux-only deps); write them, state that
plainly, never fake a pytest line.

## Deliverable

Commits on the branch + report: files, swap design (staging + flip
mechanics), convergence math for the default knobs, deviations.
