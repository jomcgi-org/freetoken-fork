# Spec: session-conditioned expert prefetch (FreeToken, patch 21)

## Workspace

A worktree on a branch off `feat/moe-disk-tier` (current tip). Commit on
the branch; do NOT push. Study first: the disk prefix store
(kvcache/disk_prefix_cache.py + its scheduler/cache.py integration), the
expert-granular residency machinery (hot_row_for_expert, ensure_experts
paths, the adaptation loop's decayed counters), and the WILLNEED prefetch
seam in moe/cpu_executor.py.

## Problem (measured)

A resumed session gets its KV/GDN state back in seconds, but its EXPERT
working set re-warms from scratch: decode ramps from <1 to ~20 tok/s over
the first ~100 tokens because the slot cache and page cache hold the
previous traffic's experts, not this session's. The information to fix
this exists and is discarded: which experts a session routed to.

## Task

1. **Capture**: track a per-session routed-expert profile during decode
   (per layer, the top-K experts by decayed count for that session; keep
   it cheap - a few KB per session, updated from data the collector
   already sees). On session park (prefix-store write), serialize it into
   the store entry (versioned field; older entries without it restore as
   today).
2. **Restore prefetch**: on prefix restore (and on VRAM radix hits that
   carry a stored profile), before or overlapping the first forward:
   - hot/pinned-tier experts in the profile: promote into the GPU slot
     cache (the normal H2D machinery, batched);
   - cold disk-tier experts: issue the coalesced WILLNEED sweep for
     their pages;
   - feed the profile into the adaptation counters (weighted injection)
     so the global hot set leans toward live sessions.
3. **Multi-lane semantics**:
   - Prefetch fires at ADMISSION (request enters the queue), so waiting
     time behind other lanes becomes warm-up time.
   - Session-aware eviction protection: experts in LIVE sessions'
     profiles get an eviction boost/protection in the slot-cache LRU,
     bounded (protect at most --session-protect-experts per live
     session, default 64) so transient traffic cannot thrash active
     conversations. Protection releases when the session parks.
4. **Strictly advisory**: prefetch and protection must never change
   outputs, never run inside CUDA-graph capture, and degrade to today's
   behavior when profiles are absent. Flag: --session-expert-prefetch
   {on,off}, default on.
5. **Stats**: resume_prefetch_experts, post-resume warm-up rate (decode
   tok/s over the first 64 steps after a restore vs the session's
   steady rate), protected_experts count on the disk stats line.
6. **Scorecard**: extend bench/realworld.py's agent-resume leg to report
   the first-64-token decode rate after the post-restart resume, so the
   ramp improvement is measured by the standing acceptance gate.

## Tests

GPU-free: profile capture/serialize/deserialize round-trip (including
versioned absence), admission-time prefetch planning, protection
bounding and release, flag gating. CUDA-gated: advisory-invariance (same
outputs with prefetch on/off). Platform note: the Mac cannot run the
package's tests (linux-only deps); write them, state that plainly, never
fake a pytest line.

## Deliverable

Commits + report: files, profile format + size math, where the
admission-time hook landed, deviations.
