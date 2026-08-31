# Spec: K=1 fused MTP speculation (FreeToken, patch 11-lite)

## Workspace

Given as a worktree on a branch off `feat/moe-disk-tier` (HEAD 3f3434d).
Commit on the branch; do NOT push.

## History (read before designing)

Full batched verify (K=3, patches 11b+11c) killed two workers on exact GDN
state rollback; do NOT attempt it. The 11a resident head is merged and
works (drafting is cheap, acceptance ~30-50% greedy). The sequential
decode-routed verify in engine/engine.py + spec_decode.py is correct but
break-even: each accepted token still costs a full extra decode step to
verify.

## Task: K=1 only, verify fused into the next step

At step t the target decodes token t normally and the resident MTP head
drafts t+1. Step t+1 runs ONE forward over TWO positions [t_accepted,
draft] using the decode-routed expert path (the machinery the sequential
verify already uses for its verify window), producing logits for both
positions:
- If the draft matches the target's argmax at its position: both tokens
  are emitted; net 2 tokens for ~1.3 steps of cost.
- If not: emit only the target's token; the draft position's state must
  be discarded EXACTLY (this is the one rollback in the design, bounded
  to a single trailing position).

GDN/mamba rollback for exactly one trailing position: prefer
recompute-over-rollback if the machinery makes that simpler (e.g. keep
step t's post-state checkpoint for the 36 GDN layers and restore it on
reject; a single-step state snapshot is O(state size), done every step,
so measure and report its cost). Bit-identical greedy outputs vs
non-speculative decode is the hard correctness bar; if any layer type
cannot roll back exactly, report the blocker precisely and stop rather
than approximating.

Flags: reuse --speculative-mtp on; --mtp-draft-tokens fixed at 1 for this
patch (reject >1 with a clear error). Stats: drafted/accepted/acceptance
in the decode log (fields exist), plus snapshot_us if the checkpoint
path is taken.

## Tests

GPU-free: bookkeeping for the fused two-position step, snapshot/restore
unit tests (synthetic state), flag validation. CUDA-gated: greedy
bit-equivalence vs non-speculative decode on the small test model.
Platform note: the Mac cannot run the package's tests (linux-only deps);
write them, state that plainly, never fake a pytest line.

## Deliverable

Commits on the branch + report: files, design chosen
(rollback vs recompute), expected speedup math at 30/50/70% acceptance,
deviations.
