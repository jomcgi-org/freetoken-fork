# Spec: MTP batched verify + persistent draft KV (FreeToken, patches 11b+11c)

## Workspace

Given as a worktree on a branch off `feat/moe-disk-tier`. Study the current
MTP path: `python/freetoken/spec_decode.py`, the verify wiring in
`engine/engine.py` (decode-routed sequential verify from 410a494 and the
follow-up), and `select_lm_head_rows()`. Commit on the branch; do NOT push.

## Problem (measured)

Speculation is parked at break-even: with the resident head (11a) drafting
works, acceptance ~30-50% greedy, but each accepted token still costs one
full sequential decode step to verify, so net throughput gain is ~0. On
node-4 (4090, 23 DISK layers, x1 9.3 tok/s) the per-step fixed costs are
huge, which is exactly what batched verify amortizes.

## Task

1. **Batched verify (11c, the win)**: verify all K draft tokens in ONE
   forward pass instead of K sequential decode steps. The verify batch is a
   short prefill-shaped extend of the target model over [last_accepted,
   draft_1..draft_K] using the existing extend/prefill machinery, but it
   MUST route MoE like decode (routed-expert fetch, not whole-bank prefill
   staging): reuse the decode-routed expert path added for sequential
   verify. Accept the longest matching prefix, rollback KV for the rest
   (the radix/mamba-slot machinery already supports truncation; follow how
   the sequential path rolls back today).
2. **Persistent draft KV (11b)**: the draft head currently re-prefills its
   context every step. Keep the draft head's KV cache alive across steps,
   appending accepted tokens only, so drafting K tokens costs K small
   decode steps on the head, not a re-prefill.
3. Flag stays `--speculative-mtp on`; add `--mtp-draft-tokens K` (default
   3). Keep greedy-only gating as is (temp<=0 greedy semantics from the
   is_greedy fix).
4. **Stats**: extend the decode log line with drafted/accepted/acceptance
   rate per interval (fields already exist; make sure batched verify
   updates them correctly) plus verify_batch_us.

## Constraints

- Mamba/GDN layers: state rollback for rejected tokens must be exact.
  Qwen3.8-Flash-Next is 36 GDN + 12 QSA; if exact GDN state rollback for
  multi-token verify is not achievable with the existing machinery,
  IMPLEMENT the largest correct subset (e.g. K=1 batched-with-decode) and
  REPORT the blocker precisely rather than approximating: correctness over
  speed, greedy outputs must be bit-identical to non-speculative decode.
- No CUDA-graph regressions for the non-speculative path: all changes
  behind the speculative flag.

## Tests

GPU-free: draft-KV bookkeeping unit tests, verify-batch shape/routing
tests, rollback bookkeeping. CUDA-gated: greedy equivalence vs
non-speculative decode (bit-identical outputs) on the small test model;
acceptance-rate sanity. Full GPU-free subset: exact pytest line, zero new
failures.

## Deliverable

Commits on the branch + a report: files touched, pytest line, expected
speedup math (fixed-cost amortization: accepted_per_step x step_cost),
deviations and any GDN rollback caveats.
