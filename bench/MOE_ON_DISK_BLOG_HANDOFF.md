# "MoE on disk" blog handoff

Post 1: **running a 125B MoE on a gaming GPU - 6x faster by teaching every
memory tier its job**. Source of truth: RESULTS.md (complete: cloud rounds,
bare-metal program 2026-08-31, program close). Joe writes final wording;
drafts are offers.

## The story in one table (all bare-metal 4090 24GB + 64GB, quality 5/5)

| | day-1 naive config | final |
|---|---|---|
| warm prefill (441 tok) | 18 tok/s | 116 |
| x1 single stream | 3.8 | 23.1 peak / ~21 sustained |
| x8 aggregate | 9.6 | 42.2 peak / ~37 sustained |
| GPU power under load | - | 138W avg (of 450W: host-bound) |

Same hardware, same model, same weights. Every gain is software, each one
measured, and five attractive ideas measured as LOSSES (that is half the
story's value).

## The narrative arc

1. Hook: Qwen3.8-Flash-Next is ~100GiB of quantized weights; a 4090 has
   24GB and the box 64GB. Upstream wants 128GB RAM. We ran it in 64 and
   then spent 36 hours finding out what each memory tier actually costs.
2. The tier laws (the educational core, all measured, in story order):
   - **The page cache is a tier**: pin budget competes with it. Pinning
     52 of 64GB collapsed throughput 3x; 40GB pinned tripled it back.
   - Disk is FREE for router-predicted expert banks (mmap + MADV_RANDOM
     + deduped WILLNEED after routing) and FATAL for per-token lookup
     tables on vGPU - but the ~105ms/step HMM tax was a vGPU ARTIFACT:
     on bare-metal open-driver, HMM WINS (x4 nearly doubled).
   - Prefill must never copy whole layers (17 min for 6 tokens -> 116
     tok/s end state), and per-layer overlap under split residency
     doubled warm prefill (54 -> 116).
   - **Routing is Zipf, exploit it per-expert not per-layer**: pinning
     the hot ~6GiB of expert rows (72% of routes) and leaving the cold
     tail on CPU broke the DDR-bandwidth ceiling: x8 34 -> 42.
   - **The hierarchy can tune itself**: online decayed counters + bounded
     background swaps recovered 62.6 -> 73.3% hot-rate under workload
     drift; no profile capture step needed at all.
3. The graveyard section (mechanisms that measured NEGATIVE, each with
   the physics): GPU-fetch decode (<48GB VRAM: slot-cache thrash),
   hybrid PCIe fetch (local CPU compute wins), UFFD as a throughput
   path (page-granular either way; it is the bigger-than-RAM capacity
   lane), lookahead prefetch (48% next-step routing predictability),
   and MTP speculation twice (marginal tokens must be cheap; a CPU tier
   makes them full price - even at 72% acceptance).
4. The measured step anatomy (--moe-step-timing): CPU phases 57-100ms vs
   GPU 19-40ms with 20-45ms already overlapped - the engines were never
   idle by accident; what remains is a scheduler-contract change or
   hardware.
5. The economics twist stays: hosted APIs crush cloud GPUs; self-hosting
   wins only on owned metal or privacy - "why a 4090" punchline. 138W
   for 37 tok/s sustained is the sustainability kicker (cheap green
   power makes this a non-issue to run 24/7).

## Attribution rules (RESULTS.md "Attribution" section)

Credit upstream FreeToken explicitly: the engine, #112 residency seams,
#257 model support, the CPU executor, the offload LRU. Ours (fork,
feat/moe-disk-tier): DISK banks + CPU prefill, PLE backends + quantized
tables, profile-guided spill, prefill overlap under split residency,
expert dedup, expert-granular residency, online hot-set adaptation,
UFFD pager, gpufetch, MTP K=1, step timing, the benchmark corpus and the
negative results.

## Claims to NOT make

- "Model-agnostic": verified on qwen4_exp ONLY.
- Peak vs sustained: 23.1/42.2 are profile-matched peaks; quote ~21/37
  as the honest sustained-diverse numbers alongside them.
- MTP: closed negative on THIS box; do not generalize to all-pinned or
  big-VRAM configs (untested there).
- Quality: the 5/5 gate is smoke-level (arith/recall/reason/longgen);
  do not claim benchmark-grade parity.
- Known cosmetic bug if screenshots show it: hot_swaps/interval prints
  0.00 while swaps are live.

## Open characterization gaps (fine to mention as future work)

Inter-interval phase variance (~2x, unexplained - the one that might
still hide throughput), first-request warm-up transient (~2x), mixed
prefill+decode interference, p99 step latency, swap observability.

## Future posts

- Post 2: DeepSeek-V4-Flash on big-RAM metal (EPYC build) - needs
  hardware; -24-class was predicted unusable (2-6 tok/s) and skipped.
- Post 3: GLM-5.3-Flash on RTX PRO 6000 - blocked on engine arch support
  (KDA + sparse MLA); the tier + expert-granular residency carry over.
