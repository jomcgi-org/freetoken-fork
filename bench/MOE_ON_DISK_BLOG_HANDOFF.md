# "MoE on disk" blog handoff

Post 1: **running a 125B MoE on a gaming GPU - what every memory tier
actually costs**. Source of truth: RESULTS.md (5 rounds + attribution +
registered predictions). Joe writes final wording; drafts are offers.

## The narrative arc

1. Hook: Qwen3.8-Flash-Next is 125B params; a 4090 has 24GB. Upstream
   FreeToken needs 128GB RAM. We made it run in 64GB - and measured what
   each shortcut costs.
2. The tier laws (the educational core, all measured):
   - Disk is FREE for router-predicted expert banks: mmap + MADV_RANDOM
     + page-deduped MADV_WILLNEED after routing = <10 major faults/step.
   - Disk is FATAL for per-token lookup tables: a flat ~105ms/step tax
     that is NOT I/O - the GPU re-faults file-backed mappings every
     CUDA-graph replay (pretouch does nothing). Quantize (47.7->28.8GB)
     and pin/hot-row-cache instead (62->70% hit rates, CLOCK eviction).
   - Prefill must never copy whole layers: 17 MINUTES for 6 tokens ->
     5.5s for 441 by computing only routed experts on CPU.
   - Batching amortizes every fixed per-step cost: 4.9->11.6 tok/s (L4
     x8), 29->88.6 (Blackwell x8). Never serve bs=1.
3. Numbers tables: L4 rounds (the 4090-class envelope) + Blackwell G4
   -24 (340 tok/s prefill / 88.6 aggregate through the full tier).
   Replace L4-proxy numbers with bare-metal 4090 measurements when the
   implementation handoff session runs.
4. Predicted-vs-measured sidebar: predictions were registered before
   each round (RESULTS.md round 5 + DSV4 predictions) - honest science
   angle.
5. The economics twist: we also priced cloud GPUs vs hosted APIs -
   self-hosting only wins on OWNED metal or privacy. That's the "why a
   4090" punchline.

## Attribution rules (RESULTS.md "Attribution" section)

Credit upstream FreeToken explicitly: the engine, #112 residency seams,
#257 model support, the CPU executor. Ours: DISK banks, CPU prefill, 4
PLE backends + hot-row cache, quantized tables, spill selection,
concurrency hardening, MTP v1, the vGPU survival guide, the benchmark
corpus.

## Claims to NOT make (until evidence exists)

- "Model-agnostic": verified on qwen4_exp ONLY (DSV4 smoke was skipped).
- e2m1 table quality: numbers exist, QUALITY GATE does not - run
  quality.sh on the e2m1 config before recommending it in print.
- MTP speedups: it is lossless but parked at break-even.
- HMM numbers are L4/vGPU-era; bare-metal HMM unmeasured until the
  implementation session.

## Future posts

- Post 2: DeepSeek-V4-Flash (already FreeToken-supported) on big-RAM
  metal (EPYC build) - needs hardware; -24-class was predicted unusable
  (2-6 tok/s) and skipped.
- Post 3: GLM-5.3-Flash on RTX PRO 6000 - blocked on engine arch
  support (KDA + sparse MLA); our tier already suffices once it exists.
