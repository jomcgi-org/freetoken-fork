# FreeToken program index

See 4090_IMPLEMENTATION_HANDOFF.md (the next work session) and
MOE_ON_DISK_BLOG_HANDOFF.md (the writing). Below: full program state.

State as of 2026-08-31. Everything durable lives in this directory or the
fork; nothing depends on the originating session.

## Assets

- **Fork**: github.com/jomcgi/FreeToken, branch `feat/moe-disk-tier`,
  25 commits on upstream 58f4b9e. Feature-complete; see RESULTS.md
  "Attribution" for the upstream-vs-ours split.
- **RESULTS.md** (here): 5 measured rounds, four outcomes, methods,
  registered predictions. Blog post 1 source material.
- **Harness** (here): bench2.sh (config bench), load.sh (concurrency),
  quality.sh (correctness gate), g4-setup.sh / recover.sh (cloud setup),
  capped-run.sh, hmm-probe.py. Known traps documented in RESULTS.md
  (pkill self-match, zombie spawn-children, sudo secure_path).
- **Spec drafts** (here): freetoken-uffd-pager-spec.md (GLM-class
  capacity play), freetoken-gpu-fetch-spec.md (spilled-layer decode on
  GPU; relevant to 4090@64GB).
- Memory: project_freetoken_disk_tier in the session memory index.

## The next session (bare-metal, post-GKE cutover, BEFORE 4090 sale)

1. node-4 freed from k8s; install open kernel modules directly (no
   gpu-operator). This enables HMM (first true HMM benchmark).
2. Clone fork, uv venv, install [accel]+ninja+pytest, CUDA 13 toolkit.
3. Models to local NVMe: RadixArk/Qwen3.8-Flash-Next-NVFP4 (126G) +
   primitive-ai/Qwen3.8-Flash-Next-PLE-quant ples_nvfp4/ (27G).
   ft checkpoint (needs >90G RAM OR run converter on a bigger box /
   accept the OOM risk at 64G - converter needs headroom, see round 5).
   NOTE: converter OOMs below ~. If 64G box: convert in cloud once or
   add swap.
4. Configs to run (all proven): budget-52 + PLE=cached PLECACHE=8-12 on
   the e2m1 table; loads x1/x4/x8; quality.sh (fix longgen thinking
   budget or raise max_tokens to 1500); one PLE=hmm run (the HMM
   first-real-benchmark); optionally 128G RAM variant if upgraded.
5. Replace RESULTS.md projections with measured numbers -> blog post 1.

## Open items (priority order)

1. e2m1 table QUALITY gate - never validated; blocks recommending the
   quantized table in print.
2. Blog post 1 draft (Joe writes final wording).
3. Tier verified on qwen4_exp ONLY - do not claim model-agnostic without
   a DSV4-family smoke.
4. MTP: parked at break-even; needs batched decode-routed verify (11c) +
   persistent draft KV (11b) to pay. Revisit on big-VRAM metal.
5. UFFD pager: spec here; dispatch when GLM-class-on-limited-RAM matters.
6. vGPU allocator autodetect (currently PYTORCH_CUDA_ALLOC_CONF=
   backend:native by hand on GCP G4).
7. Upstream PR decision (tier patches 1-5 are coherent candidates).
8. Rebase cadence: fetch upstream before any new dispatch.

## Strategy conclusions (settled 08-30/31)

- Cloud GPU self-hosting loses to hosted APIs at every shape; GCP G4
  small shapes are fractional vGPUs (-6=12GB!).
- Factory architecture: GLM-Flash API orchestrator (~$10-30/mo) + Claude/
  Codex subscription pools as muscle + owned metal for private lanes.
- Hardware endgame: EPYC 8-channel + RTX PRO 6000 (Max-Q if dual);
  4090+128GB RAM = usable 125B server (~36 tok/s); 4090 sale = logistics
  decision, not economics.
