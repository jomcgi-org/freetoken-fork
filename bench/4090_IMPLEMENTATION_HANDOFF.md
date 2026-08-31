# 4090 implementation handoff (bare-metal serving session)

Goal: serve Qwen3.8-Flash-Next on node-4's 4090 after the GKE cutover
(and BEFORE any 4090 sale), replacing the L4-proxy projections with
measured numbers. ~2 hours, zero cloud cost.

## Preconditions

- node-4 out of k8s (no gpu-operator; we own the driver).
- 4090 24GB, 64GB RAM (128GB if upgraded - changes the config, below),
  1TB local NVMe.

## Setup

1. Install NVIDIA OPEN kernel modules (Blackwell-era driver fine; open
   modules enable HMM - this session is HMM's first true benchmark).
   CUDA 13 toolkit (keyring + cuda-toolkit-13-0) + python3.12-dev +
   build-essential + ninja.
2. Clone github.com/jomcgi/FreeToken feat/moe-disk-tier (fetch upstream
   first; rebase if it moved). uv venv --python 3.12;
   uv pip install -e ".[accel]" pytest ninja.
3. Models to NVMe: RadixArk/Qwen3.8-Flash-Next-NVFP4 (126G) +
   primitive-ai/Qwen3.8-Flash-Next-PLE-quant --include "ples_nvfp4/*" (27G).
4. **Converter caveat**: `ft checkpoint` OOMs without ~90G+ headroom
   (measured: killed at 85G virtual on small boxes). At 64G RAM: add a
   big swapfile on NVMe for the conversion, or convert once on any
   >=96G machine. Output ~73G FTW.
5. e2m1 serving dir: FTW shards + freetoken_weight.json + config/
   tokenizer files + symlinks to ples_nvfp4/*.safetensors; NO
   safetensors index json in the dir (forces sidecar discovery). The
   plain-fp8 dir keeps its index. See g4round1 scripts for the exact
   assembly.

## Configs to run (harness: bench2.sh, load.sh, quality.sh here)

64GB RAM ("the blog config"):
- FREETOKEN_PIN_BUDGET_GB=52, PLE=cached PLECACHE=8-12, MODEL=e2m1 dir
  -> expect ~9-12 layers spilled to NVMe, <10 majflt/step warm.
- Loads x1/x4/x8. Projection to beat: 15-20 single / 30-40 aggregate.
- One PLE=hmm run (open driver = first real HMM numbers; L4 showed
  ~105ms/step refault tax - does bare-metal Ada do better?).
- quality.sh (bump longgen max_tokens to ~1500 for the thinking budget,
  or disable thinking) - REQUIRED for the e2m1 table quality gate,
  currently unvalidated.

128GB RAM (if upgraded): everything pins (banks 63.5 + table), zero
spill; expect ~36 single (upstream's 4090 figure) / 60-90 aggregate.

## Traps (all hit before, all documented in RESULTS.md)

pkill self-match kills your own SSH/session (bracket patterns); zombie
spawn-children hold VRAM (kill by venv path); server binds its port
before the engine is ready (readiness = a real completion, retried while
the PID lives); `ci`-style: judge runs by stats lines, not exit codes.

## Relevant open technical items

- GPU-fetch decode spec (freetoken-gpu-fetch-spec.md) - would move the
  ~10 spilled layers' compute to GPU; worth dispatching if 64GB decode
  disappoints.
- MTP parked at break-even (needs batched verify + persistent draft KV).
- vGPU allocator autodetect - irrelevant on bare metal.
