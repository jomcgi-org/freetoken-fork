# FreeToken DISK expert-bank tier: implementation + bench results

Date: 2026-08-29. Branch: jomcgi/FreeToken `feat/moe-disk-tier` (5932caf).
Bench box: GCP g2-standard-32 spot (L4 24 GB, 125 GiB RAM, 375 GB local NVMe),
europe-west2-a. Total compute spend: ~$6.50.

## Correctness

- 11/11 disk-tier tests pass on GPU, including the bitwise parity test
  (disk-mapped vs pinned banks through the real `_cpu_moe` executor) and the
  CUDA copy-plan skip.
- `tests/moe` + `tests/engine`: zero branch-only failures vs upstream/main on
  the same box (45 env-caused failures identical on both).

## Qwen3.6-35B-A3B-NVFP4 (banks ~17 GiB, all RAM-cached, 0 major faults)

| config | warm tok/s | delta |
|---|---:|---|
| baseline-pinned | 56.2 | |
| disk4 | 39.5 | -30% |
| disk8 | 31.7 | -44% |
| disk16 | 22.8 | -59% |

Measures pure CPU-executor cost of a routed layer: ~1.9 ms/token/layer.

## Qwen3.8-Flash-Next-NVFP4 (first Flash serve through the fork)

FTW = 72.7 GiB (banks 63.5); PLE table (47.7 GiB) loads from raw safetensors
shards, which must sit next to the FTW dir (symlinks work) - `ft checkpoint`
does not convert the PLE.

| config | warm tok/s | delta |
|---|---:|---|
| baseline-pinned | 11.7 | (L4; 4090 upstream benchmark = 36) |
| disk4 | 10.9 | -7% |
| disk8 | 9.6 | -18% |
| disk16 | 8.0 | -31% |

Flash CPU-layer cost ~0.9 ms/token/layer - much gentler than 35B because
Flash's per-token time is larger. All RAM-cached (125 GiB box).

## node-4 simulation (the actual question)

`FREETOKEN_PIN_BUDGET_GB=52` + cgroup `MemoryMax=64G` (verified pegged at
64.0 GiB): 47.7 PLE pinned + 3.97 GiB banks pinned (3 layers) + 59.5 GiB
file-backed (45 DISK layers).

**Decode-side prefetch mechanics are sound** (MADV_RANDOM + page-deduped
WILLNEED; counters live in the decode log). **Prefill is the blocker**: any
prefill batch triggers the per-#112 whole-layer pageable copy for every
non-pinned layer - all 59.5 GiB of disk banks per batch, regardless of prompt
length. A 6-token prompt took 17+ minutes under eviction pressure. Unusable at
this spill ratio.

## Verdict for node-4 (RTX 4090 24 GB + 64 GB RAM)

- Decode: plausibly ~20-27 tok/s with ~10-12 spilled layers (36 baseline minus
  CPU-layer cost), less with real fault traffic. Promising.
- Prefill: current path makes the tier unusable. The needed upstream change:
  prefill for DISK layers must stream only the routed experts (or run prefill
  on the CPU executor), never whole-layer copies.
- PLE: still needs upstream's PLE-to-disk work (roadmap item) - our tier only
  covers expert banks.

## Next moves

1. Patch the prefill path for DISK layers (stream routed experts only), rebench.
2. Or hand these findings + branch upstream (issue #214 / roadmap #79) and let
   them fold it into their PLE-disk work.

## Infra notes

- VM `freetoken-bench` STOPPED (not deleted). Persistent disk `ft-data` (300 GB,
  ~$33/mo if kept) holds models, FTW, venv, results; `~/recover.sh` rebuilds
  the wiped local SSD in ~15 min after any restart/preemption.
- Delete when done: `gcloud compute instances delete freetoken-bench --zone=europe-west2-a`
  and `gcloud compute disks delete ft-data --zone=europe-west2-a`.

---

# Round 2 (2026-08-29, later): CPU prefill + disk-backed PLE

Commits: 0fe89d5 (DISK-layer prefill on the CPU executor, --moe-disk-prefill),
ef052ba (--ple-backend disk: mmap'd PLE shards + staged UVA gather, CUDA
graphs disabled as v1). All on jomcgi/FreeToken feat/moe-disk-tier.

Node-4 config (Flash-Next NVFP4, 64G cgroup, FREETOKEN_PIN_BUDGET_GB=52, L4):

| metric | round 1 | + CPU prefill | + PLE disk |
|---|---:|---:|---:|
| layers spilled | 45/48 | 45/48 | 9/48 |
| prefill 441 tok (warm) | 17+ min for 6 tok | 29.2s (15 tok/s) | 5.5s (80 tok/s) |
| decode | wedged | 1.8 tok/s | 4.5-4.8 tok/s |
| warm majflt/step | n/a | 1047 | 2.5 |

Findings:
- Disk tier is SILENT at steady state after both patches (2.5 faults/step).
- Decode is now bound by the graphs-disabled eager path (~100 ms/step), not
  by disk: patch 4a (graph-compatible staged gather) is the next lever;
  projected ~9-10 tok/s on L4, ~20+ on the 4090.
- UFFD pager (patch 4b) re-scoped: capacity play (bigger-than-RAM banks,
  row-granular residency), not a current-speed play.

Bench-harness lessons: sudo secure_path strips venv (ninja JIT failures);
non-editable reinstall shadows the repo (stale-code test failures); orphaned
servers stack and CUDA-OOM each other (bench2.sh now traps EXIT and retries
warmup only while the server pid is alive).

---

# Round 3 (2026-08-29/30): HMM, staging fast path, spill selection

Commits: c93c2e9+85b5695 (graph-replayable staged gather), 04f6d88 (staging
vectorization), b48297d (--ple-backend hmm), 2e66f7f (miss-aware spill).

Decisive numbers (L4, warm decode):
- pinned PLE reference: 11.7 tok/s
- staged PLE, any variant: ~4.9-5.1
- HMM PLE, 0 spill, uncapped: 5.7  <- ~105ms/step GPU-side, NOT cold faults
  (pre-touch neutral), NOT host staging (staging_us=0), NOT spill-related
  (0-spill identical to 9-spill). The GPU re-faults file-backed mappings
  every replay on this driver/silicon.
- profile-guided spill + pretouch: neutral at bs=1.

Shape economics rewrite: -6+fork needs a pinned hot-row PLE cache (patch 9
candidate) or better Blackwell HMM behavior; -12 + primitive-ai quantized
table (28.8GB NVFP4-g16, fits 45GB RAM pinned) is the value pick at $0.53/hr.
Quality gate for the 4-bit table still pending (quality.sh comparison).

Quality harness (chat endpoint): arith/recall/reason PASS through full tier;
longgen needs thinking-budget fix (model reasons by default, xhigh).

---

# Round 4/5 (2026-08-30): feature-complete milestone

Fork: jomcgi/FreeToken feat/moe-disk-tier, 21 commits, based on upstream
58f4b9e (upstream unmoved since branch). All features live and tested.

## Shipped since round 3

- 4f137db concurrency hardening (5-target audit, 1 real fix: staging bank
  vs explicit graph sizes; the report doubles as the concurrency doc)
- fa280cd quantized PLE tables (FP8 / INT4 g16 / e2m1 g16, auto-detected)
- 5164b25 pinned hot-row PLE cache (--ple-backend cached, CLOCK eviction,
  warm profiles, full stats)
- 906574a per-shard PLE global scales (real-checkpoint layout fix)
- 4f0be4d + 410a494 MTP speculative decode (greedy bs=1, lossless;
  LM-head row-selection and is_greedy fixes; latent QSA ring bug fixed)

## Key numbers (L4, 64G-cap node-4 envelope unless noted)

| config | decode tok/s | notes |
|---|---:|---|
| pinned PLE reference | 11.7 | uncapped |
| HMM PLE, 9-layer spill | 5.7 | disk silent |
| cached PLE (fp8), cold | 4.2-4.8 | 62->70% hit rate warming |
| cached PLE aggregate x4 / x8 | 10.9 / 11.6 | batching amortizes per-step cost |
| e2m1 table + cache | 2.4 | packed miss-install ~4x cost (open item) |
| MTP on | 0.20 | 31.7% acceptance; BF16 draft experts stream per draft (open item) |

Quality: arith/recall/reason PASS through the full tier (chat endpoint);
longgen needs thinking-budget handling in the harness.

## Open performance items (ranked)

1. MTP economics: quantize/pin the 4.69 GiB draft experts (fits VRAM),
   graphs under MTP, persistent draft KV (acceptance 32% -> ?). Until then
   MTP stays default-off.
2. Packed-table miss-install cost (~4x fp8) — batch data+scale copies.
3. Hot-row cache long-uptime hit-rate measurement (short benches cap ~70%).

## Target configs (current best knowledge)

- node-4 (24G VRAM / 64G RAM / 1TB NVMe, bare metal post-migration, open
  driver): budget-52 + cached PLE (quantized table) + profile-guided spill;
  batched agent traffic ~2x single-stream. GPU-fetch decode is the next
  node-4-specific lever.
- G4 -6 (96G VRAM / 22G RAM): banks in VRAM; cached PLE on the 28.8G e2m1
  table (page-cache friendly) once miss-install is optimized; batching
  carries throughput. -12 with the table fully pinned needs zero further
  work today.

Bench-harness debt: zombie spawn-children survive pkill patterns (three
incidents); kill by venv path. pkill self-match keeps killing SSH sessions
(bracket the pattern or split kill/launch).

---

# Round 5 (2026-08-30/31): Blackwell (G4) and the four outcomes

Hardware learned the hard way: GCP G4 small shapes are FRACTIONAL vGPU
slices of the RTX PRO 6000 (g4-standard-6 = 12 GB "DC-1-12Q"; -24 = 48 GB
"DC-2-48Q"; full 96 GB needs -48). vGPU guests refuse the open kernel
module AND the plain GRID driver: only GCP's grid-gcp build works
(cuda_installer.pyz fetches it). vGPU also rejects CUDA VMM, so
FreeToken's expandable_segments allocator must be disabled
(PYTORCH_CUDA_ALLOC_CONF=backend:native) - patch candidate: auto-detect.

## The headline table (Flash-Next 125B-A6B NVFP4, full tier)

G4 -24 slice (48 GB VRAM + 88 GB RAM, $1.05/hr spot), e2m1 quantized table
hot-row cached (8 GiB pin, 62% hits cold and warming), ALL expert banks
RAM-pinned, 48 GB VRAM LRU cache, CUDA graphs on:

- prefill 441-token prompt: 1.3 s (340-352 tok/s)
- decode single stream: ~29 tok/s
- aggregate: 81 tok/s at 4 streams, 88.6 at 8
- majflt/step: 0 (the tier is invisible at steady state)

## Outcome 1: what this means for a 96 GB PRO 6000 at home

The -24 slice is HALF the card plus a vGPU tax. At home (full 96 GB, no
vGPU, open driver so HMM also works): banks go fully VRAM-resident,
removing the last host-path costs -> project 100-150+ tok/s aggregate,
400+ prefill. The tier still earns its keep there for GLM-5.3-class
models (150 GB banks > any single card) via the RAM tier + future pager.

## Outcome 2: what this means for a 4090 at home

The L4 rounds (rounds 1-4) are exactly the 4090's shape: 24 GB-class VRAM
+ 64 GB RAM. Measured there: the full stack serves the 125B model in that
envelope at 5.7-11.7 tok/s single / 11.6 aggregate on ~1/3 of a 4090's
bandwidth. Scaled to the 4090 (and its native-NVFP4 sibling numbers
upstream: 36 tok/s plain offload), expect ~20-35 tok/s with the tier and
quantized table - a genuinely usable private 125B server from a gaming
card, IF the RAM is 64 GB+ and the table is quantized.

## Outcome 3: disk vs RAM (the measured ladder)

- RAM-pinned PLE: 11.7 tok/s reference (L4).
- ANY disk-backed PLE: ~5 tok/s - a flat ~105 ms/step tax, and NOT from
  I/O: the GPU re-faults file-backed mappings every CUDA-graph replay
  (pretouch does nothing, staging_us proved the host idle). Disk tables
  are for capacity, never for speed; quantize (47.7 -> 28.8 GB) and pin
  or hot-row-cache instead.
- Expert banks on disk are DIFFERENT: with router-driven page-deduped
  MADV_WILLNEED prefetch, spilled banks run at <10 major faults/step -
  effectively free at steady state. Disk is fine for cold experts, fatal
  for per-token lookup tables.
- Prefill through disk tiers must never copy whole layers: the naive path
  took 17 MINUTES for 6 tokens; routing prefill through the CPU executor
  (touched experts only) made it 5.5 s for 441 tokens.

## Outcome 4: RAM tiering (what actually works)

- Pinned hot-row cache over a Zipf lookup table: 62% hit rate cold,
  70%+ within minutes, miss cost now 1.2x fp8 after interleaved-copy
  fix. The cache budget is the RAM allocation knob.
- Pin-budget auto-spill + per-layer traffic profiles pick WHICH layers
  leave RAM (profile endpoint shipped; matters at higher spill counts).
- Batching is the great equalizer: fixed per-step tier costs amortize
  across concurrent streams - 4.9 -> 11.6 tok/s (L4 x8), 29 -> 88.6
  (Blackwell x8). An agent factory should never serve bs=1.
- MTP speculative decode: lossless and working, but parked - draft-head
  streaming + verify routing made it net-negative until the resident-head
  + batched-verify follow-ups; only worth revisiting on big-VRAM boxes.

## Economics postscript

Cloud GPU self-hosting loses to hosted APIs at every measured shape
(GLM-5.3-Flash: $0.15/$0.50 per M list). The winning architecture:
GLM-Flash API as the 24/7 orchestrator (~$10-30/mo), subscription pools
(Claude/Codex) as implementation muscle, owned metal for private lanes.
The fork's value is metal utilisation, not cloud arbitrage.

Fork: jomcgi/FreeToken feat/moe-disk-tier, 25 commits, based on upstream
58f4b9e. Total experiment spend: ~$39 of $50.

---

# Attribution: what FreeToken had, what we added

## Upstream FreeToken (before the fork - credit where due)

- The engine itself: MoE offload backend with a GPU LRU expert-slot cache
  and PCIe miss streaming; CUDA-graph decode; radix prefix cache.
- Qwen3.8-Flash-Next support (#257): hybrid GDN/QSA layers, PINNED host
  PLE table with in-graph UVA gather, NVFP4/FP8 kernels, FTW fast-load
  format, MTP weights present-but-dropped.
- Per-layer host-bank residency (#112): PINNED/LOCKED/PAGEABLE classes,
  the CPU MoE executor (C++ worker pool, flag-handshake coordination with
  captured graphs), pin budgets for WSL/WDDM, --moe-cpu-layers.
- Elastic VRAM pools (live cache rebuild), bandwidth calibration.

Everything we built stands on those seams - especially #112's residency
plumbing and the CPU executor.

## Our fork (25 commits on feat/moe-disk-tier)

1. **DISK residency for expert banks** - read-only file mmaps of FTW
   regions, MADV_RANDOM + router-driven page-deduped MADV_WILLNEED before
   compute. The "disk banks are free" result.
2. **CPU prefill for DISK layers** - replaced whole-layer pageable copies
   (17 min for 6 tokens) with routed-experts-only CPU compute (5.5 s for
   441). Plus --moe-disk-prefill escape hatch.
3. **Four PLE-off-RAM backends**: staged gather; CUDA-graph-replayable
   staging (bit-exact host reimplementation of the n-gram hash);
   vectorized staging (process_vm_readv batched copies); HMM direct GPU
   reads of file mmaps; and the pinned hot-row cache (CLOCK eviction,
   warm profiles, hit-rate stats).
4. **Quantized PLE tables** - INT4/e2m1 group-16 in-kernel dequant,
   per-shard global scales folded at load, interleaved data+scale
   miss-install (1.2x fp8 cost).
5. **Miss-aware spill selection** - /v1/moe-layer-profile endpoint +
   profile-guided lowest-traffic layer choice.
6. **Concurrency hardening** - bs>1 audit with one real fix (staging
   capacity vs explicit graph sizes).
7. **MTP speculative decode v1** - head loading, greedy draft/verify
   (lossless), resident+quantizable head, decode-routed verify; parked at
   break-even pending batched verify.
8. **Compat + measurement corpus** - GCP vGPU survival guide (grid-gcp
   driver, VMM-free allocator), is_greedy semantics fix, and the
   five-round benchmark methodology itself.

Upstreaming posture: items 1-5 are coherent PR candidates if we choose;
the measured findings (GPU-refault law, prefill pathology) are useful to
upstream regardless of code.
