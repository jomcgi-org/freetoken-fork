# Spec: GPU-fetch decode for DISK layers (FreeToken, patch 7)

STATUS: CONFIRMED for dispatch 2026-08-31. node-4 bare-metal (4090 24GB,
64GB RAM, 7800X3D 8C/16T) measured: best config pins 33GiB banks + 6GiB PLE
cache, leaving 23 DISK layers on the CPU executor; x1 decode 9.3 tok/s,
~4.4k major faults/decode step. The spilled layers' CPU + fault cost IS the
x1 bottleneck. VRAM headroom for extra slots: ~1.9 GiB (22.1/24 used).

## Workspace

Repo `/Users/jomcgi/repos/FreeToken`, branch `feat/moe-disk-tier`. Study the
offload decode path (`OffloadMoeCache.ensure_experts` slot LRU, GPU-layer
decode), the DISK routing from 5932caf, and the HMM gather from b48297d.

## Problem

DISK layers decode on the CPU executor (~2.5 ms/token/layer on 32 Xeon
vCPUs; likely worse on node-4's 16 Zen cores). With ~9-12 spilled layers
that is ~25-30 ms/token of CPU compute that the GPU could do faster if the
routed experts' rows reached VRAM.

## Design sketch

`--moe-disk-decode {cpu,gpufetch}` (default cpu). gpufetch:

1. DISK layers keep file-backed banks, but decode routes through the normal
   GPU slot cache: on a miss for a DISK layer's expert, fetch the row from
   the mmap into a pinned staging ring, H2D-copy into the LRU slot, then
   the standard GPU GEMM path runs. The slot cache already handles
   remapping; what is new is a host-side fill (mmap read -> pinned staging)
   in front of the existing H2D machinery.
   - Alternative worth probing given the HMM result: skip staging and let
     the GPU copy engine / kernel read the mmap directly (HMM) into the
     slot — measure both if cheap, ship the faster.
2. The fill must not stall the graph: DISK-layer decode may need the same
   flag-handshake pattern the CPU executor uses (graph waits on a doorbell
   while the host fills), or run those layers eager outside the graph.
   Follow the existing house machinery; do not invent a new sync scheme.
3. Prefill for DISK layers stays on the CPU executor (already fast).
4. Expert-cache hit tracking must include DISK layers so hot experts stay
   resident in VRAM and fills amortize.

## Answered from the node-4 bench (2026-08-31)

- Spill is 23 layers at the best 64GB config (33GiB pinned banks): the CPU
  path costs ~half the step budget at x1 9.3 tok/s. Material; proceed.
- VRAM headroom: ~1.9 GiB after graphs at the serving config.
- HMM direct-read vs pinned staging: still unmeasured on the open driver;
  implement staging as the default, keep the HMM probe behind a flag if
  cheap, and we A/B on the box.

## Tests

Parity: DISK layer via gpufetch == CPU executor output (CUDA-gated);
GPU-free config/routing tests; no regression in the GPU-free subset.
