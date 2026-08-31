# Spec DRAFT: GPU-fetch decode for DISK layers (FreeToken, patch 7)

STATUS: draft — confirm against the profile-cycle numbers before dispatch
(if the spilled layers' CPU cost is no longer material after profile-guided
selection, deprioritize behind UFFD).

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

## Open questions for the bench data

- Realized per-DISK-layer decode cost after profile-guided spill.
- VRAM headroom for extra slots (L4: ~2 GiB free after graphs).
- Whether HMM direct reads into slots beat pinned staging fills.

## Tests

Parity: DISK layer via gpufetch == CPU executor output (CUDA-gated);
GPU-free config/routing tests; no regression in the GPU-free subset.
