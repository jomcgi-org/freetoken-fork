# Supported models

FreeToken loads HF safetensors checkpoints directly (plus native GGUF for
Gemma-4). The checkpoints below are known-good — the prebuilt kernels are tuned
for them; other checkpoints of the same architectures work too.

| Model | HF checkpoints |
|---|---|
| DeepSeek-V4 | [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) |
| GLM-5.2 | [nvidia/GLM-5.2-NVFP4](https://huggingface.co/nvidia/GLM-5.2-NVFP4) |
| GLM-4.7 | [nvidia/GLM-4.7-NVFP4](https://huggingface.co/nvidia/GLM-4.7-NVFP4) |
| Qwen3.8-Flash-Next | [Qwen/Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8), [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) |
| Qwen3.6 / Qwen3.5 MoE | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) ([-FP8](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8)), [nvidia/Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4), [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) ([-FP8](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8)) |
| Qwen3.8 / Qwen3.6 dense | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) ([-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)), [RadixArk/Qwen3.8-27B-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4), [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) ([-FP8](https://huggingface.co/Qwen/Qwen3.6-27B-FP8)), [nvidia/Qwen3.6-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4) |
| Qwen3-MoE | [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) |
| gpt-oss | [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b), [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) |
| Gemma-4 | [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it), [nvidia/Gemma-4-26B-A4B-NVFP4](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4), [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it), [nvidia/Gemma-4-31B-IT-NVFP4](https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4) .. |
| MiniMax-M2.5 | [nvidia/MiniMax-M2.5-NVFP4](https://huggingface.co/nvidia/MiniMax-M2.5-NVFP4) |
| Muse-Glimmer | [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B), [RedHatAI/Muse-Glimmer-30B-NVFP4](https://huggingface.co/RedHatAI/Muse-Glimmer-30B-NVFP4) |

## MoE backends

`ft serve --moe-backend {auto,fused,offload,cpu,hybrid}`:

- **fused** — experts resident on GPU (needs the VRAM); never auto-selected.
- **offload** — experts live in host RAM, an LRU cache of expert slots on GPU;
  misses stream over PCIe.
- **cpu** — misses are computed on the CPU instead of fetched.
- **hybrid** — per step, fetches some misses over PCIe and computes the rest on
  CPU, overlapped. Run `ft bench bw` once per machine to calibrate the split.
- **auto** — dense models always resolve to `fused`; MoE models resolve to
  `offload`, upgraded to `hybrid` when a cached `ft bench bw` profile
  recommends it.

### File-backed expert banks

`--moe-disk-layers` keeps the selected MoE layers as read-only mappings of their
checkpoint regions. Those layers decode on the CPU executor, while Linux loads only
touched expert pages into the page cache. The flag accepts the same explicit id list,
count, or fraction grammar as `--moe-cpu-layers`, for example
`--moe-disk-layers 48,49,50` or `--moe-disk-layers 0.25`.

DISK layers also prefill on the CPU executor by default. FreeToken first prefetches
the union of routed expert pages for the chunk, then computes only those routes without
copying the whole bank to the GPU cache. `--moe-disk-prefill copy` restores the prior
whole-layer pageable copy path for benchmarking. LOCKED and PAGEABLE layers keep their
existing GPU prefill copy behavior.

The DISK tier requires FreeToken's aligned, per-layer FTW layout. Convert a raw
safetensors checkpoint first with `ft checkpoint`; raw safetensors and GGUF banks are
not supported. When expert banks exceed `FREETOKEN_PIN_BUDGET_GB`, an FTW checkpoint
automatically spills enough tail layers to DISK. Non-FTW checkpoints retain the
existing OS-locked fallback.

### File-backed PLE table

Qwen3.8-Flash-Next defaults to `--ple-backend pinned`, which keeps its 47.7 GiB
PLE n-gram table in pinned host RAM and preserves the original CUDA graph path.
`--ple-backend disk` instead maps each PLE safetensors payload read-only, applies
random-access advice, and page-prefetches the deduplicated union of requested rows
before copying them through a small pinned staging bank. Disk PLE bytes are not
reserved from the expert-bank pin budget, so automatic MoE spilling can keep more
expert layers pinned.

Disk PLE keeps CUDA graph decode enabled: a pre-replay host hook derives the next
n-gram row ids from request token history, stages their deduplicated rows, and updates
fixed pinned compact-id buffers read by the captured gather. Set
`FREETOKEN_PLE_DISK_NO_GRAPHS=1` to restore eager decode for debugging. Prefill stages
the full requested-row union for each chunk through the unchanged eager path.

## Notes

- `ft checkpoint` conversion is optional — it pre-converts a checkpoint into
  FreeToken's fast-load format, and `ft serve --model` auto-detects the result.
- DeepSeek-V4 checkpoints must keep the `inference/config.json` subdir — the
  authoritative model args are read from there.
- Qwen3.8-Flash-Next keeps its PLE table pinned by default; use
  `--ple-backend disk` on memory-constrained hosts.
- Multimodal checkpoints are served text-only.
