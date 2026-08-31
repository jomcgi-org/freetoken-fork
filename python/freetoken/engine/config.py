from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, List

import torch
from freetoken.distributed import DistributedInfo
from freetoken.models.register import _load_attr, get_model_spec
from freetoken.utils import cached_load_hf_config

if TYPE_CHECKING:
    from freetoken.models import ModelConfig


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    tp_info: DistributedInfo
    dtype: torch.dtype
    max_running_req: int = 4
    attention_backend: str = "auto"
    moe_backend: str = "auto"
    # NVFP4 routed-expert GEMM backend (--nvfp4-backend): auto|marlin|flashinfer|triton.
    nvfp4_backend: str = "triton"
    # Expert-bank host load (--expert-load): auto|serial|parallel. "auto" reads scattered
    # experts in parallel but falls back to serial when free RAM can't cover the banks + the
    # parallel reader's extra (non-reclaimable) whole-shard buffer; "serial" forces the
    # low-memory reclaimable read; "parallel" forces the fast read.
    expert_load: str = "auto"
    moe_cache_size: int = 0
    moe_cache_rate: float | None = None
    moe_cache_auto: bool = False
    kv_reserve_tokens: int = 8192  # KV floor for --moe-cache-auto; small by design (MoE-priority)
    moe_cache_policy: str = "lru"
    moe_prefill_overlap: bool = True
    # Prefill hit/miss split: serve cache-resident experts D2D during prefill
    # prefetch instead of re-streaming the full layer over PCIe. Needs CUDA >= 12.8
    # (cudaMemcpyBatchAsync); no-op unless moe_cache_size > 2 * num_experts.
    moe_prefill_hit_d2d: bool = False
    moe_collect_stats: bool = False  # capture decode miss-rate counters into the cuda graph
    # CPU MoE backend (--moe-backend cpu): number of CPU worker threads computing
    # the decode experts. 0 = auto (physical cores). Ignored by other backends.
    moe_cpu_threads: int = 0
    # Hybrid CPU/GPU decode (--moe-backend offload only): which MoE layers decode on
    # the CPU executor instead of the GPU offload/PCIe path. Spec is an explicit id
    # list ("3,7,11"), a count ("8" -> 8 layers evenly strided across depth), or a
    # fraction ("0.5"). None/"" = all layers on GPU (plain offload). --moe-backend cpu
    # already means all layers on CPU and ignores this.
    moe_cpu_layers: str | None = None
    # File-backed FTW bank layers. Uses the same grammar as moe_cpu_layers. Decode
    # uses the CPU executor by default or the GPU slot cache under moe_disk_decode.
    moe_disk_layers: str | None = None
    # Optional per-MoE-layer traffic scores used only by automatic FTW DISK spill
    # selection. Explicit moe_disk_layers remains authoritative.
    moe_disk_layer_profile: str | None = None
    # DISK-layer prefill compute: "cpu" reads only routed experts through the CPU
    # executor; "copy" restores the whole-layer pageable GPU-copy path for benchmarks.
    moe_disk_prefill: str = "cpu"
    # DISK-layer decode: "cpu" preserves the native CPU executor path; "gpufetch"
    # faults only LRU-missing routed expert rows into a pinned staging ring, copies
    # them into the existing GPU slot cache, and runs the normal GPU expert GEMM.
    moe_disk_decode: str = "cpu"
    # DISK bank residency backend. "madvise" preserves the file-mmap path; "uffd"
    # installs FTW expert rows into anonymous mappings under a userspace LRU.
    moe_disk_pager: str = "madvise"
    moe_pager_budget_gib: float = 40.0
    # Hybrid MoE backend (--moe-backend hybrid): max experts fetched over PCIe per
    # (layer, decode step); the rest of that step's misses are computed on the CPU.
    # -1 (default) = auto: fetch the benched pcie_bw/cpu_bw fraction of each step's
    # misses so the PCIe fetch and the CPU compute finish together (perfect overlap);
    # falls back to a fixed cap of 1 without a usable `ft bench bw` profile.
    moe_hybrid_max_fetch: int = -1
    # Qwen3.8 Flash-Next PLE table storage. "pinned" is the original full-bank UVA
    # path; "cached" keeps hot mapped rows in a bounded pinned bank; "disk" stages
    # mapped rows; "hmm" gathers from mapped shards directly.
    ple_backend: str = "pinned"
    ple_cache_gib: float = 8.0
    ple_cache_warm: str | None = None
    ple_cache_profile_out: str | None = None
    # Qwen3.8-Flash-Next native multi-token prediction head. Kept as an explicit
    # on/off choice instead of a boolean so the CLI and serialized daemon config
    # have one stable spelling.
    speculative_mtp: str = "off"
    # Patch 11-lite intentionally supports one draft only. Keep the explicit
    # knob so attempts to reuse older K>1 launch commands fail loudly.
    mtp_draft_tokens: int = 1
    cuda_graph_bs: List[int] | None = None
    cuda_graph_max_bs: int | None = None
    page_size: int = 1
    memory_ratio: float = 0.9
    # Hybrid GDN models default to the HybridRadixCache (cross-request GDN-state prefix reuse);
    # `--cache-type naive` opts out. linear_state_cache_ratio sizes the GDN snapshot cache as
    # ceil(ratio * max_running_req) extra slots.
    linear_state_cache_ratio: float = 2.0
    # Window/full ratio for the SWA radix cache (`--cache-type radix` on SWA models) and the DSV4
    # window tier: the DEFAULT window-pool size = max(working-set floor, ratio x full-pool tokens).
    # < 1.0 trades retained window-prefix capacity for memory savings; must be in (0, 1]. It is the
    # DSV4 window/full ratio directly. Used only when swa_num_pages_override is None (a runtime
    # rebuild can pin an absolute window instead).
    swa_full_tokens_ratio: float = 0.2
    # Absolute window-pool size in the pool's own pages (usable, dummy excluded); None -> use the
    # ratio default above. A runtime cache rebuild sets this (num_swa_pages) to pin the window
    # regardless of the full anchor; the ratio is the startup default and the fallback.
    swa_num_pages_override: int | None = None
    distributed_timeout: float = 60.0
    use_dummy_weight: bool = False
    use_pynccl: bool = True
    max_seq_len_override: int | None = None
    num_page_override: int | None = None  # if not None, will override the number of pages
    # KV capacity in tokens; resolved into num_page_override by _adjust_config once page_size
    # is final. Mutually exclusive with num_page_override.
    num_token_override: int | None = None

    def __post_init__(self) -> None:
        from freetoken.spec_decode import (
            validate_mtp_draft_tokens,
            validate_speculative_mtp,
        )

        validate_speculative_mtp(self.speculative_mtp)
        validate_mtp_draft_tokens(self.mtp_draft_tokens)
        if self.ple_backend not in ("pinned", "cached", "disk", "hmm"):
            raise ValueError(
                "--ple-backend must be 'pinned', 'cached', 'disk', or 'hmm', got "
                f"{self.ple_backend!r}"
            )
        if not math.isfinite(float(self.ple_cache_gib)) or self.ple_cache_gib <= 0:
            raise ValueError("--ple-cache-gib must be a finite positive number")
        if self.moe_disk_prefill not in ("cpu", "copy"):
            raise ValueError(
                "--moe-disk-prefill must be 'cpu' or 'copy', got "
                f"{self.moe_disk_prefill!r}"
            )
        if self.moe_disk_decode not in ("cpu", "gpufetch"):
            raise ValueError(
                "--moe-disk-decode must be 'cpu' or 'gpufetch', got "
                f"{self.moe_disk_decode!r}"
            )
        if self.moe_disk_pager not in ("madvise", "uffd"):
            raise ValueError(
                "--moe-disk-pager must be 'madvise' or 'uffd', got "
                f"{self.moe_disk_pager!r}"
            )
        if (
            not math.isfinite(float(self.moe_pager_budget_gib))
            or self.moe_pager_budget_gib <= 0
        ):
            raise ValueError("--moe-pager-budget-gib must be a finite positive number")

    @cached_property
    def hf_config(self):
        return cached_load_hf_config(self.model_path)

    @cached_property
    def model_config(self) -> ModelConfig:
        spec = get_model_spec(self.hf_config.architectures[0])
        parse_config = _load_attr(spec.module, spec.parse_config)
        return parse_config(self.hf_config)

    @property
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.max_position

    @property
    def max_forward_len(self) -> int:
        return self.max_seq_len

    @property
    def distributed_addr(self) -> str:
        return "tcp://127.0.0.1:2333"
