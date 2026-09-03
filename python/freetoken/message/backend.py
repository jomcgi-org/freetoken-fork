from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List

import torch
from freetoken.core import SamplingParams

from .utils import deserialize_type, serialize_type


@dataclass
class BaseBackendMsg:
    def encoder(self) -> Dict:
        return serialize_type(self)

    @staticmethod
    def decoder(json: Dict) -> BaseBackendMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchBackendMsg(BaseBackendMsg):
    data: List[BaseBackendMsg]


@dataclass
class ExitMsg(BaseBackendMsg):
    pass


@dataclass
class UserMsg(BaseBackendMsg):
    uid: int
    input_ids: torch.Tensor  # CPU 1D int32 tensor
    sampling_params: SamplingParams
    # Optional precomputed multimodal soft-token embeddings (GPU tensor). Only used by
    # the in-process offline path; remains None for the (serialized) online path.
    mm_embeds: torch.Tensor | None = None
    priority: int = 0
    arrival_time: float = field(default_factory=time.monotonic)
    # Optional tokenizer-derived boundary at the end of a known coding harness's
    # stable system/tool preamble. Hybrid caches persist this boundary separately
    # so fresh sessions can reuse it even when their first user message differs.
    cache_anchor_len: int | None = None
    cache_anchor_kind: str | None = None


@dataclass
class AbortBackendMsg(BaseBackendMsg):
    uid: int
    client_disconnected: bool = False


@dataclass
class CacheRebuildBackendMsg(BaseBackendMsg):
    # tokenizer worker -> scheduler: request a runtime KV/MoE/GDN cache resize.
    request_id: str
    moe_cache_size: int | None = None
    num_pages: int | None = None
    num_mamba_slots: int | None = None
    num_swa_pages: int | None = None
    mode: str = "if_idle"  # only "if_idle" is supported; "drain" is deferred (rejected)


@dataclass
class MoeLayerProfileBackendMsg(BaseBackendMsg):
    request_id: str
