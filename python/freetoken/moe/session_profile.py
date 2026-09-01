"""Compact, versioned routed-expert profiles used as advisory cache hints."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch


SESSION_EXPERT_PROFILE_VERSION = 1
SESSION_EXPERT_PROFILE_TOPK = 8
SESSION_ADAPT_INJECTION_WEIGHT = 0.25
_VERSION_TENSOR = "expert_profile.version"
_IDS_TENSOR = "expert_profile.ids"
_COUNTS_TENSOR = "expert_profile.counts"


@dataclass(frozen=True)
class SessionExpertProfile:
    """Top routed experts for each MoE layer, with decayed relative weights."""

    ids: tuple[tuple[int, ...], ...]
    counts: tuple[tuple[float, ...], ...]
    version: int = SESSION_EXPERT_PROFILE_VERSION

    def __post_init__(self) -> None:
        if self.version != SESSION_EXPERT_PROFILE_VERSION:
            raise ValueError(f"unsupported session expert profile version {self.version}")
        if len(self.ids) != len(self.counts):
            raise ValueError("session expert profile layer dimensions disagree")
        for layer_ids, layer_counts in zip(self.ids, self.counts):
            if len(layer_ids) != len(layer_counts):
                raise ValueError("session expert profile row dimensions disagree")
            if len(layer_ids) > SESSION_EXPERT_PROFILE_TOPK:
                raise ValueError("session expert profile row exceeds its top-k bound")
            if any(expert < 0 for expert in layer_ids):
                raise ValueError("session expert profile contains a negative expert id")
            if any(expert > 32767 for expert in layer_ids):
                raise ValueError("session expert profile expert id exceeds int16 storage")
            if any(not math.isfinite(count) or count <= 0 for count in layer_counts):
                raise ValueError("session expert profile counts must be finite and positive")

    @property
    def num_layers(self) -> int:
        return len(self.ids)

    @property
    def num_experts(self) -> int:
        return sum(len(row) for row in self.ids)

    def ranked_pairs(self, limit: int | None = None) -> tuple[tuple[int, int, float], ...]:
        pairs = [
            (layer, expert, count)
            for layer, (ids, counts) in enumerate(zip(self.ids, self.counts))
            for expert, count in zip(ids, counts)
        ]
        pairs.sort(key=lambda item: (-item[2], item[0], item[1]))
        if limit is not None:
            pairs = pairs[: max(0, int(limit))]
        return tuple(pairs)

    def to_tensors(self) -> dict[str, torch.Tensor]:
        width = SESSION_EXPERT_PROFILE_TOPK
        ids = torch.full((self.num_layers, width), -1, dtype=torch.int16)
        counts = torch.zeros((self.num_layers, width), dtype=torch.float16)
        for layer, (layer_ids, layer_counts) in enumerate(zip(self.ids, self.counts)):
            n = len(layer_ids)
            if n:
                ids[layer, :n] = torch.tensor(layer_ids, dtype=torch.int16)
                counts[layer, :n] = torch.tensor(layer_counts, dtype=torch.float32).clamp(
                    max=torch.finfo(torch.float16).max
                ).to(torch.float16)
        return {
            _VERSION_TENSOR: torch.tensor([self.version], dtype=torch.int16),
            _IDS_TENSOR: ids,
            _COUNTS_TENSOR: counts,
        }

    @classmethod
    def from_tensors(
        cls, tensors: Mapping[str, torch.Tensor]
    ) -> SessionExpertProfile | None:
        """Decode a profile, returning ``None`` for a versioned-field absence."""
        if _VERSION_TENSOR not in tensors:
            return None
        version_tensor = tensors[_VERSION_TENSOR].reshape(-1)
        if version_tensor.numel() != 1:
            raise ValueError("invalid session expert profile version tensor")
        version = int(version_tensor[0])
        if version != SESSION_EXPERT_PROFILE_VERSION:
            raise ValueError(f"unsupported session expert profile version {version}")
        ids_tensor = tensors.get(_IDS_TENSOR)
        counts_tensor = tensors.get(_COUNTS_TENSOR)
        if ids_tensor is None or counts_tensor is None or ids_tensor.shape != counts_tensor.shape:
            raise ValueError("incomplete session expert profile tensors")
        if ids_tensor.ndim != 2 or ids_tensor.shape[1] > SESSION_EXPERT_PROFILE_TOPK:
            raise ValueError("invalid session expert profile geometry")
        ids_rows: list[tuple[int, ...]] = []
        count_rows: list[tuple[float, ...]] = []
        for raw_ids, raw_counts in zip(ids_tensor.tolist(), counts_tensor.tolist()):
            pairs = [
                (int(expert), float(count))
                for expert, count in zip(raw_ids, raw_counts)
                if int(expert) >= 0 and float(count) > 0
            ]
            pairs.sort(key=lambda pair: (-pair[1], pair[0]))
            ids_rows.append(tuple(expert for expert, _ in pairs))
            count_rows.append(tuple(count for _, count in pairs))
        return cls(tuple(ids_rows), tuple(count_rows), version=version)


@dataclass(frozen=True)
class SessionPrefetchPlan:
    promote: tuple[tuple[int, tuple[int, ...]], ...]
    willneed: tuple[tuple[int, tuple[int, ...]], ...]
    protected: tuple[tuple[int, int], ...]

    @property
    def expert_count(self) -> int:
        return sum(len(experts) for _, experts in self.promote) + sum(
            len(experts) for _, experts in self.willneed
        )


class SessionProtectionRegistry:
    """Host-side ownership for bounded live-session protection hints."""

    def __init__(self) -> None:
        self._live: dict[int, tuple[tuple[int, int], ...]] = {}

    def admit(self, uid: int, pairs: Sequence[tuple[int, int]]) -> None:
        self._live[int(uid)] = tuple((int(layer), int(expert)) for layer, expert in pairs)

    def release(self, uid: int) -> tuple[tuple[int, int], ...]:
        return self._live.pop(int(uid), ())

    def all(self) -> frozenset[tuple[int, int]]:
        return frozenset(pair for pairs in self._live.values() for pair in pairs)


def plan_session_prefetch(
    profile: SessionExpertProfile,
    layer_residency: Sequence[str],
    *,
    hot_experts: Mapping[int, Sequence[int]] | None = None,
    protect_limit: int = 64,
) -> SessionPrefetchPlan:
    """Split a profile into H2D and WILLNEED work without touching model state."""
    hot = {
        int(layer): frozenset(int(e) for e in experts)
        for layer, experts in (hot_experts or {}).items()
    }
    promote: list[tuple[int, tuple[int, ...]]] = []
    willneed: list[tuple[int, tuple[int, ...]]] = []
    for layer, experts in enumerate(profile.ids):
        if not experts or layer >= len(layer_residency):
            continue
        residency = layer_residency[layer]
        if residency == "pinned":
            promote.append((layer, tuple(experts)))
        elif residency == "disk":
            hot_ids = tuple(expert for expert in experts if expert in hot.get(layer, ()))
            cold_ids = tuple(expert for expert in experts if expert not in hot.get(layer, ()))
            if hot_ids:
                promote.append((layer, hot_ids))
            if cold_ids:
                willneed.append((layer, cold_ids))
    protected = tuple(
        (layer, expert) for layer, expert, _ in profile.ranked_pairs(protect_limit)
    )
    return SessionPrefetchPlan(tuple(promote), tuple(willneed), protected)


def update_profile_sketch(
    old_ids: torch.Tensor,
    old_counts: torch.Tensor,
    routes: torch.Tensor,
    *,
    decay: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized bounded heavy-hitter update used by eager and captured decode."""
    width = old_ids.shape[1]
    candidate_ids = torch.cat((old_ids, routes.to(torch.int32)), dim=1)
    candidate_values = torch.cat(
        (old_counts * float(decay), torch.ones_like(routes, dtype=torch.float32)), dim=1
    )
    valid = candidate_ids >= 0
    equal = candidate_ids.unsqueeze(2) == candidate_ids.unsqueeze(1)
    totals = (equal * candidate_values.unsqueeze(1)).sum(dim=2)
    positions = torch.arange(candidate_ids.shape[1], device=candidate_ids.device)
    duplicate = (
        equal & (positions.view(1, 1, -1) < positions.view(1, -1, 1))
    ).any(dim=2)
    scores = totals.masked_fill(~valid | duplicate, float("-inf"))
    values, indices = torch.topk(scores, width, dim=1)
    new_ids = candidate_ids.gather(1, indices)
    keep = torch.isfinite(values) & (values > 0)
    return (
        torch.where(keep, new_ids, new_ids.new_full((), -1)),
        torch.where(keep, values, values.new_zeros(())),
    )


def profile_storage_bytes(num_layers: int, top_k: int = SESSION_EXPERT_PROFILE_TOPK) -> int:
    """Serialized payload bytes, excluding safetensors header metadata."""
    return 2 + int(num_layers) * int(top_k) * (2 + 2)


PROFILE_TENSOR_NAMES = frozenset({_VERSION_TENSOR, _IDS_TENSOR, _COUNTS_TENSOR})


__all__ = [
    "PROFILE_TENSOR_NAMES",
    "SESSION_ADAPT_INJECTION_WEIGHT",
    "SESSION_EXPERT_PROFILE_TOPK",
    "SESSION_EXPERT_PROFILE_VERSION",
    "SessionExpertProfile",
    "SessionPrefetchPlan",
    "SessionProtectionRegistry",
    "plan_session_prefetch",
    "profile_storage_bytes",
    "update_profile_sketch",
]
