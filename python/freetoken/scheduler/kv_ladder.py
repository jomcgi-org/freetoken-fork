"""CUDA-free policy for growing KV by trading MoE cache slots."""

from __future__ import annotations

from dataclasses import dataclass

from freetoken.utils import div_ceil


DEFAULT_KV_LADDER_STEP_TOKENS = 32_768


class KVLadderCapacityError(ValueError):
    """The requested KV rung cannot fit with the minimum MoE cache."""


@dataclass(frozen=True)
class KVLadderEligibility:
    enabled: bool
    inactive_reasons: tuple[str, ...] = ()


def kv_ladder_eligibility(config) -> KVLadderEligibility:
    """Resolve the single config-only gate shared by engine and scheduler.

    This must run after engine config adjustment has resolved the model, MoE backend,
    cache family, and page geometry, but before auto MoE sizing treats an explicit KV
    capacity as a ladder cap. The scheduler consumes the same result after engine init.
    """
    if getattr(config, "kv_ladder", "off") != "on":
        return KVLadderEligibility(False)

    reasons = []
    if not getattr(config, "moe_cache_auto", False):
        if getattr(config, "moe_cache_rate", None) is not None:
            reasons.append(
                "MoE cache sizing uses --moe-cache-rate; --moe-cache-auto is required"
            )
        elif getattr(config, "moe_cache_size", 0) > 0:
            reasons.append(
                "MoE cache sizing uses --moe-cache-size; --moe-cache-auto is required"
            )
        else:
            reasons.append("--moe-cache-auto is not enabled")
    if config.max_running_req != 1:
        reasons.append("--max-running-requests must be 1")
    if config.tp_info.size != 1:
        reasons.append("TP must be 1")

    model_config = config.model_config
    if getattr(model_config, "dsv4_args", None) is not None:
        reasons.append("DSV4 owned KV is unsupported")

    from freetoken.moe import is_offload_moe_backend

    if not (
        getattr(model_config, "is_moe", False)
        and is_offload_moe_backend(getattr(config, "moe_backend", "auto"))
    ):
        reasons.append("there is no offloaded MoE slot cache")
    return KVLadderEligibility(not reasons, tuple(reasons))


def kv_ladder_requested(config) -> bool:
    """Compatibility predicate for callers that need only the shared decision."""
    return kv_ladder_eligibility(config).enabled


def _trim_protected_rows(
    rows_by_layer: tuple[tuple[int, int], ...], max_rows: int
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    """Keep protected rows balanced across layers and report per-layer losses."""
    remaining = max(0, int(max_rows))
    kept = {layer_id: 0 for layer_id, _ in rows_by_layer}
    capacities = dict(rows_by_layer)
    while remaining:
        progressed = False
        for layer_id, _ in rows_by_layer:
            if kept[layer_id] >= capacities[layer_id]:
                continue
            kept[layer_id] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break
    after = tuple((layer_id, kept[layer_id]) for layer_id, _ in rows_by_layer)
    lost = tuple(
        (layer_id, before - kept[layer_id])
        for layer_id, before in rows_by_layer
        if kept[layer_id] < before
    )
    return after, lost


@dataclass(frozen=True)
class KVLadderPlan:
    required_tokens: int
    current_tokens: int
    target_tokens: int
    target_pages: int
    current_moe_slots: int
    target_moe_slots: int
    protected_rows_after: tuple[tuple[int, int], ...]
    lost_protected_rows: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class KVLadderPolicy:
    step_tokens: int
    max_context_tokens: int
    page_size: int
    pool_budget_bytes: int
    kv_bytes_per_page: int
    moe_bytes_per_slot: int
    min_moe_slots: int
    prefill_overlap: bool
    protected_rows_by_layer: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        positive = (
            "step_tokens",
            "max_context_tokens",
            "page_size",
            "pool_budget_bytes",
            "kv_bytes_per_page",
            "moe_bytes_per_slot",
            "min_moe_slots",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if any(layer_id < 0 or rows < 0 for layer_id, rows in self.protected_rows_by_layer):
            raise ValueError("protected row geometry must be non-negative")

    def next_rung_tokens(self, current_tokens: int) -> int:
        """Return the next aligned rung without exceeding the configured cap."""
        return min(
            self.max_context_tokens,
            div_ceil(current_tokens + 1, self.step_tokens) * self.step_tokens,
        )

    def moe_slots_at_tokens(self, tokens: int, current_moe_slots: int) -> int:
        """Price a KV rung against the shared pool budget."""
        pages = div_ceil(tokens, self.page_size)
        # pages is usable capacity. Every pool also allocates one sentinel page.
        bytes_after_kv = self.pool_budget_bytes - (
            pages + 1
        ) * self.kv_bytes_per_page
        return min(current_moe_slots, bytes_after_kv // self.moe_bytes_per_slot)

    def plan(
        self,
        *,
        current_pages: int,
        current_moe_slots: int,
        input_tokens: int,
        max_output_tokens: int,
        protected_rows_by_layer: tuple[tuple[int, int], ...] | None = None,
        prefill_overlap: bool | None = None,
    ) -> KVLadderPlan | None:
        """Plan one or more rungs before admitting a request that exceeds current KV."""
        current_tokens = current_pages * self.page_size
        required_tokens = input_tokens + max_output_tokens
        if required_tokens <= current_tokens or current_tokens >= self.max_context_tokens:
            return None
        if input_tokens >= self.max_context_tokens:
            return None

        required_rung = div_ceil(required_tokens, self.step_tokens) * self.step_tokens
        target_tokens = min(
            self.max_context_tokens,
            max(self.next_rung_tokens(current_tokens), required_rung),
        )
        target_pages = div_ceil(target_tokens, self.page_size)
        if target_pages <= current_pages:
            return None

        target_moe_slots = self.moe_slots_at_tokens(target_tokens, current_moe_slots)
        if target_moe_slots < self.min_moe_slots:
            max_kv_pages = (
                self.pool_budget_bytes
                - self.min_moe_slots * self.moe_bytes_per_slot
            ) // self.kv_bytes_per_page - 1
            max_kv_tokens = max(0, max_kv_pages * self.page_size)
            raise KVLadderCapacityError(
                f"KV ladder rung {target_tokens} tokens cannot fit while retaining "
                f"the minimum {self.min_moe_slots} MoE slots; the budget permits at "
                f"most {max_kv_tokens} KV tokens"
            )

        overlap = self.prefill_overlap if prefill_overlap is None else prefill_overlap
        dynamic_floor = min(
            (2 if overlap else 1) * self.min_moe_slots,
            target_moe_slots // 2,
        )
        protected_room = max(0, target_moe_slots - dynamic_floor)
        protected_after, lost = _trim_protected_rows(
            (
                self.protected_rows_by_layer
                if protected_rows_by_layer is None
                else protected_rows_by_layer
            ),
            protected_room,
        )
        return KVLadderPlan(
            required_tokens=required_tokens,
            current_tokens=current_tokens,
            target_tokens=min(target_pages * self.page_size, self.max_context_tokens),
            target_pages=target_pages,
            current_moe_slots=current_moe_slots,
            target_moe_slots=target_moe_slots,
            protected_rows_after=protected_after,
            lost_protected_rows=lost,
        )
