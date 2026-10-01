from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from freetoken.memory import cgroup_memory_headroom
from freetoken.utils import init_logger


logger = init_logger(__name__)

_DEFAULT_RESERVE_FRACTION = 0.15
_MIN_RESERVE_GIB = 8.0
_PIN_SPLIT = 28
_PAGER_SPLIT = 22


@dataclass(frozen=True)
class HostMemoryInfo:
    total_gib: float
    available_gib: float
    swap_total_gib: float = 0.0
    swap_free_gib: float = 0.0

    @property
    def swap_used_gib(self) -> float:
        return max(0.0, self.swap_total_gib - self.swap_free_gib)


@dataclass(frozen=True)
class HostMemoryBudgets:
    total_gib: float
    available_gib: float
    reserve_gib: float
    ceiling_gib: float
    pin_gib: float
    hot_staging_gib: float
    prefill_scratch_gib: float
    pager_gib: float
    remainder_gib: float
    pin_derived: bool
    pager_derived: bool


def read_linux_memory_info(
    path: str | os.PathLike[str] = "/proc/meminfo",
) -> HostMemoryInfo:
    """Read the Linux host-memory counters used by the expert-tier governor."""
    values: dict[str, int] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        key, separator, raw = line.partition(":")
        if not separator:
            continue
        fields = raw.split()
        if fields:
            values[key] = int(fields[0])

    missing = [name for name in ("MemTotal", "MemAvailable") if name not in values]
    if missing:
        raise RuntimeError(
            f"{path} is missing required host-memory fields: {', '.join(missing)}"
        )

    kib_per_gib = 2**20
    return HostMemoryInfo(
        total_gib=values["MemTotal"] / kib_per_gib,
        available_gib=values["MemAvailable"] / kib_per_gib,
        swap_total_gib=values.get("SwapTotal", 0) / kib_per_gib,
        swap_free_gib=values.get("SwapFree", 0) / kib_per_gib,
    )


def apply_cgroup_bound(
    memory: HostMemoryInfo, cgroup_remaining_bytes: int | None
) -> tuple[HostMemoryInfo, str]:
    """Clamp host counters to the cgroup headroom; return ``(memory, bound)``.

    ``bound`` is ``"host"`` (no finite cgroup limit, or it is looser than the
    host figure; ``memory`` is returned unchanged) or ``"cgroup"``. Total is
    clamped too so the default reserve scales with what the cgroup can hold,
    not with the host's MemTotal. Swap counters are left as reported.
    """
    if cgroup_remaining_bytes is None:
        return memory, "host"
    cgroup_gib = cgroup_remaining_bytes / 2**30
    if cgroup_gib >= memory.available_gib:
        return memory, "host"
    return (
        HostMemoryInfo(
            # Total must stay positive; with zero headroom available=0 gates anyway.
            total_gib=(
                min(memory.total_gib, cgroup_gib) if cgroup_gib > 0 else memory.total_gib
            ),
            available_gib=cgroup_gib,
            swap_total_gib=memory.swap_total_gib,
            swap_free_gib=memory.swap_free_gib,
        ),
        "cgroup",
    )


def default_host_cache_reserve_gib(total_gib: float) -> float:
    return max(_MIN_RESERVE_GIB, total_gib * _DEFAULT_RESERVE_FRACTION)


def _validate_optional_budget(name: str, value: float | None) -> None:
    if value is None:
        return
    if not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number")


def _overflow_message(
    memory: HostMemoryInfo,
    reserve_gib: float,
    ceiling_gib: float,
    pin_gib: float,
    hot_staging_gib: float,
    prefill_scratch_gib: float,
    pager_gib: float,
) -> str:
    committed = pin_gib + hot_staging_gib + prefill_scratch_gib + pager_gib
    overflow_gib = committed - ceiling_gib
    remainder_gib = ceiling_gib - committed
    return (
        "Host memory budget exceeds fitted ceiling: "
        f"total={memory.total_gib:.2f} GiB, "
        f"available={memory.available_gib:.2f} GiB, "
        f"reserve={reserve_gib:.2f} GiB, "
        f"ceiling={ceiling_gib:.2f} GiB, "
        f"pinned_banks={pin_gib:.2f} GiB, "
        f"hot_staging={hot_staging_gib:.2f} GiB, "
        f"prefill_scratch={prefill_scratch_gib:.2f} GiB, "
        f"pager={pager_gib:.2f} GiB, "
        f"remainder={remainder_gib:.2f} GiB, "
        f"overflow={overflow_gib:.2f} GiB"
    )


def fit_host_memory_budgets(
    memory: HostMemoryInfo,
    *,
    reserve_gib: float | None,
    pin_gib: float | None,
    pager_gib: float | None,
    hot_staging_gib: float = 0.0,
    prefill_scratch_gib: float = 0.0,
) -> HostMemoryBudgets:
    """Fit explicit and derived expert-tier budgets under one host-memory ceiling."""
    _validate_optional_budget("--host-cache-reserve-gib", reserve_gib)
    _validate_optional_budget("FREETOKEN_PIN_BUDGET_GB", pin_gib)
    _validate_optional_budget("--moe-pager-budget-gib", pager_gib)
    _validate_optional_budget("HOT staging", hot_staging_gib)
    _validate_optional_budget("prefill scratch", prefill_scratch_gib)
    if memory.total_gib <= 0 or memory.available_gib < 0:
        raise ValueError(
            "MemTotal must be positive and MemAvailable must be non-negative"
        )

    resolved_reserve = (
        default_host_cache_reserve_gib(memory.total_gib)
        if reserve_gib is None
        else float(reserve_gib)
    )
    ceiling = max(0.0, min(memory.total_gib, memory.available_gib) - resolved_reserve)
    pin_derived = pin_gib is None
    pager_derived = pager_gib is None
    explicit_pin = 0.0 if pin_gib is None else float(pin_gib)
    explicit_pager = 0.0 if pager_gib is None else float(pager_gib)

    fixed = float(hot_staging_gib) + float(prefill_scratch_gib)
    if explicit_pin + explicit_pager + fixed > ceiling:
        raise ValueError(
            _overflow_message(
                memory,
                resolved_reserve,
                ceiling,
                explicit_pin,
                float(hot_staging_gib),
                float(prefill_scratch_gib),
                explicit_pager,
            )
        )

    remainder = ceiling - fixed - explicit_pin - explicit_pager
    if pin_derived and pager_derived:
        resolved_pin = remainder * _PIN_SPLIT / (_PIN_SPLIT + _PAGER_SPLIT)
        resolved_pager = remainder - resolved_pin
    elif pin_derived:
        resolved_pin = remainder
        resolved_pager = explicit_pager
    elif pager_derived:
        resolved_pin = explicit_pin
        resolved_pager = remainder
    else:
        resolved_pin = explicit_pin
        resolved_pager = explicit_pager

    return HostMemoryBudgets(
        total_gib=memory.total_gib,
        available_gib=memory.available_gib,
        reserve_gib=resolved_reserve,
        ceiling_gib=ceiling,
        pin_gib=resolved_pin,
        hot_staging_gib=float(hot_staging_gib),
        prefill_scratch_gib=float(prefill_scratch_gib),
        pager_gib=resolved_pager,
        remainder_gib=max(
            0.0,
            ceiling - fixed - resolved_pin - resolved_pager,
        ),
        pin_derived=pin_derived,
        pager_derived=pager_derived,
    )


def _hot_staging_gib(config) -> float:
    if float(getattr(config, "moe_hot_expert_budget_gib", 0.0) or 0.0) <= 0:
        return 0.0
    from freetoken.moe.hot_adapt import hot_staging_budget_bytes

    max_swap = int(float(getattr(config, "moe_hot_adapt_max_swap_gib", 0.5)) * 2**30)
    return hot_staging_budget_bytes(max_swap) / 2**30


def cpu_prefill_workspace_tokens(config) -> int:
    """Largest chunk that can take the CPU prefill path, bounding its workspace.

    Staged execution selects GPU staging for every chunk at or above the
    inclusive crossover, so the CPU workspace never needs more rows than one
    below it. Ordinary CPU prefill keeps the full scheduler chunk.
    """
    capacity = int(getattr(config, "max_extend_tokens", 2048) or 0)
    if getattr(config, "moe_disk_prefill", "cpu") == "staged":
        crossover = int(getattr(config, "moe_disk_prefill_min_tokens", 1024) or 1024)
        capacity = min(capacity, max(1, crossover - 1))
    return capacity


def _prefill_scratch_gib(config) -> float:
    """Charge the CPU MoE prefill workspaces the executor can actually allocate."""
    model = getattr(config, "model_config", None)
    hidden = int(getattr(model, "hidden_size", 0) or 0)
    intermediate = int(getattr(model, "moe_intermediate_size", 0) or 0)
    top_k = int(getattr(model, "num_experts_per_tok", 0) or 0)
    tokens = cpu_prefill_workspace_tokens(config)
    total = 0
    cpu_tier = (
        getattr(config, "moe_backend", "offload") in ("cpu", "hybrid")
        or bool(getattr(config, "moe_cpu_layers", None))
        or bool(getattr(config, "moe_disk_layers", None))
        or getattr(config, "moe_disk_prefill", "cpu") == "staged"
        or float(getattr(config, "moe_hot_expert_budget_gib", 0.0) or 0.0) > 0
    )
    if (
        cpu_tier
        and getattr(config, "moe_cpu_prefill_batch", "on") == "on"
        and hidden > 0 and intermediate > 0 and top_k > 0 and tokens > 0
    ):
        rows = tokens * top_k
        # Mirrors cpu_executor._prefill_batch_buffer_nbytes plus its pinned
        # x/ids/weights/y transfer buffers.
        total += rows * (
            4 * hidden + 3 * intermediate + hidden // 4
            + intermediate // 4 + 8 * 32 + 8
        )
        total += tokens * (4 * hidden + 8 * top_k)
    if (
        getattr(config, "moe_disk_prefill", "cpu") in ("cpu", "staged")
        and getattr(config, "moe_prefill_coalesce", "populate") == "populate"
        and (
            getattr(config, "moe_disk_layers", None)
            or getattr(config, "moe_disk_prefill", "cpu") == "staged"
            or float(getattr(config, "moe_hot_expert_budget_gib", 0.0) or 0.0) > 0
        )
    ):
        total += 32 << 20
    if getattr(config, "moe_disk_prefill", "cpu") == "staged":
        # DiskPrefillStaging owns two fixed 32 MiB pinned buffers. Reserve them
        # before fitting expert banks, in addition to CPU fallback workspaces.
        total += 64 << 20
    return total / 2**30


def govern_host_memory(
    config,
    *,
    memory: HostMemoryInfo | None = None,
    environ: Mapping[str, str] | None = None,
    _logger=logger,
) -> HostMemoryBudgets:
    """Resolve an engine config's host budgets and report startup pressure."""
    bound_note = None
    if memory is None:
        memory = read_linux_memory_info()
        host_memory = memory
        memory, bound = apply_cgroup_bound(memory, cgroup_memory_headroom())
        bound_note = (
            f"Host memory bound: {bound} "
            f"(host available={host_memory.available_gib:.2f} GiB, "
            f"effective available={memory.available_gib:.2f} GiB)"
        )
    if environ is None:
        environ = os.environ

    pin_env = environ.get("FREETOKEN_PIN_BUDGET_GB")
    try:
        pin_gib = float(pin_env) if pin_env is not None and pin_env.strip() else None
    except ValueError as exc:
        raise ValueError(
            "FREETOKEN_PIN_BUDGET_GB must be a finite non-negative number"
        ) from exc
    budgets = fit_host_memory_budgets(
        memory,
        reserve_gib=getattr(config, "host_cache_reserve_gib", None),
        pin_gib=pin_gib,
        pager_gib=getattr(config, "moe_pager_budget_gib", None),
        hot_staging_gib=_hot_staging_gib(config),
        prefill_scratch_gib=_prefill_scratch_gib(config),
    )
    if bound_note is not None:
        _logger.info_rank0(bound_note)
    object.__setattr__(config, "host_cache_reserve_gib", budgets.reserve_gib)
    object.__setattr__(config, "moe_pin_budget_gib", budgets.pin_gib)
    object.__setattr__(config, "moe_pager_budget_gib", budgets.pager_gib)

    pin_source = "derived" if budgets.pin_derived else "explicit"
    pager_source = "derived" if budgets.pager_derived else "explicit"
    _logger.info_rank0(
        "Host memory budget table: "
        f"total={budgets.total_gib:.2f} GiB, "
        f"available={budgets.available_gib:.2f} GiB, "
        f"ceiling={budgets.ceiling_gib:.2f} GiB, "
        f"pinned_banks={budgets.pin_gib:.2f} GiB ({pin_source}), "
        f"hot_staging={budgets.hot_staging_gib:.2f} GiB, "
        f"prefill_scratch={budgets.prefill_scratch_gib:.2f} GiB, "
        f"pager={budgets.pager_gib:.2f} GiB ({pager_source}), "
        f"reserve={budgets.reserve_gib:.2f} GiB, "
        f"remainder={budgets.remainder_gib:.2f} GiB, "
        f"target pin:pager={_PIN_SPLIT}:{_PAGER_SPLIT}"
    )

    if memory.swap_total_gib > 0 and memory.swap_used_gib > memory.swap_total_gib / 2:
        _logger.warning_rank0(
            "HOST MEMORY PRESSURE: swap is more than half full at startup "
            f"(used={memory.swap_used_gib:.2f} GiB, "
            f"total={memory.swap_total_gib:.2f} GiB). Likely cause: pinned banks, "
            "HOT staging, prefill scratch, and pager budgets plus file-cache demand "
            "exceeded host RAM."
        )

    return budgets
