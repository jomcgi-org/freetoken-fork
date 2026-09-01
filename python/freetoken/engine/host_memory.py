from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

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
    pager_gib: float
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
    pager_gib: float,
) -> str:
    overflow_gib = pin_gib + pager_gib - ceiling_gib
    return (
        "Host memory budget exceeds fitted ceiling: "
        f"total={memory.total_gib:.2f} GiB, "
        f"available={memory.available_gib:.2f} GiB, "
        f"reserve={reserve_gib:.2f} GiB, "
        f"ceiling={ceiling_gib:.2f} GiB, "
        f"pin={pin_gib:.2f} GiB, "
        f"pager={pager_gib:.2f} GiB, "
        f"overflow={overflow_gib:.2f} GiB"
    )


def fit_host_memory_budgets(
    memory: HostMemoryInfo,
    *,
    reserve_gib: float | None,
    pin_gib: float | None,
    pager_gib: float | None,
) -> HostMemoryBudgets:
    """Fit explicit and derived expert-tier budgets under one host-memory ceiling."""
    _validate_optional_budget("--host-cache-reserve-gib", reserve_gib)
    _validate_optional_budget("FREETOKEN_PIN_BUDGET_GB", pin_gib)
    _validate_optional_budget("--moe-pager-budget-gib", pager_gib)
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

    if explicit_pin + explicit_pager > ceiling:
        raise ValueError(
            _overflow_message(
                memory,
                resolved_reserve,
                ceiling,
                explicit_pin,
                explicit_pager,
            )
        )

    remainder = ceiling - explicit_pin - explicit_pager
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
        pager_gib=resolved_pager,
        pin_derived=pin_derived,
        pager_derived=pager_derived,
    )


def govern_host_memory(
    config,
    *,
    memory: HostMemoryInfo | None = None,
    environ: Mapping[str, str] | None = None,
    _logger=logger,
) -> HostMemoryBudgets:
    """Resolve an engine config's host budgets and report startup pressure."""
    if memory is None:
        memory = read_linux_memory_info()
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
    )
    object.__setattr__(config, "host_cache_reserve_gib", budgets.reserve_gib)
    object.__setattr__(config, "moe_pin_budget_gib", budgets.pin_gib)
    object.__setattr__(config, "moe_pager_budget_gib", budgets.pager_gib)

    if budgets.pin_derived or budgets.pager_derived:
        pin_source = "derived" if budgets.pin_derived else "explicit"
        pager_source = "derived" if budgets.pager_derived else "explicit"
        _logger.info_rank0(
            "Host memory governor derived split: "
            f"total={budgets.total_gib:.2f} GiB, "
            f"available={budgets.available_gib:.2f} GiB, "
            f"reserve={budgets.reserve_gib:.2f} GiB, "
            f"ceiling={budgets.ceiling_gib:.2f} GiB, "
            f"pin={budgets.pin_gib:.2f} GiB ({pin_source}), "
            f"pager={budgets.pager_gib:.2f} GiB ({pager_source}), "
            f"target pin:pager={_PIN_SPLIT}:{_PAGER_SPLIT}"
        )

    if memory.swap_total_gib > 0 and memory.swap_used_gib > memory.swap_total_gib / 2:
        _logger.warning_rank0(
            "HOST MEMORY PRESSURE: swap is more than half full at startup "
            f"(used={memory.swap_used_gib:.2f} GiB, "
            f"total={memory.swap_total_gib:.2f} GiB). Likely cause: expert pin and "
            "pager budgets plus file-cache demand exceeded host RAM."
        )

    return budgets
