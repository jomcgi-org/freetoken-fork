from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

from freetoken.memory import cgroup_memory_bounds
from freetoken.utils import init_logger


logger = init_logger(__name__)

_DEFAULT_RESERVE_FRACTION = 0.15
_MIN_RESERVE_GIB = 8.0
_PIN_SPLIT = 28
_PAGER_SPLIT = 22
# Page-cache demand the default reserve does not model. The disk prefix cache
# restores and writes (safetensors + fsync) through the page cache; the PLE
# table's hot rows are read through it whenever the table is not pinned.
_PREFIX_CACHE_HEADROOM_GIB = 3.0
_PLE_HOT_ROWS_GIB = 1.0
# Layer-major prefill stages each layer once per group, so only the layer being
# computed and the one being read ahead need to be cache-resident.
_LAYER_MAJOR_RESIDENT_LAYERS = 2


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
    disk_tier_cache_gib: float = 0.0
    prefix_cache_headroom_gib: float = 0.0


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
    memory: HostMemoryInfo,
    cgroup_limit_bytes: int | None,
    cgroup_remaining_bytes: int | None,
) -> tuple[HostMemoryInfo, str]:
    """Clamp host counters to the cgroup; return ``(memory, bound)``.

    ``total`` becomes min(MemTotal, cgroup limit) and ``available`` becomes
    min(MemAvailable, limit - current). Total follows the limit, not the
    headroom, so the default reserve does not shrink as the process allocates.
    ``bound`` is ``"host"`` (no finite limit, or looser than the host figures;
    ``memory`` is returned unchanged) or ``"cgroup"``. Swap is left as reported.
    """
    if cgroup_limit_bytes is None or cgroup_remaining_bytes is None:
        return memory, "host"
    limit_gib = cgroup_limit_bytes / 2**30
    remaining_gib = cgroup_remaining_bytes / 2**30
    total = min(memory.total_gib, limit_gib)
    available = min(memory.available_gib, remaining_gib)
    if total == memory.total_gib and available == memory.available_gib:
        return memory, "host"
    return (
        HostMemoryInfo(
            total_gib=total,
            available_gib=available,
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
    disk_tier_cache_gib: float = 0.0,
    prefix_cache_headroom_gib: float = 0.0,
) -> HostMemoryBudgets:
    """Fit explicit and derived expert-tier budgets under one host-memory ceiling.

    A derived reserve is ``max(default, disk_tier_cache + prefix_cache_headroom)``
    so the page cache DISK-tier reads and the prefix cache compete for is not
    handed to the pinned banks and pager. An explicit reserve is used as given.
    """
    _validate_optional_budget("--host-cache-reserve-gib", reserve_gib)
    _validate_optional_budget("FREETOKEN_PIN_BUDGET_GB", pin_gib)
    _validate_optional_budget("--moe-pager-budget-gib", pager_gib)
    _validate_optional_budget("HOT staging", hot_staging_gib)
    _validate_optional_budget("prefill scratch", prefill_scratch_gib)
    _validate_optional_budget("DISK-tier cache", disk_tier_cache_gib)
    _validate_optional_budget("prefix-cache headroom", prefix_cache_headroom_gib)
    if memory.total_gib <= 0 or memory.available_gib < 0:
        raise ValueError(
            "MemTotal must be positive and MemAvailable must be non-negative"
        )

    resolved_reserve = (
        max(
            default_host_cache_reserve_gib(memory.total_gib),
            float(disk_tier_cache_gib) + float(prefix_cache_headroom_gib),
        )
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
        disk_tier_cache_gib=float(disk_tier_cache_gib),
        prefix_cache_headroom_gib=float(prefix_cache_headroom_gib),
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


def _disk_capable_geometry(config) -> tuple[int, int, int] | None:
    """``(num_moe_layers, num_experts, expert_bytes)`` of a disk-capable checkpoint."""
    model = getattr(config, "model_config", None)
    layers = int(getattr(model, "num_moe_layers", 0) or 0)
    experts = int(getattr(model, "num_experts", 0) or 0)
    path = getattr(config, "model_path", None)
    if layers <= 0 or experts <= 0 or not path:
        return None
    try:
        from freetoken.checkpoint.safetensors_bank_index import (
            indexed_bank_byte_breakdown,
        )
        from freetoken.moe.expert_banks import ftw_bank_byte_breakdown

        breakdown = ftw_bank_byte_breakdown(path) or indexed_bank_byte_breakdown(path)
    except Exception:
        return None
    if breakdown is None or breakdown[0] % (layers * experts):
        return None
    return layers, experts, breakdown[0] // (layers * experts)


def _estimated_disk_layers(config, pin_gib: float) -> tuple[int, int, int, int] | None:
    """``(disk_layers, num_experts, expert_bytes, num_moe_layers)`` or ``None``.

    Explicit ``--moe-disk-layers`` is authoritative. Otherwise the layers the pin
    budget cannot hold go to the DISK tier, as engine residency planning does.
    """
    spec = getattr(config, "moe_disk_layers", None)
    if spec:
        model = getattr(config, "model_config", None)
        layers = int(getattr(model, "num_moe_layers", 0) or 0)
        experts = int(getattr(model, "num_experts", 0) or 0)
        geometry = _disk_capable_geometry(config)
        if geometry is not None:
            layers, experts, expert_bytes = geometry
        else:
            from freetoken.moe.expert_banks import bank_bytes_per_expert

            expert_bytes = bank_bytes_per_expert(model) or 0
        if layers <= 0 or experts <= 0 or expert_bytes <= 0:
            return None
        try:
            from freetoken.engine.engine import _parse_disk_layers_spec

            count = len(_parse_disk_layers_spec(spec, layers))
        except Exception:
            return None
        return count, experts, expert_bytes, layers
    if getattr(config, "moe_backend", "offload") not in ("offload", "hybrid"):
        return None
    geometry = _disk_capable_geometry(config)
    if geometry is None:
        return None
    layers, experts, expert_bytes = geometry
    pinned = int(pin_gib * 2**30 // (experts * expert_bytes))
    return max(0, layers - pinned), experts, expert_bytes, layers


def _disk_tier_cache_gib(config, pin_gib: float = 0.0) -> float:
    """Page cache the DISK tier's staged reads want, as a floor not the whole table.

    Per disk layer, the experts one ``max_extend_tokens`` chunk routes to at
    ``top_k`` (expected distinct under uniform routing) times the expert bytes,
    for every disk layer a chunk-major prefill re-reads each chunk. Layer-major
    prefill stages each layer once per group, so only a couple of layers need to
    be resident. Plus a fixed allowance for the PLE table's hot rows when it is
    read through the file cache.
    """
    total = 0.0
    if getattr(config, "ple_backend", "pinned") != "pinned":
        total += _PLE_HOT_ROWS_GIB
    estimate = _estimated_disk_layers(config, pin_gib)
    if estimate is None:
        return total
    disk_layers, experts, expert_bytes, _ = estimate
    top_k = int(getattr(getattr(config, "model_config", None),
                        "num_experts_per_tok", 0) or 0)
    tokens = int(getattr(config, "max_extend_tokens", 2048) or 0)
    if disk_layers <= 0 or top_k <= 0 or tokens <= 0:
        return total
    distinct = experts * (1.0 - (1.0 - min(1.0, top_k / experts)) ** tokens)
    resident = disk_layers
    if int(getattr(config, "prefill_layer_major_tokens", 0) or 0) > 0:
        resident = min(disk_layers, _LAYER_MAJOR_RESIDENT_LAYERS)
    return total + resident * distinct * expert_bytes / 2**30


def _prefix_cache_headroom_gib(config) -> float:
    if float(getattr(config, "kv_disk_cache_gib", 0.0) or 0.0) > 0:
        return _PREFIX_CACHE_HEADROOM_GIB
    return 0.0


def read_process_residency(
    path: str | os.PathLike[str] = "/proc/self/status",
) -> dict[str, float]:
    """Locked, pinned and resident GiB of this process (``VmLck``/``VmPin``/``VmRSS``)."""
    out: dict[str, float] = {}
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    for line in lines:
        key, _, raw = line.partition(":")
        if key in ("VmLck", "VmPin", "VmRSS"):
            fields = raw.split()
            if fields:
                out[key] = int(fields[0]) / 2**20
    return out


def log_measured_host_residency(
    budgets: HostMemoryBudgets, *, _logger=logger, status_path="/proc/self/status",
) -> None:
    """Report what the process holds after the banks are pinned, beside the estimate.

    Locked pages make ``free``/``MemAvailable`` understate residency, so the
    measured figures come from the process itself.
    """
    measured = read_process_residency(status_path)
    if not measured:
        return
    estimate = (
        budgets.pin_gib + budgets.hot_staging_gib
        + budgets.prefill_scratch_gib + budgets.pager_gib
    )
    _logger.info_rank0(
        "Host memory measured: "
        + ", ".join(f"{k}={v:.2f} GiB" for k, v in measured.items())
        + f" | estimate: pinned_banks={budgets.pin_gib:.2f} GiB, "
        f"committed (banks+staging+scratch+pager)={estimate:.2f} GiB, "
        f"disk_tier_cache={budgets.disk_tier_cache_gib:.2f} GiB (page cache, not held)"
    )


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
        memory, bound = apply_cgroup_bound(memory, *cgroup_memory_bounds())
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
    explicit_reserve = getattr(config, "host_cache_reserve_gib", None)
    fit_kwargs = dict(
        reserve_gib=explicit_reserve,
        pin_gib=pin_gib,
        pager_gib=getattr(config, "moe_pager_budget_gib", None),
        hot_staging_gib=_hot_staging_gib(config),
        prefill_scratch_gib=_prefill_scratch_gib(config),
    )
    prefix_headroom = _prefix_cache_headroom_gib(config)
    budgets = fit_host_memory_budgets(memory, **fit_kwargs)
    if explicit_reserve is None:
        # The disk layer count follows the pin budget, so size the DISK-tier demand
        # from the budget the default reserve yields (one pass: folding the demand
        # in shrinks the pin budget, which would only grow the estimate further).
        # A fold that would overflow explicit budgets is dropped with a warning.
        disk_cache = _disk_tier_cache_gib(config, budgets.pin_gib)
        if disk_cache + prefix_headroom > budgets.reserve_gib:
            try:
                budgets = fit_host_memory_budgets(
                    memory,
                    **fit_kwargs,
                    disk_tier_cache_gib=disk_cache,
                    prefix_cache_headroom_gib=prefix_headroom,
                )
            except ValueError:
                _logger.warning_rank0(
                    "Host cache reserve cannot cover the DISK-tier page-cache "
                    f"demand ({disk_cache + prefix_headroom:.2f} GiB) without "
                    "overflowing the explicit budgets; keeping the default reserve."
                )
    budgets = replace(
        budgets,
        disk_tier_cache_gib=_disk_tier_cache_gib(config, budgets.pin_gib),
        prefix_cache_headroom_gib=prefix_headroom,
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
        f"disk_tier_cache={budgets.disk_tier_cache_gib:.2f} GiB, "
        f"prefix_cache_headroom={budgets.prefix_cache_headroom_gib:.2f} GiB, "
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

    cache_room = budgets.reserve_gib + budgets.remainder_gib
    demand = _disk_tier_cache_gib(config, budgets.pin_gib) + prefix_headroom
    if demand > cache_room + 1e-9:
        _logger.warning_rank0(
            "HOST FILE CACHE PRESSURE: the pinned budgets fit but leave "
            f"{cache_room:.2f} GiB of page cache for an estimated "
            f"{demand:.2f} GiB of DISK-tier reads and prefix-cache I/O. The "
            "server will start, then decode and prefill will fault to disk. "
            "Lower --moe-hot-expert-budget-gib / --kv-reserve-tokens or raise "
            "--host-cache-reserve-gib."
        )

    return budgets


# Major faults per decode step above which the DISK tier is treated as crawling.
# Half of --moe-cpu-willneed-fault-ceiling (2000), the rate at which the CPU
# executor already abandons its own WILLNEED optimisation as unsafe: warning
# there would be too late to be useful, while healthy decode with warm banks
# faults orders of magnitude less. Not calibrated on hardware; see #82.
DEFAULT_PRESSURE_MAJFLT_PER_STEP = 1000.0
_PRESSURE_WARN_INTERVAL_S = 300.0
_GIB = 2**30


def _read_kv_kib(path: Path, keys: tuple[str, ...]) -> dict[str, float]:
    out: dict[str, float] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    for line in lines:
        key, _, raw = line.partition(":")
        if key in keys:
            fields = raw.split()
            if fields:
                out[key] = int(fields[0]) / 2**20  # kB -> GiB
    return out


def _read_self_faults(proc: Path) -> tuple[int, int] | None:
    """``(minflt, majflt)`` of this process from ``/proc/self/stat``."""
    try:
        tail = (proc / "self/stat").read_text(encoding="utf-8").rpartition(") ")[2].split()
        return int(tail[7]), int(tail[9])
    except (OSError, ValueError, IndexError):
        return None


def _read_pgmajfault(proc: Path) -> int | None:
    try:
        for line in (proc / "vmstat").read_text(encoding="utf-8").splitlines():
            if line.startswith("pgmajfault "):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


class HostCacheMonitor:
    """Live page-cache counters and the file-cache pressure flag.

    Reads a handful of procfs files per :meth:`sample`; callers invoke it at most
    once per status-line interval or prefill chunk, never per decode step.
    ``steps`` is the number of decode forwards (or prefill chunks) since the last
    sample, which turns the fault delta into a per-step rate.

    Pressure is true when the memory the DISK tier's page cache can occupy
    (``MemAvailable``: free plus reclaimable cache; pinned and locked pages are
    not in it) is below the startup estimate ``disk_tier_cache + prefix-cache
    headroom``, or when decode major faults per step exceed ``max_majflt_per_step``.
    A flip to true logs one WARNING, rate-limited to ``warn_interval_s``.
    """

    def __init__(
        self,
        *,
        reserve_gib: float = 0.0,
        disk_tier_cache_gib: float = 0.0,
        prefix_cache_headroom_gib: float = 0.0,
        max_majflt_per_step: float = DEFAULT_PRESSURE_MAJFLT_PER_STEP,
        warn=None,
        warn_interval_s: float = _PRESSURE_WARN_INTERVAL_S,
        proc_root: str | os.PathLike[str] = "/proc",
        clock=None,
    ) -> None:
        import time

        self.reserve_gib = float(reserve_gib)
        self.disk_tier_cache_gib = float(disk_tier_cache_gib)
        self.prefix_cache_headroom_gib = float(prefix_cache_headroom_gib)
        self.max_majflt_per_step = float(max_majflt_per_step)
        self._warn = warn if warn is not None else logger.warning_rank0
        self._warn_interval_s = warn_interval_s
        self._proc = Path(proc_root)
        self._clock = clock or time.monotonic
        self._base_faults: tuple[int, int] | None = None
        self._base_pgmajfault: int | None = None
        self._last_warn: float | None = None
        self._dirty = False
        self._prefill_since_sample = False
        self._fault_reason: str | None = None
        self.pressure = False
        self.reasons: list[str] = []
        self.latest: dict | None = None

    @classmethod
    def from_budgets(cls, budgets: HostMemoryBudgets, **kwargs) -> "HostCacheMonitor":
        return cls(
            reserve_gib=budgets.reserve_gib,
            disk_tier_cache_gib=budgets.disk_tier_cache_gib,
            prefix_cache_headroom_gib=budgets.prefix_cache_headroom_gib,
            **kwargs,
        )

    @property
    def demand_gib(self) -> float:
        return self.disk_tier_cache_gib + self.prefix_cache_headroom_gib

    def note_prefill(self) -> None:
        """A prefill chunk ran since the last sample (its faults are not decode's)."""
        self._prefill_since_sample = True

    def sample(self, kind: str, steps: int = 1) -> dict:
        """Read the counters, update the pressure flag and return the snapshot.

        ``kind`` is ``"decode"`` or ``"prefill"``. The per-step fault rate is only
        meaningful, and only judged against the threshold, for a decode sample
        whose window held no prefill chunk.
        """
        now = self._clock()
        faults = _read_self_faults(self._proc)
        pgmajfault = _read_pgmajfault(self._proc)
        mem = _read_kv_kib(self._proc / "meminfo", ("Cached", "MemAvailable"))
        majflt_delta = None
        if faults is not None and self._base_faults is not None:
            majflt_delta = faults[1] - self._base_faults[1]
        minflt_delta = (
            faults[0] - self._base_faults[0]
            if faults is not None and self._base_faults is not None else None
        )
        sys_majflt_delta = (
            pgmajfault - self._base_pgmajfault
            if pgmajfault is not None and self._base_pgmajfault is not None else None
        )
        self._base_faults, self._base_pgmajfault = faults, pgmajfault
        steps = max(1, int(steps))
        majflt_per_step = None
        if majflt_delta is not None and kind == "decode" and not self._prefill_since_sample:
            majflt_per_step = majflt_delta / steps
        self._prefill_since_sample = False

        available = mem.get("MemAvailable")
        reasons: list[str] = []
        if (
            available is not None
            and self.demand_gib > 0
            and available < self.demand_gib
        ):
            reasons.append(
                f"MemAvailable {available:.2f} GiB is below the estimated DISK-tier "
                f"page-cache demand {self.demand_gib:.2f} GiB "
                f"(disk_tier_cache={self.disk_tier_cache_gib:.2f} + "
                f"prefix_cache_headroom={self.prefix_cache_headroom_gib:.2f})"
            )
        if majflt_per_step is not None:
            # Only a clean decode window re-judges the fault condition; any other
            # sample keeps the last verdict rather than clearing it.
            self._fault_reason = (
                f"{majflt_per_step:.0f} major faults per decode step exceeds "
                f"{self.max_majflt_per_step:.0f}"
                if majflt_per_step > self.max_majflt_per_step else None
            )
        if self._fault_reason:
            reasons.append(self._fault_reason)
        flipped = bool(reasons) and not self.pressure
        self.pressure, self.reasons = bool(reasons), reasons
        if flipped and (
            self._last_warn is None or now - self._last_warn >= self._warn_interval_s
        ):
            self._last_warn = now
            self._warn(
                "HOST FILE CACHE PRESSURE (live): " + "; ".join(reasons) + ". "
                "DISK-tier reads are likely page-cache misses; lower "
                "--moe-hot-expert-budget-gib / --kv-reserve-tokens or raise "
                "--host-cache-reserve-gib."
            )
        self.latest = {
            "kind": kind,
            "pressure": self.pressure,
            "pressure_reasons": list(reasons),
            "majflt_per_step": majflt_per_step,
            "majflt_delta": majflt_delta,
            "minflt_delta": minflt_delta,
            "system_pgmajfault_delta": sys_majflt_delta,
            "cached_gib": mem.get("Cached"),
            "mem_available_gib": available,
            "steps": steps,
        }
        self._dirty = True
        return self.latest

    def pop_fresh(self) -> dict | None:
        """The ``host_memory`` block for a snapshot not yet handed out, else ``None``."""
        if not self._dirty:
            return None
        self._dirty = False
        return self.stats_block()

    def stats_block(self) -> dict:
        """The ``host_memory`` document: the startup estimate beside live values."""
        live = dict(self.latest or {})
        return {
            "estimate": {
                "reserve_gib": round(self.reserve_gib, 2),
                "disk_tier_cache_gib": round(self.disk_tier_cache_gib, 2),
                "prefix_cache_headroom_gib": round(self.prefix_cache_headroom_gib, 2),
            },
            "live": live or None,
            "pressure": bool(live.get("pressure", False)),
            "pressure_reasons": list(live.get("pressure_reasons", [])),
            "max_majflt_per_step": self.max_majflt_per_step,
        }

    def status_fragment(self, snap: dict) -> str:
        """Fields appended to a status line: only what the line did not carry."""
        parts = []
        if snap.get("kind") == "decode":
            rate = snap.get("majflt_per_step")
            parts.append(
                "majflt_per_step: " + ("n/a" if rate is None else f"{rate:.1f}")
            )
        else:
            delta = snap.get("majflt_delta")
            parts.append(
                "majflt_per_chunk: "
                + ("n/a" if delta is None else f"{delta / snap['steps']:.1f}")
            )
        for key, name in (("cached_gib", "cached_gib"),
                          ("mem_available_gib", "mem_available_gib")):
            value = snap.get(key)
            parts.append(f"{name}: " + ("n/a" if value is None else f"{value:.2f}"))
        parts.append(f"host_cache_pressure: {int(bool(snap.get('pressure')))}")
        return ", " + ", ".join(parts)
