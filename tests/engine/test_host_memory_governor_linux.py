"""GPU-free Linux tests for the /proc-backed expert-tier memory governor."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from freetoken.engine.host_memory import (
    HostMemoryInfo,
    apply_cgroup_bound,
    default_host_cache_reserve_gib,
    fit_host_memory_budgets,
    govern_host_memory,
    read_linux_memory_info,
    _prefill_scratch_gib,
)


pytestmark = pytest.mark.skipif(
    sys.platform != "linux",
    reason="host memory governor depends on Linux /proc/meminfo",
)


class _Logger:
    def __init__(self):
        self.infos = []
        self.warnings = []

    def info_rank0(self, message):
        self.infos.append(message)

    def warning_rank0(self, message):
        self.warnings.append(message)


@pytest.mark.parametrize("explicit_disk", [False, True])
def test_staging_budget_includes_ring_and_cpu_fallback_with_auto_placement(explicit_disk):
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hidden_size=2048, moe_intermediate_size=512, num_experts_per_tok=8,
        ),
        max_extend_tokens=2048, moe_disk_prefill="cpu", moe_disk_layers="all",
    )
    cpu_bytes = _prefill_scratch_gib(config) * 2**30
    assert cpu_bytes > 32 << 20
    # Staged execution bounds the CPU workspace one row below its crossover.
    config.max_extend_tokens = 1023
    bounded_bytes = _prefill_scratch_gib(config) * 2**30
    assert bounded_bytes < cpu_bytes
    config.max_extend_tokens = 2048
    config.moe_disk_prefill = "staged"
    config.moe_disk_layers = "all" if explicit_disk else None
    assert _prefill_scratch_gib(config) * 2**30 == bounded_bytes + (64 << 20)


def test_staged_scratch_charge_does_not_grow_with_chunk_size():
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hidden_size=2048, moe_intermediate_size=512, num_experts_per_tok=8,
        ),
        max_extend_tokens=2048, moe_disk_prefill="staged", moe_disk_layers=None,
        moe_disk_prefill_min_tokens=1024,
    )
    small = _prefill_scratch_gib(config)
    config.max_extend_tokens = 16384
    assert _prefill_scratch_gib(config) == small
    config.moe_disk_prefill_min_tokens = 4096
    assert _prefill_scratch_gib(config) > small


def test_fitting_arithmetic_preserves_explicit_budget_and_uses_remainder():
    budgets = fit_host_memory_budgets(
        HostMemoryInfo(total_gib=100, available_gib=90),
        reserve_gib=10,
        pin_gib=None,
        pager_gib=30,
    )

    assert budgets.ceiling_gib == 80
    assert budgets.pin_gib == 50
    assert budgets.pager_gib == 30
    assert budgets.pin_derived is True
    assert budgets.pager_derived is False


def test_explicit_sum_rejection_has_exact_arithmetic_message():
    config = SimpleNamespace(
        host_cache_reserve_gib=10,
        moe_pager_budget_gib=25,
        moe_hot_expert_budget_gib=48,
        moe_hot_adapt_max_swap_gib=0.5,
        moe_prefill_coalesce="off",
    )
    with pytest.raises(ValueError) as exc_info:
        govern_host_memory(
            config,
            memory=HostMemoryInfo(total_gib=100, available_gib=90),
            environ={"FREETOKEN_PIN_BUDGET_GB": "60"},
            _logger=_Logger(),
        )

    assert str(exc_info.value) == (
        "Host memory budget exceeds fitted ceiling: total=100.00 GiB, "
        "available=90.00 GiB, reserve=10.00 GiB, ceiling=80.00 GiB, "
        "pinned_banks=60.00 GiB, hot_staging=0.56 GiB, "
        "prefill_scratch=0.00 GiB, pager=25.00 GiB, remainder=-5.56 GiB, "
        "overflow=5.56 GiB"
    )


def test_default_derived_split_uses_28_to_22_ratio():
    budgets = fit_host_memory_budgets(
        HostMemoryInfo(total_gib=61, available_gib=61),
        reserve_gib=None,
        pin_gib=None,
        pager_gib=None,
    )

    assert budgets.reserve_gib == pytest.approx(9.15)
    assert budgets.ceiling_gib == pytest.approx(51.85)
    assert budgets.pin_gib == pytest.approx(51.85 * 28 / 50)
    assert budgets.pager_gib == pytest.approx(51.85 * 22 / 50)
    assert budgets.pin_gib / budgets.pager_gib == pytest.approx(28 / 22)


@pytest.mark.parametrize(
    ("total", "available", "expected_pager"),
    [
        (61, 61, 22.730615234375),
        (176, 160, 104.480615234375),
    ],
    ids=["node-4-61g", "g4-176g"],
)
def test_hot_tier_budget_table_arithmetic(total, available, expected_pager):
    config = SimpleNamespace(
        host_cache_reserve_gib=None,
        moe_pager_budget_gib=None,
        moe_hot_expert_budget_gib=48,
        moe_hot_adapt_max_swap_gib=0.5,
        moe_disk_layers="all",
        moe_disk_prefill="cpu",
        moe_prefill_coalesce="populate",
        moe_cpu_prefill_batch="on",
        max_extend_tokens=2048,
        model_config=SimpleNamespace(
            hidden_size=6144,
            moe_intermediate_size=1536,
            num_experts_per_tok=8,
        ),
    )
    fake_logger = _Logger()
    budgets = govern_host_memory(
        config,
        memory=HostMemoryInfo(total_gib=total, available_gib=available),
        environ={"FREETOKEN_PIN_BUDGET_GB": "28"},
        _logger=fake_logger,
    )

    assert budgets.hot_staging_gib == pytest.approx(0.5625)
    assert budgets.prefill_scratch_gib == pytest.approx(0.556884765625)
    assert budgets.pin_gib == 28
    assert budgets.pager_gib == pytest.approx(expected_pager)
    assert budgets.remainder_gib == 0
    assert "pinned_banks=28.00 GiB (explicit)" in fake_logger.infos[0]
    assert "hot_staging=0.56 GiB" in fake_logger.infos[0]
    assert "prefill_scratch=0.56 GiB" in fake_logger.infos[0]
    assert f"pager={expected_pager:.2f} GiB (derived)" in fake_logger.infos[0]


def test_default_reserve_has_eight_gib_floor():
    assert default_host_cache_reserve_gib(40) == 8
    assert default_host_cache_reserve_gib(100) == 15


def test_reads_linux_meminfo_fields(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal:       104857600 kB\n"
        "MemAvailable:    94371840 kB\n"
        "SwapTotal:       10485760 kB\n"
        "SwapFree:         4194304 kB\n"
    )

    assert read_linux_memory_info(meminfo) == HostMemoryInfo(
        total_gib=100,
        available_gib=90,
        swap_total_gib=10,
        swap_free_gib=4,
    )


def test_governor_uses_getattr_for_stub_config_and_warns_on_swap_pressure():
    config = SimpleNamespace(moe_pager_budget_gib=None)
    fake_logger = _Logger()

    budgets = govern_host_memory(
        config,
        memory=HostMemoryInfo(
            total_gib=50,
            available_gib=50,
            swap_total_gib=10,
            swap_free_gib=4,
        ),
        environ={},
        _logger=fake_logger,
    )

    assert config.host_cache_reserve_gib == 8
    assert config.moe_pin_budget_gib == budgets.pin_gib
    assert config.moe_pager_budget_gib == budgets.pager_gib
    assert len(fake_logger.infos) == 1
    assert "target pin:pager=28:22" in fake_logger.infos[0]
    assert len(fake_logger.warnings) == 1
    assert "swap is more than half full" in fake_logger.warnings[0]
    assert "Likely cause: pinned banks, HOT staging, prefill scratch" in fake_logger.warnings[0]


def _fake_host(tmp_path, monkeypatch, *, limit, current, total_gib=100, avail_gib=90):
    """Fake /proc/meminfo + cgroup v2 tree wired into the governor's readers."""
    import functools

    from freetoken.engine import host_memory

    kib = 2**20
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        f"MemTotal: {total_gib * kib} kB\nMemAvailable: {avail_gib * kib} kB\n"
    )
    root = tmp_path / "cgroup"
    (root / "svc").mkdir(parents=True)
    (root / "svc" / "memory.max").write_text(f"{limit}\n")
    (root / "svc" / "memory.current").write_text(f"{current}\n")
    proc = tmp_path / "proc-cgroup"
    proc.write_text("0::/svc\n")
    monkeypatch.setattr(
        host_memory,
        "read_linux_memory_info",
        functools.partial(host_memory.read_linux_memory_info, meminfo),
    )
    monkeypatch.setattr(
        host_memory,
        "cgroup_memory_bounds",
        functools.partial(
            host_memory.cgroup_memory_bounds,
            cgroup_root=root,
            proc_cgroup_path=proc,
        ),
    )


def test_apply_cgroup_bound_takes_the_minimum():
    host = HostMemoryInfo(total_gib=100, available_gib=90, swap_total_gib=4)
    gib = 2**30
    assert apply_cgroup_bound(host, None, None) == (host, "host")
    assert apply_cgroup_bound(host, 200 * gib, 150 * gib) == (host, "host")
    bounded, bound = apply_cgroup_bound(host, 40 * gib, 30 * gib)
    assert bound == "cgroup"
    assert bounded == HostMemoryInfo(
        total_gib=40, available_gib=30, swap_total_gib=4, swap_free_gib=0
    )


def test_governor_unlimited_cgroup_is_unchanged(tmp_path, monkeypatch):
    _fake_host(tmp_path, monkeypatch, limit="max", current=2**30)
    fake_logger = _Logger()
    budgets = govern_host_memory(
        SimpleNamespace(moe_pager_budget_gib=None), environ={}, _logger=fake_logger
    )
    expected = fit_host_memory_budgets(
        HostMemoryInfo(total_gib=100, available_gib=90),
        reserve_gib=None, pin_gib=None, pager_gib=None,
    )
    assert budgets == expected
    assert any("Host memory bound: host" in m for m in fake_logger.infos)


def test_governor_finite_cgroup_limits_derived_budgets(tmp_path, monkeypatch):
    # 40 GiB limit with 10 GiB in use -> 30 GiB headroom, host says 90.
    _fake_host(tmp_path, monkeypatch, limit=40 * 2**30, current=10 * 2**30)
    fake_logger = _Logger()
    config = SimpleNamespace(moe_pager_budget_gib=None)
    budgets = govern_host_memory(config, environ={}, _logger=fake_logger)
    assert budgets.available_gib == pytest.approx(30)
    assert budgets.total_gib == pytest.approx(40)  # the limit, not the headroom
    assert budgets.reserve_gib == 8
    assert budgets.ceiling_gib == pytest.approx(22)
    assert budgets.pin_gib + budgets.pager_gib == pytest.approx(22)
    assert any("Host memory bound: cgroup" in m for m in fake_logger.infos)


def test_governor_cgroup_usage_above_limit_leaves_no_derived_budget(
    tmp_path, monkeypatch
):
    _fake_host(tmp_path, monkeypatch, limit=2**30, current=2 * 2**30)
    budgets = govern_host_memory(
        SimpleNamespace(moe_pager_budget_gib=None), environ={}, _logger=_Logger()
    )
    assert budgets.ceiling_gib == 0
    assert budgets.pin_gib == 0 and budgets.pager_gib == 0


def test_governor_explicit_budget_still_checked_against_cgroup_ceiling(
    tmp_path, monkeypatch
):
    _fake_host(tmp_path, monkeypatch, limit=40 * 2**30, current=10 * 2**30)
    with pytest.raises(ValueError, match="exceeds fitted ceiling"):
        govern_host_memory(
            SimpleNamespace(moe_pager_budget_gib=None),
            environ={"FREETOKEN_PIN_BUDGET_GB": "28"},
            _logger=_Logger(),
        )


def test_governor_without_cgroup_files_is_host_only(tmp_path, monkeypatch):
    _fake_host(tmp_path, monkeypatch, limit="max", current=0)
    from freetoken.engine import host_memory

    monkeypatch.setattr(host_memory, "cgroup_memory_bounds", lambda: (None, None))
    fake_logger = _Logger()
    budgets = govern_host_memory(
        SimpleNamespace(moe_pager_budget_gib=None), environ={}, _logger=fake_logger
    )
    assert budgets.available_gib == 90
    assert any("Host memory bound: host" in m for m in fake_logger.infos)


def test_total_and_reserve_follow_limit_not_headroom(tmp_path, monkeypatch):
    # 200 GiB host, 100 GiB limit, 80 GiB already used: headroom 20 GiB.
    _fake_host(
        tmp_path, monkeypatch, limit=100 * 2**30, current=80 * 2**30,
        total_gib=200, avail_gib=190,
    )
    budgets = govern_host_memory(
        SimpleNamespace(moe_pager_budget_gib=None), environ={}, _logger=_Logger()
    )
    assert budgets.total_gib == pytest.approx(100)
    assert budgets.available_gib == pytest.approx(20)
    assert budgets.reserve_gib == pytest.approx(15)  # 15% of the limit
    assert budgets.ceiling_gib == pytest.approx(5)
