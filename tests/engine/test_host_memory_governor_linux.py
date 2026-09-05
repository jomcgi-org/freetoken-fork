"""GPU-free Linux tests for the /proc-backed expert-tier memory governor."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from freetoken.engine.host_memory import (
    HostMemoryInfo,
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
    config.moe_disk_prefill = "staged"
    config.moe_disk_layers = "all" if explicit_disk else None
    assert _prefill_scratch_gib(config) * 2**30 == cpu_bytes + (64 << 20)


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
