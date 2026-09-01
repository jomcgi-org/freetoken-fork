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
    config = SimpleNamespace(host_cache_reserve_gib=10, moe_pager_budget_gib=25)
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
        "pin=60.00 GiB, pager=25.00 GiB, overflow=5.00 GiB"
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
    assert "Likely cause: expert pin and pager budgets" in fake_logger.warnings[0]
