"""GPU-free tests for the live host page-cache monitor (fake /proc files)."""

from __future__ import annotations

import pytest

from freetoken.engine.host_memory import (
    HostCacheMonitor,
    HostMemoryBudgets,
    fit_host_memory_budgets,
    HostMemoryInfo,
)


class _Proc:
    """A fake /proc tree whose counters the test advances."""

    def __init__(self, root):
        self.root = root
        (root / "self").mkdir()
        self.set(majflt=0, minflt=0, pgmajfault=0, available_gib=40.0, cached_gib=20.0)

    def set(self, *, majflt, minflt, pgmajfault, available_gib, cached_gib):
        # comm may contain spaces and parens; fields after it start at state (field 3).
        fields = ["S", "1", "1", "1", "0", "-1", "0", str(minflt), "0", str(majflt), "0"]
        (self.root / "self/stat").write_text("42 (free token) " + " ".join(fields) + "\n")
        (self.root / "vmstat").write_text(f"nr_free_pages 1\npgmajfault {pgmajfault}\n")
        (self.root / "meminfo").write_text(
            f"MemTotal:       65000000 kB\nMemAvailable:   {int(available_gib * 2**20)} kB\n"
            f"Cached:         {int(cached_gib * 2**20)} kB\n"
        )


def _monitor(tmp_path, **kwargs):
    proc = _Proc(tmp_path)
    warnings: list[str] = []
    clock = {"t": 1000.0}
    kwargs.setdefault("disk_tier_cache_gib", 3.64)
    kwargs.setdefault("prefix_cache_headroom_gib", 3.0)
    mon = HostCacheMonitor(
        warn=warnings.append, proc_root=tmp_path, clock=lambda: clock["t"], **kwargs
    )
    return mon, proc, warnings, clock


def test_sample_reads_counters_and_computes_per_step_fault_rate(tmp_path):
    mon, proc, warnings, _ = _monitor(tmp_path)
    first = mon.sample("decode", 40)
    assert first["majflt_per_step"] is None  # no baseline yet
    assert first["mem_available_gib"] == pytest.approx(40.0)
    assert first["cached_gib"] == pytest.approx(20.0)
    proc.set(majflt=800, minflt=5000, pgmajfault=900, available_gib=38.0, cached_gib=21.0)
    snap = mon.sample("decode", 40)
    assert snap["majflt_per_step"] == pytest.approx(20.0)
    assert snap["majflt_delta"] == 800
    assert snap["minflt_delta"] == 5000
    assert snap["system_pgmajfault_delta"] == 900
    assert not snap["pressure"] and warnings == []


def test_pressure_when_available_cache_falls_below_estimate(tmp_path):
    mon, proc, warnings, _ = _monitor(tmp_path)
    mon.sample("decode", 40)
    proc.set(majflt=0, minflt=0, pgmajfault=0, available_gib=5.0, cached_gib=1.0)
    snap = mon.sample("decode", 40)
    assert snap["pressure"]
    assert "MemAvailable 5.00 GiB is below" in snap["pressure_reasons"][0]
    assert len(warnings) == 1
    assert warnings[0].startswith("HOST FILE CACHE PRESSURE (live)")
    assert "--host-cache-reserve-gib" in warnings[0]


def test_pressure_when_faults_per_step_exceed_threshold(tmp_path):
    mon, proc, warnings, _ = _monitor(tmp_path, max_majflt_per_step=100)
    mon.sample("decode", 10)
    proc.set(majflt=2000, minflt=0, pgmajfault=0, available_gib=40, cached_gib=20)
    snap = mon.sample("decode", 10)  # 200 per step
    assert snap["pressure"] and "major faults per decode step" in snap["pressure_reasons"][0]
    assert len(warnings) == 1


def test_warning_fires_on_flip_and_is_rate_limited(tmp_path):
    mon, proc, warnings, clock = _monitor(tmp_path, warn_interval_s=300)
    low = dict(majflt=0, minflt=0, pgmajfault=0, available_gib=1.0, cached_gib=1.0)
    ok = dict(low, available_gib=40.0)
    proc.set(**low)
    mon.sample("decode", 1)
    mon.sample("decode", 1)  # still true: no repeat
    assert len(warnings) == 1
    proc.set(**ok)
    clock["t"] += 10
    assert not mon.sample("decode", 1)["pressure"]
    proc.set(**low)
    clock["t"] += 10
    assert mon.sample("decode", 1)["pressure"]  # flipped again inside the window
    assert len(warnings) == 1
    proc.set(**ok)
    mon.sample("decode", 1)
    proc.set(**low)
    clock["t"] += 400
    mon.sample("decode", 1)
    assert len(warnings) == 2


def test_prefill_in_window_skips_fault_judgement_and_keeps_the_verdict(tmp_path):
    mon, proc, warnings, _ = _monitor(tmp_path, max_majflt_per_step=100)
    mon.sample("decode", 10)
    proc.set(majflt=5000, minflt=0, pgmajfault=0, available_gib=40, cached_gib=20)
    mon.note_prefill()  # a prefill chunk ran in this window: its faults are not decode's
    snap = mon.sample("decode", 10)
    assert snap["majflt_per_step"] is None and not snap["pressure"]
    proc.set(majflt=10000, minflt=0, pgmajfault=0, available_gib=40, cached_gib=20)
    assert mon.sample("decode", 10)["pressure"]  # clean decode window: 500/step
    proc.set(majflt=10001, minflt=0, pgmajfault=0, available_gib=40, cached_gib=20)
    # a prefill sample does not clear the decode fault verdict
    assert mon.sample("prefill", 1)["pressure"]
    proc.set(majflt=10002, minflt=0, pgmajfault=0, available_gib=40, cached_gib=20)
    assert not mon.sample("decode", 10)["pressure"]  # clean low-fault window clears it


def test_missing_proc_files_degrade_to_none_without_pressure(tmp_path):
    mon = HostCacheMonitor(
        disk_tier_cache_gib=3.0, proc_root=tmp_path / "nope", warn=lambda m: None
    )
    snap = mon.sample("decode", 40)
    assert snap["mem_available_gib"] is None and snap["majflt_per_step"] is None
    assert not snap["pressure"]
    assert "n/a" in mon.status_fragment(snap)


def test_no_estimate_means_no_cache_pressure(tmp_path):
    mon, proc, warnings, _ = _monitor(
        tmp_path, disk_tier_cache_gib=0.0, prefix_cache_headroom_gib=0.0
    )
    proc.set(majflt=0, minflt=0, pgmajfault=0, available_gib=0.1, cached_gib=0.1)
    assert not mon.sample("decode", 1)["pressure"]


def test_stats_block_schema_and_fresh_handoff(tmp_path):
    memory = HostMemoryInfo(total_gib=61.91, available_gib=57.22)
    budgets = fit_host_memory_budgets(
        memory, reserve_gib=None, pin_gib=None, pager_gib=None,
        disk_tier_cache_gib=3.64, prefix_cache_headroom_gib=3.0,
    )
    proc = _Proc(tmp_path)
    mon = HostCacheMonitor.from_budgets(budgets, proc_root=tmp_path, warn=lambda m: None)
    assert mon.pop_fresh() is None
    mon.sample("decode", 40)
    block = mon.pop_fresh()
    assert mon.pop_fresh() is None  # handed out once per sample
    assert set(block) == {
        "estimate", "live", "pressure", "pressure_reasons", "max_majflt_per_step",
    }
    assert block["estimate"] == {
        "reserve_gib": round(budgets.reserve_gib, 2),
        "disk_tier_cache_gib": 3.64,
        "prefix_cache_headroom_gib": 3.0,
    }
    assert set(block["live"]) >= {
        "majflt_per_step", "cached_gib", "mem_available_gib", "pressure",
    }
    assert block["max_majflt_per_step"] == 1000.0
    assert block["pressure"] is False


def test_status_fragment_fields(tmp_path):
    mon, proc, _, _ = _monitor(tmp_path)
    mon.sample("decode", 40)
    proc.set(majflt=400, minflt=0, pgmajfault=0, available_gib=39.5, cached_gib=20.25)
    decode = mon.status_fragment(mon.sample("decode", 40))
    assert decode == (
        ", majflt_per_step: 10.0, cached_gib: 20.25, mem_available_gib: 39.50, "
        "host_cache_pressure: 0"
    )
    proc.set(majflt=700, minflt=0, pgmajfault=0, available_gib=39.5, cached_gib=20.25)
    mon.note_prefill()
    prefill = mon.status_fragment(mon.sample("prefill", 3))
    assert prefill.startswith(", majflt_per_chunk: 100.0, ")
