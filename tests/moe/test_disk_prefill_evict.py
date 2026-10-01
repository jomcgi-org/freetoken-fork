"""GPU-free checks for --moe-disk-prefill-evict cold: range choice, flags, counters."""

import os
from types import SimpleNamespace

import pytest
import torch

from freetoken.distributed import DistributedInfo
from freetoken.engine.config import EngineConfig
from freetoken.moe.offload_cache import OffloadMoeCache
from freetoken.server.args import parse_args


def test_cold_row_ranges_split_advised_from_kept():
    from freetoken.moe.disk_prefill_staging import cold_row_ranges

    # Rows 1-3 and 6 are staged cold; 4 and 5 are decode-hot and stay.
    ranges, kept = cold_row_ranges([6, 1, 2, 3, 4, 5, 2], {4, 5, 9}, 100, base_offset=7)
    assert ranges == [(107, 300), (607, 100)]
    assert kept == 200
    assert cold_row_ranges([0, 1], {0, 1}, 8) == ([], 16)
    assert cold_row_ranges([0, 1], None, 8) == ([(0, 16)], 0)
    assert cold_row_ranges([], {1}, 8) == ([], 0)


def _staging(tmp_path, monkeypatch, evict=True):
    from freetoken.moe.disk_prefill_staging import DiskPrefillStaging

    path = tmp_path / "bank.ftw"
    path.write_bytes(bytes(4096))
    staging = object.__new__(DiskPrefillStaging)
    staging.evict_cold = evict
    staging.evict_advised_bytes = staging.evict_kept_bytes = 0
    calls = []
    monkeypatch.setattr(
        os, "posix_fadvise", lambda fd, off, length, advice: calls.append((off, length, advice))
    )
    return staging, SimpleNamespace(_file_path=str(path)), calls


def test_evict_advises_only_cold_staged_rows(tmp_path, monkeypatch):
    staging, bank, calls = _staging(tmp_path, monkeypatch)
    source = torch.empty((8, 16), dtype=torch.uint8)  # 16-byte rows
    staging._evict_cold_rows(bank, source, [0, 1, 2, 5], {1, 7}, file_offset=64)
    assert calls == [
        (64, 16, os.POSIX_FADV_DONTNEED),
        (96, 16, os.POSIX_FADV_DONTNEED),
        (144, 16, os.POSIX_FADV_DONTNEED),
    ]
    assert staging.take_evict_counters() == (48, 16)
    assert staging.take_evict_counters() == (0, 0)


def test_evict_whole_bank_when_rows_is_none(tmp_path, monkeypatch):
    staging, bank, calls = _staging(tmp_path, monkeypatch)
    source = torch.empty((4, 8), dtype=torch.uint8)
    staging._evict_cold_rows(bank, source, None, {0}, file_offset=0)
    assert calls == [(8, 24, os.POSIX_FADV_DONTNEED)]
    assert staging.take_evict_counters() == (24, 8)


def test_evict_survives_fadvise_failure(tmp_path, monkeypatch):
    staging, bank, _ = _staging(tmp_path, monkeypatch)

    def boom(*_):
        raise OSError("nope")

    monkeypatch.setattr(os, "posix_fadvise", boom)
    staging._evict_cold_rows(bank, torch.empty((2, 8), dtype=torch.uint8), None, None, 0)
    assert staging.take_evict_counters() == (0, 0)


def test_decode_hot_tracking_window():
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    executor = object.__new__(CpuMoeExecutor)
    executor.num_experts = 8
    executor.num_layers = 2
    executor._disk_banks = {1: [object()]}
    executor._disk_decode_steps = executor._disk_route_pairs = executor._disk_distinct_experts = 0
    assert executor.decode_hot_experts(1) == frozenset()
    executor.enable_decode_hot_tracking(window_steps=2)
    executor._prefetch_selected = lambda layer_id, selected: 0
    for routed in ([0, 1], [2], [3]):
        executor.prefetch_experts(1, routed)
    # Steps 0,1,2 touched {0,1},{2},{3}; a window of 2 keeps the last two.
    assert executor.decode_hot_experts(1) == frozenset({2, 3})
    executor.prefetch_experts(1, [0], is_prefill=True)  # prefill is not decode history
    assert executor.decode_hot_experts(1) == frozenset({2, 3})
    assert executor.decode_hot_experts(0) == frozenset()


def test_cli_and_config_defaults_and_validation():
    base = ["--model", "/tmp/nonexistent-model", "--dtype", "bfloat16", "--moe-disk-prefill", "staged"]
    assert parse_args(base)[0].moe_disk_prefill_evict == "off"
    assert parse_args(base + ["--moe-disk-prefill-evict", "cold"])[0].moe_disk_prefill_evict == "cold"
    with pytest.raises(SystemExit):
        parse_args(base + ["--moe-disk-prefill-evict", "all"])
    assert OffloadMoeCache.__dataclass_fields__["moe_disk_prefill_evict"].default == "off"
    kwargs = dict(
        model_path="/tmp/model", tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16,
        moe_disk_prefill="cpu", moe_disk_prefill_evict="cold",
    )
    with pytest.raises(ValueError, match="requires --moe-disk-prefill staged"):
        EngineConfig(**kwargs)
    with pytest.raises(ValueError, match="requires staged DISK prefill"):
        OffloadMoeCache(
            num_layers=1, num_experts=8, cache_size=24, device=torch.device("cpu"),
            moe_disk_prefill="cpu", moe_disk_prefill_evict="cold",
        )
    with pytest.raises(ValueError, match="must be 'off' or 'cold'"):
        EngineConfig(**{**kwargs, "moe_disk_prefill": "staged", "moe_disk_prefill_evict": "x"})


def test_summary_counters_and_keep_rows_plumbing(caplog):
    cache = OffloadMoeCache(
        num_layers=2, num_experts=8, cache_size=24, device=torch.device("cpu"),
        moe_disk_prefill="staged", moe_disk_prefill_evict="cold",
    )
    assert cache._decode_hot_rows(0) is None  # no executor attached yet
    cache.cpu_executor = SimpleNamespace(decode_hot_experts=lambda layer: {layer, 3})
    assert cache._decode_hot_rows(1) == {1, 3}
    cache.moe_disk_prefill_evict = "off"
    assert cache._decode_hot_rows(1) is None

    logged = []
    staging = SimpleNamespace(
        evict_cold=True, take_evict_counters=lambda: (3 << 20, 1 << 20),
    )
    cache._disk_prefill_staging = staging
    cache._lm_bg_staging = None
    import freetoken.moe.offload_cache as module

    monkey = module.logger
    module.logger = SimpleNamespace(info_rank0=logged.append)
    try:
        cache.log_disk_prefill_evict_summary()
    finally:
        module.logger = monkey
    assert logged == ["DISK prefill evict=cold: DONTNEED advised 3 MiB, kept 1 MiB (decode-hot)"]
