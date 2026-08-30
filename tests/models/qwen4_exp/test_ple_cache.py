"""Pinned hot-row PLE cache allocation, staging parity, profiling, and replay."""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from freetoken.models.qwen4_exp.ple import (
    CachedTable,
    ClockSlotAllocator,
    load_ple_row_profile,
    ple_cache_capacity_rows,
    ple_packed_row_nbytes,
    write_ple_row_profile,
)
from freetoken.models.qwen4_exp.weight import load_ple_table

from .common import requires_cuda
from .test_weight import (
    NGRAM_SHARDS,
    QUANT_NGRAM_DIM,
    _quantized_ple_checkpoint,
)


def _mapped_table(tmp_path, table_format: str):
    reference = _quantized_ple_checkpoint(tmp_path, table_format)
    args = SimpleNamespace(
        split_ngram_parts=NGRAM_SHARDS, ngram_head_dim=QUANT_NGRAM_DIM
    )
    return load_ple_table(str(tmp_path), args, backend="cached"), reference


def _cache(table, capacity: int, *, source_capacity: int = 16, device="cpu"):
    return CachedTable(
        table,
        capacity,
        source_capacity,
        device=torch.device(device),
        prefetch=False,
        max_decode_batch_size=2,
        rows_per_token=3,
        slab_rows=3,
    )


def test_clock_allocator_evicts_without_displacing_current_batch_rows():
    cache = ClockSlotAllocator(num_rows=10, capacity=3)
    first = cache.resolve(torch.tensor([1, 2, 3]))
    assert first.hits == 0
    assert first.misses == 3
    assert first.evictions == 0

    second = cache.resolve(torch.tensor([1, 4, 5]))
    assert second.hits == 1
    assert second.misses == 2
    assert second.evictions == 2
    assert cache.row_to_slot[1] >= 0
    assert cache.row_to_slot[4] >= 0
    assert cache.row_to_slot[5] >= 0
    assert sum(cache.row_to_slot[row] < 0 for row in (2, 3)) == 2
    assert torch.equal(
        second.slot_ids,
        cache.row_to_slot.index_select(0, torch.tensor([1, 4, 5], dtype=torch.int64)),
    )


@pytest.mark.parametrize(
    "table_format, expected_row_bytes",
    [("fp8", 36), ("int4g16", 20), ("e2m1g16", 20)],
)
def test_cache_budget_math_counts_packed_data_and_scales(
    tmp_path, table_format, expected_row_bytes
):
    table, _reference = _mapped_table(tmp_path, table_format)
    assert ple_packed_row_nbytes(table) == expected_row_bytes
    budget = (5 * expected_row_bytes + 0.25) / (1 << 30)
    assert ple_cache_capacity_rows(budget, table) == 5


@pytest.mark.parametrize("table_format", ["fp8", "int4g16", "e2m1g16"])
def test_cache_miss_install_matches_reference_rows(tmp_path, table_format):
    table, reference = _mapped_table(tmp_path, table_format)
    cache = _cache(table, capacity=7)
    ids = torch.tensor([[0, 9, 0], [table.num_rows - 1, 9, 3]])

    cache.prepare_decode(ids)
    got = cache.lookup(torch.zeros_like(ids))
    want = reference.index_select(0, ids.reshape(-1)).view(ids.shape[0], -1)
    assert torch.equal(got, want)
    assert torch.equal(cache.local_ids[:2], cache.row_to_slot.index_select(0, ids.reshape(-1)).view_as(ids))
    assert cache.cache_stats() == {
        "hits": 0,
        "misses": 6,
        "evictions": 0,
        "installed_rows": 4,
        "hit_rate": 0.0,
        "overflow_fallbacks": 0,
    }
    cache.finish_decode(record_event=False)

    cache.prepare_decode(ids)
    assert cache.cache_stats()["hit_rate"] == 0.5


def test_prefill_union_over_capacity_uses_pure_staged_fallback(tmp_path, caplog):
    table, reference = _mapped_table(tmp_path, "fp8")
    cache = _cache(table, capacity=2)
    ids = torch.tensor([[0, 1, 2]])

    got = cache.lookup(ids)
    want = reference.index_select(0, ids.reshape(-1)).view(1, -1)
    assert torch.equal(got, want)
    assert cache.overflow_fallbacks == 1
    assert "falling back to pure disk staging" in caplog.text


def test_row_profile_roundtrip_and_stable_frequency_order(tmp_path):
    path = tmp_path / "ple-profile.json"
    write_ple_row_profile(str(path), Counter({7: 4, 2: 9, 5: 4}))
    assert load_ple_row_profile(str(path), num_rows=8) == [2, 5, 7]


def test_warm_profile_installs_only_cache_capacity(tmp_path):
    table, _reference = _mapped_table(tmp_path, "fp8")
    cache = _cache(table, capacity=3)
    assert cache.warm([8, 4, 2, 1]) == 3
    assert [int(cache.row_to_slot[row]) >= 0 for row in (8, 4, 2, 1)] == [
        True,
        True,
        True,
        False,
    ]
    assert cache.cache_stats()["installed_rows"] == 0


@requires_cuda
@pytest.mark.parametrize("table_format", ["fp8", "int4g16", "e2m1g16"])
def test_cached_ple_capture_replay_tracks_evolving_row_sets(tmp_path, table_format):
    table, reference = _mapped_table(tmp_path, table_format)
    cache = _cache(table, capacity=8, device="cuda")
    placeholder = torch.zeros((2, 3), dtype=torch.int64, device="cuda")
    first = torch.tensor([[0, 1, 2], [3, 4, 0]])
    second = torch.tensor([[0, 5, 6], [7, 2, 0]])

    cache.prepare_decode(first)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = cache.lookup(placeholder)
    cache.finish_decode(record_event=False)
    # Capture records the gather without executing it: replay before asserting.
    cache.prepare_decode(first)
    graph.replay()
    cache.finish_decode(record_event=True)
    torch.cuda.synchronize()
    assert torch.equal(
        captured,
        reference.index_select(0, first.reshape(-1)).view(first.shape[0], -1).cuda(),
    )

    cache.prepare_decode(second)
    graph.replay()
    cache.finish_decode(record_event=True)
    torch.cuda.synchronize()
    assert torch.equal(
        captured,
        reference.index_select(0, second.reshape(-1)).view(second.shape[0], -1).cuda(),
    )
