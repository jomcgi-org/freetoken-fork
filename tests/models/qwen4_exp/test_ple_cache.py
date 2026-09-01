"""Pinned hot-row PLE cache allocation, staging parity, profiling, and replay."""

from __future__ import annotations

import ctypes
import time
from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from freetoken.models.qwen4_exp.ple import (
    CachedTable,
    ClockSlotAllocator,
    GpuResidentTable,
    PrefillGatherTable,
    load_ple_row_profile,
    ple_cache_capacity_rows,
    ple_packed_row_nbytes,
    write_ple_row_profile,
)
from freetoken.models.qwen4_exp.weight import load_ple_table
from freetoken.moe.host_banks import HostBank

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


def test_packed_cache_miss_install_uses_one_interleaved_readv(tmp_path):
    table, reference = _mapped_table(tmp_path, "int4g16")
    cache = _cache(table, capacity=7)
    ids = torch.tensor([[0, 9, 0], [table.num_rows - 1, 9, 3]])
    calls = []

    def process_vm_readv(_pid, local, local_count, remote, remote_count, _flags):
        calls.append((local_count, remote_count))
        copied = 0
        for index in range(local_count):
            assert local[index].length == remote[index].length
            ctypes.memmove(local[index].base, remote[index].base, local[index].length)
            copied += local[index].length
        return copied

    cache._reader._process_vm_readv = process_vm_readv
    cache.prepare_decode(ids)
    got = cache.lookup(torch.zeros_like(ids))
    want = reference.index_select(0, ids.reshape(-1)).view(ids.shape[0], -1)

    assert torch.equal(got, want)
    assert calls == [(8, 8)]


def test_packed_cache_miss_install_microbenchmark():
    """Compare 1000 synthetic packed miss rounds with the retained install oracle."""
    steps = 1000
    rows_per_step = 16
    rows_per_shard = 256
    shard_count = 64
    head_dim = 160
    data_banks = tuple(
        HostBank((rows_per_shard, head_dim // 2), torch.uint8)
        for _ in range(shard_count)
    )
    scale_banks = tuple(
        HostBank((rows_per_shard, head_dim // 16), torch.float16)
        for _ in range(shard_count)
    )
    for shard, (data, scales) in enumerate(zip(data_banks, scale_banks)):
        data.tensor.copy_(
            (
                torch.arange(data.tensor.numel(), dtype=torch.int64).view_as(data.tensor)
                + shard * 17
            ).to(torch.uint8)
        )
        scales.tensor.copy_(
            torch.arange(scales.tensor.numel(), dtype=torch.float32)
            .view_as(scales.tensor)
            .add_(shard + 1)
            .mul_(0.001)
        )
    table = SimpleNamespace(
        banks=data_banks,
        scale_banks=scale_banks,
        rows_per_shard=rows_per_shard,
        num_rows=rows_per_shard * shard_count,
        head_dim=head_dim,
        row_nbytes=head_dim // 2,
        weight_scale=1.0,
        format="int4g16",
    )

    def make_cache(source):
        return CachedTable(
            source,
            capacity_rows=rows_per_step * 2,
            source_capacity_rows=rows_per_step,
            device=torch.device("cpu"),
            prefetch=False,
            max_decode_batch_size=1,
            rows_per_token=rows_per_step,
            slab_rows=rows_per_step,
        )

    row_ids = torch.arange(steps * rows_per_step, dtype=torch.int64).view(
        steps, 1, rows_per_step
    )
    reference = make_cache(table)
    optimized = make_cache(table)
    slots = torch.arange(rows_per_step, dtype=torch.int64)

    started = time.perf_counter()
    for ids in row_ids:
        reference._copy_installed_reference(ids.reshape(-1), slots)
    reference_rate = steps / (time.perf_counter() - started)

    started = time.perf_counter()
    for ids in row_ids:
        optimized._copy_installed(ids.reshape(-1), slots)
    optimized_rate = steps / (time.perf_counter() - started)
    assert torch.equal(
        reference._data_slabs[0].tensor, optimized._data_slabs[0].tensor
    )
    assert torch.equal(
        reference._scale_slabs[0].tensor, optimized._scale_slabs[0].tensor
    )

    fp8_banks = tuple(
        HostBank((rows_per_shard, head_dim), torch.uint8)
        for _ in range(shard_count)
    )
    fp8_table = SimpleNamespace(
        banks=fp8_banks,
        scale_banks=(),
        rows_per_shard=rows_per_shard,
        num_rows=rows_per_shard * shard_count,
        head_dim=head_dim,
        row_nbytes=head_dim,
        weight_scale=1.0,
        format="fp8",
    )
    fp8 = make_cache(fp8_table)
    started = time.perf_counter()
    for ids in row_ids:
        fp8._copy_installed(ids.reshape(-1), slots)
    fp8_rate = steps / (time.perf_counter() - started)
    print(
        f"PLE packed miss-install micro-benchmark: before={reference_rate:.1f} steps/sec, "
        f"after={optimized_rate:.1f} steps/sec, speedup={optimized_rate / reference_rate:.2f}x, "
        f"fp8={fp8_rate:.1f} steps/sec, packed/fp8 cost={fp8_rate / optimized_rate:.2f}x"
    )


def test_prefill_union_over_capacity_uses_pure_staged_fallback(tmp_path, caplog):
    table, reference = _mapped_table(tmp_path, "fp8")
    cache = _cache(table, capacity=2)
    ids = torch.tensor([[0, 1, 2]])

    got = cache.lookup(ids)
    want = reference.index_select(0, ids.reshape(-1)).view(1, -1)
    assert torch.equal(got, want)
    assert cache.overflow_fallbacks == 1
    assert "falling back to pure disk staging" in caplog.text


def test_prefill_gather_deduplicates_and_decode_delegates(tmp_path):
    table, reference = _mapped_table(tmp_path, "fp8")
    fallback = GpuResidentTable(reference, dtype=torch.bfloat16)
    gather = PrefillGatherTable(
        fallback,
        table,
        max_prefill_tokens=2,
        rows_per_token=3,
        device=torch.device("cpu"),
    )
    ids = torch.tensor([[0, 2, 0], [table.num_rows - 1, 2, 3]])
    calls = []

    def process_vm_readv(_pid, local, local_count, remote, remote_count, _flags):
        calls.append((local_count, remote_count))
        copied = 0
        for index in range(local_count):
            ctypes.memmove(local[index].base, remote[index].base, local[index].length)
            copied += local[index].length
        return copied

    gather._reader._process_vm_readv = process_vm_readv
    assert gather._reader._stage_bank.tensor.shape[0] == 2 * 3

    assert gather.prepare_prefill(ids)
    placeholder = torch.zeros_like(ids)
    gather.prefetch(placeholder)
    got = gather.lookup(placeholder)
    want = reference.index_select(0, ids.reshape(-1)).view(ids.shape[0], -1)
    assert torch.equal(got, want)
    assert gather.prefill_gather_rows == 4
    assert gather.prefill_gather_ms >= 0
    assert calls == [(8, 8)]

    decode_ids = torch.tensor([[1, 4, 5]])
    assert torch.equal(gather.lookup(decode_ids), fallback.lookup(decode_ids))


def test_prefill_gather_allocation_failure_degrades_to_fallback(
    tmp_path, monkeypatch, caplog
):
    import freetoken.models.qwen4_exp.ple as ple

    table, reference = _mapped_table(tmp_path, "fp8")
    fallback = GpuResidentTable(reference, dtype=torch.bfloat16)

    def fail_allocation(*args, **kwargs):
        raise RuntimeError("synthetic staging OOM")

    monkeypatch.setattr(ple, "DiskStagedTable", fail_allocation)
    gather = PrefillGatherTable(
        fallback,
        table,
        max_prefill_tokens=2,
        rows_per_token=3,
        device=torch.device("cpu"),
    )
    ids = torch.tensor([[0, 1, 2]])

    assert not gather.enabled
    assert not gather.prepare_prefill(ids)
    gather.prefetch(ids)
    assert torch.equal(gather.lookup(ids), fallback.lookup(ids))
    assert gather.prefill_gather_fallbacks == 1
    assert "synthetic staging OOM" in caplog.text


def test_prefill_gather_chunk_failure_degrades_only_that_chunk(tmp_path):
    table, reference = _mapped_table(tmp_path, "fp8")
    fallback = GpuResidentTable(reference, dtype=torch.bfloat16)
    gather = PrefillGatherTable(
        fallback,
        table,
        max_prefill_tokens=2,
        rows_per_token=3,
        device=torch.device("cpu"),
    )
    ids = torch.tensor([[0, 1, 2]])
    real_stage = gather._reader.stage_prefill_rows

    def fail_chunk(_ids):
        raise RuntimeError("synthetic per-chunk allocation failure")

    gather._reader.stage_prefill_rows = fail_chunk

    assert not gather.prepare_prefill(ids)
    gather.prefetch(ids)
    assert torch.equal(gather.lookup(ids), fallback.lookup(ids))

    gather._reader.stage_prefill_rows = real_stage
    assert gather.prepare_prefill(ids)
    gather.prefetch(ids)
    assert torch.equal(gather.lookup(ids), fallback.lookup(ids))


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
def test_prefill_gather_cuda_matches_synthetic_shards(tmp_path, table_format):
    table, reference = _mapped_table(tmp_path, table_format)
    fallback = GpuResidentTable(reference.cuda(), dtype=torch.bfloat16)
    gather = PrefillGatherTable(
        fallback,
        table,
        max_prefill_tokens=2,
        rows_per_token=3,
        device=torch.device("cuda"),
    )
    ids_host = torch.tensor([[0, 2, 0], [table.num_rows - 1, 2, 3]])
    ids_device = ids_host.cuda()

    assert gather.prepare_prefill(ids_host)
    gather.prefetch(ids_device)
    got = gather.lookup(ids_device)
    want = fallback.lookup(ids_device)
    assert torch.equal(got, want)


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
