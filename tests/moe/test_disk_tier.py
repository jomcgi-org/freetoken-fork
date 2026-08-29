"""File-backed FTW expert-bank residency tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def _write_bf16_ftw(path, layers: list[tuple[torch.Tensor, torch.Tensor]]):
    from freetoken.checkpoint.ftw import FTWWriter, layer_bank_entry_name

    writer = FTWWriter(str(path))
    for layer_id, (gate_up, down) in enumerate(layers):
        writer.add_tensor(
            layer_bank_entry_name("gate_up", layer_id), gate_up,
            kind="experts_bank",
        )
        writer.add_tensor(
            layer_bank_entry_name("down", layer_id), down,
            kind="experts_bank",
        )
    return writer.finalize({"quant_format": "bf16", "num_moe_layers": len(layers)})


def test_ftw_disk_banks_are_aligned_mapped_and_byte_exact(tmp_path):
    from freetoken.checkpoint.ftw import ALIGN, load_ftw_banks
    from freetoken.moe.host_banks import HostResidency

    layers = [
        (
            torch.arange(4 * 16 * 8, dtype=torch.int32).view(4, 16, 8),
            torch.arange(4 * 8 * 8, dtype=torch.int32).view(4, 8, 8) + 10_000 * layer_id,
        )
        for layer_id in range(2)
    ]
    index = _write_bf16_ftw(tmp_path, layers)
    bank_entries = [t for t in index["tensors"] if t["kind"] == "experts_bank"]
    assert bank_entries
    assert all(entry["global_off"] % ALIGN == 0 for entry in bank_entries)

    banks = load_ftw_banks(
        str(tmp_path), num_layers=2,
        layer_residency=[HostResidency.DISK.value] * 2,
    )
    assert banks is not None
    assert banks.layer_residency == [HostResidency.DISK.value] * 2
    for layer_id, (gate_up, down) in enumerate(layers):
        assert torch.equal(banks.sources["gate_up"][layer_id], gate_up)
        assert torch.equal(banks.sources["down"][layer_id], down)
        mapped = banks.sources["gate_up"][layer_id]._freetoken_host_bank
        assert mapped.residency is HostResidency.DISK
        assert mapped.prefetch_experts([0, 0]) >= 1


def test_disk_residency_rejects_non_ftw_checkpoint(tmp_path):
    from freetoken.moe.expert_banks import load_expert_banks
    from freetoken.moe.host_banks import HostResidency

    config = SimpleNamespace(num_moe_layers=1)
    with pytest.raises(ValueError, match=r"ft checkpoint"):
        load_expert_banks(
            str(tmp_path), config, device=torch.device("cpu"), dtype=torch.bfloat16,
            layer_residency=[HostResidency.DISK.value],
        )


def test_ftw_writer_keeps_mmap_padding_in_one_shard(tmp_path):
    from freetoken.checkpoint.ftw import FTWReader, FTWWriter

    writer = FTWWriter(str(tmp_path), shard_limit=8192)
    writer.add_tensor("first", torch.zeros(4096, dtype=torch.uint8))
    writer.add_tensor("bank#L00000", torch.zeros(5000, dtype=torch.uint8))
    index = writer.finalize({})
    entry = next(t for t in index["tensors"] if t["name"] == "bank#L00000")
    reader = FTWReader(str(tmp_path))
    path, file_offset, mapped_length = reader.file_region(entry)
    assert path.endswith("freetoken-00001.ftw")
    assert file_offset == 0
    assert mapped_length == 8192


@pytest.mark.parametrize(
    ("ids", "stride", "expected"),
    [
        ([0, 3, 3, -1], 1024, [(0, 4096)]),
        ([0, 4], 1024, [(0, 8192)]),
        ([0, 8], 1024, [(0, 4096), (8192, 4096)]),
        ([0, 1], 6000, [(0, 12288)]),
    ],
)
def test_disk_prefetch_page_dedup_and_coalescing(ids, stride, expected):
    from freetoken.moe.host_banks import coalesced_page_ranges

    assert coalesced_page_ranges(ids, stride) == expected


def test_disk_prefetch_stats_are_per_layer_and_flush_major_faults(monkeypatch):
    import freetoken.moe.cpu_executor as cpu_executor

    executor = cpu_executor.CpuMoeExecutor.__new__(cpu_executor.CpuMoeExecutor)
    executor.num_layers = 3
    executor._disk_banks = {1: [object()], 2: [object()]}
    executor._disk_prefetch_calls = [0, 3, 3]
    executor._disk_prefetch_pages = [0, 17, 19]
    executor._disk_decode_steps = 6
    executor._disk_major_fault_base = 10
    monkeypatch.setattr(cpu_executor, "_major_faults", lambda: 16)

    stats = executor.disk_prefetch_stats(reset=True)
    assert stats["prefetch_calls"] == 6
    assert stats["pages_requested"] == 36
    assert stats["major_faults"] == 6
    assert stats["major_faults_per_decode_step"] == 2.0
    assert stats["per_layer"] == [
        {"layer": 1, "prefetch_calls": 3, "pages_requested": 17},
        {"layer": 2, "prefetch_calls": 3, "pages_requested": 19},
    ]
    assert executor._disk_prefetch_calls == [0, 0, 0]
    assert executor._disk_major_fault_base == 16


def test_disk_layer_routes_to_cpu_and_uses_pageable_prefill_copy():
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.moe.host_banks import HostResidency
    from freetoken.moe.offload_cache import OffloadMoeCache

    if try_get_tp_info() is None:
        set_tp_info(0, 1)
    cache = OffloadMoeCache(
        num_layers=2, num_experts=4, cache_size=8, device=torch.device("cpu"),
        prefill_overlap=False,
    )
    cache.cpu_layer_ids = frozenset({1})
    sources = {
        "gate_up": [torch.randn(4, 16, 8) for _ in range(2)],
        "down": [torch.randn(4, 8, 8) for _ in range(2)],
    }
    cache.set_bank_sources(
        sources,
        layer_residency=[HostResidency.PINNED.value, HostResidency.DISK.value],
    )
    assert cache.is_cpu_layer(1)
    assert cache.is_unpinned_layer(1)

    cache._pending_src_layer = 1
    cache._pending_whole_layer = True
    cache.copy_missing()
    gate_up_cache, down_cache = (bank for _, bank in cache.banks)
    assert torch.equal(gate_up_cache[:4], sources["gate_up"][1])
    assert torch.equal(down_cache[:4], sources["down"][1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_copy_plan_skips_disk_layer():
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.moe.host_banks import HostResidency
    from freetoken.moe.offload_cache import OffloadMoeCache

    if try_get_tp_info() is None:
        set_tp_info(0, 1)
    cache = OffloadMoeCache(
        num_layers=2, num_experts=4, cache_size=8, device=torch.device("cuda"),
        prefill_overlap=False,
    )
    cache.cpu_layer_ids = frozenset({1})
    sources = {
        "gate_up": [torch.randn(4, 16, 8, device="cuda"), torch.randn(4, 16, 8)],
        "down": [torch.randn(4, 8, 8, device="cuda"), torch.randn(4, 8, 8)],
    }
    cache.set_bank_sources(
        sources,
        layer_residency=[HostResidency.PINNED.value, HostResidency.DISK.value],
    )
    assert cache._copy_fused_ok
    assert (cache._copy_src_ptrs[1] == 0).all()
    assert (cache._copy_src_ptrs[0] != 0).all()


def test_cpu_executor_disk_mapping_matches_anonymous_bank(tmp_path):
    from freetoken.checkpoint.ftw import load_ftw_banks

    try:
        from freetoken.kernel import _cpu_moe
    except ImportError:
        pytest.skip("CPU MoE extension is not built")
    if not hasattr(_cpu_moe.CpuMoeExecutor, "set_pre_run_callback"):
        pytest.skip("CPU MoE extension needs rebuilding for DISK prefetch")
    from freetoken.moe.cpu_executor import CpuMoeExecutor
    from freetoken.moe.host_banks import HostBank, HostResidency

    torch.manual_seed(7)
    experts, hidden, inter, top_k = 4, 32, 32, 2
    gate_up = torch.randn(experts, 2 * inter, hidden, dtype=torch.bfloat16)
    down = torch.randn(experts, hidden, inter, dtype=torch.bfloat16)
    _write_bf16_ftw(tmp_path, [(gate_up, down)])
    disk = load_ftw_banks(
        str(tmp_path), num_layers=1,
        layer_residency=[HostResidency.DISK.value],
    )
    assert disk is not None

    gate_up_ram = HostBank(tuple(gate_up.shape), gate_up.dtype)
    down_ram = HostBank(tuple(down.shape), down.dtype)
    gate_up_ram.tensor.copy_(gate_up)
    down_ram.tensor.copy_(down)
    ram_sources = {"gate_up": [gate_up_ram.tensor], "down": [down_ram.tensor]}

    def run(sources):
        cache = SimpleNamespace(
            quant_format="bf16", bank_sources=sources,
            num_layers=1, num_experts=experts,
        )
        executor = CpuMoeExecutor(
            cache, top_k=top_k, activation="silu",
            apply_router_weight_on_input=False, num_threads=1, max_tokens=2,
            device=torch.device("cpu"),
        )
        x = torch.randn(2, hidden, dtype=torch.bfloat16)
        ids = torch.tensor([[0, 2], [3, 1]], dtype=torch.int32)
        weights = torch.tensor([[0.6, 0.4], [0.25, 0.75]], dtype=torch.float32)
        out = torch.empty_like(x)
        task = executor._ext.create_task(
            0, 2, x.data_ptr(), ids.data_ptr(), weights.data_ptr(), out.data_ptr(),
        )
        executor._ext.run_task(task)
        return x, ids, weights, out

    x, ids, weights, ram_out = run(ram_sources)

    disk_cache = SimpleNamespace(
        quant_format="bf16", bank_sources=disk.sources,
        num_layers=1, num_experts=experts,
    )
    disk_executor = CpuMoeExecutor(
        disk_cache, top_k=top_k, activation="silu",
        apply_router_weight_on_input=False, num_threads=1, max_tokens=2,
        device=torch.device("cpu"),
    )
    disk_out = torch.empty_like(x)
    task = disk_executor._ext.create_task(
        0, 2, x.data_ptr(), ids.data_ptr(), weights.data_ptr(), disk_out.data_ptr(),
    )
    disk_executor._ext.run_task(task)
    assert torch.equal(disk_out, ram_out)
