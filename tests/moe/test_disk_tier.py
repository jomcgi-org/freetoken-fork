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


def test_ftw_load_builds_compact_hot_rows_and_keeps_cold_rows_file_backed(
    tmp_path, monkeypatch,
):
    from freetoken.checkpoint.ftw import load_ftw_banks
    from freetoken.moe.host_banks import HostResidency

    monkeypatch.setenv("FREETOKEN_SKIP_BANK_PIN", "1")
    gate_up = torch.arange(5 * 4 * 3, dtype=torch.int32).view(5, 4, 3)
    down = torch.arange(5 * 3 * 2, dtype=torch.int32).view(5, 3, 2)
    _write_bf16_ftw(tmp_path, [(gate_up, down)])

    banks = load_ftw_banks(
        str(tmp_path),
        num_layers=1,
        layer_residency=[HostResidency.DISK.value],
        hot_expert_ids={0: (1, 4)},
    )
    assert banks is not None
    assert banks.hot_expert_ids == {0: (1, 4)}
    assert torch.equal(banks.hot_sources["gate_up"][0], gate_up[[1, 4]])
    assert torch.equal(banks.hot_sources["down"][0], down[[1, 4]])
    assert banks.sources["gate_up"][0]._freetoken_host_bank.residency is HostResidency.DISK
    assert banks.hot_sources["gate_up"][0]._freetoken_host_bank.residency is not HostResidency.DISK


def test_ftw_load_allocates_all_cold_hot_capacity(tmp_path, monkeypatch):
    from freetoken.checkpoint.ftw import load_ftw_banks
    from freetoken.moe.host_banks import HostResidency

    monkeypatch.setenv("FREETOKEN_SKIP_BANK_PIN", "1")
    gate_up = torch.arange(5 * 4 * 3, dtype=torch.int32).view(5, 4, 3)
    down = torch.arange(5 * 3 * 2, dtype=torch.int32).view(5, 3, 2)
    _write_bf16_ftw(tmp_path, [(gate_up, down)])

    banks = load_ftw_banks(
        str(tmp_path),
        num_layers=1,
        layer_residency=[HostResidency.DISK.value],
        hot_expert_ids={0: ()},
        hot_expert_capacity={0: 2},
    )
    assert banks is not None
    assert banks.hot_expert_ids == {0: ()}
    assert banks.hot_expert_capacity == {0: 2}
    assert banks.hot_sources["gate_up"][0].shape == (2, 4, 3)


def test_disk_residency_rejects_non_ftw_checkpoint(tmp_path):
    from freetoken.moe.expert_banks import load_expert_banks
    from freetoken.moe.host_banks import HostResidency

    config = SimpleNamespace(num_moe_layers=1)
    with pytest.raises(ValueError, match=r"ft checkpoint"):
        load_expert_banks(
            str(tmp_path), config, device=torch.device("cpu"), dtype=torch.bfloat16,
            layer_residency=[HostResidency.DISK.value],
        )


def test_cache_registers_compact_hot_sources_and_static_row_map():
    from freetoken.moe.host_banks import HostResidency
    from freetoken.moe.offload_cache import OffloadMoeCache

    cache = OffloadMoeCache(
        num_layers=1, num_experts=5, cache_size=5, device=torch.device("cpu"),
        prefill_overlap=False, decode_target="cpu",
    )
    cache.cpu_layer_ids = frozenset({0})
    sources = {
        "gate_up": [torch.arange(5 * 4 * 3).view(5, 4, 3)],
        "down": [torch.arange(5 * 3 * 2).view(5, 3, 2)],
    }
    hot_sources = {
        "gate_up": [sources["gate_up"][0][[1, 4]].contiguous()],
        "down": [sources["down"][0][[1, 4]].contiguous()],
    }
    cache.set_bank_sources(
        sources,
        layer_residency=[HostResidency.DISK.value],
        hot_sources=hot_sources,
        hot_expert_ids={0: (1, 4)},
    )

    assert cache.is_hot_split_layer(0)
    assert cache.hot_row_for_expert[0].tolist() == [-1, 0, -1, -1, 1]
    ids = torch.tensor([[4, 2], [1, 4]], dtype=torch.int32)
    cache.ensure_experts_hot(0, ids)
    assert ids[0, 1].item() == -1
    assert ids[0, 0].item() >= 0
    assert ids[1, 0].item() >= 0
    assert cache.src_indices[:2].tolist() == [0, 1]


def test_disk_stats_report_hot_pair_rate_and_reset():
    from freetoken.moe.offload_cache import OffloadMoeCache

    class Executor:
        _disk_banks = {0: [object()]}

        def disk_prefetch_stats(self, reset=False):
            return {"prefetch_calls": 0}

    cache = OffloadMoeCache(
        num_layers=1, num_experts=4, cache_size=4, device=torch.device("cpu"),
        prefill_overlap=False,
    )
    cache.cpu_executor = Executor()
    cache.stat_hot_pairs.fill_(7)
    cache.stat_hot_total_pairs.fill_(10)
    cache.hot_adapt_enabled = True
    cache._hot_slot_owners = {0: [1]}
    cache.decayed_decode_freq[0] = torch.tensor([1.0, 3.0, 2.0, 4.0])
    cache.hot_adapt_ticks = 2
    cache.hot_adapt_swaps = 3

    stats = cache.disk_prefetch_stats(reset=True)
    assert stats["hot_pair_rate"] == pytest.approx(0.7)
    assert stats["hot_pairs"] == 7
    assert stats["routed_pairs"] == 10
    assert stats["hot_swaps_per_interval"] == pytest.approx(1.5)
    assert stats["decayed_hot_pair_rate"] == pytest.approx(0.3)
    assert cache.stat_hot_pairs.item() == 0
    assert cache.stat_hot_total_pairs.item() == 0


def test_ftw_writer_keeps_mmap_padding_in_one_shard(tmp_path):
    from freetoken.checkpoint.ftw import FTWReader, FTWWriter

    writer = FTWWriter(str(tmp_path), shard_limit=8192)
    writer.add_tensor("first", torch.zeros(4096, dtype=torch.uint8))
    writer.add_tensor("bank#L00000", torch.zeros(5000, dtype=torch.uint8))
    index = writer.finalize({})
    entry = next(t for t in index["tensors"] if t["name"] == "bank#L00000")
    reader = FTWReader(str(tmp_path))
    path, file_offset, source_length = reader.file_region(entry)
    assert path.endswith("freetoken-00001.ftw")
    assert file_offset == 0
    assert source_length == 5000

    path, file_offset, source_length = reader.file_region({
        "name": "misaligned",
        "global_off": 1234,
        "nbytes": 2700,
    })
    assert path.endswith("freetoken-00000.ftw")
    assert file_offset == 1234
    assert source_length == 2700


@pytest.mark.parametrize(
    ("ids", "stride", "expected"),
    [
        ([0, 3, 3, -1], 1024, [(0, 4096)]),
        ([0, 4], 1024, [(0, 8192)]),
        ([0, 8], 1024, [(0, 4096), (8192, 4096)]),
        ([0, 1], 6000, [(0, 12288)]),
        # Repeated tiny PLE rows sharing a page collapse to one advice range.
        ([0, 1, 1], 160, [(0, 4096)]),
    ],
)
def test_disk_prefetch_page_dedup_and_coalescing(ids, stride, expected):
    from freetoken.moe.host_banks import coalesced_page_ranges

    assert coalesced_page_ranges(ids, stride) == expected


def test_disk_prefetch_page_dedup_accounts_for_unaligned_tensor_view():
    from freetoken.moe.host_banks import coalesced_page_ranges

    assert coalesced_page_ranges(
        [0, 1, 1], 160, page_offset=4032,
    ) == [(0, 8192)]


def test_populate_row_ranges_sort_dedupe_and_join_only_adjacent_rows():
    from freetoken.moe.host_banks import coalesced_row_ranges

    assert coalesced_row_ranges(
        [4, 2, 3, 3, -1, 0], 10, limit=50, base_offset=100
    ) == [(100, 10), (120, 30)]


def test_disk_populate_reads_coalesced_range_through_bounded_scratch(
    tmp_path, monkeypatch,
):
    import freetoken.moe.host_banks as host_banks

    source = tmp_path / "populate.ftw"
    source.write_bytes(b"pre" + bytes(range(30)))
    bank = host_banks.HostBank(
        (6, 5),
        torch.uint8,
        backing="file",
        file_path=str(source),
        file_offset=3,
    )
    original_preadv = host_banks.os.preadv
    calls = []

    def tracked_preadv(fd, buffers, offset):
        calls.append((offset, len(buffers[0])))
        return original_preadv(fd, buffers, offset)

    monkeypatch.setattr(host_banks.os, "preadv", tracked_preadv)
    scratch = bytearray(7)

    assert bank.populate_rows([4, 2, 3, 3], scratch) == 15
    assert calls == [(13, 7), (20, 7), (27, 1)]
    assert max(length for _offset, length in calls) <= len(scratch)


def test_disk_prefill_release_marks_only_selected_page_ranges_noreuse(
    tmp_path, monkeypatch,
):
    import freetoken.moe.host_banks as host_banks

    source = tmp_path / "experts.ftw"
    source.write_bytes(b"x" * 8192)
    bank = host_banks.HostBank(
        (2, 4096), torch.uint8, backing="file", file_path=str(source),
    )
    calls = []
    monkeypatch.setattr(host_banks.os, "POSIX_FADV_NOREUSE", 5, raising=False)
    monkeypatch.setattr(
        host_banks.os,
        "posix_fadvise",
        lambda fd, offset, length, advice: calls.append(
            (offset, length, advice)
        ),
        raising=False,
    )

    assert bank.release_rows([1, 1]) == 1
    assert calls == [(4096, 4096, 5)]


def test_disk_prefetch_stats_are_per_layer_and_flush_major_faults(monkeypatch):
    import freetoken.moe.cpu_executor as cpu_executor

    executor = cpu_executor.CpuMoeExecutor.__new__(cpu_executor.CpuMoeExecutor)
    executor.num_layers = 3
    executor._disk_banks = {1: [object()], 2: [object()]}
    executor._disk_prefetch_calls = [0, 3, 3]
    executor._disk_prefetch_pages = [0, 17, 19]
    executor._disk_decode_steps = 6
    executor._disk_route_pairs = 48
    executor._disk_distinct_experts = 30
    executor._prefill_coalesce_experts = 9
    executor._prefill_coalesce_ns = 2_500_000
    executor._prefill_coalesce_degrades = 1
    executor._prefill_populate_bytes = 590_000_000
    executor._prefill_populate_ns = 175_000_000
    executor._disk_major_fault_base = 10
    executor._gpufetch_tasks = {}
    monkeypatch.setattr(cpu_executor, "_major_faults", lambda: 16)

    stats = executor.disk_prefetch_stats(reset=True)
    assert stats["prefetch_calls"] == 6
    assert stats["pages_requested"] == 36
    assert stats["major_faults"] == 6
    assert stats["major_faults_per_decode_step"] == 2.0
    assert stats["distinct_experts_per_step"] == 5.0
    assert stats["dedup_ratio"] == 1.6
    assert stats["moe_prefill_coalesce_experts"] == 9
    assert stats["moe_prefill_coalesce_ms"] == 2.5
    assert stats["moe_prefill_coalesce_degrades"] == 1
    assert stats["moe_prefill_populate_bytes"] == 590_000_000
    assert stats["moe_prefill_populate_ms"] == 175.0
    assert stats["gpufetch_fills_per_step"] == 0.0
    assert stats["gpufetch_fill_us"] == 0.0
    assert stats["per_layer"] == [
        {"layer": 1, "prefetch_calls": 3, "pages_requested": 17},
        {"layer": 2, "prefetch_calls": 3, "pages_requested": 19},
    ]
    assert executor._disk_prefetch_calls == [0, 0, 0]
    assert executor._disk_route_pairs == 0
    assert executor._disk_distinct_experts == 0
    assert executor._prefill_coalesce_experts == 0
    assert executor._prefill_coalesce_ns == 0
    assert executor._prefill_coalesce_degrades == 0
    assert executor._prefill_populate_bytes == 0
    assert executor._prefill_populate_ns == 0
    assert executor._disk_major_fault_base == 16


def test_gpufetch_staging_capacity_tracks_max_distinct_decode_routes():
    from freetoken.moe.offload_cache import disk_gpufetch_capacity

    assert disk_gpufetch_capacity(
        max_tokens=4, top_k=8, num_experts=128, cache_size=256,
    ) == 32
    assert disk_gpufetch_capacity(
        max_tokens=64, top_k=8, num_experts=128, cache_size=96,
    ) == 96
    with pytest.raises(ValueError, match="positive"):
        disk_gpufetch_capacity(
            max_tokens=0, top_k=8, num_experts=128, cache_size=256,
        )


def test_disk_gpufetch_residency_does_not_require_cpu_decode_membership():
    from freetoken.moe.host_banks import HostResidency
    from freetoken.moe.offload_cache import OffloadMoeCache

    cache = OffloadMoeCache(
        num_layers=1,
        num_experts=4,
        cache_size=4,
        device=torch.device("cpu"),
        moe_disk_decode="gpufetch",
    )
    cache.set_bank_sources(
        {
            "gate_up": [torch.randn(4, 16, 8)],
            "down": [torch.randn(4, 8, 8)],
        },
        layer_residency=[HostResidency.DISK.value],
    )

    assert cache.cpu_layer_ids == frozenset()
    assert cache.is_gpufetch_layer(0)
    assert not cache.is_cpu_layer(0)


def test_gpufetch_stats_are_reported_per_decode_step(monkeypatch):
    import freetoken.moe.cpu_executor as cpu_executor

    class Extension:
        def gpufetch_stats(self, reset):
            assert reset
            return (12, 6, 900_000)

    executor = cpu_executor.CpuMoeExecutor.__new__(cpu_executor.CpuMoeExecutor)
    executor.num_layers = 3
    executor._disk_banks = {1: [object()], 2: [object()]}
    executor._disk_prefetch_calls = [0, 0, 0]
    executor._disk_prefetch_pages = [0, 0, 0]
    executor._disk_decode_steps = 0
    executor._disk_route_pairs = 0
    executor._disk_distinct_experts = 0
    executor._disk_major_fault_base = 5
    executor._gpufetch_tasks = {1: (10, 1), 2: (11, 2)}
    executor._ext = Extension()
    monkeypatch.setattr(cpu_executor, "_major_faults", lambda: 11)

    stats = executor.disk_prefetch_stats(reset=True)

    assert stats["gpufetch_fills_per_step"] == 4.0
    assert stats["gpufetch_fill_us"] == 300.0
    assert stats["major_faults_per_decode_step"] == 2.0


def test_gpufetch_decode_uses_lru_gpu_path_even_for_hybrid_cache():
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.layers.moe import OffloadMoELayer

    if try_get_tp_info() is None:
        set_tp_info(0, 1)
    calls = []
    output = torch.randn(1, 8)

    class Cache:
        decode_target = "hybrid"

        def is_gpufetch_layer(self, layer_id):
            return True

        def is_cpu_layer(self, layer_id):
            return False

        def ensure_experts(self, layer_id, ids):
            calls.append(("ensure", layer_id, ids))

        def copy_missing(self):
            calls.append(("copy",))

        def bank_views(self):
            return ()

        def alphas_for_slots(self, layer_id):
            return None

    layer = OffloadMoELayer(
        layer_id=0, num_experts=4, top_k=2,
        hidden_size=8, intermediate_size=8,
    )
    layer.offload_cache = Cache()
    layer._expert_gemm = lambda *args, **kwargs: calls.append(("gemm",)) or output
    hidden = torch.randn(1, 8)
    weights = torch.tensor([[0.6, 0.4]])
    ids = torch.tensor([[1, 3]], dtype=torch.int32)

    assert layer._decode_routed(hidden, weights, ids) is output
    assert [call[0] for call in calls] == ["ensure", "copy", "gemm"]


def test_cpu_prefill_io_reuses_one_grow_to_largest_buffer():
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    executor = CpuMoeExecutor.__new__(CpuMoeExecutor)
    executor.device = torch.device("cpu")
    executor.H = 8
    executor.top_k = 2
    executor._prefill_io = None
    executor._prefill_capacity = 0

    first = executor._prefill_io_for(3)
    first_ptr = first["x"].data_ptr()
    smaller = executor._prefill_io_for(2)
    assert smaller["x"].data_ptr() == first_ptr
    assert executor._prefill_capacity == 3

    larger = executor._prefill_io_for(5)
    assert larger["x"].shape == (5, 8)
    assert executor._prefill_capacity == 5
    assert len(executor._prefill_io) == 4


def test_disk_prefetch_deduplicates_route_union_across_batch():
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    class Bank:
        def __init__(self, pages):
            self.pages = pages
            self.calls = []

        def prefetch_experts(self, expert_ids):
            self.calls.append(expert_ids)
            return self.pages

    gate_up = Bank(5)
    down = Bank(7)
    executor = CpuMoeExecutor.__new__(CpuMoeExecutor)
    executor.num_layers = 2
    executor.num_experts = 4
    executor._disk_banks = {1: [gate_up, down]}
    executor._disk_prefetch_calls = [0, 0]
    executor._disk_prefetch_pages = [0, 0]
    executor._disk_decode_steps = 0
    executor._disk_route_pairs = 0
    executor._disk_distinct_experts = 0

    routes = torch.tensor(
        [[3, 1], [3, -1], [2, 1], [2, 3]], dtype=torch.int32
    )
    assert executor.prefetch_experts(1, routes, is_prefill=True) == 12
    assert gate_up.calls == [[1, 2, 3]]
    assert down.calls == [[1, 2, 3]]
    assert executor._disk_prefetch_calls == [0, 1]
    assert executor._disk_prefetch_pages == [0, 12]
    assert executor._disk_decode_steps == 0


def test_cpu_prefill_coalesce_dedupes_and_releases_after_layer():
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    class Bank:
        def __init__(self):
            self.calls = []
            self.releases = []

        def prefetch_experts(self, expert_ids):
            self.calls.append(list(expert_ids))
            return len(expert_ids)

        def release_rows(self, expert_ids):
            self.releases.append(tuple(expert_ids))

    bank = Bank()
    executor = CpuMoeExecutor.__new__(CpuMoeExecutor)
    executor.num_experts = 8
    executor._disk_banks = {0: [bank]}
    executor._disk_prefetch_calls = [0]
    executor._disk_prefetch_pages = [0]
    executor._prefill_coalesce_enabled = True
    executor._prefill_coalesce_mode = "on"
    executor._prefill_coalesce_limits = {0: 8}
    executor._prefill_coalesce_experts = 0
    executor._prefill_coalesce_ns = 0
    executor._prefill_coalesce_degrades = 0
    executor._prefill_coalesce_warned = False

    routes = torch.tensor([[7, 2], [7, 1], [2, -1]], dtype=torch.int32)
    lease = executor.prepare_prefill_layer(0, routes)

    assert lease.experts == (1, 2, 7)
    assert bank.calls == [[1, 2, 7]]
    assert executor._prefill_coalesce_experts == 3
    executor.release_prefill_layer(lease)
    assert bank.releases == [(1, 2, 7)]


def test_cpu_prefill_coalesce_byte_ceiling_caps_distinct_experts():
    from freetoken.moe.cpu_executor import _prefill_coalesce_limit

    bank = SimpleNamespace(
        nbytes=8 * 4096,
        tensor=SimpleNamespace(shape=(8, 4096)),
        _view_offset=0,
    )

    assert _prefill_coalesce_limit(
        [bank], 8, byte_ceiling=3 * 4096, page_size=4096,
    ) == (3, 4096)
    assert _prefill_coalesce_limit(
        [bank], 8, byte_ceiling=4095, page_size=4096,
    ) == (0, 4096)


def test_cpu_prefill_zero_ceiling_returns_lease_without_sweeping():
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    executor = CpuMoeExecutor.__new__(CpuMoeExecutor)
    executor.num_experts = 8
    executor._disk_banks = {0: [object()]}
    executor._prefill_coalesce_enabled = True
    executor._prefill_coalesce_mode = "populate"
    executor._prefill_coalesce_limits = {0: 0}
    executor._prefill_coalesce_experts = 0
    executor._prefill_coalesce_ns = 0
    executor._prefill_coalesce_degrades = 0
    executor._prefill_coalesce_warned = False

    lease = executor.prepare_prefill_layer(0, [1, 2])

    assert lease.experts == ()
    assert executor._prefill_coalesce_experts == 0


@pytest.mark.parametrize(
    ("prefill_mode", "coalesce", "expected_calls"),
    [
        ("cpu", "populate", 1),
        ("cpu", "on", 1),
        ("cpu", "off", 0),
        ("copy", "populate", 0),
    ],
)
def test_cpu_prefill_coalesce_flag_gating(prefill_mode, coalesce, expected_calls):
    from freetoken.moe.offload_cache import OffloadMoeCache

    class Executor:
        def __init__(self):
            self.calls = []

        def prepare_prefill_layer(self, layer_id, expert_ids):
            self.calls.append((layer_id, expert_ids))
            return object()

    cache = OffloadMoeCache.__new__(OffloadMoeCache)
    cache.layer_residency = ["disk"]
    cache.moe_disk_prefill = prefill_mode
    cache.moe_prefill_coalesce = coalesce
    cache.cpu_executor = Executor()

    cache.prepare_disk_prefill(0, [1, 2])
    assert len(cache.cpu_executor.calls) == expected_calls


def test_cpu_prefill_coalesce_allocation_failure_degrades_to_fault_path(monkeypatch):
    import freetoken.moe.cpu_executor as cpu_executor

    executor = cpu_executor.CpuMoeExecutor.__new__(cpu_executor.CpuMoeExecutor)
    executor.num_experts = 8
    executor._disk_banks = {0: [object()]}
    executor._prefill_coalesce_enabled = True
    executor._prefill_coalesce_limits = {0: 8}
    executor._prefill_coalesce_experts = 0
    executor._prefill_coalesce_ns = 0
    executor._prefill_coalesce_degrades = 0
    executor._prefill_coalesce_warned = False
    monkeypatch.setattr(
        cpu_executor,
        "_dedupe_decode_routes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(MemoryError("bounded alloc")),
    )

    lease = executor.prepare_prefill_layer(0, [1, 2])
    assert lease.experts == ()
    assert executor._prefill_coalesce_degrades == 1
    assert executor._prefill_coalesce_experts == 0


def test_cpu_prefill_populate_reuses_bounded_scratch_and_accounts_bytes():
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    class Bank:
        _pager = None

        def __init__(self):
            self.scratch_ids = []

        def populate_experts(self, expert_ids, scratch):
            self.scratch_ids.append((id(scratch), len(scratch), tuple(expert_ids)))
            return 100 * len(expert_ids)

    bank = Bank()
    executor = CpuMoeExecutor.__new__(CpuMoeExecutor)
    executor.num_experts = 8
    executor._disk_banks = {0: [bank]}
    executor._disk_prefetch_calls = [0]
    executor._disk_prefetch_pages = [0]
    executor._prefill_coalesce_enabled = True
    executor._prefill_coalesce_mode = "populate"
    executor._prefill_coalesce_limits = {0: 8}
    executor._prefill_populate_scratch_bytes = 17
    executor._prefill_populate_scratch = None
    executor._prefill_coalesce_experts = 0
    executor._prefill_coalesce_ns = 0
    executor._prefill_coalesce_degrades = 0
    executor._prefill_coalesce_warned = False
    executor._prefill_populate_bytes = 0
    executor._prefill_populate_ns = 0

    first = executor.prepare_prefill_layer(0, [3, 1, 3])
    second = executor.prepare_prefill_layer(0, [2])

    assert first.experts == (1, 3)
    assert second.experts == (2,)
    assert bank.scratch_ids[0][0] == bank.scratch_ids[1][0]
    assert [entry[1] for entry in bank.scratch_ids] == [17, 17]
    assert executor._prefill_populate_bytes == 300
    assert executor._prefill_populate_ns > 0
    assert executor._prefill_coalesce_degrades == 0


def test_cpu_prefill_populate_failure_falls_back_to_willneed():
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    events = []

    class Bank:
        _pager = None

        def populate_experts(self, expert_ids, scratch):
            events.append(("populate", tuple(expert_ids)))
            raise OSError("pread failed")

        def prefetch_experts(self, expert_ids):
            events.append(("willneed", tuple(expert_ids)))
            return 2

    executor = CpuMoeExecutor.__new__(CpuMoeExecutor)
    executor.num_experts = 8
    executor._disk_banks = {0: [Bank()]}
    executor._disk_prefetch_calls = [0]
    executor._disk_prefetch_pages = [0]
    executor._prefill_coalesce_enabled = True
    executor._prefill_coalesce_mode = "populate"
    executor._prefill_coalesce_limits = {0: 8}
    executor._prefill_populate_scratch_bytes = 16
    executor._prefill_populate_scratch = None
    executor._prefill_coalesce_experts = 0
    executor._prefill_coalesce_ns = 0
    executor._prefill_coalesce_degrades = 0
    executor._prefill_coalesce_warned = False
    executor._prefill_populate_bytes = 0
    executor._prefill_populate_ns = 0

    lease = executor.prepare_prefill_layer(0, [2, 1])

    assert lease.experts == (1, 2)
    assert events == [("populate", (1, 2)), ("willneed", (1, 2))]
    assert executor._prefill_coalesce_degrades == 1
    assert executor._disk_prefetch_pages == [2]


def test_cpu_prefill_populate_degrades_to_willneed_then_faults():
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    class Bank:
        _pager = None

        def populate_experts(self, expert_ids, scratch):
            raise OSError("pread failed")

        def prefetch_experts(self, expert_ids):
            raise OSError("madvise failed")

    executor = CpuMoeExecutor.__new__(CpuMoeExecutor)
    executor.num_experts = 8
    executor._disk_banks = {0: [Bank()]}
    executor._disk_prefetch_calls = [0]
    executor._disk_prefetch_pages = [0]
    executor._prefill_coalesce_enabled = True
    executor._prefill_coalesce_mode = "populate"
    executor._prefill_coalesce_limits = {0: 8}
    executor._prefill_populate_scratch_bytes = 16
    executor._prefill_populate_scratch = None
    executor._prefill_coalesce_experts = 0
    executor._prefill_coalesce_ns = 0
    executor._prefill_coalesce_degrades = 0
    executor._prefill_coalesce_warned = False
    executor._prefill_populate_bytes = 0
    executor._prefill_populate_ns = 0

    lease = executor.prepare_prefill_layer(0, [2, 1])

    assert lease.experts == (1, 2)
    assert executor._prefill_coalesce_degrades == 2
    assert executor._prefill_coalesce_experts == 2
    assert executor._prefill_populate_bytes == 0


def test_cpu_prefill_populate_keeps_uffd_pager_path():
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    class Pager:
        def __init__(self):
            self.calls = []

        def prefetch(self, banks, expert_ids):
            self.calls.append((banks, tuple(expert_ids)))
            return 7

    pager = Pager()
    bank = SimpleNamespace(_pager=pager)
    executor = CpuMoeExecutor.__new__(CpuMoeExecutor)
    executor.num_experts = 8
    executor._disk_banks = {0: [bank]}
    executor._disk_prefetch_calls = [0]
    executor._disk_prefetch_pages = [0]
    executor._prefill_coalesce_enabled = True
    executor._prefill_coalesce_mode = "populate"
    executor._prefill_coalesce_limits = {0: 8}
    executor._prefill_populate_scratch_bytes = 16
    executor._prefill_populate_scratch = None
    executor._prefill_coalesce_experts = 0
    executor._prefill_coalesce_ns = 0
    executor._prefill_coalesce_degrades = 0
    executor._prefill_coalesce_warned = False
    executor._prefill_populate_bytes = 0
    executor._prefill_populate_ns = 0

    lease = executor.prepare_prefill_layer(0, [3, 1, 3])

    assert lease.experts == (1, 3)
    assert pager.calls == [([bank], (1, 3))]
    assert executor._disk_prefetch_pages == [7]
    assert executor._prefill_populate_bytes == 0


@pytest.mark.parametrize("model_type", ["qwen4_exp", "glm4_moe"])
def test_cpu_prefill_coalesce_uses_shared_layer_seam_for_architectures(model_type):
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.layers.moe import OffloadMoELayer

    if try_get_tp_info() is None:
        set_tp_info(0, 1)
    calls = []
    output = torch.zeros((2, 8), dtype=torch.bfloat16)

    class Executor:
        def prefill(self, layer_id, hidden_states, topk_weights, topk_ids):
            calls.append(("compute", model_type, layer_id))
            return output

    lease = object()
    cache = SimpleNamespace(
        model_config=SimpleNamespace(model_type=model_type),
        layer_residency=["disk"],
        moe_disk_prefill="cpu",
        prefill_overlap=False,
        cpu_executor=Executor(),
        prepare_disk_prefill=lambda layer_id, ids: calls.append(
            ("prepare", model_type, layer_id)
        ) or lease,
        release_disk_prefill=lambda actual: calls.append(
            ("release", model_type, actual)
        ),
    )
    layer = OffloadMoELayer(
        layer_id=0,
        num_experts=4,
        top_k=2,
        hidden_size=8,
        intermediate_size=8,
    )
    layer.offload_cache = cache
    hidden = torch.zeros((2, 8), dtype=torch.bfloat16)
    weights = torch.full((2, 2), 0.5)
    ids = torch.tensor([[1, 2], [2, 3]], dtype=torch.int32)

    assert layer._prefill_routed(hidden, weights, ids) is output
    assert calls == [
        ("prepare", model_type, 0),
        ("compute", model_type, 0),
        ("release", model_type, lease),
    ]


def test_disk_decode_route_dedup_stats_track_heavy_recurrence():
    from freetoken.moe.cpu_executor import CpuMoeExecutor, _dedupe_decode_routes

    class Bank:
        def prefetch_experts(self, expert_ids):
            assert expert_ids == [1, 2, 7]
            return 3

    executor = CpuMoeExecutor.__new__(CpuMoeExecutor)
    executor.num_layers = 1
    executor.num_experts = 8
    executor._disk_banks = {0: [Bank()]}
    executor._disk_prefetch_calls = [0]
    executor._disk_prefetch_pages = [0]
    executor._disk_decode_steps = 0
    executor._disk_route_pairs = 0
    executor._disk_distinct_experts = 0
    executor._disk_major_fault_base = None
    executor._gpufetch_tasks = {}

    routes = torch.tensor(
        [[1, 1, 2, 7], [1, 2, 2, 7], [1, 1, 1, -1]], dtype=torch.int32
    )
    selected, route_pairs = _dedupe_decode_routes(routes, executor.num_experts)
    assert selected == [1, 2, 7]
    # Native decode supplies this compact list from the existing routing D2H,
    # along with the pre-dedup pair count used by the ratio counter.
    assert executor.prefetch_experts(0, selected, route_pairs=route_pairs) == 3

    stats = executor.disk_prefetch_stats()
    assert stats["distinct_experts_per_step"] == 3.0
    assert stats["dedup_ratio"] == pytest.approx(11 / 3)


def _make_lookahead_executor(*, enabled=True):
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    class Bank:
        def __init__(self):
            self.calls = []

        def prefetch_experts(self, expert_ids):
            selected = list(expert_ids)
            self.calls.append(selected)
            return len(selected)

    banks = [Bank(), Bank()]
    executor = CpuMoeExecutor.__new__(CpuMoeExecutor)
    executor.num_layers = 2
    executor.num_experts = 8
    executor._disk_banks = {0: [banks[0]], 1: [banks[1]]}
    executor._disk_pagers = set()
    executor._disk_prefetch_calls = [0, 0]
    executor._disk_prefetch_pages = [0, 0]
    executor._disk_decode_steps = 0
    executor._disk_route_pairs = 0
    executor._disk_distinct_experts = 0
    executor._disk_lookahead_hits = 0
    executor._disk_lookahead_routes = 0
    executor._disk_delta_pages = 0
    executor._disk_major_fault_base = None
    executor._gpufetch_tasks = {}
    executor._disk_lookahead_enabled = enabled
    executor._disk_previous_experts = {}
    executor._disk_predicted_experts = {}
    return executor, banks


def test_disk_lookahead_tracks_each_layer_and_prefetches_only_route_delta():
    executor, banks = _make_lookahead_executor()

    # No history on the first step means the existing full reactive sweep remains.
    assert executor.begin_decode_step() == 0
    assert executor.prefetch_experts(0, [3, 1, 3]) == 2
    assert executor.prefetch_experts(1, [4, 2]) == 2
    assert executor._disk_previous_experts == {0: (1, 3), 1: (2, 4)}

    # The next step starts with both per-layer predictions. Layer 0 then requests
    # only expert 4, while layer 1's perfect prediction has no reactive call.
    assert executor.begin_decode_step() == 4
    assert executor.prefetch_experts(0, [4, 3]) == 1
    assert executor.prefetch_experts(1, [2, 4]) == 0
    assert banks[0].calls == [[1, 3], [1, 3], [4]]
    assert banks[1].calls == [[2, 4], [2, 4]]
    assert executor._disk_previous_experts == {0: (3, 4), 1: (2, 4)}

    stats = executor.disk_prefetch_stats(reset=True)
    assert stats["lookahead_hit_rate"] == pytest.approx(3 / 4)
    assert stats["delta_pages_per_step"] == 2.5
    assert executor._disk_lookahead_hits == 0
    assert executor._disk_lookahead_routes == 0
    assert executor._disk_delta_pages == 0
    # Telemetry flushes must not turn the next live step into a cold prediction.
    assert executor._disk_previous_experts == {0: (3, 4), 1: (2, 4)}


def test_disk_lookahead_defaults_to_madvise_only():
    from freetoken.moe.cpu_executor import _disk_lookahead_allowed

    assert _disk_lookahead_allowed(True, set())


@pytest.mark.parametrize(
    ("requested", "pagers"),
    [(False, set()), (True, {"uffd"})],
    ids=["flag-off", "uffd-backend"],
)
def test_disk_lookahead_disabled_keeps_reactive_prefetch(requested, pagers):
    from freetoken.moe.cpu_executor import _disk_lookahead_allowed

    assert not _disk_lookahead_allowed(requested, pagers)
    executor, banks = _make_lookahead_executor(enabled=False)
    executor._disk_previous_experts = {0: (1, 2), 1: (3,)}

    assert executor.begin_decode_step() == 0
    assert executor.prefetch_experts(0, [1, 2, 4]) == 3
    assert banks[0].calls == [[1, 2, 4]]
    assert executor._disk_previous_experts == {0: (1, 2), 1: (3,)}


def test_disk_lookahead_prefill_boundary_restores_first_step_fallback():
    executor, banks = _make_lookahead_executor()
    executor.prefetch_experts(0, [1, 2])

    # Explicit prefill advice neither consumes nor replaces decode prediction state.
    executor.prefetch_experts(0, [7], is_prefill=True)
    assert executor._disk_previous_experts == {0: (1, 2)}

    executor.reset_disk_lookahead()
    assert executor.begin_decode_step() == 0
    assert executor.prefetch_experts(0, [2, 5]) == 2
    assert banks[0].calls == [[1, 2], [7], [2, 5]]


def test_uffd_prefetch_groups_all_layer_banks_into_one_budget_request():
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    class Pager:
        def __init__(self):
            self.calls = []

        def prefetch(self, banks, expert_ids):
            self.calls.append((banks, expert_ids))
            return 9

    pager = Pager()
    gate_up = SimpleNamespace(_pager=pager)
    down = SimpleNamespace(_pager=pager)
    executor = CpuMoeExecutor.__new__(CpuMoeExecutor)
    executor.num_layers = 1
    executor.num_experts = 8
    executor._disk_banks = {0: [gate_up, down]}
    executor._disk_prefetch_calls = [0]
    executor._disk_prefetch_pages = [0]
    executor._disk_decode_steps = 0
    executor._disk_route_pairs = 0
    executor._disk_distinct_experts = 0

    assert executor.prefetch_experts(0, [3, 1, 3, -1]) == 9
    assert pager.calls == [([gate_up, down], [1, 3])]
    assert executor._disk_prefetch_pages == [9]


def test_uffd_host_bank_uses_anonymous_region_and_residency_bitmap_hook(tmp_path):
    from freetoken.moe.host_banks import HostBank, HostResidency

    class Pager:
        def __init__(self):
            self.registration = None
            self.prefetch_call = None

        def register_bank(self, bank, **kwargs):
            self.registration = (bank, kwargs)
            return 7

        def prefetch(self, banks, expert_ids):
            self.prefetch_call = (banks, expert_ids)
            return 2

    source = tmp_path / "bank.ftw"
    source.write_bytes(b"x" * 8192)
    pager = Pager()
    bank = HostBank(
        (2, 4096), torch.uint8, backing="uffd",
        file_path=str(source), file_offset=0, disk_pager=pager,
    )

    assert bank.residency is HostResidency.DISK
    assert bank._pager is pager
    assert bank._pager_region == 7
    assert pager.registration[1] == {
        "file_path": str(source),
        "file_offset": 0,
        "row_bytes": 4096,
        "num_rows": 2,
    }
    assert bank.prefetch_experts([1]) == 2
    assert pager.prefetch_call == ([bank], [1])


def test_uffd_stats_are_merged_into_disk_telemetry(monkeypatch):
    import freetoken.moe.cpu_executor as cpu_executor

    class Pager:
        def stats(self, *, reset=False):
            assert reset
            return {
                "fills": 5,
                "fills_from_prefetch": 4,
                "fault_driven": 1,
                "evictions": 2,
                "resident_bytes": 12288,
                "pages_installed": 3,
                "rows_spanning_pages": 2,
                "fill_latency_histogram": {
                    "buckets_us": [50, 100],
                    "counts": [1, 3, 1],
                },
            }

    executor = cpu_executor.CpuMoeExecutor.__new__(cpu_executor.CpuMoeExecutor)
    executor.num_layers = 1
    executor._disk_banks = {0: [object()]}
    executor._disk_pagers = {Pager()}
    executor._disk_prefetch_calls = [2]
    executor._disk_prefetch_pages = [7]
    executor._disk_decode_steps = 1
    executor._disk_route_pairs = 0
    executor._disk_distinct_experts = 0
    executor._disk_major_fault_base = 10
    monkeypatch.setattr(cpu_executor, "_major_faults", lambda: 10)

    stats = executor.disk_prefetch_stats(reset=True)
    assert stats["pager_backend"] == "uffd"
    assert stats["fills"] == 5
    assert stats["fills_from_prefetch"] == 4
    assert stats["fault_driven"] == 1
    assert stats["evictions"] == 2
    assert stats["resident_bytes"] == 12288
    assert stats["pages_installed"] == 3
    assert stats["rows_spanning_pages"] == 2
    assert stats["fill_latency_histogram"] == {
        "buckets_us": [50, 100],
        "counts": [1, 3, 1],
    }


def test_disk_decode_tasks_are_cached_per_layer_and_batch_size():
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    class Extension:
        def __init__(self):
            self.calls = []

        def create_task(self, *args):
            self.calls.append(args)
            return len(self.calls)

    executor = CpuMoeExecutor.__new__(CpuMoeExecutor)
    executor.device = torch.device("cpu")
    executor.H = 8
    executor.top_k = 2
    executor._io = {}
    executor._tasks = {}
    executor._disk_banks = {0: [object()], 1: [object()]}
    executor._ext = Extension()
    executor._flag_sync = False

    layer0_bs3 = executor._task_for(0, 3)
    assert executor._task_for(0, 3) == layer0_bs3
    layer1_bs3 = executor._task_for(1, 3)
    layer0_bs2 = executor._task_for(0, 2)

    assert (layer0_bs3, layer1_bs3, layer0_bs2) == (1, 2, 3)
    assert [(call[0], call[1]) for call in executor._ext.calls] == [
        (0, 3), (1, 3), (0, 2)
    ]
    # Layers at the same batch size intentionally share one fixed IO bank. Stream
    # ordering and each layer's submit/sync pair prevent simultaneous reuse.
    assert executor._ext.calls[0][2:] == executor._ext.calls[1][2:]
    assert executor._ext.calls[0][2:] != executor._ext.calls[2][2:]


@pytest.mark.parametrize(
    "mode", [pytest.param(None, id="default-cpu"), pytest.param("copy", id="copy")],
)
@pytest.mark.parametrize("residency", ["disk", "locked", "pageable"])
def test_prefill_routing_changes_only_disk_cpu_mode(mode, residency):
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.layers.moe import OffloadMoELayer
    from freetoken.moe.offload_cache import OffloadMoeCache

    if try_get_tp_info() is None:
        set_tp_info(0, 1)
    if mode is None:
        mode = OffloadMoeCache.__dataclass_fields__["moe_disk_prefill"].default
        assert mode == "cpu"
    calls = []
    cpu_out = torch.full((3, 8), 11, dtype=torch.bfloat16)
    gpu_out = torch.full((3, 8), 22, dtype=torch.bfloat16)

    class Executor:
        def prefill(self, layer_id, hidden_states, topk_weights, topk_ids):
            calls.append(("cpu", layer_id, hidden_states, topk_weights, topk_ids))
            return cpu_out

    cache = SimpleNamespace(
        layer_residency=[residency],
        moe_disk_prefill=mode,
        prefill_overlap=False,
        cpu_executor=Executor(),
        prefetch_disk_experts=lambda layer_id, ids: calls.append(
            ("prefetch", layer_id, ids)
        ),
        materialize_layer=lambda layer_id: calls.append(("materialize", layer_id)),
        copy_missing=lambda: calls.append(("copy",)),
        bank_views=lambda n: (),
        alphas_for_layer=lambda layer_id: None,
    )
    layer = OffloadMoELayer(
        layer_id=0, num_experts=4, top_k=2,
        hidden_size=8, intermediate_size=8,
    )
    layer.offload_cache = cache
    layer._expert_gemm = lambda *args, **kwargs: calls.append(("gemm",)) or gpu_out
    hidden = torch.randn(3, 8, dtype=torch.bfloat16)
    ids = torch.tensor([[0, 1], [2, 3], [1, 3]], dtype=torch.int32)
    weights = torch.tensor(
        [[0.7, 0.3], [0.25, 0.75], [0.6, 0.4]], dtype=torch.float32,
    )

    out = layer._prefill_routed(hidden, weights, ids)
    names = [call[0] for call in calls]

    if residency == "disk" and mode == "cpu":
        assert out is cpu_out
        assert names == ["prefetch", "cpu"]
        assert calls[0][2] is ids
        assert calls[1][2] is hidden
        assert calls[1][3] is weights
        assert calls[1][4] is ids
    else:
        assert out is gpu_out
        expected = ["materialize", "copy", "gemm"]
        if residency == "disk":
            expected.insert(0, "prefetch")
        assert names == expected


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
    if not hasattr(_cpu_moe.CpuMoeExecutor, "run_task_sync"):
        pytest.skip("CPU MoE extension needs rebuilding for DISK CPU prefill")
    from freetoken.moe.cpu_executor import CpuMoeExecutor
    from freetoken.moe.host_banks import HostBank, HostResidency

    torch.manual_seed(7)
    experts, hidden, inter, top_k, tokens = 4, 32, 32, 2, 5
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

    def make_executor(sources):
        cache = SimpleNamespace(
            quant_format="bf16", bank_sources=sources,
            num_layers=1, num_experts=experts,
        )
        executor = CpuMoeExecutor(
            cache, top_k=top_k, activation="silu",
            apply_router_weight_on_input=False, num_threads=1, max_tokens=2,
            device=torch.device("cpu"),
        )
        return executor

    x = torch.randn(tokens, hidden, dtype=torch.bfloat16)
    ids = torch.tensor(
        [[0, 2], [3, 1], [1, 0], [2, 3], [3, 0]], dtype=torch.int32,
    )
    weights = torch.tensor(
        [[0.6, 0.4], [0.25, 0.75], [0.8, 0.2], [0.45, 0.55], [0.1, 0.9]],
        dtype=torch.float32,
    )
    ram_executor = make_executor(ram_sources)
    ram_out = ram_executor.prefill(0, x, weights, ids)

    disk_cache = SimpleNamespace(
        quant_format="bf16", bank_sources=disk.sources,
        num_layers=1, num_experts=experts,
    )
    disk_executor = make_executor(disk_cache.bank_sources)
    disk_executor.prefetch_experts(0, ids, is_prefill=True)
    disk_out = disk_executor.prefill(0, x, weights, ids)

    assert disk_out.shape == x.shape
    assert disk_out.dtype == x.dtype
    assert torch.equal(disk_out, ram_out)
    assert disk_executor.disk_prefetch_stats()["prefetch_calls"] == 1


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_disk_gpufetch_decode_matches_cpu_executor(tmp_path):
    from freetoken.checkpoint.ftw import load_ftw_banks
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.kernel import _cpu_moe
    from freetoken.layers.moe import OffloadMoELayer
    from freetoken.moe.cpu_executor import CpuMoeExecutor
    from freetoken.moe.host_banks import HostResidency
    from freetoken.moe.offload_cache import OffloadMoeCache

    required = (
        "create_gpufetch_task",
        "register_flag_gpufetch_task",
        "gpufetch_with_cuda_stream",
    )
    if not all(hasattr(_cpu_moe.CpuMoeExecutor, name) for name in required):
        pytest.skip("CPU MoE extension needs rebuilding for DISK gpufetch")
    if try_get_tp_info() is None:
        set_tp_info(0, 1)

    torch.manual_seed(31)
    experts, hidden, inter, top_k, batch = 4, 128, 128, 2, 2
    gate_up = torch.randn(
        experts, 2 * inter, hidden, dtype=torch.bfloat16,
    ) * 0.1
    down = torch.randn(experts, hidden, inter, dtype=torch.bfloat16) * 0.1
    _write_bf16_ftw(tmp_path, [(gate_up, down)])
    banks = load_ftw_banks(
        str(tmp_path),
        num_layers=1,
        layer_residency=[HostResidency.DISK.value],
    )
    assert banks is not None

    device = torch.device("cuda")
    cache = OffloadMoeCache(
        num_layers=1,
        num_experts=experts,
        cache_size=experts,
        device=device,
        prefill_overlap=False,
        moe_disk_decode="gpufetch",
        decode_target="gpu",
    )
    cache.set_bank_sources(
        banks.sources,
        layer_residency=[HostResidency.DISK.value],
    )
    executor = CpuMoeExecutor(
        cache,
        top_k=top_k,
        activation="silu",
        apply_router_weight_on_input=False,
        num_threads=1,
        max_tokens=batch,
        device=device,
    )
    cache.set_cpu_executor(executor)
    cache.init_disk_gpufetch(executor, max_tokens=batch, top_k=top_k)
    layer = OffloadMoELayer(
        layer_id=0,
        num_experts=experts,
        top_k=top_k,
        hidden_size=hidden,
        intermediate_size=inter,
    )
    layer.offload_cache = cache

    x = torch.randn(batch, hidden, device=device, dtype=torch.bfloat16)
    ids = torch.tensor([[0, 2], [3, 1]], device=device, dtype=torch.int32)
    weights = torch.tensor(
        [[0.6, 0.4], [0.25, 0.75]], device=device, dtype=torch.float32,
    )
    cpu_out = executor.decode(0, x, weights, ids).float()
    torch.cuda.synchronize()

    executor.reset_disk_stats()
    gpu_out = layer._decode_routed(x, weights, ids.clone()).float()
    torch.cuda.synchronize()
    first = executor.disk_prefetch_stats(reset=True)
    hot_out = layer._decode_routed(x, weights, ids.clone()).float()
    torch.cuda.synchronize()
    hot = executor.disk_prefetch_stats(reset=True)

    rel = (cpu_out - gpu_out).abs().max() / (cpu_out.abs().max() + 1e-6)
    assert rel < 2e-2, f"relative error {rel.item()}"
    torch.testing.assert_close(hot_out, gpu_out, rtol=0, atol=0)
    assert first["gpufetch_fills_per_step"] == experts
    assert hot["gpufetch_fills_per_step"] == 0


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_disk_hot_cold_split_matches_pure_cpu_decode(tmp_path):
    from freetoken.checkpoint.ftw import load_ftw_banks
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.layers.moe import OffloadMoELayer
    from freetoken.moe.cpu_executor import CpuMoeExecutor
    from freetoken.moe.host_banks import HostResidency
    from freetoken.moe.offload_cache import OffloadMoeCache

    if try_get_tp_info() is None:
        set_tp_info(0, 1)
    torch.manual_seed(43)
    experts, hidden, inter, top_k, batch = 4, 128, 128, 2, 2
    gate_up = torch.randn(
        experts, 2 * inter, hidden, dtype=torch.bfloat16,
    ) * 0.1
    down = torch.randn(
        experts, hidden, inter, dtype=torch.bfloat16,
    ) * 0.1
    _write_bf16_ftw(tmp_path, [(gate_up, down)])
    banks = load_ftw_banks(
        str(tmp_path),
        num_layers=1,
        layer_residency=[HostResidency.DISK.value],
        hot_expert_ids={0: (0, 2)},
    )
    assert banks is not None

    device = torch.device("cuda")
    cache = OffloadMoeCache(
        num_layers=1,
        num_experts=experts,
        cache_size=experts,
        device=device,
        prefill_overlap=False,
        decode_target="cpu",
    )
    cache.cpu_layer_ids = frozenset({0})
    cache.set_bank_sources(
        banks.sources,
        layer_residency=banks.layer_residency,
        hot_sources=banks.hot_sources,
        hot_expert_ids=banks.hot_expert_ids,
    )
    executor = CpuMoeExecutor(
        cache,
        top_k=top_k,
        activation="silu",
        apply_router_weight_on_input=False,
        num_threads=1,
        max_tokens=batch,
        device=device,
    )
    cache.set_cpu_executor(executor)
    layer = OffloadMoELayer(
        layer_id=0,
        num_experts=experts,
        top_k=top_k,
        hidden_size=hidden,
        intermediate_size=inter,
    )
    layer.offload_cache = cache

    x = torch.randn(batch, hidden, device=device, dtype=torch.bfloat16)
    ids = torch.tensor([[0, 1], [2, 3]], device=device, dtype=torch.int32)
    weights = torch.tensor(
        [[0.6, 0.4], [0.25, 0.75]], device=device, dtype=torch.float32,
    )
    cpu_out = executor.decode(0, x, weights, ids).float()
    split_out = layer._decode_routed(x, weights, ids.clone()).float()
    torch.cuda.synchronize()

    torch.testing.assert_close(split_out, cpu_out, rtol=2e-2, atol=2e-2)
    stats = cache.disk_prefetch_stats(reset=True)
    assert stats["hot_pair_rate"] == pytest.approx(0.5)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_disk_hot_adaptation_forced_tick_preserves_decode_parity(tmp_path):
    from freetoken.checkpoint.ftw import load_ftw_banks
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.layers.moe import OffloadMoELayer
    from freetoken.moe.cpu_executor import CpuMoeExecutor
    from freetoken.moe.host_banks import HostResidency
    from freetoken.moe.offload_cache import OffloadMoeCache

    if try_get_tp_info() is None:
        set_tp_info(0, 1)
    torch.manual_seed(47)
    experts, hidden, inter, top_k, batch = 4, 128, 128, 2, 2
    gate_up = torch.randn(experts, 2 * inter, hidden, dtype=torch.bfloat16) * 0.1
    down = torch.randn(experts, hidden, inter, dtype=torch.bfloat16) * 0.1
    _write_bf16_ftw(tmp_path, [(gate_up, down)])
    banks = load_ftw_banks(
        str(tmp_path),
        num_layers=1,
        layer_residency=[HostResidency.DISK.value],
        hot_expert_ids={0: (0, 2)},
        hot_expert_capacity={0: 2},
    )
    assert banks is not None

    device = torch.device("cuda")
    cache = OffloadMoeCache(
        num_layers=1, num_experts=experts, cache_size=experts, device=device,
        prefill_overlap=False, decode_target="cpu",
    )
    cache.cpu_layer_ids = frozenset({0})
    cache.set_bank_sources(
        banks.sources,
        layer_residency=banks.layer_residency,
        hot_sources=banks.hot_sources,
        hot_expert_ids=banks.hot_expert_ids,
        hot_expert_capacity=banks.hot_expert_capacity,
    )
    row_bytes = sum(
        source[0][0].numel() * source[0].element_size()
        for source in banks.sources.values()
    )
    cache.configure_hot_adaptation(
        half_life_steps=2, interval_steps=1,
        max_swap_bytes=2 * row_bytes, expert_bytes=row_bytes,
    )
    executor = CpuMoeExecutor(
        cache, top_k=top_k, activation="silu",
        apply_router_weight_on_input=False, num_threads=1,
        max_tokens=batch, device=device,
    )
    cache.set_cpu_executor(executor)
    layer = OffloadMoELayer(
        layer_id=0, num_experts=experts, top_k=top_k,
        hidden_size=hidden, intermediate_size=inter,
    )
    layer.offload_cache = cache

    x = torch.randn(batch, hidden, device=device, dtype=torch.bfloat16)
    ids = torch.tensor([[1, 3], [3, 1]], device=device, dtype=torch.int32)
    weights = torch.tensor(
        [[0.6, 0.4], [0.25, 0.75]], device=device, dtype=torch.float32,
    )
    expected = executor.decode(0, x, weights, ids).float()
    torch.cuda.synchronize()
    before_ptr = cache.hot_row_for_expert.data_ptr()
    cache.decayed_decode_freq.zero_()
    cache.decayed_decode_freq[0, 1] = 10
    cache.decayed_decode_freq[0, 3] = 9

    cache.hot_adapt_step_boundary()
    cache._hot_adapt_future.result(timeout=10)
    cache.hot_adapt_step_boundary()
    cache._hot_adapt_future.result(timeout=10)
    cache.hot_adapt_step_boundary()
    assert cache.hot_row_for_expert.data_ptr() == before_ptr
    assert cache.hot_row_for_expert[0].tolist() == [-1, 1, -1, 0]

    actual = layer._decode_routed(x, weights, ids.clone()).float()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
