from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import torch

from freetoken.core import Batch, Context, Req, SamplingParams
from freetoken.scheduler.cache import CacheManager


class _Pool:
    """Small CPU double for the scheduler-visible LinearStatePool surface."""

    def __init__(self, num_slots=8):
        self.conv_states = torch.zeros(1, num_slots, 1, 3)
        self.recurrent_states = torch.zeros(1, num_slots, 1, 1)
        self.slot_states = {}
        self._free_slots = list(range(1, num_slots))

    def alloc(self, n=1):
        return [self._free_slots.pop() for _ in range(n)]


def _pool(num_slots=8):
    return _Pool(num_slots)


def _stub_fla_metadata_dependencies(monkeypatch):
    """Expose scheduler metadata helpers without importing Triton kernels."""
    fla = ModuleType("freetoken.kernel.fla")
    fla.__path__ = []
    chunk = ModuleType("freetoken.kernel.fla.chunk")
    chunk.CHUNK_SIZE = 64
    index = ModuleType("freetoken.kernel.fla.index")

    def prepare_chunk_offsets(cu_seqlens, chunk_size):
        lens = cu_seqlens[1:] - cu_seqlens[:-1]
        chunks = torch.div(lens + chunk_size - 1, chunk_size, rounding_mode="floor")
        return torch.cat([cu_seqlens.new_tensor([0]), chunks]).cumsum(-1)

    index.prepare_chunk_offsets = prepare_chunk_offsets
    monkeypatch.setitem(sys.modules, "freetoken.kernel.fla", fla)
    monkeypatch.setitem(sys.modules, "freetoken.kernel.fla.chunk", chunk)
    monkeypatch.setitem(sys.modules, "freetoken.kernel.fla.index", index)


def test_harness_anchor_wins_over_deepest_prefill_track(monkeypatch):
    import freetoken.core as core
    from freetoken.attention.linear import build_fla_metadata
    from freetoken.scheduler.prefill import ChunkedReq

    _stub_fla_metadata_dependencies(monkeypatch)
    pool = _pool()
    monkeypatch.setattr(
        core,
        "_GLOBAL_CTX",
        Context(page_size=64, linear_state_pool=pool),
    )
    req = ChunkedReq(
        input_ids=torch.arange(160, dtype=torch.int32),
        table_idx=0,
        cached_len=0,
        output_len=1,
        uid=1,
        sampling_params=SamplingParams(),
        cache_handle=None,
        cache_anchor_len=64,
        cache_anchor_kind="opencode",
        cache_anchor_persistable=True,
    )
    req.linear_slot_idx = pool.alloc(1)[0]
    req.mamba_ping_pong = tuple(pool.alloc(2))
    batch = Batch(reqs=[req], phase="prefill")
    batch.padded_reqs = batch.reqs

    metadata = build_fla_metadata(batch, torch.device("cpu"))

    assert metadata.track_boundary_row.tolist() == [64]
    assert req.mamba_last_track_seqlen == 64


def test_final_and_single_chunk_harnesses_keep_the_normal_deepest_track(monkeypatch):
    import freetoken.core as core
    from freetoken.attention.linear import build_fla_metadata

    _stub_fla_metadata_dependencies(monkeypatch)

    def tracked_boundary(anchor, persistable):
        pool = _pool()
        monkeypatch.setattr(
            core,
            "_GLOBAL_CTX",
            Context(page_size=64, linear_state_pool=pool),
        )
        req = Req(
            input_ids=torch.arange(160, dtype=torch.int32),
            table_idx=0,
            cached_len=0,
            output_len=1,
            uid=1,
            sampling_params=SamplingParams(),
            cache_handle=None,
            cache_anchor_len=anchor,
            cache_anchor_kind="opencode" if anchor is not None else None,
            cache_anchor_persistable=persistable,
        )
        req.linear_slot_idx = pool.alloc(1)[0]
        req.mamba_ping_pong = tuple(pool.alloc(2))
        batch = Batch(reqs=[req], phase="prefill")
        batch.padded_reqs = batch.reqs
        return build_fla_metadata(batch, torch.device("cpu")).track_boundary_row.tolist()

    normal = tracked_boundary(None, False)
    assert tracked_boundary(64, False) == normal == [128]
    # Even a wrongly marked final Req cannot override the normal tracked boundary.
    assert tracked_boundary(64, True) == normal


def test_prefill_admission_aligns_and_carries_harness_anchor(monkeypatch):
    from freetoken.message import UserMsg
    from freetoken.scheduler.prefill import PrefillManager

    _stub_fla_metadata_dependencies(monkeypatch)
    cache = SimpleNamespace(
        is_hybrid=True,
        disk_prefix_store=object(),
        admit_expert_profile=lambda _uid, _ids: None,
    )
    manager = PrefillManager(
        cache_manager=cache,
        table_manager=SimpleNamespace(),
        decode_manager=SimpleNamespace(),
    )
    manager.add_one_req(
        UserMsg(
            uid=3,
            input_ids=torch.arange(200, dtype=torch.int32),
            sampling_params=SamplingParams(),
            cache_anchor_len=127,
            cache_anchor_kind="opencode",
        )
    )

    assert manager.pending_list[0].cache_anchor_len == 64
    assert manager.pending_list[0].cache_anchor_kind == "opencode"


def test_prefill_admission_drops_anchor_without_disk_store_and_counts(monkeypatch):
    from freetoken.message import UserMsg
    from freetoken.scheduler.prefill import PrefillManager

    _stub_fla_metadata_dependencies(monkeypatch)
    manager_cache = object.__new__(CacheManager)
    manager_cache.is_hybrid = True
    manager_cache.disk_prefix_store = None
    manager_cache._harness_anchor_stats = {}
    manager_cache.admit_expert_profile = lambda _uid, _ids: None
    manager = PrefillManager(
        cache_manager=manager_cache,
        table_manager=SimpleNamespace(),
        decode_manager=SimpleNamespace(),
    )
    manager.add_one_req(
        UserMsg(
            uid=4,
            input_ids=torch.arange(200, dtype=torch.int32),
            sampling_params=SamplingParams(),
            cache_anchor_len=64,
            cache_anchor_kind="opencode",
        )
    )

    pending = manager.pending_list[0]
    assert pending.cache_anchor_len is None
    assert pending.cache_anchor_kind is None
    assert manager_cache.harness_anchor_stats()["harness_anchor_skipped_no_store"] == 1


def test_persistence_guard_counts_missing_store():
    manager = object.__new__(CacheManager)
    manager.is_hybrid = True
    manager.disk_prefix_store = None
    manager.page_size = 64
    manager._harness_anchor_stats = {}
    req = SimpleNamespace(cache_anchor_len=64)

    manager.persist_intermediate_cache_anchor(req)

    assert manager.harness_anchor_stats()["harness_anchor_skipped_no_store"] == 1


def test_final_chunk_drops_anchor_and_counts_skip(tmp_path):
    from freetoken.kvcache.disk_prefix_cache import DiskPrefixStore
    from freetoken.scheduler.prefill import PrefillAdder
    from freetoken.scheduler.utils import PendingReq

    store = DiskPrefixStore(tmp_path, 1 << 20, identity="final-counter")
    cache = SimpleNamespace(
        swa_paged=False,
        prefill_chunk_align=1,
        disk_prefix_store=store,
        note_harness_anchor=store.note_harness_anchor,
    )
    page_table = torch.zeros(1, 128, dtype=torch.int32)
    table = SimpleNamespace(token_pool=page_table)
    pending = PendingReq(
        5,
        torch.arange(100, dtype=torch.int32),
        SamplingParams(max_tokens=1),
        cache_anchor_len=64,
        cache_anchor_kind="opencode",
    )
    req = PrefillAdder(100, 0, cache, table)._add_one_req(
        pending,
        cache_handle=None,
        table_idx=0,
        cached_len=0,
    )

    assert type(req) is Req
    assert req.cache_anchor_len is None
    assert not req.cache_anchor_persistable
    assert store.stats()["harness_anchor_skipped_final_chunk"] == 1
    store.close()


def test_only_nonfinal_chunk_strictly_containing_anchor_opts_in():
    from freetoken.scheduler.prefill import ChunkedReq, PrefillAdder
    from freetoken.scheduler.utils import PendingReq

    cache = SimpleNamespace(
        swa_paged=False,
        prefill_chunk_align=1,
        disk_prefix_store=object(),
        note_harness_anchor=lambda _outcome: None,
    )
    table = SimpleNamespace(token_pool=torch.zeros(1, 256, dtype=torch.int32))

    def make(anchor):
        pending = PendingReq(
            6,
            torch.arange(200, dtype=torch.int32),
            SamplingParams(max_tokens=1),
            cache_anchor_len=anchor,
            cache_anchor_kind="opencode",
        )
        return PrefillAdder(128, 0, cache, table)._add_one_req(
            pending,
            cache_handle=None,
            table_idx=0,
            cached_len=0,
        )

    inside = make(64)
    at_end = make(128)

    assert isinstance(inside, ChunkedReq)
    assert inside.cache_anchor_persistable
    assert inside.cache_anchor_len == 64
    assert isinstance(at_end, ChunkedReq)
    assert not at_end.cache_anchor_persistable
    assert at_end.cache_anchor_len is None


def test_chunked_anchor_persistence_does_not_touch_radix_ownership(monkeypatch):
    class Store:
        def __init__(self):
            self.stats = {}

        def contains(self, _token_ids):
            return False

        def note_harness_anchor(self, outcome):
            self.stats[outcome] = self.stats.get(outcome, 0) + 1

    page_table = torch.arange(4 * 256, dtype=torch.int32).view(4, 256)
    manager = object.__new__(CacheManager)
    manager.is_hybrid = True
    manager.disk_prefix_store = Store()
    manager.page_size = 64
    manager.page_table = page_table
    manager.free_slots = torch.arange(1024, dtype=torch.int32)
    manager.prefix_cache = SimpleNamespace(
        full_evictable_size=0,
        mamba_evictable_size=0,
    )
    pool = _pool()
    from freetoken.scheduler.prefill import ChunkedReq

    req = ChunkedReq(
        input_ids=torch.arange(129, dtype=torch.int32),
        table_idx=0,
        cached_len=128,
        output_len=1,
        uid=2,
        sampling_params=SamplingParams(),
        cache_handle=None,
        cache_anchor_len=64,
        cache_anchor_kind="pi",
        cache_anchor_persistable=True,
    )
    req.linear_slot_idx = pool.alloc(1)[0]
    req.mamba_ping_pong = tuple(pool.alloc(2))
    req.mamba_next_track_idx = 1
    req.mamba_last_track_seqlen = 64
    queued = []
    monkeypatch.setattr(
        manager,
        "_queue_disk_prefix",
        lambda request, length, indices, frozen: (
            queued.append((request, length, indices.clone(), frozen)) or True
        ),
    )
    free_before = manager.free_slots.clone()

    manager.persist_intermediate_cache_anchor(req)

    assert len(queued) == 1
    assert queued[0][0] is req
    assert queued[0][1] == 64
    assert queued[0][2].tolist() == page_table[0, :64].tolist()
    assert queued[0][3] == req.mamba_ping_pong[0]
    assert torch.equal(manager.free_slots, free_before)
    assert manager.prefix_cache.full_evictable_size == 0
    assert manager.prefix_cache.mamba_evictable_size == 0
    assert manager.disk_prefix_store.stats["persisted"] == 1


def test_existing_disk_anchor_is_not_rewritten(monkeypatch):
    class Store:
        def note_harness_anchor(self, _outcome):
            raise AssertionError("existing roots are not a skip outcome")

        def contains(self, _token_ids):
            return True

    page_table = torch.zeros(4, 256, dtype=torch.int32)
    manager = object.__new__(CacheManager)
    manager.is_hybrid = True
    manager.disk_prefix_store = Store()
    manager.page_size = 64
    manager.page_table = page_table
    from freetoken.scheduler.prefill import ChunkedReq

    req = ChunkedReq(
        input_ids=torch.arange(192, dtype=torch.int32),  # the chunk after the 128 cached tokens
        table_idx=0,
        cached_len=128,
        output_len=1,
        uid=9,
        sampling_params=SamplingParams(),
        cache_handle=None,
        cache_anchor_len=64,
        cache_anchor_persistable=True,
    )
    req.mamba_last_track_seqlen = 64
    req.mamba_ping_pong = (1, 2)
    req.mamba_next_track_idx = 1
    queued = []
    monkeypatch.setattr(manager, "_queue_disk_prefix", lambda *args: queued.append(args))

    manager.persist_intermediate_cache_anchor(req)

    assert queued == []


def test_intermediate_anchor_guards_unaligned_and_invalid_table(tmp_path):
    from freetoken.kvcache.disk_prefix_cache import DiskPrefixStore
    from freetoken.scheduler.prefill import ChunkedReq

    store = DiskPrefixStore(tmp_path, 1 << 20, identity="guard-counters")
    manager = object.__new__(CacheManager)
    manager.is_hybrid = True
    manager.disk_prefix_store = store
    manager.page_size = 128
    manager.page_table = torch.zeros(1, 256, dtype=torch.int32)
    manager._queue_disk_prefix = lambda *_args: (_ for _ in ()).throw(
        AssertionError("guarded anchors must not stage")
    )
    req = ChunkedReq(
        input_ids=torch.arange(129, dtype=torch.int32),
        table_idx=0,
        cached_len=128,
        output_len=1,
        uid=10,
        sampling_params=SamplingParams(),
        cache_handle=None,
        cache_anchor_len=64,
        cache_anchor_persistable=True,
    )
    req.mamba_last_track_seqlen = 64
    req.mamba_ping_pong = (1, 2)

    manager.persist_intermediate_cache_anchor(req)
    assert store.stats()["harness_anchor_skipped_unaligned"] == 1

    manager.page_size = 64
    req.table_idx = -1
    manager.persist_intermediate_cache_anchor(req)
    assert store.stats()["harness_anchor_skipped_unaligned"] == 1
    store.close()


def test_real_store_round_trip_for_intermediate_harness_root(tmp_path):
    from freetoken.kvcache.disk_prefix_cache import DiskPrefixStore
    from freetoken.scheduler.prefill import ChunkedReq

    store = DiskPrefixStore(tmp_path, 1 << 20, identity="harness-root")
    pool = _pool()
    kv_cache = SimpleNamespace(
        _kv_buffer=torch.arange(2 * 1 * 256, dtype=torch.float32).view(
            2, 1, 256, 1, 1, 1
        ),
        _pending_ring=torch.arange(4 * 1 * 4, dtype=torch.float32).view(4, 1, 4),
        device=torch.device("cpu"),
        _page_size=1,
    )
    page_table = torch.arange(4 * 256, dtype=torch.int32).view(4, 256)
    manager = CacheManager(
        1024,
        1,
        page_table,
        "hybrid_radix",
        linear_state_pool=pool,
        kv_cache=kv_cache,
        disk_prefix_store=store,
    )
    root = torch.arange(64, dtype=torch.int32)
    request_a_ids = torch.cat((root, torch.arange(64, 129, dtype=torch.int32)))
    req = ChunkedReq(
        input_ids=request_a_ids,
        table_idx=0,
        cached_len=128,
        output_len=1,
        uid=11,
        sampling_params=SamplingParams(),
        cache_handle=None,
        cache_anchor_len=64,
        cache_anchor_kind="opencode",
        cache_anchor_persistable=True,
    )
    req.mamba_ping_pong = tuple(pool.alloc(2))
    req.mamba_next_track_idx = 1
    req.mamba_last_track_seqlen = 64
    frozen = req.mamba_ping_pong[0]
    pool.conv_states[:, frozen].fill_(7)
    pool.recurrent_states[:, frozen].fill_(11)

    manager.persist_intermediate_cache_anchor(req)
    store.flush()
    assert store.stats()["harness_anchor_persisted"] == 1
    store.close()

    reader = DiskPrefixStore(tmp_path, 1 << 20, identity="harness-root")
    request_b_ids = torch.cat((root, torch.tensor([999, 1000], dtype=torch.int32)))
    entry = reader.lookup_longest(request_b_ids)

    assert entry is not None
    assert entry.length == 64
    assert torch.equal(entry.tensors["token_ids"], root)
    reader.close()
