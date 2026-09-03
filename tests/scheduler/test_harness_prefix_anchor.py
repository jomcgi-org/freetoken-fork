from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.core import Batch, Context, Req, SamplingParams
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.scheduler.cache import CacheManager


def _pool(num_slots=8):
    group = LinearGatedDeltaGroupConfig(
        name="linear",
        layer_ids=(0,),
        num_key_heads=2,
        num_value_heads=4,
        key_head_dim=16,
        value_head_dim=16,
        conv_kernel_dim=4,
        output_gate="silu",
    )
    return LinearStatePool(
        group,
        num_slots=num_slots,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        tp_size=1,
    )


def test_harness_anchor_wins_over_deepest_prefill_track(monkeypatch):
    import freetoken.core as core
    from freetoken.attention.linear import build_fla_metadata

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
        cache_anchor_len=64,
        cache_anchor_kind="opencode",
    )
    req.linear_slot_idx = pool.alloc(1)[0]
    req.mamba_ping_pong = tuple(pool.alloc(2))
    batch = Batch(reqs=[req], phase="prefill")
    batch.padded_reqs = batch.reqs

    metadata = build_fla_metadata(batch, torch.device("cpu"))

    assert metadata.track_boundary_row.tolist() == [64]
    assert req.mamba_last_track_seqlen == 64


def test_prefill_admission_aligns_and_carries_harness_anchor():
    from freetoken.message import UserMsg
    from freetoken.scheduler.prefill import PrefillManager

    cache = SimpleNamespace(
        is_hybrid=True,
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


def test_chunked_anchor_persistence_does_not_touch_radix_ownership(monkeypatch):
    class Store:
        def contains(self, _token_ids):
            return False

    pool = _pool()
    page_table = torch.arange(4 * 256, dtype=torch.int32).view(4, 256)
    manager = CacheManager(
        num_pages=1024,
        page_size=1,
        page_table=page_table,
        type="hybrid_radix",
        linear_state_pool=pool,
        disk_prefix_store=Store(),
    )
    req = Req(
        input_ids=torch.arange(129, dtype=torch.int32),
        table_idx=0,
        cached_len=128,
        output_len=1,
        uid=2,
        sampling_params=SamplingParams(),
        cache_handle=None,
        cache_anchor_len=64,
        cache_anchor_kind="pi",
    )
    req.linear_slot_idx = pool.alloc(1)[0]
    req.mamba_ping_pong = tuple(pool.alloc(2))
    req.mamba_next_track_idx = 1
    req.mamba_last_track_seqlen = 64
    queued = []
    monkeypatch.setattr(
        manager,
        "_queue_disk_prefix",
        lambda request, length, indices, frozen: queued.append(
            (request, length, indices.clone(), frozen)
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


def test_existing_disk_anchor_is_not_rewritten(monkeypatch):
    class Store:
        def contains(self, _token_ids):
            return True

    pool = _pool()
    page_table = torch.zeros(4, 256, dtype=torch.int32)
    manager = CacheManager(
        num_pages=1024,
        page_size=1,
        page_table=page_table,
        type="hybrid_radix",
        linear_state_pool=pool,
        disk_prefix_store=Store(),
    )
    req = SimpleNamespace(
        cache_anchor_len=64,
        mamba_last_track_seqlen=64,
        mamba_ping_pong=(1, 2),
        mamba_next_track_idx=1,
        cached_len=128,
        input_ids=torch.arange(128, dtype=torch.int32),
        table_idx=0,
    )
    queued = []
    monkeypatch.setattr(manager, "_queue_disk_prefix", lambda *args: queued.append(args))

    manager.persist_intermediate_cache_anchor(req)

    assert queued == []
