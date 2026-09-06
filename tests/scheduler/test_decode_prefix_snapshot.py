"""State identity and ownership for retained aligned decode checkpoints."""

from types import SimpleNamespace

import pytest
import torch

from freetoken.core import Context, Req, SamplingParams
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.models.config import LinearGatedDeltaGroupConfig, SlotStateSpec
from freetoken.scheduler.cache import CacheManager


def setup(monkeypatch, *, enabled=True, page_size=4, num_slots=12):
    import freetoken.core as core

    group = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0, 1), num_key_heads=1, num_value_heads=1,
        key_head_dim=4, value_head_dim=4, conv_kernel_dim=4, output_gate="silu")
    pool = LinearStatePool(
        group, num_slots, torch.bfloat16, torch.device("cpu"), tp_size=1,
        slot_states=(SlotStateSpec(name="ple_conv", shape=(2, 3), layer_ids=(1,)),))
    monkeypatch.setattr(core, "_GLOBAL_CTX", Context(page_size=page_size, linear_state_pool=pool))
    table = torch.zeros(4, 128, dtype=torch.int32)
    manager = CacheManager(32, page_size, table, "hybrid_radix", linear_state_pool=pool,
                           decode_prefix_snapshot=enabled)
    return manager, pool


def mark(pool, slot, length):
    pool.conv_states[:, slot].fill_(length)
    pool.recurrent_states[:, slot].fill_(length + 0.25)
    pool.slot_states["ple_conv"][:, slot].fill_(length + 0.5)


def assert_state(pool, slot, length):
    assert torch.all(pool.conv_states[:, slot] == length)
    assert torch.all(pool.recurrent_states[:, slot] == length + 0.25)
    assert torch.all(pool.slot_states["ple_conv"][:, slot] == length + 0.5)


def pending(length):
    ids = torch.arange(1, length + 1, dtype=torch.int32)
    return SimpleNamespace(input_ids=ids, input_len=length, mm_embeds=None)


def start(manager, pool, *, uid=1, table_idx=0):
    prompt = pending(5)
    match = manager.match_req(prompt)
    req = Req(prompt.input_ids, table_idx, match.cuda_handle.cached_len, 48, uid,
              SamplingParams(max_tokens=48), match.cuda_handle)
    manager.lock(req.cache_handle)
    req.linear_slot_idx = pool.alloc(1)[0]
    req.mamba_ping_pong = tuple(pool.alloc(2))
    if req.cached_len:
        manager.page_table[table_idx, :req.cached_len] = match.cuda_handle.get_matched_indices()
    manager.allocate_paged([req])
    mark(pool, req.linear_slot_idx, 5)
    if not req.cached_len:
        mark(pool, req.mamba_ping_pong[0], 4)
        req.mamba_last_track_seqlen = 4
        req.mamba_next_track_idx = 1
    req.complete_one()
    req.append_host(torch.tensor([6], dtype=torch.int32))
    manager.cache_req(req, finished=False)
    return req


def step(manager, pool, req, *, append=True):
    manager.allocate_paged([req])
    mark(pool, req.linear_slot_idx, req.device_len)
    req.complete_one()
    manager.snapshot_decode_prefix([req])
    if append:
        req.append_host(torch.tensor([req.device_len], dtype=torch.int32))


def finish(manager, req):
    manager.cache_req(req, finished=True)
    req.table_idx = -1
    manager.check_integrity()


def test_unaligned_finish_retains_exact_earlier_state_without_extra_slots(monkeypatch):
    manager, pool = setup(monkeypatch)
    req = start(manager, pool)
    available = pool.num_free_slots
    while req.cached_len < 10:
        step(manager, pool, req)
        assert pool.num_free_slots == available
    assert req.mamba_last_track_seqlen == 8
    assert_state(pool, req.linear_slot_idx, 10)
    frozen = req.mamba_ping_pong[1 - req.mamba_next_track_idx]
    assert_state(pool, frozen, 8)
    finish(manager, req)
    hit = manager.match_req(pending(12))
    assert hit.cuda_handle.cached_len == 8
    assert_state(pool, hit.mamba_value, 8)
    # A future live slot must own a copy, preserving the cached state as it advances.
    live = pool.alloc(1)[0]
    pool.copy_from(hit.mamba_value, live)
    mark(pool, live, 11)
    assert_state(pool, hit.mamba_value, 8)
    pool.free(live)


def test_new_boundary_replaces_snapshot_but_unaligned_steps_do_not(monkeypatch):
    manager, pool = setup(monkeypatch)
    req = start(manager, pool)
    while req.cached_len < 13:
        step(manager, pool, req)
    assert req.mamba_last_track_seqlen == 12
    assert_state(pool, req.mamba_ping_pong[1 - req.mamba_next_track_idx], 12)
    finish(manager, req)
    hit = manager.match_req(pending(15))
    assert hit.cuda_handle.cached_len == 12
    assert_state(pool, hit.mamba_value, 12)


def test_duplicate_boundary_does_not_rotate_or_copy(monkeypatch):
    manager, pool = setup(monkeypatch)
    req = start(manager, pool)
    while req.cached_len < 8:
        step(manager, pool, req)
    index = req.mamba_next_track_idx
    monkeypatch.setattr(pool, "copy_from", lambda *_: pytest.fail("duplicate copy"))
    manager.snapshot_decode_prefix([req])
    assert req.mamba_next_track_idx == index


def test_disabled_path_never_copies_state(monkeypatch):
    manager, pool = setup(monkeypatch, enabled=False)
    req = start(manager, pool)
    monkeypatch.setattr(pool, "copy_from", lambda *_: pytest.fail("disabled snapshot copied"))
    while req.cached_len < 10:
        step(manager, pool, req)
    assert req.mamba_last_track_seqlen is None
    finish(manager, req)
    assert manager.match_req(pending(12)).cuda_handle.cached_len == 4


@pytest.mark.parametrize("page_size,cache_type", [(1, "hybrid_radix"), (4, "radix"), (4, "naive")])
def test_option_only_activates_for_paged_hybrid_cache(monkeypatch, page_size, cache_type):
    manager, pool = setup(monkeypatch)
    other = CacheManager(32, page_size, manager.page_table, cache_type,
                         linear_state_pool=pool, decode_prefix_snapshot=True)
    assert not other.decode_prefix_snapshot
    other.snapshot_decode_prefix([object()])


@pytest.mark.parametrize("anchor", [8, 10])
def test_tool_anchor_keeps_last_aligned_state_through_opener(monkeypatch, anchor):
    manager, pool = setup(monkeypatch)
    req = start(manager, pool)
    while req.cached_len < 13:
        if req.cached_len == anchor - 1:
            req.toolcall_anchor_len = anchor
        manager.snapshot_toolcall_anchor([req])
        step(manager, pool, req)
    assert req.mamba_last_track_seqlen == 8
    finish(manager, req)
    hit = manager.match_req(pending(15))
    assert hit.cuda_handle.cached_len == 8
    assert_state(pool, hit.mamba_value, 8)


def test_identical_concurrent_finishes_deduplicate_without_double_free(monkeypatch):
    manager, pool = setup(monkeypatch)
    first = start(manager, pool)
    second = start(manager, pool, uid=2, table_idx=1)
    for req in (first, second):
        while req.cached_len < 10:
            step(manager, pool, req)
        manager.cache_req(req, finished=True)
        req.table_idx = -1
    # The integrity check is idle-only; the second request owns uncached pages
    # while the first one finishes.
    manager.check_integrity()
    assert len(pool._free_slots) == len(set(pool._free_slots))
    assert pool.num_free_slots == pool.num_slots - 3  # sink plus two cached boundaries
    manager.ensure_mamba_slots(pool.num_slots - 1)
    manager.check_integrity()
    assert pool.num_free_slots == pool.num_slots - 1
    assert manager.match_req(pending(12)).cuda_handle.cached_len == 0


def test_failed_request_discards_pending_decode_snapshot(monkeypatch):
    manager, pool = setup(monkeypatch)
    req = start(manager, pool)
    while req.cached_len < 10:
        step(manager, pool, req)
    req.device_len = req.cached_len
    manager.cache_req(req, finished=True, failed=True)
    manager.check_integrity()
    assert manager.match_req(pending(12)).cuda_handle.cached_len == 4
    assert len(pool._free_slots) == len(set(pool._free_slots))


def test_abort_with_host_ids_behind_never_attaches_state_to_shorter_key(monkeypatch):
    manager, pool = setup(monkeypatch)
    req = start(manager, pool)
    while req.cached_len < 7:
        step(manager, pool, req)
    # Overlap has consumed the next token before the previous host result drains.
    # An abort skips that drain, leaving fewer host IDs than the frozen boundary.
    req.input_ids = req.input_ids[:7]
    step(manager, pool, req, append=False)
    assert req.cached_len == 8 and req.mamba_last_track_seqlen == 8
    req.aborted = True
    finish(manager, req)
    hit = manager.match_req(pending(12))
    assert hit.cuda_handle.cached_len == 4
    assert_state(pool, hit.mamba_value, 4)


def test_changed_history_falls_back_to_matching_earlier_state(monkeypatch):
    manager, pool = setup(monkeypatch)
    req = start(manager, pool)
    while req.cached_len < 10:
        step(manager, pool, req)
    finish(manager, req)
    edited = pending(12)
    edited.input_ids[6] = 99
    hit = manager.match_req(edited)
    assert hit.cuda_handle.cached_len == 4
    assert_state(pool, hit.mamba_value, 4)


def test_first_decode_snapshot_survives_previous_prefill_drain(monkeypatch):
    from freetoken.core import Batch
    from freetoken.scheduler.decode import DecodeManager
    from freetoken.scheduler.scheduler import ForwardInput, Scheduler

    manager, pool = setup(monkeypatch)
    prompt = pending(7)
    match = manager.match_req(prompt)
    req = Req(prompt.input_ids, 0, 0, 16, 1, SamplingParams(max_tokens=16), match.cuda_handle)
    manager.lock(req.cache_handle)
    req.linear_slot_idx = pool.alloc(1)[0]
    req.mamba_ping_pong = tuple(pool.alloc(2))
    manager.allocate_paged([req])
    mark(pool, req.linear_slot_idx, 7)
    mark(pool, req.mamba_ping_pong[0], 4)
    req.mamba_last_track_seqlen, req.mamba_next_track_idx = 4, 1
    req.complete_one()  # prefill has launched, but its host result has not drained

    sent = []
    stub = Scheduler.__new__(Scheduler)
    stub.cache_manager = manager
    stub.decode_manager = DecodeManager(4)
    stub.toolcall_anchor_id = None
    stub.token_pool = torch.arange(1, 129, dtype=torch.int32).view(1, -1)

    def forward(batch, _args):
        assert req.cached_len == 7
        mark(pool, req.linear_slot_idx, 8)
        req.complete_one()
        return SimpleNamespace(next_tokens_gpu=torch.tensor([9], dtype=torch.int32))

    stub.engine = SimpleNamespace(forward_batch=forward, moe_offload_cache=None)
    batch = Batch(reqs=[req], phase="decode")
    idx = torch.tensor([0])
    fi = ForwardInput(batch, None, (idx, torch.tensor([7])), (idx, torch.tensor([8])))
    Scheduler._forward(stub, fi)
    assert req.mamba_last_track_seqlen == 8
    assert req.input_ids.numel() == 7
    assert_state(pool, req.mamba_ping_pong[1 - req.mamba_next_track_idx], 8)

    stub.__dict__.update(
        finished_reqs=set(), eos_token_ids=set(), disk_prefix_store=None,
        config=SimpleNamespace(page_size=4), send_result=sent.extend,
        status_reporter=SimpleNamespace(report_batch=lambda *_, **__: None),
        _kv_usage_pages=manager.page_usage, _mamba_slot_usage=lambda: None,
        _swa_token_usage=lambda: None, _gpu_mem_bytes=lambda: 0,
        _queue_stats=lambda: (0, {}, 0.0),
    )
    prefill = Batch(reqs=[req], phase="prefill")
    Scheduler._process_last_data(stub, (
        SimpleNamespace(batch=prefill),
        (None, torch.tensor([8], dtype=torch.int32), SimpleNamespace(synchronize=lambda: None))))
    assert req.input_ids.numel() == 8
    assert req.cache_handle.cached_len == 8
    assert req.mamba_last_track_seqlen is None
    hit = manager.match_req(pending(10))
    assert_state(pool, hit.mamba_value, 8)
    finish(manager, req)
