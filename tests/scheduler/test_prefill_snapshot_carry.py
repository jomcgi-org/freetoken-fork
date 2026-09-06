"""Retain a prefill checkpoint when later chunks contain no new FLA boundary.

CPU state/ownership tests use the real prefill scheduler and FLA metadata.
Sentinel tensor contents stand in for the forward's states. Kernel numerical
parity is covered by the existing GDN, PLE and QSA integration checks.
"""

from types import SimpleNamespace

import pytest
import torch

from freetoken.attention.linear import build_fla_metadata
from freetoken.core import Context, SamplingParams
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.models.config import LinearGatedDeltaGroupConfig, SlotStateSpec
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.decode import DecodeManager
from freetoken.scheduler.prefill import ChunkedReq, PrefillManager
from freetoken.scheduler.table import TableManager
from freetoken.scheduler.utils import PendingReq


def tensors(pool, slot):
    return [pool.recurrent_states[:, slot], pool.conv_states[:, slot],
            *(state[:, slot] for state in pool.slot_states.values())]


def fill(pool, slot, boundary):
    for i, tensor in enumerate(tensors(pool, slot)):
        tensor.fill_(boundary + i)


def setup(monkeypatch):
    import freetoken.core as core

    # No CUDA context or pinned allocations, including on the GPU test host.
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    group = LinearGatedDeltaGroupConfig(
        name='linear', layer_ids=(0, 1), num_key_heads=1, num_value_heads=1,
        key_head_dim=4, value_head_dim=4, conv_kernel_dim=4, output_gate='silu')
    pool = LinearStatePool(
        group, 12, torch.bfloat16, torch.device('cpu'), tp_size=1,
        slot_states=(SlotStateSpec(name='ple_conv', shape=(2, 3), layer_ids=(1,)),
                     SlotStateSpec(name='ple_ngram_ctx', shape=(2,), dtype=torch.int32)))
    monkeypatch.setattr(core, '_GLOBAL_CTX', Context(page_size=64, linear_state_pool=pool))
    table = torch.zeros(2, 1024, dtype=torch.int32)
    cm = CacheManager(16, 64, table, 'hybrid_radix', linear_state_pool=pool)
    tm = TableManager(max_running_reqs=1, page_table=table)
    pm = PrefillManager(cm, tm, DecodeManager(page_size=64))
    return cm, tm, pm, pool


def prefill(monkeypatch, prompt_len, budgets):
    cm, tm, pm, pool = setup(monkeypatch)
    ids = torch.arange(prompt_len, dtype=torch.int32)
    pm.pending_list = [PendingReq(1, ids, SamplingParams(max_tokens=4))]
    last_boundary = None
    snapshots = []
    for budget in budgets:
        batch = pm.schedule_next_batch(budget)
        assert batch is not None and len(batch.reqs) == 1
        req = batch.reqs[0]
        batch.padded_reqs = batch.reqs
        cm.allocate_paged(batch.reqs)
        metadata = build_fla_metadata(batch, torch.device('cpu'))
        fill(pool, req.linear_slot_idx, req.device_len)
        if metadata.track_dst is not None:
            last_boundary = req.mamba_last_track_seqlen
            dst = int(metadata.track_dst[0])
            fill(pool, dst, last_boundary)
            snapshots.append(last_boundary)
        req.complete_one()
        if isinstance(req, ChunkedReq):
            # The next chunk is constructed before the old chunk drains. Like
            # Scheduler._process_last_data, do not insert intermediate chunks.
            assert cm.prefix_cache.size_info.evictable_size == 0
    assert not pm.runnable and not isinstance(req, ChunkedReq)
    return cm, tm, pm, pool, req, last_boundary, snapshots


@pytest.mark.parametrize('tail', [1, 7, 64, 65])
def test_final_chunk_preserves_or_replaces_frozen_state_and_reuses_it(monkeypatch, tail):
    cm, tm, pm, pool, req, boundary, snapshots = prefill(monkeypatch, 128 + tail, [128, 128])
    assert snapshots == ([64] if tail <= 64 else [64, 192])
    frozen = req.mamba_ping_pong[1 - req.mamba_next_track_idx]
    expected = [t.clone() for t in tensors(pool, frozen)]
    expected_pages = cm.page_table[req.table_idx, :boundary].clone()

    # Final-prefill commit must publish the existing freeze without copying
    # the newer live state onto an earlier key.
    req.append_host(torch.tensor([777], dtype=torch.int32))
    cm.cache_req(req, finished=False)
    assert req.cache_handle.cached_len == boundary
    assert req.mamba_last_track_seqlen is None
    assert cm.prefix_cache.match_prefix(req.input_ids).mamba_value == frozen

    # One ordinary decode ends unaligned, preventing a live-state donation
    # from hiding a missing prefill checkpoint.
    cm.allocate_paged([req])
    fill(pool, req.linear_slot_idx, req.device_len)
    req.complete_one()
    req.append_host(torch.tensor([778], dtype=torch.int32))
    cm.cache_req(req, finished=True)
    tm.free(req.table_idx)
    cm.check_integrity()
    assert pool.num_free_slots == pool.num_slots - 2  # padding and one cached snapshot
    assert all(torch.equal(a, b) for a, b in zip(tensors(pool, frozen), expected))

    # A real second admission restores from that frozen slot, with the exact
    # prefix pages and state, while retaining independent live storage.
    pm.pending_list = [PendingReq(2, torch.arange(128 + tail + 5, dtype=torch.int32),
                                  SamplingParams(max_tokens=4))]
    resumed = pm.schedule_next_batch(1024).reqs[0]
    assert resumed.cached_len == boundary and resumed.mamba_restore_src == frozen
    assert torch.equal(cm.page_table[resumed.table_idx, :boundary], expected_pages)
    pool.copy_from(frozen, resumed.linear_slot_idx)
    assert all(torch.equal(a, b) for a, b in zip(tensors(pool, resumed.linear_slot_idx), expected))
    cm.cache_req(resumed, finished=True)
    tm.free(resumed.table_idx)
    cm.check_integrity()


def test_marker_survives_more_than_one_chunk_without_a_new_boundary(monkeypatch):
    cm, tm, _pm, pool, req, boundary, snapshots = prefill(monkeypatch, 200, [128, 32, 64])
    assert boundary == 64 and snapshots == [64]
    cm.cache_req(req, finished=False)
    assert req.cache_handle.cached_len == 64
    cm.cache_req(req, finished=True)
    tm.free(req.table_idx)
    cm.check_integrity()
    assert pool.num_free_slots == pool.num_slots - 2


def test_finish_before_prefill_commit_donates_the_carried_freeze(monkeypatch):
    cm, tm, _pm, pool, req, boundary, _snapshots = prefill(monkeypatch, 135, [128, 128])
    frozen = req.mamba_ping_pong[1 - req.mamba_next_track_idx]
    expected = [t.clone() for t in tensors(pool, frozen)]
    req.aborted = True
    cm.cache_req(req, finished=True)
    tm.free(req.table_idx)
    cm.check_integrity()
    match = cm.match_req(SimpleNamespace(input_ids=torch.arange(140, dtype=torch.int32),
                                         input_len=140, mm_embeds=None))
    assert match.cuda_handle.cached_len == boundary == 64
    assert match.mamba_value == frozen
    assert all(torch.equal(a, b) for a, b in zip(tensors(pool, frozen), expected))


@pytest.mark.parametrize('tail', [7, 65])
def test_failed_final_forward_discards_pending_state(monkeypatch, tail):
    cm, tm, pm, pool = setup(monkeypatch)
    pm.pending_list = [PendingReq(1, torch.arange(128 + tail, dtype=torch.int32),
                                  SamplingParams(max_tokens=4))]
    for first in (True, False):
        batch = pm.schedule_next_batch(128)
        batch.padded_reqs = batch.reqs
        req = batch.reqs[0]
        cm.allocate_paged(batch.reqs)
        metadata = build_fla_metadata(batch, torch.device('cpu'))
        if first:
            fill(pool, req.linear_slot_idx, req.device_len)
            fill(pool, int(metadata.track_dst[0]), req.mamba_last_track_seqlen)
            req.complete_one()
    # Metadata may point at the earlier freeze or an unwritten new one. The
    # OOM cleanup path must discard both, without publishing partial state.
    pm.abort_req(req.uid)
    cm.cache_req(req, finished=True, failed=True)
    tm.free(req.table_idx)
    cm.check_integrity()
    assert cm.prefix_cache.size_info.evictable_size == 0
    assert pool.num_free_slots == pool.num_slots - 1  # padding only


def profile_source(cm, profile):
    calls = []
    cm.moe_offload_cache = SimpleNamespace(
        session_profile_enabled=profile is not None,
        export_session_profile=lambda table_idx: calls.append(('export', table_idx)) or profile,
        release_session_profile=lambda uid: calls.append(('release', uid)),
        admit_session_profile=lambda uid, value: calls.append(('admit', uid, value)))
    return calls


@pytest.mark.parametrize('tail', [7, 64, 65])
def test_finish_keeps_profile_on_committed_prefix_without_changing_state(monkeypatch, tail):
    from freetoken.moe.session_profile import SessionExpertProfile

    cm, tm, _pm, pool, req, boundary, _snapshots = prefill(monkeypatch, 128 + tail, [128, 128])
    profile = SessionExpertProfile(ids=((1, 2),), counts=((3.0, 1.0),))
    calls = profile_source(cm, profile)
    cm.cache_req(req, finished=False)
    node = req.cache_handle.node
    frozen = node.mamba_value
    expected = [t.clone() for t in tensors(pool, frozen)]
    expected_pages = cm.page_table[req.table_idx, :boundary].clone()
    assert calls == [] and node.expert_profile is None

    cm.cache_req(req, finished=True)
    tm.free(req.table_idx)
    cm.check_integrity()
    assert calls == [('export', req.table_idx), ('release', req.uid)]
    assert node.expert_profile is profile
    assert all(torch.equal(a, b) for a, b in zip(tensors(pool, frozen), expected))

    # The earlier boundary remains useful even when an aligned finish can also
    # donate a longer prefix. Admission uses its advice with the same KV/state.
    ids = torch.arange(boundary + 1, dtype=torch.int32)
    match = cm.prefix_cache.match_prefix(ids)
    assert match.cached_len == boundary and match.mamba_value == frozen
    assert torch.equal(match.kv_indices, expected_pages)
    assert cm.admit_expert_profile(2, ids) is profile
    assert calls[-1] == ('admit', 2, profile)
    assert cm.lookup_expert_profile(torch.arange(133 + tail, dtype=torch.int32)) is profile


@pytest.mark.parametrize('failed', [False, True])
def test_finish_without_export_preserves_existing_prefix_profile(monkeypatch, failed):
    from freetoken.moe.session_profile import SessionExpertProfile

    cm, tm, _pm, _pool, req, _boundary, _snapshots = prefill(monkeypatch, 135, [128, 128])
    cm.cache_req(req, finished=False)
    node = req.cache_handle.node
    old_profile = SessionExpertProfile(ids=((1,),), counts=((2.0,),))
    node.expert_profile = old_profile
    calls = profile_source(cm, old_profile if failed else None)

    cm.cache_req(req, finished=True, failed=failed)
    tm.free(req.table_idx)
    cm.check_integrity()
    assert node.expert_profile is old_profile
    assert calls == ([('release', req.uid)] if failed else
                     [('export', req.table_idx), ('release', req.uid)])


def test_finish_without_reusable_prefix_does_not_attach_profile_to_root(monkeypatch):
    from freetoken.moe.session_profile import SessionExpertProfile

    cm, tm, _pm, pool, req, boundary, _snapshots = prefill(monkeypatch, 7, [8])
    assert boundary is None and req.cache_handle.cached_len == 0
    root = req.cache_handle.node
    profile_source(cm, SessionExpertProfile(ids=((1,),), counts=((2.0,),)))
    cm.cache_req(req, finished=True)
    tm.free(req.table_idx)
    cm.check_integrity()
    assert root.expert_profile is None
    assert cm.lookup_expert_profile(torch.arange(12, dtype=torch.int32)) is None
    assert pool.num_free_slots == pool.num_slots - 1
