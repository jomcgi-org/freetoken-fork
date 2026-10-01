from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from freetoken.core import Req, SamplingParams
from freetoken.message import UserMsg
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.prefill import (
    LAST_MESSAGE_ANCHOR_MIN_GAIN,
    ChunkedReq,
    PrefillAdder,
    PrefillManager,
)
from freetoken.scheduler.utils import PendingReq


@pytest.fixture(autouse=True)
def _stub_fla(monkeypatch):
    fla = ModuleType("freetoken.kernel.fla")
    fla.__path__ = []
    chunk = ModuleType("freetoken.kernel.fla.chunk")
    chunk.CHUNK_SIZE = 64
    monkeypatch.setitem(sys.modules, "freetoken.kernel.fla", fla)
    monkeypatch.setitem(sys.modules, "freetoken.kernel.fla.chunk", chunk)


class _Outcomes(list):
    def __call__(self, outcome):
        self.append(outcome)


def _manager(page_size=1, store=True):
    outcomes = _Outcomes()
    cache = SimpleNamespace(
        is_hybrid=True,
        disk_prefix_store=object() if store else None,
        page_size=page_size,
        admit_expert_profile=lambda _uid, _ids: None,
        note_harness_anchor=outcomes,
    )
    manager = PrefillManager(
        cache_manager=cache, table_manager=SimpleNamespace(), decode_manager=SimpleNamespace()
    )
    return manager, outcomes


def _admit(manager, *, root=None, last=None, n=4000):
    manager.add_one_req(
        UserMsg(
            uid=1,
            input_ids=torch.arange(n, dtype=torch.int32),
            sampling_params=SamplingParams(),
            cache_anchor_len=root,
            cache_anchor_kind="opencode" if root else None,
            cache_last_anchor_len=last,
        )
    )
    return manager.pending_list[-1]


def test_last_message_anchor_is_aligned_to_chunk_and_page_grid():
    manager, outcomes = _manager(page_size=128)
    pending = _admit(manager, root=640, last=3000)
    assert pending.cache_last_anchor_len == 2944  # 3000 rounded down to lcm(64, 128) * k
    assert pending.cache_anchor_len == 640
    assert outcomes == []


def test_unaligned_shallow_and_storeless_anchors_are_counted_and_dropped():
    manager, outcomes = _manager(page_size=128)
    assert _admit(manager, last=100).cache_last_anchor_len is None
    assert _admit(manager, root=2048, last=2048 + LAST_MESSAGE_ANCHOR_MIN_GAIN - 128 + 5
                  ).cache_last_anchor_len is None
    manager, storeless = _manager(store=False)
    assert _admit(manager, last=3000).cache_last_anchor_len is None
    assert outcomes == ["skipped_last_message_unaligned", "skipped_last_message_shallow"]
    assert storeless == ["skipped_no_store"]


def _adder(budget, cache=None):
    cache = cache or SimpleNamespace(
        swa_paged=False,
        page_size=1,
        prefill_chunk_align=1,
        disk_prefix_store=object(),
        note_harness_anchor=_Outcomes(),
    )
    table = SimpleNamespace(token_pool=torch.zeros(1, 8192, dtype=torch.int32))
    return PrefillAdder(budget, 0, cache, table), cache


def _pending(root, last, n=4096, chunked=None):
    return PendingReq(
        7,
        torch.arange(n, dtype=torch.int32),
        SamplingParams(max_tokens=1),
        chunked_req=chunked,
        cache_anchor_len=root,
        cache_anchor_kind="opencode" if root else None,
        cache_last_anchor_len=last,
    )


def test_two_anchors_in_one_chunk_split_so_each_is_tracked_in_its_own_chunk():
    adder, _ = _adder(8192)
    pending = _pending(root=1024, last=3008)
    first = adder._add_one_req(pending, cache_handle=None, table_idx=0, cached_len=0)
    assert isinstance(first, ChunkedReq)
    assert first.extend_len == 1088  # first chunk edge past the root anchor
    assert (first.cache_anchor_len, first.cache_anchor_kind) == (1024, "opencode")
    assert first.cache_anchor_persistable

    adder, _ = _adder(8192)
    pending.chunked_req = first
    second = adder._add_one_req(pending, cache_handle=None, table_idx=0, cached_len=1088)
    assert type(second) is Req  # the deeper anchor sits inside the final chunk
    assert second.cached_len + second.extend_len == 4096
    assert (second.cache_anchor_len, second.cache_anchor_kind) == (3008, "last_message")
    assert second.cache_anchor_persistable


def test_split_respects_chunk_alignment_unit():
    cache = SimpleNamespace(
        swa_paged=False, page_size=1, prefill_chunk_align=256,
        disk_prefix_store=object(), note_harness_anchor=_Outcomes(),
    )
    adder, _ = _adder(8192, cache)
    first = adder._add_one_req(_pending(1024, 3008), cache_handle=None, table_idx=0, cached_len=0)
    assert first.extend_len == 1280
    assert first.cache_anchor_len == 1024


def test_a_single_last_message_anchor_does_not_split_the_prompt():
    adder, _ = _adder(8192)
    req = adder._add_one_req(_pending(None, 3008), cache_handle=None, table_idx=0, cached_len=0)
    assert type(req) is Req and req.extend_len == 4096
    assert (req.cache_anchor_len, req.cache_anchor_kind) == (3008, "last_message")


def test_anchor_close_to_the_cache_hit_is_shed_and_counted():
    adder, cache = _adder(8192)
    gain = LAST_MESSAGE_ANCHOR_MIN_GAIN
    pending = _pending(None, 3008)
    req = adder._add_one_req(pending, cache_handle=None, table_idx=0, cached_len=3008 - gain + 64)
    assert req.cache_anchor_len is None and not req.cache_anchor_persistable
    assert pending.cache_last_anchor_len is None
    assert list(cache.note_harness_anchor) == ["skipped_last_message_shallow"]

    adder, cache = _adder(8192)
    req = adder._add_one_req(_pending(None, 3008), cache_handle=None, table_idx=0,
                             cached_len=3008 - gain)
    assert req.cache_anchor_len == 3008
    assert list(cache.note_harness_anchor) == []


def test_last_message_persist_is_counted_separately(monkeypatch):
    class Store:
        def __init__(self):
            self.stats = {}

        def contains(self, _ids):
            return False

        def note_harness_anchor(self, outcome):
            self.stats[outcome] = self.stats.get(outcome, 0) + 1

    manager = object.__new__(CacheManager)
    manager.is_hybrid = True
    manager.disk_prefix_store = Store()
    manager.page_size = 64
    manager.page_table = torch.arange(4 * 256, dtype=torch.int32).view(4, 256)
    monkeypatch.setattr(manager, "_queue_disk_prefix", lambda *args: True)
    req = ChunkedReq(
        input_ids=torch.arange(129, dtype=torch.int32), table_idx=0, cached_len=128,
        output_len=1, uid=2, sampling_params=SamplingParams(), cache_handle=None,
        cache_anchor_len=64, cache_anchor_kind="last_message", cache_anchor_persistable=True,
    )
    req.mamba_ping_pong = (1, 2)
    req.mamba_next_track_idx = 1
    req.mamba_last_track_seqlen = 64

    manager.persist_intermediate_cache_anchor(req)

    assert manager.disk_prefix_store.stats == {
        "persisted": 1,
        "persisted_intermediate": 1,
        "persisted_last_message": 1,
    }
