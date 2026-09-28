"""Layer-major prefill: grouping consecutive chunks of one request.

The scheduler must prepare each chunk of a group exactly as chunk-major serving would,
stop at the token budget, the final chunk and an intermediate harness root, and leave the
ordinary path untouched whenever no second chunk joins.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

CHUNK = 8
WIDTH = 128
MAX_RUNNING = 4
UID = 11


def _setup_context() -> None:
    from freetoken.core import Context, get_global_ctx, set_global_ctx

    try:
        get_global_ctx()
    except AssertionError:
        set_global_ctx(Context(page_size=1))


def _scheduler(budget: int, *, prompt_len: int, supports: bool = True, anchor=None):
    from freetoken.core import SamplingParams
    from freetoken.scheduler.cache import CacheManager
    from freetoken.scheduler.decode import DecodeManager
    from freetoken.scheduler.prefill import PrefillManager
    from freetoken.scheduler.scheduler import ForwardInput, Scheduler
    from freetoken.scheduler.table import TableManager
    from freetoken.scheduler.utils import PendingReq

    _setup_context()
    pt = torch.zeros((MAX_RUNNING + 1, WIDTH), dtype=torch.int32, device="cpu")
    cm = CacheManager(num_pages=WIDTH * 2, page_size=1, page_table=pt, type="radix")
    tm = TableManager(max_running_reqs=MAX_RUNNING, page_table=pt)
    dm = DecodeManager(page_size=1)
    pm = PrefillManager(cm, tm, dm)
    pm.pending_list = [
        PendingReq(
            uid=UID,
            input_ids=torch.arange(prompt_len, dtype=torch.int32),
            sampling_params=SamplingParams(max_tokens=4),
            cache_anchor_len=anchor,
        )
    ]
    sched = Scheduler.__new__(Scheduler)
    sched.config = SimpleNamespace(prefill_layer_major_tokens=budget, speculative_mtp="off")
    sched.engine = SimpleNamespace(model=SimpleNamespace(supports_layer_major_prefill=supports))
    sched.prefill_manager = pm
    sched.decode_manager = dm
    sched.cache_manager = cm
    sched.prefill_budget = CHUNK
    prepared = []

    def prepare(batch):
        cm.allocate_paged(batch.reqs)
        prepared.append([(r.cached_len, r.extend_len) for r in batch.reqs])
        return ForwardInput(batch=batch, sample_args=None, input_tuple=None, write_tuple=None)

    sched._prepare_batch = prepare
    sched._report_prompt_admissions = lambda batch: None
    return sched, pm, prepared


def _first(sched, pm):
    batch = pm.schedule_next_batch(CHUNK)
    return sched._prepare_batch(batch)


def test_group_prepares_consecutive_chunks_up_to_the_budget():
    from freetoken.scheduler.prefill import ChunkedReq

    sched, pm, prepared = _scheduler(3 * CHUNK, prompt_len=5 * CHUNK)
    group = sched._schedule_layer_major_group(_first(sched, pm))
    assert group is not None and len(group) == 3
    assert prepared == [[(0, CHUNK)], [(CHUNK, CHUNK)], [(2 * CHUNK, CHUNK)]]
    reqs = [fi.batch.reqs[0] for fi in group]
    assert all(isinstance(r, ChunkedReq) for r in reqs)
    # Host bookkeeping advanced as each chunk's forward launch would.
    assert [r.cached_len for r in reqs] == [CHUNK, 2 * CHUNK, 3 * CHUNK]
    # The continuation stays pending for the next group.
    assert pm.pending_list[0].chunked_req is reqs[-1]


def test_group_ends_at_the_final_chunk():
    from freetoken.scheduler.prefill import ChunkedReq

    sched, pm, prepared = _scheduler(100 * CHUNK, prompt_len=3 * CHUNK + 3)
    group = sched._schedule_layer_major_group(_first(sched, pm))
    assert [len(p) for p in prepared] == [1, 1, 1, 1]
    assert prepared[-1] == [(3 * CHUNK, 3)]
    final = group[-1].batch.reqs[0]
    assert not isinstance(final, ChunkedReq)
    assert final.can_decode and not pm.runnable


def test_no_group_leaves_the_first_chunk_untouched():
    sched, pm, prepared = _scheduler(CHUNK + 1, prompt_len=4 * CHUNK)
    first = _first(sched, pm)
    assert sched._schedule_layer_major_group(first) is None
    req = first.batch.reqs[0]
    # The ordinary forward still owns complete_one for this chunk.
    assert req.cached_len == 0 and len(prepared) == 1


@pytest.mark.parametrize("change", ["off", "unsupported", "decode_running"])
def test_ineligible_batches_use_the_ordinary_path(change):
    sched, pm, prepared = _scheduler(
        4 * CHUNK, prompt_len=4 * CHUNK, supports=change != "unsupported"
    )
    if change == "off":
        sched.config.prefill_layer_major_tokens = 0
    first = _first(sched, pm)
    if change == "decode_running":
        class Running:
            remain_len = 1

        sched.decode_manager.running_reqs = {Running()}
    assert sched._schedule_layer_major_group(first) is None
    assert len(prepared) == 1


def test_group_stops_at_an_intermediate_harness_root():
    sched, pm, prepared = _scheduler(8 * CHUNK, prompt_len=6 * CHUNK)
    sched.cache_manager.disk_prefix_store = object()
    pm.pending_list[0].cache_anchor_len = 2 * CHUNK - 3
    group = sched._schedule_layer_major_group(_first(sched, pm))
    assert group is not None
    # The chunk holding the root drains before any later chunk can reuse its slot.
    assert [p[0] for p in prepared] == [(0, CHUNK), (CHUNK, CHUNK)]
    assert group[-1].batch.reqs[0].cache_anchor_persistable


def test_group_drain_processes_chunks_in_order_and_exposes_the_last():
    from freetoken.scheduler.scheduler import Scheduler

    sched = Scheduler.__new__(Scheduler)
    seen = []
    original = Scheduler._process_last_data

    def record(self, last_data):
        if isinstance(last_data, list):
            return original(self, last_data)
        seen.append(last_data)

    sched._process_last_data = record.__get__(sched)
    items = [("a", 1), ("b", 1), ("c", 1)]
    Scheduler._process_last_data(sched, items)
    assert seen == items
