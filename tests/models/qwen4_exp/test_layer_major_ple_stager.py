"""Layer-major PLE staging: a request-level host thread stages every remaining chunk's
rows into a ring of bank slots.

A slot is restaged only after the chunk that used it released its gather, row ids are
the host hash of each planned chunk's prompt range, chunks that ran outside a group
free their slots, an unplanned chunk is refused, and a staging failure reaches the chunk
that needs it (and every later chunk) so the forward path can take over.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest
import torch

from freetoken.models.qwen4_exp.model import _PleRequestStager, _PleStageMismatch

HEADS = 3
STEP = 4
UID = 7


class _Embedding:
    def host_prefill_row_ids(self, reqs, max_tokens):
        assert max_tokens == STEP
        (req,) = reqs
        assert req.input_ids.numel() == req.device_len
        rows = req.device_len - req.cached_len
        return torch.full((rows, HEADS), req.cached_len, dtype=torch.int64)


class _Table:
    def __init__(self, fail_at=None):
        self.staged = []
        self.fail_at = fail_at
        self.lock = threading.Lock()

    def stage_prefill_slot(self, ids, slot, slot_rows, out):
        begin = int(ids[0, 0])
        if begin == self.fail_at:
            raise OSError("read failed")
        with self.lock:
            self.staged.append((begin, slot))
        local = out.view(-1)[: ids.numel()].view(ids.shape)
        local.copy_(ids + slot * slot_rows)
        return local


def _stager(table, *, prompt_len=5 * STEP + 2, start=0, slots=2):
    prompt = torch.arange(prompt_len)
    return _PleRequestStager(
        _Embedding(), table, prompt, UID, start, STEP, HEADS, slots, torch.device("cpu")
    )


def _batch(begin, end, uid=UID):
    return SimpleNamespace(reqs=[SimpleNamespace(uid=uid, cached_len=begin, device_len=end)])


def _wait_for(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.005)
    return predicate()


def test_plans_every_remaining_chunk_and_rotates_slots_after_release():
    table = _Table()
    stager = _stager(table, start=STEP)
    try:
        assert stager._ranges == [(4, 8), (8, 12), (12, 16), (16, 20), (20, 22)]
        index, _token, local = stager.take(_batch(4, 8))
        assert index == 0 and int(local[0, 0]) == 4
        index, _token, local = stager.take(_batch(8, 12))
        assert index == 1 and int(local[0, 0]) == 8 + STEP * HEADS
        time.sleep(0.05)
        assert table.staged == [(4, 0), (8, 1)]
        stager.release(0)
        assert _wait_for(lambda: len(table.staged) == 3)
        assert table.staged[2] == (12, 0)
    finally:
        stager.close()
    assert not stager._thread.is_alive()


def test_the_final_partial_chunk_is_planned_and_finishes_the_request():
    stager = _stager(_Table(), slots=8)
    try:
        for index, (begin, end) in enumerate(stager._ranges):
            got, _token, local = stager.take(_batch(begin, end))
            assert got == index and local.shape == (end - begin, HEADS)
            assert not stager.finished
            stager.release(got)
        assert stager.finished
    finally:
        stager.close()


def test_a_chunk_that_ran_outside_a_group_frees_its_slot():
    table = _Table()
    stager = _stager(table)
    try:
        stager.take(_batch(0, 4))
        stager.release(0)
        # Chunk 1 ran chunk-major; chunk 2 (slot 0) and chunk 3 (slot 1) still stage.
        index, _token, _local = stager.take(_batch(8, 12))
        assert index == 2
        assert _wait_for(lambda: (12, 1) in table.staged)
    finally:
        stager.close()


def test_unplanned_chunks_are_refused():
    stager = _stager(_Table())
    try:
        assert stager.index_of(_batch(0, 4, uid=UID + 1)) is None
        assert stager.index_of(_batch(2, 6)) is None
        assert stager.index_of(_batch(0, 3)) is None
        with pytest.raises(_PleStageMismatch):
            stager.take(_batch(4, 9))
    finally:
        stager.close()


def test_failure_reaches_the_chunk_and_every_later_one():
    stager = _stager(_Table(fail_at=4), slots=4)
    try:
        stager.take(_batch(0, 4))
        for begin in (4, 8, 12):
            with pytest.raises(OSError):
                stager.take(_batch(begin, begin + STEP))
    finally:
        stager.close()


def test_close_unblocks_a_thread_waiting_for_a_release():
    table = _Table()
    stager = _stager(table)
    assert _wait_for(lambda: len(table.staged) == 2)
    stager.close()
    assert not stager._thread.is_alive()
    assert len(table.staged) == 2
