"""Layer-major PLE staging: a host thread stages chunk rows into rotating bank slots.

A slot is restaged only after the chunk that used it released its gather, row ids are
the host hash of each chunk's own request, and a staging failure reaches the chunk
that needs it (and every later chunk) so the forward path can take over.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest
import torch

from freetoken.models.qwen4_exp.model import _PleGroupStager

HEADS = 3
TOKENS = 4


class _Embedding:
    def host_prefill_row_ids(self, reqs, max_tokens):
        assert max_tokens == TOKENS
        (req,) = reqs
        return torch.full((TOKENS, HEADS), req.uid, dtype=torch.int64)


class _Table:
    def __init__(self, fail_at=None):
        self.staged = []
        self.fail_at = fail_at
        self.lock = threading.Lock()

    def stage_prefill_slot(self, ids, slot, slot_rows, out):
        uid = int(ids[0, 0])
        if uid == self.fail_at:
            raise OSError("read failed")
        with self.lock:
            self.staged.append((uid, slot))
        out.copy_(ids + slot * slot_rows)
        return out


def _batches(count):
    return [
        SimpleNamespace(input_ids=torch.zeros(TOKENS), reqs=[SimpleNamespace(uid=index)])
        for index in range(count)
    ]


def _wait_for(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.005)
    return predicate()


def test_slots_rotate_only_after_release():
    table = _Table()
    stager = _PleGroupStager(_Embedding(), table, _batches(5), TOKENS, HEADS, 2)
    try:
        for index in range(2):
            _token, local = stager.take(index)
            assert torch.equal(local, torch.full((TOKENS, HEADS), index + (index % 2) * TOKENS * HEADS))
        # Chunk 2 reuses chunk 0's slot, so it waits for chunk 0's release.
        time.sleep(0.05)
        assert table.staged == [(0, 0), (1, 1)]
        stager.release(0)
        assert _wait_for(lambda: len(table.staged) == 3)
        assert table.staged[2] == (2, 0)
        _token, local = stager.take(2)
        assert int(local[0, 0]) == 2
    finally:
        stager.close()
    assert not stager._thread.is_alive()


def test_tokens_are_distinct_per_take():
    stager = _PleGroupStager(_Embedding(), _Table(), _batches(2), TOKENS, HEADS, 2)
    try:
        first, _ = stager.take(0)
        second, _ = stager.take(1)
        assert first is not second
    finally:
        stager.close()


def test_failure_reaches_the_chunk_and_every_later_one():
    stager = _PleGroupStager(_Embedding(), _Table(fail_at=1), _batches(4), TOKENS, HEADS, 2)
    try:
        stager.take(0)
        for index in (1, 2, 3):
            with pytest.raises(OSError):
                stager.take(index)
    finally:
        stager.close()


def test_close_unblocks_a_thread_waiting_for_a_release():
    table = _Table()
    stager = _PleGroupStager(_Embedding(), table, _batches(6), TOKENS, HEADS, 2)
    assert _wait_for(lambda: len(table.staged) == 2)
    stager.close()
    assert not stager._thread.is_alive()
    assert len(table.staged) == 2
