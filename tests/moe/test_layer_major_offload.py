"""Offload-cache group mode for layer-major prefill."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_HOT = Path(__file__).with_name("test_hot_adapt.py")


def _cache(monkeypatch):
    spec = importlib.util.spec_from_file_location("_hot_adapt_helpers", _HOT)
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    cls = helpers._offload_cache_class_without_triton(monkeypatch)
    cache = cls.__new__(cls)
    cache._layer_major_group = False
    cache._layer_major_begun = False
    cache._layer_major_staged = None
    return cache


def test_missing_rows_stage_each_expert_once_per_group(monkeypatch):
    cache = _cache(monkeypatch)
    assert cache._layer_major_missing_rows(3, [1, 2]) == [1, 2]  # no group: unchanged
    cache.begin_layer_major_group()
    assert cache._layer_major_missing_rows(3, [1, 2, 5]) == [1, 2, 5]
    assert cache._layer_major_missing_rows(3, [2, 5, 7]) == [7]
    assert cache._layer_major_missing_rows(3, [1, 7]) == []
    # A different layer owns the scratch rows now.
    assert cache._layer_major_missing_rows(4, [1, 7]) == [1, 7]
    cache.end_layer_major_group()
    assert cache._layer_major_staged is None
    cache.begin_layer_major_group()
    assert cache._layer_major_missing_rows(4, [1]) == [1]


def test_buffer_zero_copy_invalidates_staged_rows(monkeypatch):
    import torch

    cache = _cache(monkeypatch)
    cache.num_experts = 4
    cache.id_of_slot = torch.full((8,), -1, dtype=torch.int64)
    cache.slot_for_id = torch.full((2, 4), -1, dtype=torch.int64)
    cache.usage = torch.zeros(8)
    cache.begin_layer_major_group()
    cache._layer_major_missing_rows(3, [1])
    cache._invalidate_prefill_buffer(1)
    assert cache._layer_major_staged is not None
    cache._invalidate_prefill_buffer(0)
    assert cache._layer_major_staged is None


def test_begin_prefill_runs_once_per_group(monkeypatch):
    cache = _cache(monkeypatch)
    cache.moe_disk_prefill = "cpu"
    cache.prefill_selective_max_tokens = 0
    cache.prefill_copy_stream = None
    cache._prefill_overlap_buffer_ids = [-1]
    cache.begin_layer_major_group()
    cache.begin_prefill(8192)
    assert cache._layer_major_begun
    # Any second call inside the group returns before touching prefill state.
    cache.moe_disk_prefill = None
    cache.begin_prefill(100)
    cache.end_layer_major_group()
    assert not cache._layer_major_begun



class _Staging:
    def __init__(self):
        self.calls = []

    def copy_bank(self, source, destination, rows=None):
        self.calls.append(list(rows))
        for row in rows:
            destination[row].copy_(source[row])
        return len(rows)


class _Now:
    """Executor double: runs each job at submission, like a finished worker."""

    def submit(self, fn):
        from concurrent.futures import Future

        fn()
        done = Future()
        done.set_result(None)
        return done


def _predictive_cache(monkeypatch):
    import torch

    cache = _cache(monkeypatch)
    cache.num_layers = 3
    cache.num_experts = 4
    cache.layer_residency = ["disk", "disk", "pinned"]
    cache.moe_disk_prefill = "staged"
    cache.prefill_overlap = True
    cache.prefill_copy_stream = object()
    cache.prefill_hit_d2d = False
    cache.collect_stats = False
    cache._disk_prefill_staging = _Staging()
    cache._lm_bg_staging = _Staging()
    cache._lm_executor = _Now()
    cache._lm_copy_stream = None
    cache._lm_predict = {}
    cache.prefill_ready_events = []
    cache.prefill_release_events = []
    cache._prefill_buffer_layer = [None, None]
    cache._prefill_buffer_released = [True, True]
    cache._prefill_buffer_has_release_event = [False, False]
    cache.id_of_slot = torch.full((8,), -1, dtype=torch.int64)
    cache.slot_for_id = torch.full((3, 4), -1, dtype=torch.int64)
    cache.usage = torch.zeros(8)
    sources = [torch.arange(8.0).view(4, 2) + 10 * layer for layer in range(3)]
    cache.banks = [(sources, None)]
    cache.prefill_bank_buffers = [torch.zeros(2, 4, 2)]
    cache._configure_prefill_overlap_layers = lambda: None
    cache.device = torch.device("cpu")
    cache.begin_layer_major_group()
    cache._prefill_overlap_buffer_ids = [0, 1, 0]
    return cache, sources


def test_predictive_staging_is_off_without_disk_layers_or_by_switch(monkeypatch):
    cache, _ = _predictive_cache(monkeypatch)
    assert cache._lm_predictive
    monkeypatch.setenv("FREETOKEN_LAYER_MAJOR_PREDICT", "0")
    cache.begin_layer_major_group()
    assert not cache._lm_predictive
    monkeypatch.delenv("FREETOKEN_LAYER_MAJOR_PREDICT")
    cache.layer_residency = ["pinned"] * 3
    cache.begin_layer_major_group()
    assert not cache._lm_predictive


def test_first_group_stages_routed_rows_then_predicts_the_next(monkeypatch):
    import torch

    cache, sources = _predictive_cache(monkeypatch)
    # No prediction yet: rows are staged on demand, each once per group.
    views = cache.layer_major_disk_views(1, torch.tensor([[2, 0], [2, 3]]))
    assert cache._disk_prefill_staging.calls == [[0, 2, 3]]
    for row in (0, 2, 3):
        assert torch.equal(views[0][row], sources[1][row])
    cache.layer_major_disk_views(1, torch.tensor([[3, 1]]))
    assert cache._disk_prefill_staging.calls == [[0, 2, 3], [1]]
    cache.end_layer_major_group()
    assert cache._lm_predict == {1: {0, 1, 2, 3}}

    # Next group: the prediction is staged ahead, only misses are staged on demand.
    cache._prefill_overlap_buffer_ids = [0, 1, 0]
    cache._lm_predict = {1: {0, 2}}
    cache.begin_layer_major_group()
    cache._prefill_overlap_buffer_ids = [0, 1, 0]
    cache._disk_prefill_staging.calls.clear()
    cache.prefetch_prefill_layer(1)
    assert cache._lm_bg_staging.calls == [[0, 2]]
    views = cache.layer_major_disk_views(1, torch.tensor([[2, 3]]))
    assert cache._disk_prefill_staging.calls == [[3]]
    for row in (0, 2, 3):
        assert torch.equal(views[0][row], sources[1][row])


def test_buffer_reuse_requires_release(monkeypatch):
    import torch

    cache, _ = _predictive_cache(monkeypatch)
    cache.layer_major_disk_views(0, torch.tensor([[1]]))  # layer 0 holds buffer 0
    with pytest.raises(AssertionError):
        cache._lm_claim_buffer(2)  # layer 2 shares buffer 0, not yet released
