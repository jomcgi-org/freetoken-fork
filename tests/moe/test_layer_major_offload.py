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
