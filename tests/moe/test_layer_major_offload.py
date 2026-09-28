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
    cache._layer_major_full = False
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


def _full_cache(monkeypatch, *, staged=True, stream=True):
    cache = _cache(monkeypatch)
    cache.moe_disk_prefill = "staged" if staged else "cpu"
    cache._disk_prefill_staging = object()
    cache.prefill_overlap = True
    cache.prefill_copy_stream = object() if stream else None
    cache.prefill_hit_d2d = False
    return cache


@pytest.mark.parametrize(
    ("staged", "stream", "expected"), [(True, True, True), (False, True, False), (True, False, False)]
)
def test_whole_layer_streaming_needs_staged_disk_prefill_and_a_copy_stream(
    monkeypatch, staged, stream, expected
):
    cache = _full_cache(monkeypatch, staged=staged, stream=stream)
    cache.begin_layer_major_group()
    assert cache._layer_major_full is expected


def test_whole_layer_group_alternates_buffers_and_restores_the_schedule(monkeypatch):
    cache = _full_cache(monkeypatch)
    cache.num_layers = 5
    cache._prefill_overlap_buffer_ids = [-1, 0, -1, 1, -1]
    cache._staged_prefill_active = True
    cache.prefill_selective_max_tokens = 0
    cache.moe_disk_prefill_min_tokens = 1024
    configured = []
    cache._configure_prefill_overlap_layers = lambda: configured.append(True)
    cache.begin_layer_major_group()
    # begin_prefill applies the group schedule after any staged reconfiguration.
    cache.prefill_copy_stream = None  # skip the CUDA fence in this CPU test
    cache.begin_prefill(8192)
    assert cache._prefill_overlap_buffer_ids == [0, 1, 0, 1, 0]
    cache._prefill_buffer_layer = [3, 4]
    cache._prefill_buffer_released = [False, True]
    cache.end_layer_major_group()
    assert configured and cache._prefill_buffer_layer == [None, None]
    assert cache._prefill_buffer_released == [True, True]


def test_layer_done_streams_the_next_layer_only_in_whole_layer_mode(monkeypatch):
    cache = _full_cache(monkeypatch)
    cache.num_layers = 3
    cache._prefill_overlap_buffer_ids = [0, 1, 0]
    fetched = []
    cache.prefetch_prefill_layer = fetched.append
    cache.layer_major_layer_done(0)
    assert fetched == []  # not in a group
    cache.begin_layer_major_group()
    cache.layer_major_layer_done(0)
    cache.layer_major_layer_done(2)  # no layer 3
    assert fetched == [1]


def test_disk_layers_stream_from_file_in_whole_layer_mode(monkeypatch):
    import torch

    cache = _full_cache(monkeypatch)
    cache.prefill_copy_stream = None
    cache.num_layers = 2
    cache.num_experts = 2
    cache.layer_residency = ["disk", "pinned"]
    cache._prefill_overlap_buffer_ids = [0, 1]
    cache._prefill_buffer_layer = [None, None]
    cache._prefill_buffer_released = [True, True]
    cache._prefill_hit_d2d_active = False
    cache.collect_stats = False
    cache.id_of_slot = torch.full((4,), -1, dtype=torch.int64)
    cache.slot_for_id = torch.full((2, 2), -1, dtype=torch.int64)
    cache.usage = torch.zeros(4)
    sources = [torch.arange(4.0).view(2, 2) + 10, torch.arange(4.0).view(2, 2) + 20]
    cache.banks = [(sources, None)]
    cache.prefill_bank_buffers = [torch.zeros(2, 2, 2)]
    calls = []

    class Staging:
        def copy_bank(self, source, destination, rows=None):
            calls.append(rows)
            destination.copy_(source)
            return source.numel() * source.element_size()

    cache._disk_prefill_staging = Staging()
    cache._layer_major_full = True
    cache.prefetch_prefill_layer(0)
    cache.prefetch_prefill_layer(1)
    assert calls == [None]  # the DISK layer, whole, from its file
    assert torch.equal(cache.prefill_bank_buffers[0][0], sources[0])
    assert torch.equal(cache.prefill_bank_buffers[0][1], sources[1])
