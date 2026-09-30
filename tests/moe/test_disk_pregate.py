"""Decode pre-gating (--moe-disk-pregate-experts): prediction, hand-off and advice."""

from __future__ import annotations

import queue

import torch
from torch import nn

from freetoken.moe import cpu_executor
from freetoken.moe.cpu_executor import CpuMoeExecutor


class _Bank:
    def __init__(self, pages: int = 3):
        self.pages = pages
        self.calls: list[list[int]] = []

    def prefetch_experts(self, expert_ids):
        self.calls.append(list(expert_ids))
        return self.pages


def _executor(disk_layers=(0, 2, 5), *, experts=8, hidden=4, fetch=2, max_tokens=1):
    executor = CpuMoeExecutor.__new__(CpuMoeExecutor)
    executor.num_layers = 6
    executor.num_experts = experts
    executor.H = hidden
    executor.max_tokens = max_tokens
    executor.device = torch.device("cpu")
    executor.core_ids = []
    executor._disk_banks = {layer: [_Bank()] for layer in disk_layers}
    executor._disk_prefetch_calls = [0] * 6
    executor._disk_prefetch_pages = [0] * 6
    executor._disk_decode_steps = 0
    executor._disk_route_pairs = 0
    executor._disk_distinct_experts = 0
    executor._moe_cpu_willneed = "always"
    executor.disk_pregate_experts = fetch
    executor._pregate_next = {}
    executor._pregate_gates = {}
    executor._pregate_hot = None
    executor._pregate_host = None
    executor._pregate_pad = None
    executor._pregate_queue = None
    executor._pregate_pending = {}
    executor._pregate_seq = [0] * 6
    executor._pregate_stop = False
    executor._pregate_warned = False
    executor._pregate_steps = 0
    executor._pregate_vec = None
    executor._pregate_checked_at = {}
    executor._reset_pregate_stats()
    return executor


def _stop(executor):
    executor._pregate_stop = True
    executor._pregate_queue.put(None)
    executor._pregate_thread.join(timeout=5)


def _gates(experts=8, hidden=4, layers=(0, 2, 5)):
    return {layer: torch.randn(experts, hidden) for layer in layers}


def test_configure_pairs_each_disk_layer_with_the_next_disk_layer():
    executor = _executor()
    try:
        assert executor.configure_disk_pregate(_gates()) == 2
        assert executor._pregate_next == {0: 2, 2: 5}
        assert executor._pregate_host.shape == (6, 1, 2)
        assert bool((executor._pregate_host == -1).all())
    finally:
        _stop(executor)


def test_configure_disables_without_matching_routers():
    executor = _executor()
    assert executor.configure_disk_pregate({0: torch.zeros(8, 4)}) == 0
    assert executor.disk_pregate_experts == 0
    assert executor._pregate_host is None


def test_issue_masks_hot_experts_of_the_target_layer_and_pads_rows():
    executor = _executor(max_tokens=2)
    hidden = torch.eye(4)[:1]  # picks router column 0
    gate = torch.zeros(8, 4)
    gate[:, 0] = torch.tensor([8.0, 7, 6, 5, 4, 3, 2, 1])
    hot = torch.full((6, 8), -1, dtype=torch.int32)
    hot[2, 0] = 0  # expert 0 is HOT at the target layer 2
    try:
        executor.configure_disk_pregate({0: gate, 2: gate, 5: gate}, hot)
        executor._pregate_issue(0, 2, hidden)
        assert sorted(executor._pregate_host[0, 0].tolist()) == [1, 2]
        assert executor._pregate_host[0, 1].tolist() == [-1, -1]
    finally:
        _stop(executor)


def test_callback_forwards_prediction_and_scores_it_at_the_target_layer():
    executor = _executor()
    executor.configure_disk_pregate = None  # not used: wire the pieces by hand
    executor._pregate_next = {0: 2, 2: 5}
    executor._pregate_host = torch.full((6, 1, 2), -1, dtype=torch.int32)
    executor._pregate_queue = queue.SimpleQueue()
    executor._pregate_host[0, 0] = torch.tensor([3, 6], dtype=torch.int32)

    executor.prefetch_experts(0, torch.tensor([[1, -1]], dtype=torch.int32))
    dst, ids, seq = executor._pregate_queue.get_nowait()
    assert (dst, ids, seq) == (2, [3, 6], 0)
    assert executor._pregate_pending[2] == frozenset({3, 6})

    # The advice runs while layer 2 has not executed yet.
    executor._pregate_advise(dst, ids, seq)
    assert executor._disk_banks[2][0].calls == [[3, 6]]
    assert executor._pregate_advised == 2
    assert executor._pregate_pages == 3

    # Layer 2 routes experts 3 and 4 to the CPU: one of two was predicted.
    executor.prefetch_experts(2, torch.tensor([[3, 4]], dtype=torch.int32))
    stats = executor.pregate_stats()
    assert stats["coverage_cold"] == 0.5
    assert 2 not in executor._pregate_pending

    # A repeat prediction within the recheck window is not probed again.
    executor._pregate_advise(2, [3, 6], executor._pregate_seq[2])
    assert executor._disk_banks[2][0].calls == [[3, 6], [3, 4]]
    assert executor._pregate_recheck_skips == 2

    # A prediction whose target layer already ran is dropped.
    executor._pregate_advise(2, [7], 0)
    assert executor._pregate_late == 1
    # Only the pre-gated advice and layer 2's own reactive advice were issued.
    assert executor._disk_banks[2][0].calls == [[3, 6], [3, 4]]


def test_recent_mode_scores_nonrecent_and_advises_by_residency_not_recency():
    executor = _executor()
    executor._pregate_next = {0: 2}
    executor._pregate_host = torch.full((6, 1, 2), -1, dtype=torch.int32)
    executor._pregate_queue = queue.SimpleQueue()
    executor._moe_cpu_willneed = "recent"
    executor._willneed_recent_steps = 4
    never = -(1 << 60)
    executor._willneed_last_touch = {layer: [never] * 8 for layer in (0, 2, 5)}
    executor._willneed_layer_steps = [10] * 6
    executor._willneed_guard_steps_remaining = 0
    executor._willneed_skipped_experts = 0
    executor._willneed_advised_experts = 0
    executor._willneed_last_touch[2][3] = 9  # recently used at layer 2

    class ProbeBank(_Bank):
        tensor = torch.zeros(8, 64 * 1024, dtype=torch.uint8)

        def prefetch_nonresident_rows(self, row_ids, vec, probe_pages=0):
            self.calls.append(list(row_ids))
            return 5, [6]  # expert 6 had uncached pages

    executor._disk_banks[2] = [ProbeBank()]
    executor._pregate_host[0, 0] = torch.tensor([3, 6], dtype=torch.int32)
    executor.prefetch_experts(0, torch.tensor([[1, -1]], dtype=torch.int32))
    executor._pregate_advise(*executor._pregate_queue.get_nowait())
    # Recently used expert 3 is still checked: eviction ignores recency.
    assert executor._disk_banks[2][0].calls[0] == [3, 6]
    assert executor._pregate_checked == 2
    assert executor._pregate_advised == 1
    assert executor._pregate_pages == 5

    executor.prefetch_experts(2, torch.tensor([[3, 6]], dtype=torch.int32))
    stats = executor.pregate_stats()
    assert stats["coverage_cold"] == 1.0
    # Only expert 6 was non-recent, and it was predicted.
    assert stats["coverage_nonrecent"] == 1.0
    assert executor._pregate_stale_routes == 1


def test_nonresident_page_runs_and_sampled_probe_follow_mincore():
    import ctypes
    import mmap

    from freetoken.moe.host_banks import _nonresident_page_runs, _sampled_pages_resident

    area = mmap.mmap(-1, 8 * 4096)
    view = (ctypes.c_char * len(area)).from_buffer(area)
    base = ctypes.addressof(view)
    for page in (0, 1, 4, 7):
        area[page * 4096] = 1  # fault in
    vec = (ctypes.c_char * 8)()
    assert _nonresident_page_runs(base, 8, vec) == [(2, 2), (5, 2)]
    assert not _sampled_pages_resident(base, 8, 8, vec)
    assert _sampled_pages_resident(base, 8, 2, vec)  # pages 0 and 7 only
    del view
    area.close()


def test_prefill_callbacks_do_not_touch_pregate_state():
    executor = _executor()
    executor._pregate_next = {0: 2}
    executor._pregate_host = torch.full((6, 1, 2), 1, dtype=torch.int32)
    executor._pregate_queue = queue.SimpleQueue()
    executor.prefetch_experts(0, torch.tensor([[1, 2]]), is_prefill=True)
    assert executor._pregate_queue.empty()


def test_router_gates_by_layer_finds_moe_routers():
    from freetoken.engine.engine import _router_gates_by_layer

    class Experts(nn.Module):
        def __init__(self, layer_id):
            super().__init__()
            self.layer_id = layer_id

    class Block(nn.Module):
        def __init__(self, layer_id):
            super().__init__()
            self.gate = nn.Linear(4, 8, bias=False)
            self.experts = Experts(layer_id)

    model = nn.ModuleList([Block(0), nn.Linear(2, 2), Block(3)])
    gates = _router_gates_by_layer(model)
    assert sorted(gates) == [0, 3]
    assert gates[3] is model[2].gate.weight


def test_pregate_thread_drains_queue_and_exits():
    executor = _executor()
    try:
        executor.configure_disk_pregate(_gates())
        executor._pregate_queue.put((2, [1, 4], 0))
    finally:
        _stop(executor)
    assert executor._disk_banks[2][0].calls == [[1, 4]]
    assert not executor._pregate_thread.is_alive()
    assert cpu_executor._PREGATE_LOG_STEPS > 0


def test_router_gates_by_layer_walks_baseop_trees():
    from freetoken.engine.engine import _router_gates_by_layer
    from freetoken.layers import BaseOP

    class Experts(BaseOP):
        def __init__(self, layer_id):
            self.layer_id = layer_id

    class Gate(BaseOP):
        def __init__(self):
            self.weight = torch.zeros(8, 4)

    class Block(BaseOP):
        def __init__(self, layer_id):
            self.gate = Gate()
            self.experts = Experts(layer_id)

    class Model(BaseOP):
        def __init__(self):
            self.layers = [Block(1), Block(4)]

    model = Model()
    gates = _router_gates_by_layer(model)
    assert sorted(gates) == [1, 4]
    assert gates[4] is model.layers[1].gate.weight


def test_willneed_splits_advice_into_readahead_sized_calls(monkeypatch):
    from freetoken.moe import host_banks

    calls = []

    class Libc:
        def madvise(self, address, size, advice):
            calls.append((address, size))
            return 0

    monkeypatch.setattr(host_banks, "_libc", lambda: Libc())
    monkeypatch.setattr(host_banks, "_WILLNEED_CHUNK", 128 * 1024)
    host_banks._willneed(1 << 20, 300 * 1024)
    assert calls == [(1 << 20, 128 * 1024), ((1 << 20) + 128 * 1024, 128 * 1024),
                     ((1 << 20) + 256 * 1024, 44 * 1024)]
