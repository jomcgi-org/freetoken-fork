"""GPU-free coverage for batch-aware CPU MoE decode route grouping."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
import torch


def _wait_for_coordinator(done: torch.Tensor) -> None:
    deadline = time.monotonic() + 3.0
    while int(done[0]) == 0 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert int(done[0]) == 1


def _make_bf16_cache(experts: int, hidden: int, intermediate: int):
    gate_up = torch.randn(
        experts, 2 * intermediate, hidden, dtype=torch.bfloat16,
    ).mul_(0.05).contiguous()
    down = torch.randn(
        experts, hidden, intermediate, dtype=torch.bfloat16,
    ).mul_(0.05).contiguous()
    return SimpleNamespace(
        quant_format="bf16",
        bank_sources={"gate_up": [gate_up], "down": [down]},
        num_layers=1,
        num_experts=experts,
        decode_target="cpu",
        cpu_executor=None,
    )


def _run_grouped_decode_on_cpu(executor, hidden, weights, ids):
    """Run a persistent task directly, avoiding the CUDA transport wrapper."""
    batch = hidden.shape[0]
    io = executor._io_for(batch)
    io["x"].copy_(hidden)
    io["ids"].copy_(ids)
    io["w"].copy_(weights)
    executor._ext.run_task(executor._task_for(0, batch))
    return io["y"].clone()


@pytest.mark.parametrize("apply_on_input", [False, True])
def test_grouped_decode_matches_ungrouped_reference_with_recurrent_routes(apply_on_input):
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    torch.manual_seed(1308 + int(apply_on_input))
    experts, hidden_size, intermediate, top_k, batch = 16, 64, 64, 4, 8
    cache = _make_bf16_cache(experts, hidden_size, intermediate)
    executor = CpuMoeExecutor(
        cache,
        top_k=top_k,
        activation="silu",
        apply_router_weight_on_input=apply_on_input,
        num_threads=3,
        max_tokens=batch,
        device=torch.device("cpu"),
    )
    hidden = torch.randn(batch, hidden_size, dtype=torch.bfloat16)
    weights = torch.rand(batch, top_k, dtype=torch.float32)
    # Experts 0 and 1 recur in every token. The remaining routes cover enough
    # experts to exercise multiple expert-major work items and stable reduction.
    ids = torch.tensor(
        [
            [0, 1, 2, 3],
            [0, 1, 4, 5],
            [0, 1, 6, 7],
            [0, 1, 8, 9],
            [0, 1, 2, 4],
            [0, 1, 3, 5],
            [0, 1, 6, 8],
            [0, 1, 7, 9],
        ],
        dtype=torch.int32,
    )

    grouped = _run_grouped_decode_on_cpu(executor, hidden, weights, ids)
    ungrouped = executor.prefill(0, hidden, weights, ids)

    # Both schedules evaluate each route identically and reduce in original
    # top-k order, so the native executor's determinism contract is bit exact.
    assert torch.equal(grouped, ungrouped)


def test_grouped_decode_skips_invalid_routes_like_ungrouped_reference():
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    torch.manual_seed(1310)
    experts, hidden_size, intermediate, top_k, batch = 8, 64, 64, 4, 4
    cache = _make_bf16_cache(experts, hidden_size, intermediate)
    executor = CpuMoeExecutor(
        cache,
        top_k=top_k,
        activation="silu",
        apply_router_weight_on_input=False,
        num_threads=2,
        max_tokens=batch,
        device=torch.device("cpu"),
    )
    hidden = torch.randn(batch, hidden_size, dtype=torch.bfloat16)
    weights = torch.rand(batch, top_k, dtype=torch.float32)
    ids = torch.tensor(
        [[0, 1, -1, 2], [0, -1, 1, 3], [0, 1, 2, -1], [0, 1, -1, 3]],
        dtype=torch.int32,
    )

    grouped = _run_grouped_decode_on_cpu(executor, hidden, weights, ids)
    ungrouped = executor.prefill(0, hidden, weights, ids)

    assert torch.equal(grouped, ungrouped)


def test_coordinator_empty_skip_zeros_partial_and_preserves_routed_tasks():
    try:
        from freetoken.kernel import _cpu_moe
    except (ImportError, OSError) as exc:
        pytest.skip(f"CPU MoE extension is not built: {exc}")
    if not hasattr(_cpu_moe.CpuMoeExecutor, "set_empty_skip"):
        pytest.skip("CPU MoE extension needs rebuilding for empty-skip coverage")

    from freetoken.moe.cpu_executor import CpuMoeExecutor

    torch.manual_seed(1311)
    experts, hidden_size, intermediate, top_k, batch = 8, 64, 64, 4, 2
    cache = _make_bf16_cache(experts, hidden_size, intermediate)
    executor = CpuMoeExecutor(
        cache,
        top_k=top_k,
        activation="silu",
        apply_router_weight_on_input=False,
        num_threads=2,
        max_tokens=batch,
        device=torch.device("cpu"),
        step_timing=True,
        moe_cpu_empty_skip="on",
    )
    hidden = torch.randn(batch, hidden_size, dtype=torch.bfloat16)
    weights = torch.rand(batch, top_k, dtype=torch.float32)
    valid_ids = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.int32)
    reference = executor.prefill(0, hidden, weights, valid_ids)

    io = executor._io_for(batch)
    task = executor._task_for(0, batch)
    callback_calls = []

    def callback(*args):
        callback_calls.append(args)

    executor._ext.set_pre_run_callback(callback)
    ready = torch.zeros(1, dtype=torch.int64)
    done = torch.zeros(1, dtype=torch.int64)
    executor._ext.register_flag_task(0, task)
    executor._ext.start_flag_coordinator(
        ready.data_ptr(), done.data_ptr(), 1, -1
    )

    io["x"].copy_(hidden)
    io["w"].copy_(weights)
    io["ids"].fill_(-1)
    io["y"].fill_(7)
    ready[0] = 1
    _wait_for_coordinator(done)

    assert torch.count_nonzero(io["y"]) == 0
    assert callback_calls == []
    empty_timing = executor._ext.step_timing_snapshot_and_reset()[0]
    assert empty_timing["tasks"] == 1
    assert empty_timing["empty_tasks"] == 1
    assert empty_timing["empty_us"] >= 0
    assert empty_timing["wake_us"] == 0
    assert empty_timing["compute_us"] == 0
    assert empty_timing["signal_us"] == 0

    done[0] = 0
    io["ids"].copy_(valid_ids)
    ready[0] = 1
    _wait_for_coordinator(done)

    assert torch.equal(io["y"], reference)
    assert len(callback_calls) == 1
    routed_timing = executor._ext.step_timing_snapshot_and_reset()[0]
    assert routed_timing["tasks"] == 1
    assert routed_timing["empty_tasks"] == 0
