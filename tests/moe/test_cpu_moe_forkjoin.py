"""GPU-free parity for the CPU MoE fork-join options (fused combine, adaptive wake)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

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
    batch = hidden.shape[0]
    io = executor._io_for(batch)
    io["x"].copy_(hidden)
    io["ids"].copy_(ids)
    io["w"].copy_(weights)
    executor._ext.run_task(executor._task_for(0, batch))
    return io["y"].clone()


_FLAGS = (
    "FREETOKEN_CPU_MOE_FUSE",
    "FREETOKEN_CPU_MOE_BYTES_PER_WORKER",
    "FREETOKEN_CPU_MOE_P1_BALANCE",
    "FREETOKEN_CPU_MOE_SYNC_SPIN_US",
)


def _decode(env, monkeypatch, hidden, weights, ids, experts, intermediate, threads):
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    for name in _FLAGS:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    torch.manual_seed(7)
    cache = _make_bf16_cache(experts, hidden.shape[1], intermediate)
    executor = CpuMoeExecutor(
        cache,
        top_k=ids.shape[1],
        activation="silu",
        apply_router_weight_on_input=False,
        num_threads=threads,
        max_tokens=hidden.shape[0],
        device=torch.device("cpu"),
    )
    return [
        _run_grouped_decode_on_cpu(executor, hidden, weights, ids) for _ in range(3)
    ]


_CONFIGS = [
    {"FREETOKEN_CPU_MOE_FUSE": "1"},
    {"FREETOKEN_CPU_MOE_BYTES_PER_WORKER": "20000"},
    {"FREETOKEN_CPU_MOE_P1_BALANCE": "1"},
    {"FREETOKEN_CPU_MOE_SYNC_SPIN_US": "50"},
    {
        "FREETOKEN_CPU_MOE_FUSE": "1",
        "FREETOKEN_CPU_MOE_BYTES_PER_WORKER": "20000",
        "FREETOKEN_CPU_MOE_P1_BALANCE": "1",
        "FREETOKEN_CPU_MOE_SYNC_SPIN_US": "50",
    },
]


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("config", _CONFIGS, ids=lambda c: "+".join(
    k.removeprefix("FREETOKEN_CPU_MOE_") for k in c))
def test_forkjoin_options_are_bit_identical(config, threads, monkeypatch):
    experts, hidden_size, intermediate, top_k = 12, 96, 80, 4
    torch.manual_seed(11)
    # Batch sizes 1 and 3 (shared and disjoint experts), one invalid route, and an
    # all-invalid layer (the fused path must still write the zero rows).
    for ids in (
        torch.tensor([[2, 5, -1, -1]], dtype=torch.int32),
        torch.tensor([[0, 1, 2, 3], [0, 4, 5, -1], [6, 7, 8, 1]], dtype=torch.int32),
        torch.tensor([[-1, -1, -1, -1]], dtype=torch.int32),
    ):
        batch = ids.shape[0]
        hidden = torch.randn(batch, hidden_size, dtype=torch.bfloat16)
        weights = torch.rand(batch, top_k, dtype=torch.float32)
        args = (hidden, weights, ids, experts, intermediate, threads)
        expected = _decode({}, monkeypatch, *args)
        actual = _decode(config, monkeypatch, *args)
        for want, got in zip(expected, actual):
            assert torch.equal(want, got)
