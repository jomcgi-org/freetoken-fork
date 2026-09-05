"""Verify copied CPU inputs, true GPU/CPU overlap, and native NVFP4 parity."""

import pytest
import torch

from freetoken.moe.cpu_executor import CpuMoeExecutor


def bare_executor(device):
    executor = object.__new__(CpuMoeExecutor)
    executor.H = 64
    executor.top_k = 2
    executor.device = torch.device(device)
    executor._gpu_prequant = False
    executor._prefill_io = None
    executor._prefill_capacity = 0
    executor._prefill_batch_enabled = False
    return executor


def test_ready_callback_observes_copied_inputs_before_cpu_compute():
    executor = bare_executor("cpu")
    calls = []
    hidden = torch.randn(3, 64, dtype=torch.bfloat16)
    original = hidden.clone()
    ids = torch.tensor([[0, -1], [2, 3], [-1, 1]], dtype=torch.int32)
    weights = torch.rand(3, 2)

    class Extension:
        def run_task_sync(self, *args):
            calls.append("compute")
            executor._prefill_io["y"].copy_(executor._prefill_io["x"])

    executor._ext = Extension()

    def ready():
        calls.append("ready")
        assert torch.equal(executor._prefill_io["x"], original)
        assert torch.equal(executor._prefill_io["ids"], ids)
        assert torch.equal(executor._prefill_io["w"], weights)
        hidden.zero_()

    output = executor.prefill(0, hidden, weights, ids, on_inputs_ready=ready)
    assert calls == ["ready", "compute"]
    assert torch.equal(output, original)


def test_ready_callback_failure_does_not_enter_cpu_fallback():
    executor = bare_executor("cpu")
    executor._prefill_batch_enabled = True

    class Extension:
        def run_task_sync(self, *args):
            pytest.fail("CPU compute must not start after the GPU callback fails")

    executor._ext = Extension()

    def fail():
        raise torch.OutOfMemoryError("synthetic GPU callback OOM")

    with pytest.raises(torch.OutOfMemoryError, match="callback OOM"):
        executor.prefill(
            0, torch.zeros(3, 64, dtype=torch.bfloat16), torch.ones(3, 2),
            torch.zeros(3, 2, dtype=torch.int32), on_inputs_ready=fail,
        )
    assert executor._prefill_batch_enabled


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cpu_compute_starts_while_gpu_partial_is_in_flight():
    executor = bare_executor("cuda")
    done = torch.cuda.Event()
    calls = []

    class Extension:
        def run_task_sync(self, *args):
            assert not done.query(), "a blocking operation waited for the GPU partial"
            calls.append("cpu overlaps gpu")
            executor._prefill_io["y"].copy_(executor._prefill_io["x"])

    executor._ext = Extension()

    def launch():
        # Diagnostic-only delay makes accidental synchronization observable.
        torch.cuda._sleep(200_000_000)
        done.record()

    hidden = torch.randn(3, 64, dtype=torch.bfloat16, device="cuda")
    output = executor.prefill(
        0, hidden, torch.ones(3, 2, device="cuda"),
        torch.zeros(3, 2, dtype=torch.int32, device="cuda"), on_inputs_ready=launch,
    )
    assert calls == ["cpu overlaps gpu"]
    assert done.query()
    assert torch.equal(output, hidden)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("tokens,hidden,intermediate", [
    (8, 128, 128), (65, 128, 128), (129, 128, 128), (64, 2560, 640),
])
def test_native_nvfp4_split_is_bitwise_equal_across_schedules(monkeypatch, tokens, hidden, intermediate):
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.layers import moe
    from freetoken.moe.offload_cache import OffloadMoeCache

    if try_get_tp_info() is None:
        set_tp_info(0, 1)
    experts, top_k = 8, 4
    rng = torch.Generator().manual_seed(4090 + tokens)
    sources = {}
    for name, shape, dtype in (
        ("gate_up_packed", (experts, 2 * intermediate, hidden // 2), torch.uint8),
        ("gate_up_scale", (experts, 2 * intermediate, hidden // 16), torch.float8_e4m3fn),
        ("gate_up_global", (experts, 2 * intermediate), torch.float16),
        ("down_packed", (experts, hidden, intermediate // 2), torch.uint8),
        ("down_scale", (experts, hidden, intermediate // 16), torch.float8_e4m3fn),
        ("down_global", (experts, hidden), torch.float16),
    ):
        sources[name] = [
            torch.randint(0, 256, shape, dtype=dtype, generator=rng)
            if dtype == torch.uint8 else (0.025 + 0.025 * torch.rand(shape, generator=rng)).to(dtype)
        ]
    cache = OffloadMoeCache(
        num_layers=1, num_experts=experts, cache_size=experts + 4,
        device=torch.device("cuda"), quant_format="nvfp4", decode_target="cpu",
    )
    cache.cpu_layer_ids = frozenset({0})
    cache.set_bank_sources(sources, layer_residency=["disk"], hot_expert_ids={0: (0, 2, 4, 6)})
    expert_bytes = sum(rows[0][0].numel() * rows[0].element_size() for rows in sources.values())
    cache.configure_hot_adaptation(
        half_life_steps=2000, interval_steps=0,
        max_swap_bytes=expert_bytes, expert_bytes=expert_bytes,
    )
    executor = CpuMoeExecutor(
        cache, top_k=top_k, activation="silu", apply_router_weight_on_input=False,
        num_threads=3, max_tokens=1, max_prefill_tokens=tokens, device=torch.device("cuda"),
    )
    assert executor._prefill_batch_enabled, "NVFP4 native batch kernel is required"
    cache.set_cpu_executor(executor)
    layer = moe.OffloadMoELayer(
        layer_id=0, num_experts=experts, top_k=top_k,
        hidden_size=hidden, intermediate_size=intermediate,
    )
    layer.offload_cache = cache
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16, generator=rng).cuda()
    # Include repeated experts, all-cold token rows, and every HOT row.
    ids = torch.randint(0, experts, (tokens, top_k), generator=rng, dtype=torch.int32).cuda()
    ids[0] = 1
    ids[1] = torch.tensor([0, 2, 4, 6], device="cuda", dtype=torch.int32)
    weights = torch.rand(tokens, top_k, generator=rng).cuda()
    monkeypatch.setattr(moe, "_PREFILL_HOT_OVERLAP", False)
    expected = layer._prefill_hot_split(cache, x.clone(), weights, ids.clone())
    monkeypatch.setattr(moe, "_PREFILL_HOT_OVERLAP", True)
    actual = layer._prefill_hot_split(cache, x.clone(), weights, ids.clone())
    again = layer._prefill_hot_split(cache, x.clone(), weights, ids.clone())
    torch.cuda.synchronize()
    assert torch.isfinite(expected).all()
    assert torch.equal(actual, expected)
    assert torch.equal(again, expected)
