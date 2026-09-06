"""Native decode parity, including FP32 bits before BF16 rounding can hide errors."""

import importlib.util
from pathlib import Path

import pytest
import torch


def fixture_module(filename):
    spec = importlib.util.spec_from_file_location(filename, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extension():
    try:
        from freetoken.kernel import _cpu_moe
    except ImportError:
        pytest.skip("Linux CPU MoE extension is not built")
    assert hasattr(_cpu_moe, "run_decode_weight_reuse_parity"), "rebuild the CPU MoE extension"
    return _cpu_moe


def executor_for(hidden, intermediate, *, threads=1, activation="silu", apply_on_input=False):
    extension()
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    fixtures = fixture_module("test_cpu_moe_prefill_batch.py")
    cache = fixtures._make_nvfp4_cache(12, hidden, intermediate, seed=451)
    executor = CpuMoeExecutor(
        cache, top_k=4, activation=activation,
        apply_router_weight_on_input=apply_on_input, num_threads=threads,
        max_tokens=8, device=torch.device("cpu"), prefill_batch="off",
    )
    return executor, cache


def run_task(executor, x, weights, ids):
    io = executor._io_for(len(x))
    io["x"].copy_(x)
    io["ids"].copy_(ids)
    io["w"].copy_(weights)
    executor._ext.run_task(executor._task_for(0, len(x)))
    return io["y"].clone()


@pytest.mark.parametrize("hidden", [16, 48, 64, 80, 240, 256, 288, 640, 2560])
@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_fp32_bits_match_serial_decode_for_each_accumulator_tail(hidden, count):
    native = extension()
    executor, _ = executor_for(64, 64)
    if not executor._ext.decode_weight_reuse_available():
        pytest.skip("AVX-512 VNNI is unavailable")
    torch.manual_seed(552 + hidden + count)
    rows = 7
    packed = torch.randint(0, 256, (rows, hidden // 2), dtype=torch.uint8)
    scales = torch.randint(0, 127, (rows, hidden // 16), dtype=torch.uint8)
    globals_ = (torch.rand(rows) * 0.05).to(torch.float16)
    globals_[0] = 0
    acts = torch.randint(-127, 128, (count, hidden), dtype=torch.int8)
    act_scales = torch.rand(count, hidden // 16, dtype=torch.float32)
    act_scales[0, ::3] = 0
    grouped = torch.empty(rows, count, dtype=torch.float32)
    singles = torch.empty_like(grouped)
    native.run_decode_weight_reuse_parity(
        grouped.data_ptr(), singles.data_ptr(), packed.data_ptr(), scales.data_ptr(),
        globals_.data_ptr(), acts.data_ptr(), act_scales.data_ptr(), rows, count, hidden,
    )
    assert torch.isfinite(grouped).all()
    assert torch.equal(grouped.view(torch.int32), singles.view(torch.int32))


@pytest.mark.parametrize("hidden,intermediate,threads,activation,apply_on_input", [
    (80, 112, 1, "silu", False),
    (288, 336, 3, "gelu", True),
    (64, 64, 3, "swigluoai", True),
    (64, 64, 3, "clamped_silu", False),
    (2560, 640, 14, "silu", False),
])
def test_persistent_tasks_preserve_bits_across_repeated_sparse_and_empty_routes(
    hidden, intermediate, threads, activation, apply_on_input,
):
    executor, cache = executor_for(hidden, intermediate, threads=threads,
                                   activation=activation, apply_on_input=apply_on_input)
    if not executor._ext.decode_weight_reuse_available():
        pytest.skip("AVX-512 VNNI is unavailable")
    original_banks = {key: [bank.clone() for bank in banks] for key, banks in cache.bank_sources.items()}
    for turn, count in enumerate((1, 2, 3, 4, 5, 7, 4, 4)):
        x = torch.randn(count, hidden, dtype=torch.bfloat16)
        x[0].zero_()
        weights = torch.rand(count, 4, dtype=torch.float32)
        weights[:, 0] = 0
        ids = torch.randint(0, 12, (count, 4), dtype=torch.int32)
        ids[:, 1] = 3  # Recurrent expert with different token inputs.
        ids[-1, 2] = 3  # Duplicate routes must retain separate weights and reductions.
        if turn == 5:
            ids[0].fill_(-1)
            ids[-1, -1] = 12  # Out-of-range routes are absent, too.
        elif turn == 6:
            ids.fill_(-1)  # Reused output buffers must erase earlier partials.
        originals = (x.clone(), weights.clone(), ids.clone())
        outputs = {}
        for enabled in ((False, True) if turn % 2 == 0 else (True, False)):
            executor._ext.set_decode_weight_reuse(enabled)
            outputs[enabled] = run_task(executor, x, weights, ids)
        assert torch.isfinite(outputs[True]).all()
        assert torch.equal(outputs[False].view(torch.int16), outputs[True].view(torch.int16))
        if turn == 6:
            assert torch.count_nonzero(outputs[True]) == 0
        for original, current in zip(originals, (x, weights, ids), strict=True):
            assert torch.equal(original, current)
    for key, banks in cache.bank_sources.items():
        for original, current in zip(original_banks[key], banks, strict=True):
            assert torch.equal(original.view(torch.uint8), current.view(torch.uint8))


def test_disabling_vnni_keeps_the_existing_fallback(monkeypatch):
    monkeypatch.setenv("FREETOKEN_CPU_MOE_NO_VNNI", "1")
    executor, _ = executor_for(64, 64)
    assert not executor._ext.decode_weight_reuse_available()
    x = torch.randn(3, 64, dtype=torch.bfloat16)
    weights = torch.rand(3, 4)
    ids = torch.tensor([[0, 1, 2, 3]] * 3, dtype=torch.int32)
    outputs = []
    for enabled in (False, True):
        executor._ext.set_decode_weight_reuse(enabled)
        outputs.append(run_task(executor, x, weights, ids))
    assert torch.equal(outputs[0].view(torch.int16), outputs[1].view(torch.int16))
