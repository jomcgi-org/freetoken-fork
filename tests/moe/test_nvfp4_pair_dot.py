"""Exact CPU dot parity across vector loops, tails, signs and scale encodings."""

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def probe():
    try:
        from freetoken.kernel import _cpu_moe
    except ImportError:
        pytest.skip("CPU extension is not built")
    fn = getattr(_cpu_moe, "nvfp4_pair_dot_probe", None)
    if fn is None:
        pytest.skip("CPU extension needs rebuilding for pair dot probe")

    def call(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except RuntimeError as exc:
            if "requires AVX-512 VNNI" in str(exc):
                pytest.skip(str(exc))
            raise
    return call


def inputs(hidden, rows=17, seed=4090):
    rng = torch.Generator().manual_seed(seed)
    packed = torch.randint(0, 256, (rows, hidden // 2), dtype=torch.uint8, generator=rng)
    # Cover all finite E4M3 encodings, including subnormals and signed zeros.
    scales = (torch.arange(rows * (hidden // 16)) % 256).to(torch.uint8)
    scales[scales == 127] = 126
    scales[scales == 255] = 254
    scales = scales.reshape(rows, hidden // 16)
    globals_ = (torch.rand(rows, generator=rng) * .04).half().float()
    acts = torch.randint(-127, 128, (2, hidden), dtype=torch.int8, generator=rng)
    act_scales = torch.rand(2, hidden // 16, generator=rng) * .03
    return packed, scales, globals_, acts, act_scales


@pytest.mark.parametrize("hidden", [16, 32, 48, 64, 80, 112, 128, 240, 256, 272, 320, 496, 640, 2560, 4352])
@pytest.mark.parametrize("kind", ["random", "zeros", "extremes"])
def test_pair_matches_two_ordinary_dots_bit_for_bit(probe, hidden, kind):
    values = inputs(hidden)
    packed, scales, globals_, acts, act_scales = values
    if kind == "zeros":
        acts[0].zero_()
        act_scales[1].zero_()
        globals_[::2] = -0.
    elif kind == "extremes":
        acts[0, ::2] = 127
        acts[0, 1::2] = -127
        acts[1].copy_(-acts[0])
        packed.flatten()[::2] = 0xF7
        packed.flatten()[1::2] = 0x80
        globals_[::2] *= -1
    result = probe(*values)
    assert torch.isfinite(result["singles"]).all()
    assert torch.equal(result["paired"].view(torch.int32), result["singles"].view(torch.int32))


@pytest.mark.parametrize("change", ["rank", "dtype", "rows", "inputs", "iterations", "contiguous"])
def test_probe_rejects_invalid_tensor_contracts(probe, change):
    values = list(inputs(256))
    kwargs = {}
    if change == "rank":
        values[0] = values[0].flatten()
    elif change == "dtype":
        values[0] = values[0].float()
    elif change == "rows":
        values[2] = values[2][:-1]
    elif change == "inputs":
        values[3] = values[3][:1]
    elif change == "iterations":
        kwargs["iterations"] = 0
    else:
        values[3] = values[3][:, ::2]
    with pytest.raises(RuntimeError, match="pair dot"):
        probe(*values, **kwargs)


def make_executor(hidden, intermediate, batch, activation="silu", apply_on_input=False, threads=3):
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    spec = importlib.util.spec_from_file_location(
        "pair_dot_fixtures", Path(__file__).with_name("test_cpu_moe_prefill_batch.py"))
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    cache = fixtures._make_nvfp4_cache(16, hidden, intermediate, seed=4090)
    return CpuMoeExecutor(cache, top_k=3, activation=activation,
                          apply_router_weight_on_input=apply_on_input,
                          num_threads=threads, max_tokens=batch, device=torch.device("cpu"),
                          swiglu_limit=1.3, prefill_batch="off")


def executor_inputs(hidden, batch, routes):
    rng = torch.Generator().manual_seed(4091)
    x = torch.randn(batch, hidden, dtype=torch.bfloat16, generator=rng)
    weights = torch.rand(batch, 3, generator=rng)
    if routes == "shared":
        ids = torch.tensor([[0, 1, 2]] * batch, dtype=torch.int32)
    elif routes == "disjoint":
        ids = torch.arange(batch * 3, dtype=torch.int32).reshape(batch, 3)
    else:
        ids = torch.tensor([[0, 1, -1], [0, 4, 4], [1, 2, 16], [0, 2, 3], [-1, -1, -1]],
                           dtype=torch.int32)[:batch].clone()
    return x, weights, ids


def check_executor(hidden, intermediate, batch, activation, apply_on_input, routes):
    executor = make_executor(hidden, intermediate, batch, activation, apply_on_input)
    if not hasattr(executor._ext, "set_nvfp4_pair_dot"):
        pytest.skip("CPU extension needs rebuilding for pair executor")
    if not executor._ext.set_nvfp4_pair_dot(True):
        pytest.skip("pair executor requires NVFP4 and AVX-512 VNNI")
    io = executor._io_for(batch)
    x, weights, ids = executor_inputs(hidden, batch, routes)
    io["x"].copy_(x)
    io["ids"].copy_(ids)
    io["w"].copy_(weights)
    task = executor._task_for(0, batch)
    executor._ext.run_task(task)
    paired = io["y"].clone()
    assert executor._ext.set_nvfp4_pair_dot(False)
    executor._ext.run_task(task)
    ordinary = io["y"].clone()
    assert torch.isfinite(ordinary).all()
    assert torch.equal(paired.view(torch.int16), ordinary.view(torch.int16))


@pytest.mark.parametrize("batch", [1, 2, 5])
@pytest.mark.parametrize("activation", ["silu", "swigluoai", "clamped_silu"])
@pytest.mark.parametrize("apply_on_input", [False, True])
@pytest.mark.parametrize("routes", ["shared", "disjoint", "mixed"])
def test_complete_expert_pair_schedule_is_exact(probe, batch, activation, apply_on_input, routes):
    # Partial output tiles and inner-loop tails exercise both pair and remainder paths.
    check_executor(272, 80, batch, activation, apply_on_input, routes)


@pytest.mark.parametrize("batch", [1, 2, 5])
@pytest.mark.parametrize("apply_on_input", [False, True])
def test_model_expert_dimensions_are_exact(probe, batch, apply_on_input):
    check_executor(2560, 640, batch, "silu", apply_on_input, "mixed")
