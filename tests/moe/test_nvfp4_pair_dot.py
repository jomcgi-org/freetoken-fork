"""Exact CPU dot parity across vector loops, tails, signs and scale encodings."""

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
