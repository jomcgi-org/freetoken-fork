"""GPU-free coverage for expert-batched CPU MoE prefill.

The native parity test needs the Linux CPU extension but never needs CUDA. It is
skipped when the extension is not built, which is the expected state on macOS.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def _pack_nvfp4(codes: torch.Tensor) -> torch.Tensor:
    lo = codes[..., 0::2]
    hi = codes[..., 1::2]
    return (lo | (hi << 4)).contiguous().to(torch.uint8)


def _make_nvfp4_cache(experts: int, hidden: int, intermediate: int, seed: int = 0):
    torch.manual_seed(seed)

    def rows(output: int, input_size: int):
        codes = torch.randint(
            0, 16, (experts, output, input_size), dtype=torch.uint8
        )
        packed = _pack_nvfp4(codes)
        scale = (
            0.5 + torch.rand(experts, output, input_size // 16)
        ).to(torch.float8_e4m3fn).contiguous()
        global_scale = (
            0.02 + 0.01 * torch.rand(experts, output)
        ).to(torch.float16).contiguous()
        return packed, scale, global_scale

    gup, gus, gug = rows(2 * intermediate, hidden)
    dnp, dns, dng = rows(hidden, intermediate)
    return SimpleNamespace(
        quant_format="nvfp4",
        bank_sources={
            "gate_up_packed": [gup],
            "gate_up_scale": [gus],
            "gate_up_global": [gug],
            "down_packed": [dnp],
            "down_scale": [dns],
            "down_global": [dng],
        },
        num_layers=1,
        num_experts=experts,
    )


def test_grouping_preserves_every_route_and_router_weight():
    from freetoken.moe.cpu_executor import _group_prefill_routes

    ids = torch.tensor(
        [[2, 0, 3], [1, 3, 2], [0, 2, 1], [3, 1, 0]], dtype=torch.int32
    )
    weights = torch.tensor(
        [[0.2, 0.3, 0.5], [0.1, 0.6, 0.3],
         [0.7, 0.2, 0.1], [0.4, 0.35, 0.25]],
        dtype=torch.float32,
    )

    groups = _group_prefill_routes(ids, weights, num_experts=4)
    observed = sorted(
        (token, expert, weight)
        for expert, routes in groups.items()
        for token, weight in routes
    )
    expected = sorted(
        (token, int(ids[token, slot]), float(weights[token, slot]))
        for token in range(ids.shape[0])
        for slot in range(ids.shape[1])
    )

    assert observed == expected
    assert len(observed) == ids.shape[0] * ids.shape[1]


def test_buffer_bound_matches_native_geometry_formula():
    from freetoken.moe.cpu_executor import _prefill_batch_buffer_nbytes

    # qwen4_exp production geometry: H=2560, I=640, 2048 tokens, top_k=10.
    assert _prefill_batch_buffer_nbytes(2048, 10, 2560, 640) == 270_827_520


class _SetupExtension:
    def __init__(self, *, setup_result=True, setup_error=None):
        self.setup_result = setup_result
        self.setup_error = setup_error
        self.setup_calls = []

    def setup_prefill_batch(self, capacity):
        self.setup_calls.append(capacity)
        if self.setup_error is not None:
            raise self.setup_error
        return self.setup_result

    def run_prefill_batch_sync(self, *args):
        return [0, 0]

    def prefill_batch_buffer_bytes(self):
        return 1234


def _bare_executor(extension, *, requested=True):
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    executor = CpuMoeExecutor.__new__(CpuMoeExecutor)
    executor._ext = extension
    executor._prefill_batch_requested = requested
    executor._prefill_batch_enabled = False
    executor._prefill_batch_warned = False
    executor._prefill_batch_degrades = 0
    executor._prefill_batch_capacity = 2048
    executor._prefill_batch_buffer_bytes = 0
    executor.quant_format = "nvfp4"
    return executor


def test_flag_off_does_not_allocate_batch_buffers():
    extension = _SetupExtension()
    executor = _bare_executor(extension, requested=False)

    executor._configure_prefill_batch()

    assert extension.setup_calls == []
    assert executor._prefill_batch_enabled is False


@pytest.mark.parametrize(
    "extension",
    [
        SimpleNamespace(setup_prefill_batch=lambda capacity: True),
        _SetupExtension(setup_result=False),
        _SetupExtension(setup_error=MemoryError("synthetic allocation failure")),
    ],
)
def test_missing_kernel_or_setup_failure_degrades_to_serial(extension):
    executor = _bare_executor(extension)

    executor._configure_prefill_batch()

    assert executor._prefill_batch_enabled is False
    assert executor._prefill_batch_degrades == 1


def test_successful_setup_is_one_time_and_reports_native_bytes():
    extension = _SetupExtension()
    executor = _bare_executor(extension)

    executor._configure_prefill_batch()

    assert extension.setup_calls == [2048]
    assert executor._prefill_batch_enabled is True
    assert executor._prefill_batch_buffer_bytes == 1234


def test_native_dispatch_uses_vnni_when_isa_flags_expose_it():
    """The injected CPUID seam checks pointer selection without CUDA or VNNI hardware."""
    try:
        from freetoken.kernel import _cpu_moe
    except ImportError:
        pytest.skip("Linux CPU MoE extension is not built")

    probe = getattr(_cpu_moe, "prefill_batch_kernel_for_isa_flags", None)
    if probe is None:
        pytest.skip("CPU MoE extension needs rebuilding for dispatch probe")

    # AVX-512 VNNI is a distinct feature from AVX-VNNI. This is the node-4 case
    # that previously selected the scalar batch pointer despite serial using VNNI.
    assert (
        probe(has_avx512vnni=True, has_avxvnni=False)
        == "batch_nvfp4_i8_avx512vnni_rows"
    )
    assert (
        probe(has_avx512vnni=True, has_avxvnni=True)
        == "batch_nvfp4_i8_avx512vnni_rows"
    )
    assert (
        probe(has_avx512vnni=False, has_avxvnni=True)
        == "batch_nvfp4_i8_vnni_rows"
    )
    assert (
        probe(has_avx512vnni=False, has_avxvnni=False)
        == "batch_nvfp4_i8_scalar_rows"
    )


def test_native_weight_rows_match_single_row_kernel_exactly():
    try:
        from freetoken.kernel import _cpu_moe
    except ImportError:
        pytest.skip("Linux CPU MoE extension is not built")

    run = getattr(_cpu_moe, "run_prefill_batch_rows_parity", None)
    if run is None:
        pytest.skip("CPU MoE extension needs rebuilding for row-block parity")

    torch.manual_seed(47)
    rows, activation_rows, hidden = 7, 11, 64
    codes = torch.randint(0, 16, (rows, hidden), dtype=torch.uint8)
    packed = _pack_nvfp4(codes)
    scales = torch.randint(0, 127, (rows, hidden // 16), dtype=torch.uint8)
    globals_ = (0.02 + torch.rand(rows) * 0.01).to(torch.float16)
    acts = torch.randint(
        -127, 128, (activation_rows, hidden), dtype=torch.int8
    )
    act_scales = torch.rand(
        activation_rows, hidden // 16, dtype=torch.float32
    )
    blocked = torch.empty(rows, activation_rows, dtype=torch.float32)
    singles = torch.empty_like(blocked)

    run(
        blocked.data_ptr(), singles.data_ptr(), packed.data_ptr(),
        scales.data_ptr(), globals_.data_ptr(), acts.data_ptr(),
        act_scales.data_ptr(), rows, activation_rows, hidden,
    )

    torch.testing.assert_close(blocked, singles, rtol=0, atol=0)


@pytest.mark.parametrize("with_callback", [False, True])
def test_batch_run_failure_disables_it_and_retries_serial(with_callback):
    class RunExtension:
        def __init__(self):
            self.batch_calls = 0
            self.serial_calls = 0

        def run_prefill_batch_sync(self, *args):
            self.batch_calls += 1
            raise RuntimeError("synthetic kernel failure")

        def run_task_sync(self, *args):
            self.serial_calls += 1

    extension = RunExtension()
    executor = _bare_executor(extension)
    executor.H = 16
    executor.top_k = 2
    executor.device = torch.device("cpu")
    executor._gpu_prequant = False
    executor._prefill_io = None
    executor._prefill_capacity = 0
    executor._prefill_batch_enabled = True
    executor._prefill_batch_rows = 0
    executor._prefill_batch_gemms = 0

    hidden = torch.randn(3, 16, dtype=torch.bfloat16)
    ids = torch.tensor([[0, 1], [1, 2], [2, 0]], dtype=torch.int32)
    weights = torch.rand(3, 2, dtype=torch.float32)
    ready_calls = []
    output = executor.prefill(
        0, hidden, weights, ids,
        on_inputs_ready=(lambda: ready_calls.append(True)) if with_callback else None,
    )

    assert output.shape == hidden.shape
    assert extension.batch_calls == 1
    assert extension.serial_calls == 1
    assert executor._prefill_batch_enabled is False
    assert executor._prefill_batch_degrades == 1
    # A degraded CPU batch must not launch the independent GPU partial twice.
    assert ready_calls == ([True] if with_callback else [])


def test_native_batch_matches_serial_and_reuses_workspace():
    try:
        from freetoken.kernel import _cpu_moe
    except ImportError:
        pytest.skip("Linux CPU MoE extension is not built")
    required = (
        "setup_prefill_batch",
        "run_prefill_batch_sync",
        "prefill_batch_buffer_bytes",
    )
    if not all(hasattr(_cpu_moe.CpuMoeExecutor, name) for name in required):
        pytest.skip("CPU MoE extension needs rebuilding for batched prefill")

    from freetoken.moe.cpu_executor import (
        CpuMoeExecutor,
        _prefill_batch_buffer_nbytes,
    )

    torch.manual_seed(29)
    experts, hidden, intermediate, top_k, tokens = 8, 64, 64, 3, 12
    cache = _make_nvfp4_cache(experts, hidden, intermediate, seed=29)
    common = dict(
        top_k=top_k,
        activation="silu",
        apply_router_weight_on_input=False,
        num_threads=3,
        max_tokens=4,
        max_prefill_tokens=tokens,
        device=torch.device("cpu"),
    )
    serial = CpuMoeExecutor(cache, prefill_batch="off", **common)
    batched = CpuMoeExecutor(cache, prefill_batch="on", **common)
    if not batched._prefill_batch_enabled:
        pytest.skip("native batched NVFP4 kernel unavailable on this CPU build")

    x = torch.randn(tokens, hidden, dtype=torch.bfloat16)
    ids = torch.stack([torch.randperm(experts)[:top_k] for _ in range(tokens)]).to(
        torch.int32
    )
    weights = torch.rand(tokens, top_k, dtype=torch.float32)
    before = batched._ext.prefill_batch_buffer_bytes()
    assert before == _prefill_batch_buffer_nbytes(
        tokens, top_k, hidden, intermediate
    )

    expected = serial.prefill(0, x, weights, ids).float()
    actual = batched.prefill(0, x, weights, ids).float()
    again = batched.prefill(0, x, weights, ids).float()

    # W4A8 dot products accumulate in fp32. The activated intermediate and each
    # per-route down result are stored as bf16 before the stable fp32 scatter sum.
    # The tolerance covers that extra per-route bf16 store and reduction ordering.
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=6e-2)
    torch.testing.assert_close(again, actual, rtol=0, atol=0)
    assert batched._ext.prefill_batch_buffer_bytes() == before
    assert batched._prefill_batch_rows == 2 * tokens * top_k
    assert batched._prefill_batch_gemms == 4 * len(torch.unique(ids))


def test_server_cli_defaults_batch_on_and_accepts_off():
    from freetoken.engine.config import EngineConfig
    from freetoken.server.args import parse_args

    assert EngineConfig.__dataclass_fields__["moe_cpu_prefill_batch"].default == "on"
    args, _ = parse_args([
        "--model", "/tmp/nonexistent-model",
        "--dtype", "bfloat16",
        "--moe-cpu-prefill-batch", "off",
    ])
    assert args.moe_cpu_prefill_batch == "off"
