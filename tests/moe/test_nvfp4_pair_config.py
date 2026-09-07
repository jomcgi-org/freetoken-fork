"""The paired kernel is a startup opt-in, including with older native libraries."""

from types import SimpleNamespace

import pytest
import torch

from freetoken.moe.cpu_executor import CpuMoeExecutor


@pytest.mark.parametrize("has_setter", [False, True])
def test_disabled_pair_never_requires_or_calls_the_native_setter(has_setter):
    def unexpected(enabled):
        pytest.fail("default-off startup called the native pair setter")

    executor = CpuMoeExecutor.__new__(CpuMoeExecutor)
    executor._ext = SimpleNamespace(**({"set_nvfp4_pair_dot": unexpected} if has_setter else {}))
    executor._configure_nvfp4_pair(False)
    assert executor._nvfp4_pair_enabled is False


def test_enabled_pair_is_forwarded_to_the_fresh_native_executor():
    calls = []

    def enable(value):
        calls.append(value)
        return True

    executor = CpuMoeExecutor.__new__(CpuMoeExecutor)
    executor._ext = SimpleNamespace(set_nvfp4_pair_dot=enable)
    executor._configure_nvfp4_pair(True)
    assert calls == [True]
    assert executor._nvfp4_pair_enabled is True


@pytest.mark.parametrize("support", ["missing", "unsupported"])
def test_enabled_pair_rejects_missing_native_support(support):
    executor = CpuMoeExecutor.__new__(CpuMoeExecutor)
    executor._ext = SimpleNamespace()
    if support == "unsupported":
        executor._ext.set_nvfp4_pair_dot = lambda enabled: False
    with pytest.raises(RuntimeError, match="rebuilding" if support == "missing" else "AVX-512 VNNI"):
        executor._configure_nvfp4_pair(True)
    assert executor._nvfp4_pair_enabled is False


@pytest.mark.parametrize("mode,fmt", [("auto", "nvfp4"), ("on", "bf16"), ("on", "mxfp4_triton")])
def test_bad_pair_configuration_rejects_before_bank_or_native_allocation(mode, fmt):
    # Only the format is supplied, so accessing bank geometry would fail this test.
    cache = SimpleNamespace(quant_format=fmt)
    with pytest.raises(ValueError, match="nvfp4.pair"):
        CpuMoeExecutor(cache, top_k=2, activation="silu", apply_router_weight_on_input=False,
                       num_threads=1, max_tokens=1, device=torch.device("cpu"),
                       moe_cpu_nvfp4_pair=mode)


def test_engine_config_pair_default_and_validation():
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    options = dict(model_path="unused", tp_info=DistributedInfo(rank=0, size=1),
                   dtype=torch.bfloat16, max_running_req=1)
    assert EngineConfig(**options).moe_cpu_nvfp4_pair == "off"
    assert EngineConfig(**options, moe_cpu_nvfp4_pair="on").moe_cpu_nvfp4_pair == "on"
    with pytest.raises(ValueError, match="moe-cpu-nvfp4-pair"):
        EngineConfig(**options, moe_cpu_nvfp4_pair="auto")


@pytest.mark.parametrize("mode", [None, "off", "on"])
def test_server_cli_pair_default_and_explicit_modes(mode):
    from freetoken.server.args import parse_args

    argv = ["--model", "/tmp/nonexistent-model", "--dtype", "bfloat16"]
    if mode is not None:
        argv += ["--moe-cpu-nvfp4-pair", mode]
    args, _ = parse_args(argv)
    assert args.moe_cpu_nvfp4_pair == (mode or "off")
