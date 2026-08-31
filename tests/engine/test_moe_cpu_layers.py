"""Resolver for the hybrid CPU/GPU MoE decode split (--moe-cpu-layers).

CPU-only: exercises _parse_cpu_layers_spec / _resolve_cpu_layers without a GPU.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from freetoken.engine.engine import _parse_cpu_layers_spec as parse
from freetoken.engine.engine import _parse_disk_layers_spec as parse_disk
from freetoken.engine.engine import _resolve_cpu_layers as resolve
from freetoken.engine.engine import _resolve_disk_layers as resolve_disk
from freetoken.engine.engine import _auto_cpu_layers as auto_layers
from freetoken.engine.engine import _validate_disk_prefill_task_size as validate_chunk

L = 40


def test_explicit_list():
    assert parse("3,7,11", L) == frozenset({3, 7, 11})
    assert parse("3, 7 ,11,", L) == frozenset({3, 7, 11})  # whitespace + trailing comma
    assert parse("5,5,5", L) == frozenset({5})  # dups collapse


def test_count_evenly_strided():
    assert parse("8", L) == frozenset({0, 5, 10, 15, 20, 25, 30, 35})
    assert parse("1", L) == frozenset({0})
    assert len(parse(str(L), L)) == L  # all layers
    assert parse("0", L) == frozenset()


def test_fraction():
    assert len(parse("0.5", L)) == L // 2
    assert len(parse("1.0", L)) == L
    assert parse("0.0", L) == frozenset()


def test_disk_layers_use_the_same_grammar():
    assert parse_disk("3,7,11", L) == frozenset({3, 7, 11})
    assert len(parse_disk("8", L)) == 8
    assert len(parse_disk("0.5", L)) == L // 2


def test_empty():
    assert parse("", L) == frozenset()
    assert parse("   ", L) == frozenset()


@pytest.mark.parametrize("spec", ["99", "40,1", "-1", "1.5"])
def test_out_of_range_raises(spec):
    with pytest.raises(ValueError):
        parse(spec, L)


def _cfg(backend, spec=None, disk=None):
    return SimpleNamespace(
        moe_backend=backend, moe_cpu_layers=spec, moe_disk_layers=disk,
    )


def test_resolve_backend_dispatch():
    # cpu backend -> every layer, ignoring any spec
    assert resolve(_cfg("cpu"), L) == frozenset(range(L))
    assert resolve(_cfg("cpu", "8"), L) == frozenset(range(L))
    # offload + spec -> parsed subset
    assert len(resolve(_cfg("offload", "8"), L)) == 8
    # offload, no spec -> none (plain offload)
    assert resolve(_cfg("offload", None), L) == frozenset()
    # non-offload backend ignores the spec (validation lives in _adjust_config)
    assert resolve(_cfg("fused", "8"), L) == frozenset()


def test_resolve_disk_layers_always_targets_cpu_capable_backends():
    assert resolve_disk(_cfg("offload", disk="3,7"), L) == frozenset({3, 7})
    assert resolve_disk(_cfg("hybrid", disk="2"), L) == frozenset({0, 20})
    assert resolve_disk(_cfg("cpu", disk="1.0"), L) == frozenset(range(L))
    assert resolve_disk(_cfg("fused", disk="3,7"), L) == frozenset()


def _write_ftw_index(path, num_layers):
    index = {
        "format": "freetoken_weight",
        "tensors": [
            {"kind": "experts_bank", "nbytes": 100, "name": f"bank-{i}"}
            for i in range(num_layers)
        ],
    }
    (path / "freetoken_weight.json").write_text(json.dumps(index))


def _auto_config(path, profile=None):
    return SimpleNamespace(
        model_path=str(path),
        model_config=SimpleNamespace(),
        moe_disk_layer_profile=str(profile) if profile is not None else None,
    )


def test_auto_budget_spills_ftw_head_and_tail_layers_to_disk(tmp_path, monkeypatch):
    _write_ftw_index(tmp_path, 4)
    monkeypatch.setenv("FREETOKEN_PIN_BUDGET_GB", str(201 / 2**30))
    monkeypatch.setattr(
        "freetoken.engine.engine._cpu_moe_executor_viable", lambda model_config: True,
    )
    assert auto_layers(_auto_config(tmp_path), 4) == frozenset({0, 3})


def test_auto_budget_uses_lowest_profile_scores_with_stable_ties(tmp_path, monkeypatch):
    _write_ftw_index(tmp_path, 6)
    profile = tmp_path / "traffic.json"
    profile.write_text(json.dumps({"0": 5, "1": 1, "2": 1, "3": 2, "4": 3, "5": 0}))
    monkeypatch.setenv("FREETOKEN_PIN_BUDGET_GB", str(401 / 2**30))
    monkeypatch.setattr(
        "freetoken.engine.engine._cpu_moe_executor_viable", lambda model_config: True,
    )
    logs = []
    monkeypatch.setattr("freetoken.engine.engine.logger.info_rank0", logs.append)

    assert auto_layers(_auto_config(tmp_path, profile), 6) == frozenset({1, 5})
    assert "layer scores {1: 1.0, 5: 0.0}" in logs[-1]
    assert "([1, 5])" in logs[-1]


@pytest.mark.parametrize(
    "contents",
    ["{not-json", json.dumps({"0": 1, "1": 2, "2": 3})],
    ids=["malformed", "incomplete"],
)
def test_bad_profile_warns_and_falls_back(tmp_path, monkeypatch, caplog, contents):
    import logging
    import freetoken.engine.engine as engine

    _write_ftw_index(tmp_path, 4)
    profile = tmp_path / "traffic.json"
    profile.write_text(contents)
    monkeypatch.setenv("FREETOKEN_PIN_BUDGET_GB", str(201 / 2**30))
    monkeypatch.setattr(engine, "_cpu_moe_executor_viable", lambda model_config: True)
    monkeypatch.setattr(engine.logger, "propagate", True)
    caplog.set_level(logging.WARNING, logger=engine.logger.name)

    assert auto_layers(_auto_config(tmp_path, profile), 4) == frozenset({0, 3})
    assert "falling back to head+tail DISK selection" in caplog.text


def test_ple_disk_zero_reservation_expands_expert_pin_budget(tmp_path, monkeypatch):
    _write_ftw_index(tmp_path, 4)
    monkeypatch.setenv("FREETOKEN_PIN_BUDGET_GB", str(201 / 2**30))
    monkeypatch.setattr(
        "freetoken.engine.engine._cpu_moe_executor_viable", lambda model_config: True,
    )
    config = _auto_config(tmp_path)

    assert auto_layers(config, 4, reserved=0) == frozenset({0, 3})
    assert auto_layers(config, 4, reserved=200) == frozenset(range(4))


@pytest.mark.parametrize("mode", ["cpu", "copy"])
def test_engine_config_accepts_disk_prefill_modes(mode):
    import torch

    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    config = EngineConfig(
        model_path="/tmp/model",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        moe_disk_prefill=mode,
    )
    assert config.moe_disk_prefill == mode


@pytest.mark.parametrize("mode", ["cpu", "gpufetch"])
def test_engine_config_accepts_disk_decode_modes(mode):
    import torch

    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    config = EngineConfig(
        model_path="/tmp/model",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        moe_disk_decode=mode,
    )
    assert config.moe_disk_decode == mode


def test_engine_config_defaults_disk_prefill_to_cpu():
    import torch

    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    config = EngineConfig(
        model_path="/tmp/model",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
    )
    assert config.moe_disk_prefill == "cpu"
    assert config.moe_disk_decode == "cpu"


def test_engine_config_rejects_invalid_disk_prefill_mode():
    import torch

    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    with pytest.raises(ValueError, match="--moe-disk-prefill.*cpu.*copy"):
        EngineConfig(
            model_path="/tmp/model",
            tp_info=DistributedInfo(0, 1),
            dtype=torch.bfloat16,
            moe_disk_prefill="gpu",
        )


def test_engine_config_rejects_invalid_disk_decode_mode():
    import torch

    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    with pytest.raises(ValueError, match="--moe-disk-decode.*cpu.*gpufetch"):
        EngineConfig(
            model_path="/tmp/model",
            tp_info=DistributedInfo(0, 1),
            dtype=torch.bfloat16,
            moe_disk_decode="eager",
        )


@pytest.mark.parametrize("pager", ["madvise", "uffd"])
def test_engine_config_accepts_disk_pagers(pager):
    import torch

    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    config = EngineConfig(
        model_path="/tmp/model",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        moe_disk_pager=pager,
        moe_pager_budget_gib=7.5,
    )
    assert config.moe_disk_pager == pager
    assert config.moe_pager_budget_gib == 7.5


def test_engine_config_defaults_to_madvise_disk_pager():
    import torch

    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    config = EngineConfig(
        model_path="/tmp/model",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
    )
    assert config.moe_disk_pager == "madvise"
    assert config.moe_pager_budget_gib == 40.0


def test_engine_config_rejects_invalid_uffd_settings():
    import torch

    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    base = dict(
        model_path="/tmp/model",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
    )
    with pytest.raises(ValueError, match="--moe-disk-pager"):
        EngineConfig(**base, moe_disk_pager="kernel")
    with pytest.raises(ValueError, match="--moe-pager-budget-gib"):
        EngineConfig(**base, moe_pager_budget_gib=0)


@pytest.mark.parametrize("backend", ["pinned", "cached", "disk", "hmm"])
def test_engine_config_accepts_ple_backends(backend):
    import torch

    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    config = EngineConfig(
        model_path="/tmp/model",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        ple_backend=backend,
    )
    assert config.ple_backend == backend


def test_engine_config_defaults_ple_backend_to_pinned():
    import torch

    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    config = EngineConfig(
        model_path="/tmp/model",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
    )
    assert config.ple_backend == "pinned"


def test_engine_config_rejects_invalid_ple_backend():
    import torch

    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    with pytest.raises(ValueError, match="--ple-backend.*pinned.*cached.*disk.*hmm"):
        EngineConfig(
            model_path="/tmp/model",
            tp_info=DistributedInfo(0, 1),
            dtype=torch.bfloat16,
            ple_backend="ram",
        )


@pytest.mark.parametrize("budget", [0, -1, float("inf"), float("nan")])
def test_engine_config_rejects_invalid_ple_cache_budget(budget):
    import torch

    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    with pytest.raises(ValueError, match="--ple-cache-gib.*finite positive"):
        EngineConfig(
            model_path="/tmp/model",
            tp_info=DistributedInfo(0, 1),
            dtype=torch.bfloat16,
            ple_cache_gib=budget,
        )


def _disk_ple_adjust_config(backend="disk"):
    import torch

    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    config = EngineConfig(
        model_path="/tmp/model",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        ple_backend=backend,
        attention_backend="triton",
        cuda_graph_bs=[1, 2, 4],
        cuda_graph_max_bs=4,
    )
    object.__setattr__(
        config,
        "model_config",
        SimpleNamespace(
            single_stream_only=False,
            dsv4_args=None,
            is_moe=False,
            expert_quant="none",
            has_swa_attention=False,
            has_linear_attention=False,
            qwen4_args=SimpleNamespace(ple_layer_ids=(2,)),
        ),
    )
    return config


def test_disk_ple_keeps_cuda_graph_config_enabled(monkeypatch):
    from freetoken.engine.engine import _adjust_config

    monkeypatch.delenv("FREETOKEN_PLE_DISK_NO_GRAPHS", raising=False)
    config = _disk_ple_adjust_config()
    _adjust_config(config)
    assert config.cuda_graph_bs == [1, 2, 4]
    assert config.cuda_graph_max_bs == 4


def test_hmm_ple_keeps_cuda_graph_config_enabled(monkeypatch):
    from freetoken.engine.engine import _adjust_config

    monkeypatch.setenv("FREETOKEN_PLE_DISK_NO_GRAPHS", "1")
    config = _disk_ple_adjust_config("hmm")
    _adjust_config(config)
    assert config.cuda_graph_bs == [1, 2, 4]
    assert config.cuda_graph_max_bs == 4


def test_cached_ple_keeps_cuda_graph_config_enabled(monkeypatch):
    from freetoken.engine.engine import _adjust_config

    monkeypatch.delenv("FREETOKEN_PLE_DISK_NO_GRAPHS", raising=False)
    config = _disk_ple_adjust_config("cached")
    _adjust_config(config)
    assert config.cuda_graph_bs == [1, 2, 4]
    assert config.cuda_graph_max_bs == 4


def test_disk_ple_no_graphs_env_restores_eager_fallback(monkeypatch):
    from freetoken.engine.engine import _adjust_config

    monkeypatch.setenv("FREETOKEN_PLE_DISK_NO_GRAPHS", "1")
    config = _disk_ple_adjust_config()
    _adjust_config(config)
    assert config.cuda_graph_bs == []
    assert config.cuda_graph_max_bs == 0


def test_disk_cpu_prefill_validates_scheduler_chunk_against_task_limit():
    from freetoken.moe.cpu_executor import CPU_MOE_MAX_TASK_TOKENS

    cache = SimpleNamespace(layer_residency=["pinned", "disk"])
    config = SimpleNamespace(moe_disk_prefill="cpu", max_extend_tokens=8192)
    validate_chunk(config, cache)

    for invalid in (0, CPU_MOE_MAX_TASK_TOKENS + 1):
        config.max_extend_tokens = invalid
        with pytest.raises(ValueError, match="max-prefill-length.*token-field range"):
            validate_chunk(config, cache)

    config.moe_disk_prefill = "copy"
    validate_chunk(config, cache)

    config.moe_disk_prefill = "cpu"
    validate_chunk(config, SimpleNamespace(layer_residency=["pinned"]))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
