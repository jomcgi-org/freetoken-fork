"""Preserve compute placement and arithmetic while changing GPU host backing."""

from types import SimpleNamespace

import pytest
import torch

from freetoken.distributed import DistributedInfo, set_tp_info, try_get_tp_info
from freetoken.engine.config import EngineConfig
from freetoken.engine.engine import _gpu_source_staging_layers
from freetoken.layers.moe import OffloadMoELayer
from freetoken.moe.offload_cache import OffloadMoeCache
from freetoken.server.args import parse_args


@pytest.fixture(autouse=True)
def _single_rank():
    if try_get_tp_info() is None:
        set_tp_info(0, 1)


def _sources(layers=3, experts=8, hidden=32, inter=32):
    shapes = {
        "gate_up_packed": (2 * inter, hidden // 2),
        "gate_up_scale": (2 * inter, hidden // 16),
        "gate_up_global": (2 * inter,),
        "down_packed": (hidden, inter // 2),
        "down_scale": (hidden, inter // 16),
        "down_global": (hidden,),
    }
    return {
        name: [torch.zeros((experts, *shape), dtype=(
            torch.float16 if name.endswith("global") else torch.uint8
        )) for _ in range(layers)]
        for name, shape in shapes.items()
    }


def _mixed_cache():
    cache = OffloadMoeCache(
        num_layers=3, num_experts=8, cache_size=24, device=torch.device("cpu"),
        quant_format="nvfp4", decode_target="cpu", prefill_overlap=False,
        moe_disk_prefill="staged", moe_disk_prefill_min_tokens=8,
        gpu_staging_layer_ids=frozenset({2}),
    )
    cache.cpu_layer_ids = frozenset({0, 1})
    cache.set_bank_sources(_sources(), layer_residency=["disk"] * 3,
                           hot_expert_ids={1: (2,)}, hot_expert_capacity={1: 2})
    return cache


def test_cli_defaults_to_full_pinned_gpu_sources_and_accepts_opt_in():
    base = ["--model", "/tmp/nonexistent-model", "--dtype", "bfloat16"]
    default, _ = parse_args(base)
    staged, _ = parse_args(base + ["--moe-gpu-source", "staged",
                                  "--moe-backend", "offload",
                                  "--moe-disk-prefill", "staged"])
    assert default.moe_gpu_source == "pinned"
    assert staged.moe_gpu_source == "staged"
    assert EngineConfig.__dataclass_fields__["moe_gpu_source"].default == "pinned"


@pytest.mark.parametrize("override", [
    {"moe_gpu_source": "pageable"}, {"moe_backend": "hybrid"},
    {"moe_backend": "cpu"}, {"moe_disk_prefill": "cpu"},
    {"moe_disk_decode": "gpufetch"}, {"moe_disk_pager": "uffd"},
])
def test_config_rejects_changes_to_the_required_execution_contract(override):
    fields = dict(model_path="/tmp/model", tp_info=DistributedInfo(0, 1),
                  dtype=torch.bfloat16, moe_backend="offload",
                  moe_gpu_source="staged", moe_disk_prefill="staged")
    with pytest.raises(ValueError):
        EngineConfig(**(fields | override))


def test_file_backing_uses_the_already_selected_gpu_partition():
    config = SimpleNamespace(moe_gpu_source="staged",
                             model_config=SimpleNamespace(expert_quant="nvfp4"))
    cpu = frozenset({0, 2})
    staged = _gpu_source_staging_layers(config, 4, cpu, bank_source="ftw", custom_cache=False)
    assert staged == frozenset({1, 3})
    assert staged.isdisjoint(cpu) and staged | cpu == frozenset(range(4))
    config.moe_gpu_source = "pinned"
    assert not _gpu_source_staging_layers(config, 4, cpu, bank_source="ftw", custom_cache=False)


@pytest.mark.parametrize("bank_source,custom,quant,cpu", [
    ("auto", False, "nvfp4", frozenset({0})),
    ("ftw", True, "nvfp4", frozenset({0})),
    ("ftw", False, "bf16", frozenset({0})),
    ("ftw", False, "nvfp4", frozenset({0, 1})),
])
def test_unsupported_backing_is_refused_before_bank_loading(bank_source, custom, quant, cpu):
    config = SimpleNamespace(moe_gpu_source="staged",
                             model_config=SimpleNamespace(expert_quant=quant))
    with pytest.raises(ValueError):
        _gpu_source_staging_layers(config, 2, cpu, bank_source=bank_source, custom_cache=custom)


@pytest.mark.parametrize("selected", [frozenset({0}), frozenset({3})])
def test_cache_refuses_staging_cpu_or_nonexistent_layers(selected):
    cache = _mixed_cache()
    cache.gpu_staging_layer_ids = selected
    with pytest.raises(ValueError):
        cache.set_bank_sources(_sources(), layer_residency=["disk"] * 3)


def test_mixed_decode_keeps_cpu_hot_and_gpu_paths(monkeypatch):
    if try_get_tp_info() is None:
        set_tp_info(0, 1)
    cache = _mixed_cache()
    assert cache.cpu_layer_ids == frozenset({0, 1})
    assert [cache.is_gpufetch_layer(i) for i in range(3)] == [False, False, True]
    assert cache.hot_expert_capacity == {1: 2}
    calls = []
    output = torch.ones(1, 32)
    cache.cpu_executor = SimpleNamespace(decode=lambda layer, *_: calls.append(("cpu", layer)) or output)
    monkeypatch.setattr(cache, "ensure_experts", lambda layer, _: calls.append(("gpu", layer)))
    monkeypatch.setattr(cache, "copy_missing", lambda: None)
    for layer_id in range(3):
        layer = OffloadMoELayer(layer_id=layer_id, num_experts=8, top_k=2,
                               hidden_size=32, intermediate_size=32)
        layer.offload_cache = cache
        monkeypatch.setattr(layer, "_decode_hot_split",
                            lambda *_: calls.append(("hot", layer_id)) or output)
        monkeypatch.setattr(layer, "_expert_gemm", lambda *_args, **_kw: output)
        result = layer._decode_routed(torch.ones(1, 32), torch.ones(1, 2),
                                      torch.tensor([[1, 3]], dtype=torch.int32))
        assert result is output
    assert calls == [("cpu", 0), ("hot", 1), ("gpu", 2)]


def test_gpu_only_sources_can_attach_the_native_copy_coordinator():
    cache = OffloadMoeCache(
        num_layers=1, num_experts=8, cache_size=24, device=torch.device("cpu"),
        quant_format="nvfp4", decode_target="gpu", moe_disk_prefill="staged",
        gpu_staging_layer_ids=frozenset({0}),
    )
    cache.set_bank_sources(_sources(layers=1), layer_residency=["disk"])
    executor = object()
    cache.set_cpu_executor(executor)
    assert cache.cpu_executor is executor
    assert cache.is_gpufetch_layer(0) and not cache.is_cpu_layer(0)


def test_short_chunks_keep_gpu_math_and_protect_the_following_overlap_buffer():
    cache = OffloadMoeCache(
        num_layers=4, num_experts=8, cache_size=24, device=torch.device("cpu"),
        quant_format="nvfp4", moe_disk_prefill="staged", moe_disk_prefill_min_tokens=8,
        gpu_staging_layer_ids=frozenset({0}),
    )
    cache.layer_residency = ["disk", "pinned", "disk", "pinned"]
    cache._unpinned_layers = frozenset({0, 2})
    cache._configure_prefill_overlap_layers()
    for tokens in (1, 7, 8, 16, 1):
        cache.begin_prefill(tokens)
        assert cache.disk_prefill_mode(0) == "staged"
        assert cache.disk_prefill_mode(2) == ("cpu" if tokens < 8 else "staged")
        assert cache._prefill_overlap_buffer_ids == (
            [-1, 1, -1, 0] if tokens < 8 else [-1, 1, -1, 1]
        )
        assert cache.prefill_path_counts() == ((2, 1, 1) if tokens < 8 else (2, 2, 0))


def test_session_advice_keeps_gpu_warming_for_staged_sources(monkeypatch):
    from freetoken.moe import offload_kernels
    from freetoken.moe.session_profile import SessionExpertProfile

    cache = _mixed_cache()
    cache.session_profile_enabled = True
    profile = SessionExpertProfile(((1,), (2,), (3,)), ((1.0,), (1.0,), (1.0,)))
    gpu, cpu = [], []
    monkeypatch.setattr(offload_kernels, "ensure_experts", lambda *_a, **_k: None)
    monkeypatch.setattr(offload_kernels, "ensure_experts_hot", lambda *_a, **_k: None)
    monkeypatch.setattr(cache, "copy_missing", lambda: gpu.append(cache._pending_src_layer))
    cache.cpu_executor = SimpleNamespace(_prefetch_selected=lambda layer, ids: cpu.append((layer, ids)))
    cache.admit_session_profile(41, profile)
    assert gpu == [1, 2]
    assert cpu == [(0, (1,))]


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_nvfp4_staged_gpu_layers_match_pinned_prefill_and_captured_decode(tmp_path):
    from freetoken.moe.cpu_executor import CpuMoeExecutor
    from freetoken.moe.host_banks import HostBank

    if try_get_tp_info() is None:
        set_tp_info(0, 1)
    torch.manual_seed(173)
    experts, hidden, inter, top_k, batch = 8, 256, 128, 2, 4
    pinned = _sources(hidden=hidden, inter=inter)
    mapped = {name: [] for name in pinned}
    for name, layers in pinned.items():
        for layer_id, value in enumerate(layers):
            if name.endswith("packed"):
                value.random_(0, 256)
            elif name.endswith("scale"):
                value.fill_(0x20)
            else:
                value.fill_(0.5)
                value[:, value.shape[1] // 2:] = 0.25
            path = tmp_path / f"{name}-{layer_id}.bin"
            path.write_bytes(value.view(torch.uint8).numpy().tobytes())
            bank = HostBank(value.shape, value.dtype, backing="file", file_path=str(path))
            mapped[name].append(bank.tensor)
            layers[layer_id] = value.pin_memory()

    rigs = []
    device = torch.device("cuda")
    for staged in (False, True):
        cache = OffloadMoeCache(
            num_layers=3, num_experts=experts, cache_size=experts + 2, device=device,
            quant_format="nvfp4", decode_target="cpu", prefill_overlap=False,
            moe_disk_prefill="staged", moe_disk_prefill_min_tokens=8,
            gpu_staging_layer_ids=frozenset({1, 2}) if staged else frozenset(),
        )
        cache.cpu_layer_ids = frozenset({0})
        sources = {name: [mapped[name][0], *(mapped[name][1:] if staged else values[1:])]
                   for name, values in pinned.items()}
        cache.set_bank_sources(sources, layer_residency=["disk"] * 3 if staged else ["disk", "pinned", "pinned"],
                               hot_expert_ids={0: (1, 2)}, hot_expert_capacity={0: 2})
        expert_bytes = sum(t[0][0].numel() * t[0].element_size() for t in sources.values())
        cache.configure_hot_adaptation(half_life_steps=2000, interval_steps=0,
                                       max_swap_bytes=2 * expert_bytes, expert_bytes=expert_bytes)
        executor = CpuMoeExecutor(cache, top_k=top_k, activation="silu",
                                  apply_router_weight_on_input=False, num_threads=2,
                                  max_tokens=batch, device=device, disk_lookahead=False,
                                  step_timing=False, prefill_coalesce="off", prefill_batch="off")
        cache.set_cpu_executor(executor)
        cache.init_disk_prefill_staging()
        cache.init_disk_gpufetch(executor, max_tokens=batch, top_k=top_k)
        assert set(executor._gpufetch_tasks) == ({1, 2} if staged else set())
        layers = []
        for layer_id in (1, 2):
            layer = OffloadMoELayer(layer_id=layer_id, num_experts=experts, top_k=top_k,
                                   hidden_size=hidden, intermediate_size=inter)
            layer.offload_cache = cache
            layers.append(layer)
        rigs.append((cache, executor, layers))

    try:
        # Below, at and above the ordinary CPU-prefill crossover, then back
        # below it. Both source modes must run the same GPU arithmetic.
        for tokens in (1, 7, 8, 16, 1):
            x = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16)
            ids = (torch.arange(tokens * top_k, device=device, dtype=torch.int32)
                   .reshape(tokens, top_k) * 3) % experts
            weights = torch.full((tokens, top_k), 0.5, device=device)
            outputs = []
            for cache, _executor, layers in rigs:
                cache.begin_prefill(tokens)
                outputs.append([layer._prefill_routed(x, weights, ids.clone()) for layer in layers])
            torch.cuda.synchronize()
            for direct, staged in zip(*outputs, strict=True):
                assert torch.isfinite(direct).all()
                torch.testing.assert_close(staged, direct, rtol=0, atol=0)

        x = torch.randn(batch, hidden, device=device, dtype=torch.bfloat16)
        ids = torch.arange(experts, device=device, dtype=torch.int32).reshape(batch, top_k)
        weights = torch.full((batch, top_k), 0.5, device=device)
        captured = []
        for cache, _executor, layers in rigs:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    for layer in layers:
                        layer._decode_routed(x, weights, ids.clone())
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                outputs = [layer._decode_routed(x, weights, ids.clone()) for layer in layers]
            captured.append((graph, outputs))
        executor = rigs[1][1]
        for _ in range(8):
            ids.add_(1).remainder_(experts)
            executor._ext.gpufetch_stats(True)
            for graph, _outputs in captured:
                graph.replay()
            torch.cuda.synchronize()
            for direct, staged in zip(captured[0][1], captured[1][1], strict=True):
                assert torch.isfinite(direct).all()
                torch.testing.assert_close(staged, direct, rtol=0, atol=0)
            fills, calls, _ns = executor._ext.gpufetch_stats(True)
            assert calls == 2 and fills == 2 * experts
            assert executor._ext.gpufetch_error_code() == 0
            assert rigs[1][0].cpu_layer_ids == frozenset({0})
            assert rigs[1][0].hot_expert_capacity == {0: 2}
    finally:
        torch.cuda.synchronize()
        for cache, _executor, _layers in rigs:
            cache.synchronize_disk_prefill_staging()
            cache.shutdown_hot_adaptation()
