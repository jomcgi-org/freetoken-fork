"""Compact rollback owns its inputs and reproduces ordinary recurrent updates."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


def load(name, filename):
    path = Path(__file__).parents[1] / "bench" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


seed = load("compact_seed", "target_seed_checkpoint.py")
compact = load("compact_rollback", "target_compact_rollback.py")
multi = load("compact_cost", "target_multitoken.py")
Compact = compact.make_checkpoint_type(seed.SeedCheckpoint)


def test_compact_module_is_lazy():
    path = Path(__file__).parents[1] / "bench/target_compact_rollback.py"
    subprocess.run([sys.executable, "-c",
                    "import runpy,sys; runpy.run_path(sys.argv[1]); assert 'torch' not in sys.modules",
                    str(path)], check=True)


def test_compact_summary_requires_all_rejection_outcomes():
    rows = [dict(case="2", mode=name, warmup=False, checks_passed=True, wall_s=1.)
            for name in multi.modes(5)]
    assert not multi.summarize(rows, 5, True)["2"]["checks_passed"]
    rows += [dict(rows[0], mode=name) for name in multi.modes(5, True) if name.startswith("compact_")]
    assert multi.summarize(rows, 5, True)["2"]["checks_passed"]
    rows.append(dict(rows[-1], warmup=True, checks_passed=False))
    assert not multi.summarize(rows, 5, True)["2"]["checks_passed"]


def test_reconstructed_recurrence_is_part_of_numerical_qualification():
    checks = {"compact": {"tokens_equal": True, "logits": {"exact": True},
                           "restored_prefix_state": {"3": {"recurrent": {"exact": False}}}}}
    assert multi.numerical_failures(checks) == ["/compact/restored_prefix_state/3/recurrent"]


def update(q, k, v, a, b, *, A_log, dt_bias, state_source, indices, cu_seqlens, scale):
    delta = (q + 2 * k + 3 * v + 4 * a + 5 * b).sum(-1, keepdim=True) * scale
    state_source.index_add_(0, indices, delta.expand(1, state_source.shape[-1]))


@pytest.fixture
def state():
    torch = pytest.importorskip("torch")
    source = torch.full((3, 4096), -3.)
    views = {"conv": torch.zeros(1, 2), "recurrent": source[2:3],
             "slot/ple_conv": torch.zeros(1, 2), "slot/ple_ngram_ctx": torch.zeros(1, 2, dtype=torch.int64),
             "qsa_pending": torch.zeros(1, 8, 2)}
    mapping = dict(gdn_sources={source.data_ptr(): 0}, ple_layers={5: 0}, qsa_layers={7: 0}, width=5)
    kwargs = dict(A_log=torch.zeros(2), dt_bias=torch.zeros(2), state_source=source,
                  indices=torch.tensor([2]), cu_seqlens=torch.tensor([0, 1]), scale=0.5)
    return SimpleNamespace(cp=Compact(views, **mapping), views=views, source=source,
                           kwargs=kwargs, mapping=mapping)


def generation(state, value):
    torch = pytest.importorskip("torch")
    cp, views = state.cp, state.views
    views["recurrent"].fill_(value)
    cp.begin()
    reference, inputs = [], []
    context = torch.tensor([[7, 8]])
    ids = torch.arange(10 + value, 10 + value + cp.width)
    for step in range(cp.width):
        args = tuple(torch.full((1, 2), float(value + step + i)) for i in range(5))
        inputs.append(args)
        views["conv"][0].copy_(args[0][0])
        update(*args, **state.kwargs)
        cp.capture_gdn(state.source, args=args, kwargs=state.kwargs, update=update)
        views["slot/ple_conv"].fill_(step + value)
        cp.capture_ple(5)
        views["qsa_pending"].fill_(step + 2 * value)
        cp.capture_qsa(7)
        views["slot/ple_ngram_ctx"].copy_(torch.cat((context, ids[:step + 1].reshape(1, -1)), 1)[:, -2:])
        reference.append({name: tensor.clone() for name, tensor in views.items()})
    cp.capture_ngram(SimpleNamespace(input_ids=ids, ngram_context=context))
    cp.finish()
    return reference, inputs


def test_compact_replays_every_prefix_and_refreshes_between_forwards(state):
    torch = pytest.importorskip("torch")
    for value in (2, 19):
        reference, inputs = generation(state, value)
        # Simulate reusable activation storage being overwritten after the target.
        for args in inputs:
            for tensor in args:
                tensor.fill_(999)
        for prefix_len in range(1, state.cp.width):
            state.cp.replay_eager(prefix_len)
            for name, tensor in state.views.items():
                assert torch.equal(tensor, reference[prefix_len - 1][name])
            assert bool((state.source[:2] == -3).all())
    full = seed.SeedCheckpoint(state.views, **state.mapping)
    assert all("recurrent" not in prefix for prefix in state.cp.prefixes)
    assert state.cp.owned_tensor_bytes() < full.owned_tensor_bytes()


@pytest.mark.parametrize("change", ["geometry", "binding", "function"])
def test_changed_update_contract_invalidates_checkpoint(state, change):
    torch = pytest.importorskip("torch")
    generation(state, 2)
    state.cp.begin()
    args = tuple(torch.zeros(1, 2) for _ in range(5))
    kwargs = dict(state.kwargs)
    fn = update
    if change == "geometry":
        args = (torch.zeros(1, 3),) + args[1:]
    elif change == "binding":
        kwargs["indices"] = kwargs["indices"].clone()
    else:
        fn = lambda *a, **kw: update(*a, **kw)
    with pytest.raises(RuntimeError, match="changed"):
        state.cp.capture_gdn(state.source, args=args, kwargs=kwargs, update=fn)
    with pytest.raises(RuntimeError, match="not ready"):
        state.cp.replay_eager(1)


def test_measured_restore_never_falls_back_to_eager(state):
    generation(state, 2)
    with pytest.raises(RuntimeError, match="unavailable"):
        state.cp.restore(1)
    for prefix_len in (0, state.cp.width):
        with pytest.raises(ValueError, match="retained range"):
            state.cp.replay_eager(prefix_len)


@pytest.mark.skipif(os.environ.get("FREETOKEN_COMPACT_ROLLBACK_CUDA_TEST") != "1",
                    reason="requires explicit exclusive GPU validation")
def test_cuda_rollback_graphs_use_current_captured_gdn_inputs():
    import torch
    from freetoken.models.qwen3_5_moe.gdn_kernels import gdn_decode_fla

    assert torch.cuda.is_available(), "exclusive GPU check requires CUDA"
    torch.manual_seed(19)
    width, layers, heads, dim = 5, 2, 4, 128
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        recurrent = torch.zeros(layers, 3, heads, dim, dim, device="cuda", dtype=torch.float32)
        views = {"recurrent": recurrent[:, 2], "conv": torch.zeros(layers, heads, device="cuda"),
                 "slot/ple_conv": torch.zeros(1, heads, device="cuda"),
                 "qsa_pending": torch.zeros(1, heads, device="cuda"),
                 "slot/ple_ngram_ctx": torch.zeros(1, 2, device="cuda", dtype=torch.int64)}
        cp = Compact(views, gdn_sources={recurrent[i].data_ptr(): i for i in range(layers)},
                     ple_layers={5: 0}, qsa_layers={7: 0}, width=width)
        inputs = [[(torch.randn(1, 1, heads, dim, device="cuda", dtype=torch.bfloat16),
                    torch.randn(1, 1, heads, dim, device="cuda", dtype=torch.bfloat16),
                    torch.randn(1, 1, heads, dim, device="cuda", dtype=torch.bfloat16),
                    torch.randn(1, heads, device="cuda"), torch.randn(1, heads, device="cuda"))
                   for _ in range(width)] for _ in range(layers)]
        kwargs = [dict(A_log=torch.zeros(heads, device="cuda"), dt_bias=torch.zeros(heads, device="cuda"),
                       state_source=recurrent[i], indices=torch.tensor([2], device="cuda", dtype=torch.int32),
                       cu_seqlens=torch.tensor([0, 1], device="cuda", dtype=torch.int32), scale=dim ** -0.5)
                  for i in range(layers)]
        meta = SimpleNamespace(input_ids=torch.arange(width, device="cuda"),
                               ngram_context=torch.tensor([[7, 8]], device="cuda"))

        def target():
            cp.begin()
            for index in range(layers):
                for step, args in enumerate(inputs[index]):
                    views["conv"][index].copy_(args[3][0])
                    gdn_decode_fla(*args, **kwargs[index])
                    cp.capture_gdn(recurrent[index], args=args, kwargs=kwargs[index], update=gdn_decode_fla)
            for step in range(width):
                views["slot/ple_conv"][0].copy_(inputs[0][step][3][0])
                cp.capture_ple(5)
                views["qsa_pending"][0].copy_(inputs[0][step][4][0])
                cp.capture_qsa(7)
            cp.capture_ngram(meta)
            cp.finish()

        target()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            target()
        cp.capture_restore_graphs(stream)
        for generation_id in (1, 2):
            for layer_inputs in inputs:
                for args in layer_inputs:
                    for value in args:
                        value.normal_()
            expected = {}
            for length in range(1, width + 1):
                recurrent.zero_()
                recurrent[:, 2].fill_(generation_id * 0.125)
                for index in range(layers):
                    for step in range(length):
                        gdn_decode_fla(*inputs[index][step], **kwargs[index])
                expected[length] = recurrent.clone()
            recurrent.zero_()
            recurrent[:, 2].fill_(generation_id * 0.125)
            graph.replay()
            stream.synchronize()
            assert torch.equal(recurrent, expected[width])
            for layer_inputs in inputs:
                for args in layer_inputs:
                    for value in args:
                        value.fill_(42)
            for length in range(1, width):
                cp.restore(length)
                stream.synchronize()
                assert torch.equal(recurrent, expected[length])
        graph.reset()
        cp.close(stream)
