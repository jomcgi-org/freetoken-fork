"""Relocated checkpoint reads and restores must leave other requests untouched."""

import copy
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / "bench" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


seed = load("binding_seed", "target_seed_checkpoint.py")
compact = load("binding_compact", "target_compact_rollback.py")
base = load("binding_graph", "target-verify-cost.py")
Compact = compact.make_checkpoint_type(seed.SeedCheckpoint)


def slabs_on(torch, device="cpu"):
    return {"conv": torch.full((2, 4, 3), -1., device=device),
            "recurrent": torch.full((2, 4, 3), -2., device=device),
            "slot/ple_conv": torch.full((2, 4, 3), -3., device=device),
            "slot/ple_ngram_ctx": torch.full((1, 4, 2), -4, dtype=torch.int64, device=device),
            "qsa_pending": torch.full((4, 2, 8, 3), -5., device=device)}


def views_at(slabs, linear, request):
    return {name: slab[request] if name == "qsa_pending" else slab[:, linear]
            for name, slab in slabs.items()}


def update(q, k, v, a, b, *, A_log, dt_bias, state_source, indices, cu_seqlens, scale):
    delta = (q + 2 * k + 3 * v + 4 * a + 5 * b) * scale
    state_source.index_add_(0, indices, delta)


def make_checkpoint(torch, cls, width, device="cpu", recurrent_shape=None):
    slabs = slabs_on(torch, device)
    if recurrent_shape is not None:
        slabs["recurrent"] = torch.zeros((2, 4, *recurrent_shape), device=device)
    linear = torch.tensor([0], dtype=torch.int64, device=device)
    request = torch.tensor([1], dtype=torch.int64, device=device)
    bindings = seed.SlotStateBindings(slabs, linear, request)
    cp = cls(views_at(slabs, 0, 1), width=width,
             gdn_sources={slabs["recurrent"][i].data_ptr(): i for i in range(2)},
             ple_layers={5: 0, 11: 1}, qsa_layers={3: 0, 7: 1})
    cp.bind_state(bindings)
    return cp, slabs, linear, request


def run_target(torch, cp, slabs, linear, request, inputs, meta, kwargs, reference=None):
    cp.begin()
    for step in range(cp.width):
        for layer in range(2):
            slabs["conv"][layer].index_copy_(0, linear, inputs[layer][step][0])
            update(*inputs[layer][step], **kwargs[layer])
            cp.capture_gdn(slabs["recurrent"][layer], args=inputs[layer][step],
                           kwargs=kwargs[layer], update=update)
            slabs["slot/ple_conv"][layer].index_copy_(0, linear, inputs[layer][step][1])
            cp.capture_ple((5, 11)[layer])
            pending = inputs[layer][step][2].unsqueeze(1).expand(1, 8, 3)
            slabs["qsa_pending"][:, layer].index_copy_(0, request, pending)
            cp.capture_qsa((3, 7)[layer])
        context = torch.cat((meta.ngram_context, meta.input_ids[:step + 1].reshape(1, -1)), 1)[:, -2:]
        slabs["slot/ple_ngram_ctx"][0].index_copy_(0, linear, context)
        if reference is not None:
            reference.append({name: value.clone() for name, value in slabs.items()})
    cp.capture_ngram(meta)
    cp.finish()


def target_inputs(torch, cp, slabs, linear, device="cpu"):
    inputs = [[tuple(torch.full((1, 3), float(20 * layer + 5 * step + i), device=device)
                     for i in range(5)) for step in range(cp.width)] for layer in range(2)]
    kwargs = [dict(A_log=torch.zeros(3, device=device), dt_bias=torch.zeros(3, device=device),
                   state_source=slabs["recurrent"][layer], indices=linear,
                   cu_seqlens=torch.tensor([0, 1], device=device), scale=0.5) for layer in range(2)]
    meta = SimpleNamespace(input_ids=torch.arange(cp.width, device=device),
                           ngram_context=torch.tensor([[70, 80]], device=device))
    return inputs, meta, kwargs


@pytest.mark.parametrize("cls", [seed.SeedCheckpoint, Compact])
@pytest.mark.parametrize("width", [2, 3, 5])
def test_every_prefix_follows_current_request_in_both_slot_spaces(cls, width):
    torch = pytest.importorskip("torch")
    cp, slabs, linear, request = make_checkpoint(torch, cls, width)
    inputs, meta, kwargs = target_inputs(torch, cp, slabs, linear)
    for generation, (linear_slot, request_slot) in enumerate(((0, 1), (3, 2), (1, 0))):
        linear.fill_(linear_slot)
        request.fill_(request_slot)
        meta.input_ids.add_(10)
        for values in inputs:
            for row in values:
                for value in row:
                    value.add_(generation + 1)
        before = {name: value.clone() for name, value in slabs.items()}
        expected = []
        run_target(torch, cp, slabs, linear, request, inputs, meta, kwargs, expected)
        if cls is Compact:
            for values in inputs:
                for row in values:
                    for value in row:
                        value.fill_(999)
        for length in range(1, width):
            if cls is Compact:
                cp.replay_eager(length)
            else:
                cp.restore(length)
            for name, slab in slabs.items():
                assert torch.equal(slab, expected[length - 1][name]), (generation, length, name)
                for slot in range(4):
                    own = request_slot if name == "qsa_pending" else linear_slot
                    if slot != own:
                        actual = slab[slot] if name == "qsa_pending" else slab[:, slot]
                        prior = before[name][slot] if name == "qsa_pending" else before[name][:, slot]
                        assert torch.equal(actual, prior), (name, slot)


def test_binding_rejects_changed_geometry_and_rebinding_after_use():
    torch = pytest.importorskip("torch")
    cp, slabs, linear, request = make_checkpoint(torch, seed.SeedCheckpoint, 3)
    with pytest.raises(RuntimeError, match="once"):
        cp.bind_state(cp.state_bindings)
    bad = dict(slabs, conv=slabs["conv"][:, :, :2])
    with pytest.raises(ValueError, match="geometry"):
        seed.SlotStateBindings(bad, linear, request).validate_views(cp.views)
    with pytest.raises(ValueError, match="int64"):
        seed.SlotStateBindings(slabs, linear.to(torch.int32), request)
    for lin, req in ((-1, 0), (4, 0), (0, -1), (0, 4)):
        with pytest.raises(ValueError, match="outside"):
            cp.state_bindings.validate_request(SimpleNamespace(linear_slot_idx=lin, table_idx=req))
    inputs, meta, kwargs = target_inputs(torch, cp, slabs, linear)
    run_target(torch, cp, slabs, linear, request, inputs, meta, kwargs)
    with pytest.raises(RuntimeError, match="once"):
        cp.bind_state(cp.state_bindings)


def make_batch(torch, *, width, linear, request, position):
    req = SimpleNamespace(linear_slot_idx=linear, table_idx=request,
                          cached_len=position, input_ids=torch.arange(position + width))
    md = [SimpleNamespace(block_table=torch.tensor([[4, 1, 7]], dtype=torch.int32),
                           seq_lens=torch.tensor([position + i + 1], dtype=torch.int32),
                           ring_slots=torch.tensor([request], dtype=torch.int32),
                           token_to_req=torch.tensor([0], dtype=torch.int32),
                           cu_seqlens=torch.tensor([0, 1], dtype=torch.int32)) for i in range(width)]
    return SimpleNamespace(reqs=[req], padded_reqs=[req], phase="decode", mtp_fused=True,
                            input_ids=torch.arange(width, dtype=torch.int32),
                            positions=torch.arange(position, position + width, dtype=torch.int32),
                            out_loc=torch.arange(width, dtype=torch.int32) + 100,
                            linear_table_idx=torch.tensor([linear], dtype=torch.int32),
                            active_table_idx=torch.full((width,), request, dtype=torch.int32),
                            mtp_qsa_metadata=md, mtp_original_cached_len=position,
                            mtp_original_device_len=position + 1)


def staging_fixture(torch):
    cp, slabs, linear, request = make_checkpoint(torch, Compact, 5)
    graph = base.FusedGraph.__new__(base.FusedGraph)
    graph.batch = make_batch(torch, width=5, linear=0, request=1, position=61)
    graph.request_key = graph._request_key(graph.batch)
    graph.width, graph.state_checkpoint = 5, cp
    graph.linear_state_index, graph.request_state_index = linear, request
    return graph


def test_graph_stages_new_history_slots_positions_and_noncontiguous_pages():
    torch = pytest.importorskip("torch")
    graph = staging_fixture(torch)
    incoming = make_batch(torch, width=5, linear=3, request=2, position=127)
    incoming.input_ids.add_(12)
    incoming.out_loc.copy_(torch.tensor([191, 64, 65, 66, 67], dtype=torch.int32))
    for md in incoming.mtp_qsa_metadata:
        md.block_table.copy_(torch.tensor([[7, 2, 1]], dtype=torch.int32))
    original = graph.batch.reqs[0]
    buffers = [(graph.batch, name) for name in ("input_ids", "positions", "out_loc",
                                               "linear_table_idx", "active_table_idx")]
    buffers += [(md, name) for md in graph.batch.mtp_qsa_metadata
                for name in ("block_table", "seq_lens", "ring_slots", "token_to_req", "cu_seqlens")]
    pointers = [getattr(obj, name).data_ptr() for obj, name in buffers]
    graph._stage(incoming)
    assert pointers == [getattr(obj, name).data_ptr() for obj, name in buffers]
    assert graph.batch.reqs[0] is incoming.reqs[0] and graph.batch.reqs[0] is not original
    assert graph.linear_state_index.tolist() == [3]
    assert graph.request_state_index.tolist() == [2]
    contexts = torch.empty((5, 2), dtype=torch.int64)
    base.host_contexts(graph.batch.reqs[0].input_ids, graph.batch.reqs[0].cached_len,
                       graph.batch.input_ids.long(), contexts, -1)
    assert contexts.tolist() == [[125, 126], [126, 12], [12, 13], [13, 14], [14, 15]]
    for name in ("input_ids", "positions", "out_loc", "linear_table_idx", "active_table_idx"):
        assert torch.equal(getattr(graph.batch, name), getattr(incoming, name))
    for dest, source in zip(graph.batch.mtp_qsa_metadata, incoming.mtp_qsa_metadata):
        for name in ("block_table", "seq_lens", "ring_slots", "token_to_req", "cu_seqlens"):
            assert torch.equal(getattr(dest, name), getattr(source, name))


@pytest.mark.parametrize("failure", ["row_missing", "shape", "dtype", "slot", "phase", "fixed",
                                    "lazy_batch", "lazy_request"])
def test_invalid_replay_does_not_partially_update_captured_buffers(failure):
    torch = pytest.importorskip("torch")
    graph = staging_fixture(torch)
    incoming = make_batch(torch, width=5, linear=3, request=2, position=127)
    original = copy.deepcopy(graph.batch)
    if failure == "row_missing":
        incoming.mtp_qsa_metadata.pop()
    elif failure == "shape":
        incoming.mtp_qsa_metadata[-1].block_table = torch.zeros(1, 4, dtype=torch.int32)
    elif failure == "dtype":
        incoming.positions = incoming.positions.long()
    elif failure == "slot":
        incoming.reqs[0].linear_slot_idx = 4
    elif failure == "phase":
        incoming.phase = "prefill"
    elif failure == "lazy_batch":
        incoming.lazy_restore_pending = True
    elif failure == "lazy_request":
        incoming.reqs[0].lazy_kv_restore = SimpleNamespace(complete=False)
    else:
        graph.state_checkpoint = None
    with pytest.raises((RuntimeError, ValueError)):
        graph._stage(incoming)
    for name in ("input_ids", "positions", "out_loc", "linear_table_idx", "active_table_idx"):
        assert torch.equal(getattr(graph.batch, name), getattr(original, name))
    for actual, expected in zip(graph.batch.mtp_qsa_metadata, original.mtp_qsa_metadata):
        assert torch.equal(actual.block_table, expected.block_table)
    assert graph.linear_state_index.tolist() == [0]
    assert graph.request_state_index.tolist() == [1]


@pytest.mark.skipif(os.environ.get("FREETOKEN_RELOCATABLE_STATE_CUDA_TEST") != "1",
                    reason="requires explicit exclusive GPU validation")
def test_cuda_target_and_restore_graphs_follow_new_request_slots():
    import torch

    assert torch.cuda.is_available(), "exclusive GPU check requires CUDA"
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        cp, slabs, linear, request = make_checkpoint(torch, Compact, 5, "cuda")
        inputs, meta, kwargs = target_inputs(torch, cp, slabs, linear, "cuda")
        run_target(torch, cp, slabs, linear, request, inputs, meta, kwargs)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            run_target(torch, cp, slabs, linear, request, inputs, meta, kwargs)
        cp.capture_restore_graphs(stream)
        for linear_slot, request_slot in ((0, 1), (3, 2), (1, 0)):
            linear.fill_(linear_slot)
            request.fill_(request_slot)
            before = {name: value.clone() for name, value in slabs.items()}
            expected = []
            run_target(torch, cp, slabs, linear, request, inputs, meta, kwargs, expected)
            for name, value in slabs.items():
                value.copy_(before[name])
            graph.replay()
            stream.synchronize()
            for name, value in slabs.items():
                assert torch.equal(value, expected[-1][name])
            for length in range(1, cp.width):
                cp.restore(length)
                stream.synchronize()
                for name, value in slabs.items():
                    assert torch.equal(value, expected[length - 1][name]), (linear_slot, length, name)
        graph.reset()
        cp.close(stream)


@pytest.mark.skipif(os.environ.get("FREETOKEN_RELOCATABLE_STATE_CUDA_TEST") != "1",
                    reason="requires explicit exclusive GPU validation")
def test_cuda_real_gdn_rollback_uses_relocated_slot_and_latest_inputs():
    import torch
    from freetoken.models.qwen3_5_moe.gdn_kernels import gdn_decode_fla

    assert torch.cuda.is_available(), "exclusive GPU check requires CUDA"
    torch.manual_seed(29)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        heads, dim, width = 4, 128, 5
        cp, slabs, linear, request = make_checkpoint(
            torch, Compact, width, "cuda", recurrent_shape=(heads, dim, dim))
        recurrent = slabs["recurrent"]
        # The model consumes an int32 slot buffer; checkpoint indexing is int64.
        model_index = linear.to(torch.int32)
        inputs = [[(torch.randn(1, 1, heads, dim, device="cuda", dtype=torch.bfloat16),
                    torch.randn(1, 1, heads, dim, device="cuda", dtype=torch.bfloat16),
                    torch.randn(1, 1, heads, dim, device="cuda", dtype=torch.bfloat16),
                    torch.randn(1, heads, device="cuda"), torch.randn(1, heads, device="cuda"))
                   for _ in range(width)] for _ in range(2)]
        kwargs = [dict(A_log=torch.zeros(heads, device="cuda"), dt_bias=torch.zeros(heads, device="cuda"),
                       state_source=recurrent[i], indices=model_index,
                       cu_seqlens=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
                       scale=dim ** -0.5) for i in range(2)]
        meta = SimpleNamespace(input_ids=torch.arange(width, device="cuda"),
                               ngram_context=torch.tensor([[7, 8]], device="cuda"))

        def target():
            cp.begin()
            for layer in range(2):
                for args in inputs[layer]:
                    gdn_decode_fla(*args, **kwargs[layer])
                    cp.capture_gdn(recurrent[layer], args=args, kwargs=kwargs[layer], update=gdn_decode_fla)
                for _ in range(width):
                    cp.capture_ple((5, 11)[layer])
                    cp.capture_qsa((3, 7)[layer])
            cp.capture_ngram(meta)
            cp.finish()

        target()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            target()
        cp.capture_restore_graphs(stream)
        for generation, (linear_slot, request_slot) in enumerate(((0, 1), (3, 2), (1, 0)), 1):
            linear.fill_(linear_slot)
            model_index.copy_(linear)
            request.fill_(request_slot)
            recurrent.fill_(-0.25)
            recurrent[:, linear_slot].fill_(generation * 0.125)
            initial = recurrent.clone()
            for layer in inputs:
                for args in layer:
                    for value in args:
                        value.normal_()
            expected = {}
            for length in range(1, width + 1):
                recurrent.copy_(initial)
                for layer in range(2):
                    for step in range(length):
                        gdn_decode_fla(*inputs[layer][step], **kwargs[layer])
                expected[length] = recurrent.clone()
            recurrent.copy_(initial)
            graph.replay()
            stream.synchronize()
            assert torch.equal(recurrent, expected[width])
            for layer in inputs:
                for args in layer:
                    for value in args:
                        value.fill_(42)
            for length in range(1, width):
                cp.restore(length)
                stream.synchronize()
                assert torch.equal(recurrent, expected[length]), (linear_slot, length)
        graph.reset()
        cp.close(stream)
