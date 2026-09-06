"""Intermediate-state retention, slot isolation and diagnostic hook checks."""

import importlib.util
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest


SOURCE = Path(__file__).parents[1] / "bench/target_seed_checkpoint.py"
spec = importlib.util.spec_from_file_location("target_seed_checkpoint", SOURCE)
checkpoint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checkpoint)


def test_loading_checkpoint_module_does_not_import_torch():
    subprocess.run([sys.executable, "-c",
                    "import runpy,sys; runpy.run_path(sys.argv[1]); assert 'torch' not in sys.modules",
                    str(SOURCE)], check=True)


@pytest.fixture
def state():
    torch = pytest.importorskip("torch")
    slabs = {"conv": torch.full((2, 4, 3, 2), -1.0),
             "recurrent": torch.full((2, 4, 2, 2), -1.0),
             "slot/ple_conv": torch.full((2, 4, 3, 3), -1.0),
             "slot/ple_ngram_ctx": torch.full((1, 4, 3), -1, dtype=torch.int64),
             "qsa_pending": torch.full((4, 2, 8, 3), -1.0)}
    # Linear slot differs from request table index; both have live neighbours.
    views = {name: value[1] if name == "qsa_pending" else value[:, 2]
             for name, value in slabs.items()}
    views["slot/ple_ngram_ctx"].copy_(torch.tensor([[9, 10, 11]]))
    cp = checkpoint.SeedCheckpoint(
        views, gdn_sources={slabs["recurrent"][i].data_ptr(): i for i in range(2)},
        ple_layers={5: 0, 11: 1}, qsa_layers={3: 0, 7: 1})
    return SimpleNamespace(cp=cp, slabs=slabs, views=views)


def advance(state, *, omit_second_qsa=False):
    torch = pytest.importorskip("torch")
    cp = state.cp
    cp.begin()
    expected = {name: value.clone() for name, value in state.views.items()}
    for index in range(2):
        for step in range(2):
            for name, offset in (("conv", 10), ("recurrent", 20)):
                state.views[name][index].fill_(offset + index + 100 * step)
                expected[name][index].fill_(offset + index)
            cp.capture_gdn(state.slabs["recurrent"][index])
        for step in range(2):
            state.views["slot/ple_conv"][index].fill_(30 + index + 100 * step)
            expected["slot/ple_conv"][index].fill_(30 + index)
            cp.capture_ple((5, 11)[index])
        for step in range(2):
            state.views["qsa_pending"][index].fill_(40 + index + 100 * step)
            expected["qsa_pending"][index].fill_(40 + index)
            if not (omit_second_qsa and index == 1 and step == 1):
                cp.capture_qsa((3, 7)[index])
    cp.capture_ngram(SimpleNamespace(ngram_context=torch.tensor([[9, 10, 11]]),
                                      input_ids=torch.tensor([100, 999])))
    expected["slot/ple_ngram_ctx"].copy_(torch.tensor([[10, 11, 100]]))
    state.views["slot/ple_ngram_ctx"].copy_(torch.tensor([[11, 100, 999]]))
    cp.finish()
    return expected


def test_checkpoint_restores_seed_state_without_touching_other_request_slots(state):
    torch = pytest.importorskip("torch")
    expected = advance(state)
    for name in state.views:
        assert torch.equal(state.cp.saved[name], expected[name])
        assert not torch.equal(state.views[name], expected[name])
    state.cp.restore()
    for name, slab in state.slabs.items():
        assert torch.equal(state.views[name], expected[name])
        for slot in range(4):
            own_slot = 1 if name == "qsa_pending" else 2
            if slot != own_slot:
                other = slab[slot] if name == "qsa_pending" else slab[:, slot]
                assert bool((other == -1).all())


def test_missing_state_update_never_reuses_a_previous_checkpoint(state):
    advance(state)
    with pytest.raises(RuntimeError, match="both updates"):
        advance(state, omit_second_qsa=True)
    with pytest.raises(RuntimeError, match="not ready"):
        state.cp.restore()


def test_checkpoint_rejects_extra_updates_and_unknown_state(state):
    advance(state)
    with pytest.raises(RuntimeError, match="unexpected checkpoint"):
        state.cp.capture_ple(5)
    with pytest.raises(ValueError, match="exactly the supported"):
        checkpoint.SeedCheckpoint(dict(state.views, additional=state.views["conv"]),
                                  gdn_sources={}, ple_layers={}, qsa_layers={})


def test_checkpoint_requires_complete_layer_mapping(state):
    with pytest.raises(ValueError, match="layer mapping"):
        checkpoint.SeedCheckpoint(state.views, gdn_sources=state.cp.gdn_sources,
                                  ple_layers={5: 0}, qsa_layers=state.cp.qsa_layers)


def test_capture_context_clears_after_failure_and_rejects_nesting():
    first, second = object(), object()
    with pytest.raises(ValueError, match="test failure"):
        with checkpoint.capture_context(first):
            assert checkpoint._ACTIVE is first
            with pytest.raises(RuntimeError, match="nested"):
                with checkpoint.capture_context(second):
                    pass
            raise ValueError("test failure")
    assert checkpoint._ACTIVE is None


def test_hooks_preserve_outputs_and_only_capture_inside_explicit_context(monkeypatch):
    calls = []
    token_ids = SimpleNamespace(numel=lambda: 2)
    batch = SimpleNamespace(mtp_fused=True)
    output = object()
    fake_gdn = SimpleNamespace(gdn_decode_fla=lambda *a, **k: output)
    fake_ple = SimpleNamespace(PLELayer=type("PLELayer", (), {
        "_decode_conv": lambda *a: output}), commit_ngram_context=lambda *a: output)
    fake_qsa = type("QSASparseAttnBackend", (), {"_qsa_forward_one": lambda *a: output})

    def forward(*args):
        assert fake_gdn.gdn_decode_fla(state_source="state") is output
        assert fake_ple.PLELayer._decode_conv(SimpleNamespace(layer_id=5), 1, 2, 3) is output
        assert fake_qsa._qsa_forward_one(None, 1, 2, 3, 4, 7, batch) is output
        assert fake_ple.commit_ngram_context("meta", None) is output
        return output

    fake_model = SimpleNamespace(Qwen4ExpModel=type("Qwen4ExpModel", (), {"forward": forward}))
    package = ModuleType("freetoken.models.qwen4_exp")
    package.gdn, package.model, package.ple = fake_gdn, fake_model, fake_ple
    sparse = ModuleType("freetoken.attention.qsa_sparse")
    sparse.QSASparseAttnBackend = fake_qsa
    for name in ("freetoken", "freetoken.models", "freetoken.attention"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, sparse.__name__, sparse)
    checkpoint.install()
    network = fake_model.Qwen4ExpModel()
    assert network.forward(token_ids, batch) is output
    assert calls == []
    cp = SimpleNamespace(**{name: (lambda *a, name=name: calls.append((name, a)))
                            for name in ("begin", "finish", "capture_gdn", "capture_ple",
                                         "capture_qsa", "capture_ngram")})
    with checkpoint.capture_context(cp):
        assert network.forward(token_ids, batch) is output
    assert calls == [("begin", ()), ("capture_gdn", ("state",)), ("capture_ple", (5,)),
                     ("capture_qsa", (7,)), ("capture_ngram", ("meta",)), ("finish", ())]
    assert checkpoint._ACTIVE is None
