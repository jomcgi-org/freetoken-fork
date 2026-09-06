"""Wider diagnostic addressing, prefix restoration and qualification checks."""

from dataclasses import make_dataclass
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / "bench" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


multi = load("target_multitoken", "target_multitoken.py")
base = load("target_verify_wide", "target-verify-cost.py")
seed = load("target_seed_wide", "target_seed_checkpoint.py")


@pytest.mark.parametrize("width", [2, 3, 5])
def test_explicit_width_configuration(monkeypatch, width):
    monkeypatch.setenv(base.WIDTH_ENV, str(width))
    assert base.verification_width() == width


@pytest.mark.parametrize("width", ["0", "1", "4", "6", "invalid"])
def test_unsupported_width_is_rejected(monkeypatch, width):
    monkeypatch.setenv(base.WIDTH_ENV, width)
    with pytest.raises(ValueError):
        base.verification_width()


def test_four_windows_and_final_target_row_must_fit_one_page():
    assert multi.eligible_window(58, 64, 3, 6)
    assert not multi.eligible_window(59, 64, 3, 6)
    assert not multi.eligible_window(58, 64, 3, 5)
    assert multi.eligible_window(56, 64, 5, 8)
    assert not multi.eligible_window(57, 64, 5, 8)


def test_summary_requires_every_partial_acceptance_and_keeps_warmup_failures():
    rows = [dict(case="2", mode=name, warmup=False, checks_passed=True, wall_s=i + 1.)
            for i, name in enumerate(multi.modes(3))]
    result = multi.summarize(rows, 3)["2"]
    assert result["checks_passed"] and not result["model_wall_qualified"]
    assert result["component_cost_over_one"]["reject_1"] == 5
    assert not multi.summarize(rows[:-1], 3)["2"]["checks_passed"]
    rows.append(dict(rows[-1], warmup=True, checks_passed=False))
    failed = multi.summarize(rows, 3)["2"]
    assert not failed["checks_passed"] and "component_cost_over_one" not in failed


def test_pair_trial_order_compares_neighbors_and_reverses_execution_order():
    even = multi.trial_order(5, True, True, 0)
    odd = multi.trial_order(5, True, True, 1)
    assert even[:4] == ("graph_one", "pair_graph_one", "graph_all", "pair_graph_all")
    assert odd == tuple(reversed(even))
    assert set(even) == set(multi.modes(5, True, True))
    assert len(even) == len(set(even))
    for ordinary, paired in zip(even[::2], even[1::2]):
        assert paired == "pair_" + ordinary


@pytest.mark.parametrize("failure", [None, "missing", "dispatch", "warmup"])
def test_pair_summary_requires_both_dispatches_and_retains_failures(failure):
    rows = [dict(case="2", mode=name, warmup=False, checks_passed=True,
                 cpu_pair_enabled=name.startswith("pair_"),
                 wall_s=3. if name.startswith("pair_") else 4.)
            for name in multi.modes(5, True, True)]
    if failure == "missing":
        rows.pop()
    elif failure == "dispatch":
        rows[-1]["cpu_pair_enabled"] = False
    elif failure == "warmup":
        rows.append(dict(rows[-1], warmup=True, checks_passed=False))
    result = multi.summarize(rows, 5, True, True)["2"]
    assert result["checks_passed"] is (failure is None)
    assert not result["model_wall_qualified"]
    if failure is None:
        assert set(result["cpu_pair_reduction_percent"].values()) == {25.}
    else:
        assert "cpu_pair_reduction_percent" not in result


@pytest.mark.parametrize("elapsed", [0., -1., float("nan"), float("inf")])
def test_invalid_component_time_cannot_qualify(elapsed):
    rows = [dict(case="2", mode=name, warmup=False, checks_passed=True, wall_s=1.)
            for name in multi.modes(5)]
    rows[0]["wall_s"] = elapsed
    result = multi.summarize(rows, 5)["2"]
    assert not result["checks_passed"]
    assert "component_cost_over_one" not in result


@pytest.mark.parametrize("supported", [True, False])
def test_cpu_pair_switch_waits_for_inflight_graph_and_rejects_unsupported_native(supported):
    events = []

    def setter(value):
        events.append(("setter", value))
        return supported

    engine = SimpleNamespace(stream=SimpleNamespace(synchronize=lambda: events.append("sync")),
                             cpu_moe_executor=SimpleNamespace(_ext=SimpleNamespace(set_nvfp4_pair_dot=setter)))
    if supported:
        multi.set_cpu_pair(engine, True)
    else:
        with pytest.raises(RuntimeError, match="AVX-512 NVFP4"):
            multi.set_cpu_pair(engine, True)
    assert events == ["sync", ("setter", True)]


def test_missing_pair_native_fails_before_measurement():
    engine = SimpleNamespace(cpu_moe_executor=SimpleNamespace(_ext=SimpleNamespace()))
    with pytest.raises(RuntimeError, match="native setter"):
        multi.set_cpu_pair(engine, True)


def test_pair_prefix_state_failure_is_numerically_load_bearing():
    checks = {"cpu_pair": {"compact": {"tokens_equal": True, "logits": {"exact": True},
               "restored_prefix_state": {"3": {"recurrent": {"exact": False}}}}}}
    assert multi.numerical_failures(checks) == ["/cpu_pair/compact/restored_prefix_state/3/recurrent"]


def test_fused_configuration_keeps_one_request_with_multiple_positions():
    tensor = lambda n: SimpleNamespace(numel=lambda: n)
    req = SimpleNamespace(cached_len=0, device_len=0)
    batch = SimpleNamespace(reqs=[req], mtp_original_cached_len=42, mtp_original_device_len=43)
    ids, positions, locations = tensor(3), tensor(3), tensor(3)
    multi.configure_fused(batch, ids, positions, locations, 3)
    assert (req.cached_len, req.device_len, batch.phase, batch.mtp_fused) == (42, 45, "decode", True)
    assert batch.input_ids is ids and batch.positions is positions and batch.out_loc is locations
    with pytest.raises(ValueError, match="configured verification width"):
        multi.configure_fused(batch, ids, positions, tensor(2), 3)


def test_wider_host_contexts_use_only_preceding_tokens():
    torch = pytest.importorskip("torch")
    contexts = torch.empty((5, 3), dtype=torch.int64)
    base.host_contexts(torch.tensor([10]), 1, torch.tensor([20, 21, 22, 23, 24]), contexts, -1)
    assert contexts.tolist() == [[-1, -1, 10], [-1, 10, 20], [10, 20, 21],
                                 [20, 21, 22], [21, 22, 23]]


@pytest.mark.parametrize("width", [3, 5])
def test_every_rejection_prefix_restores_its_own_state(width):
    torch = pytest.importorskip("torch")
    source = torch.zeros(1, 3, 2, 2)
    views = {"conv": torch.zeros(1, 2, 2), "recurrent": source[:, 2],
             "slot/ple_conv": torch.zeros(1, 2, 2), "slot/ple_ngram_ctx": torch.zeros(1, 3, dtype=torch.int64),
             "qsa_pending": torch.zeros(1, 8, 2)}
    cp = seed.SeedCheckpoint(views, gdn_sources={source[0].data_ptr(): 0},
                             ple_layers={5: 0}, qsa_layers={7: 0}, width=width)
    cp.begin()
    for step in range(width):
        for name in ("conv", "recurrent", "slot/ple_conv", "qsa_pending"):
            views[name].fill_(step + 10)
        cp.capture_gdn(source[0])
        cp.capture_ple(5)
        cp.capture_qsa(7)
    ids = torch.arange(20, 20 + width)
    cp.capture_ngram(SimpleNamespace(input_ids=ids, ngram_context=torch.tensor([[1, 2, 3]])))
    cp.finish()
    for length in range(1, width):
        cp.restore(length)
        for name in ("conv", "recurrent", "slot/ple_conv", "qsa_pending"):
            assert bool((views[name] == length + 9).all())
        assert views["slot/ple_ngram_ctx"].tolist() == [([1, 2, 3] + ids[:length].tolist())[-3:]]
        assert bool((source[:, :2] == 0).all())
    for bad in (0, width):
        with pytest.raises(ValueError, match="retained range"):
            cp.restore(bad)


@pytest.fixture
def installed(monkeypatch):
    torch = pytest.importorskip("torch")
    batch = SimpleNamespace(mtp_fused=True)
    calls = []

    def forward(layer, hidden):
        assert not batch.mtp_fused
        calls.append(hidden.shape[0])
        if getattr(layer, "fail", False):
            raise RuntimeError("primitive failed")
        return hidden + 2

    class Backend:
        def __init__(self):
            self.device = "cpu"
            self.snapshots = []
            self.rows = []

        def prepare_metadata(self, batch):
            return "ordinary"

        def _snapshot_decode(self, md, batch):
            md.seq_lens = md.kv_len_cpu.clone()
            self.snapshots.append(md)

        def _qsa_forward_one(self, q, k, v, index, layer_id, one):
            self.rows.append((one.positions.tolist(), one.out_loc.tolist(),
                              one.attn_metadata.seq_lens.tolist(), index.k.tolist()))
            return q + 100

    core = ModuleType("freetoken.core")
    core.get_global_ctx = lambda: SimpleNamespace(batch=batch)
    gdn = ModuleType("freetoken.models.qwen4_exp.gdn")
    gdn.Qwen4ExpGatedDeltaNet = type("GDN", (), {"forward": forward})
    sparse = ModuleType("freetoken.attention.qsa_sparse")
    sparse.QSASparseAttnBackend, sparse.QSASparseMetadata = Backend, SimpleNamespace
    for name in ("freetoken", "freetoken.models", "freetoken.models.qwen4_exp", "freetoken.attention"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    for module in (core, gdn, sparse):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    original_tensor = torch.tensor

    def tensor(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return original_tensor(*args, **kwargs)

    monkeypatch.setattr(torch, "tensor", tensor)
    multi.install(3)
    return SimpleNamespace(batch=batch, calls=calls, gdn=gdn.Qwen4ExpGatedDeltaNet, backend=Backend)


def test_wider_gdn_keeps_width_one_calls_and_restores_classification_on_failure(installed):
    torch = pytest.importorskip("torch")
    layer = installed.gdn()
    x = torch.arange(6).reshape(3, 2)
    assert torch.equal(layer.forward(x), x + 2)
    assert installed.calls == [1, 1, 1] and installed.batch.mtp_fused
    layer.fail = True
    with pytest.raises(RuntimeError, match="primitive failed"):
        layer.forward(x)
    assert installed.batch.mtp_fused


def test_qsa_positions_keep_independent_lengths_and_use_matching_rows(installed):
    torch = pytest.importorskip("torch")
    batch, backend = installed.batch, installed.backend()
    batch.reqs = batch.padded_reqs = [object()]
    batch.input_ids, batch.positions = torch.tensor([1, 2, 3]), torch.tensor([62, 63, 64])
    batch.out_loc, batch.mtp_original_device_len = torch.tensor([190, 191, 256]), 63
    backend.prepare_metadata(batch)
    assert [md.seq_lens.item() for md in batch.mtp_qsa_metadata] == [63, 64, 65]
    assert len({md.seq_lens.data_ptr() for md in batch.mtp_qsa_metadata}) == 3
    q = torch.arange(3).reshape(3, 1)
    Index = make_dataclass("Index", ["q", "k"])
    index = Index(q=q + 30, k=q + 40)
    result = backend._qsa_forward_mtp_k1(q, q + 10, q + 20, index, 7, batch)
    assert result.tolist() == [[100], [101], [102]]
    assert backend.rows == [([62], [190], [63], [[40]]), ([63], [191], [64], [[41]]),
                            ([64], [256], [65], [[42]])]
    batch.mtp_fused = False
    assert backend.prepare_metadata(batch) == "ordinary"
