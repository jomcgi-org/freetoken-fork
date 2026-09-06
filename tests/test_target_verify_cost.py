"""Focused diagnostic qualification and rollback-boundary checks."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


SOURCE = Path(__file__).parents[1] / "bench/target-verify-cost.py"
spec = importlib.util.spec_from_file_location("target_verify_cost", SOURCE)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.mark.parametrize("position,expected", [(2, True), (6, True), (58, True),
                                               (62, False), (63, False), (64, False),
                                               (66, True)])
def test_windows_cross_compression_boundaries_without_allocating_a_new_page(position, expected):
    assert probe.eligible_position(position, page_size=64, ratio=4, remaining=5) == expected


@pytest.mark.parametrize("kwargs", [dict(page_size=63, ratio=4, remaining=5),
                                    dict(page_size=64, ratio=1, remaining=5),
                                    dict(page_size=64, ratio=4, remaining=4)])
def test_ineligible_geometry_or_output_limit(kwargs):
    assert not probe.eligible_position(2, **kwargs)


def records():
    costs = dict(graph_one=1.0, graph_two=2.0, snapshot=0.1, accept=1.4, reject=2.5)
    return [dict(case="2", mode=mode, wall_s=cost, warmup=False, checks_passed=True)
            for mode, cost in costs.items()]


def test_break_even_uses_rejection_and_does_not_claim_model_throughput():
    summary = probe.summarize(records())["2"]
    assert summary["checks_passed"] and summary["complete"]
    assert summary["break_even_acceptance_excluding_proposer"] == pytest.approx(1.5 / 2.1)
    assert not summary["model_wall_qualified"]


@pytest.mark.parametrize("failure", ["missing", "measured", "warmup"])
def test_incomplete_or_failed_work_suppresses_break_even(failure):
    rows = records()
    if failure == "missing":
        rows.pop()
    elif failure == "measured":
        rows[0]["checks_passed"] = False
    else:
        rows.append(dict(rows[0], warmup=True, checks_passed=False))
    result = probe.summarize(rows)["2"]
    assert not result["checks_passed"]
    assert "break_even_acceptance_excluding_proposer" not in result


def test_graph_mode_requires_both_graph_outcomes():
    rows = records()
    assert not probe.summarize(rows, graph_enabled=True)["2"]["complete"]
    rows += [dict(rows[0], mode=mode) for mode in probe.GRAPH_MODES]
    assert probe.summarize(rows, graph_enabled=True)["2"]["complete"]
    rows[-1]["checks_passed"] = False
    assert not probe.summarize(rows, graph_enabled=True)["2"]["checks_passed"]


def test_report_is_private_and_replaced_atomically(tmp_path):
    probe.save(tmp_path, {"completed": False})
    probe.save(tmp_path, {"completed": True})
    path = tmp_path / (str(os.getpid()) + ".json")
    assert json.loads(path.read_text()) == {"completed": True}
    assert path.stat().st_mode & 0o777 == 0o600
    assert list(tmp_path.iterdir()) == [path]


def test_import_hook_is_explicit_and_does_not_import_torch(tmp_path):
    script = """
import os, runpy, sys
module = runpy.run_path(sys.argv[1])
assert 'torch' not in sys.modules
os.environ.pop(module['OUTPUT_ENV'], None)
try:
    module['install_import_hook']()
except RuntimeError:
    pass
else:
    raise AssertionError('missing opt-in accepted')
os.environ[module['OUTPUT_ENV']] = sys.argv[2]
module['install_import_hook']()
assert 'torch' not in sys.modules
assert sys.meta_path[0].find_spec('json') is None
"""
    subprocess.run([sys.executable, "-c", script, str(SOURCE), str(tmp_path)], check=True)


def state_window(monkeypatch, *, position=6):
    torch = pytest.importorskip("torch")
    # Slots and pages deliberately differ, catching accidental request-index addressing.
    req = SimpleNamespace(table_idx=1, linear_slot_idx=2)
    kv = SimpleNamespace(index_ratio=4, cmp_scratch_base=4,
                         _kv_buffer=torch.zeros(2, 2, 2, 8, 1, 2, dtype=torch.uint8),
                         _cmp_k_buffer=torch.zeros(2, 6, 3))
    engine = SimpleNamespace(config=SimpleNamespace(page_size=8), kv_cache=kv, num_pages=2,
                             page_table=torch.tensor([[0] * 8, list(range(8, 16))]))
    views = {"recurrent": torch.zeros(2, 3), "slot/ple_ngram": torch.zeros(2, 4, dtype=torch.int64),
             "qsa_pending": torch.zeros(2, 8, 3)}
    monkeypatch.setattr(probe, "state_views", lambda e, r: views)
    return probe.StateWindow(engine, req, position)


def test_rejection_checks_committed_state_but_allows_unreachable_future_writes(monkeypatch):
    window = state_window(monkeypatch)
    saved = window.capture()
    window.views["kv_page"][:, :, 7].fill_(17)
    window.views["cmp_page"][:, 1].fill_(19)
    window.views["cmp_scratch"].fill_(23)
    assert window.compare(saved, committed_end=7) == []
    assert set(window.compare(saved, committed_end=8)) == {"kv_page", "cmp_page"}
    window.views["kv_page"][:, :, 6].fill_(29)
    assert window.compare(saved, committed_end=7) == ["kv_page"]


@pytest.mark.parametrize("name", ["recurrent", "slot/ple_ngram", "qsa_pending"])
def test_rejection_never_ignores_mutable_recurrence_or_integer_history(monkeypatch, name):
    window = state_window(monkeypatch)
    saved = window.capture()
    window.views[name].reshape(-1)[-1] = 1
    assert window.compare(saved, committed_end=7) == [name]
    window.reset(saved)
    assert window.compare(saved, committed_end=8) == []


def test_window_rejects_unallocated_or_noncontiguous_page(monkeypatch):
    window = state_window(monkeypatch)
    window.engine.page_table[1, 7] = 0
    with pytest.raises(RuntimeError, match="fully allocated contiguous page"):
        probe.StateWindow(window.engine, window.req, 6)


def test_window_rejects_dummy_page(monkeypatch):
    window = state_window(monkeypatch)
    window.engine.num_pages = 1
    with pytest.raises(RuntimeError, match="dummy KV page"):
        probe.StateWindow(window.engine, window.req, 6)


def test_float_error_metrics_do_not_relax_exact_comparison():
    torch = pytest.importorskip("torch")
    expected = torch.tensor([1.0, 2.0])
    actual = torch.tensor([1.0, 2.5])
    result = probe.difference_metrics(actual, expected)
    assert not result["exact"] and result["finite"]
    assert result["different_elements"] == 1
    assert result["max_abs"] == 0.5
    assert result["rms"] == pytest.approx((0.25 / 2) ** 0.5)
    assert result["relative_rms"] == pytest.approx((0.25 / 5) ** 0.5)


def test_integer_metrics_preserve_low_bits_above_float_precision():
    torch = pytest.importorskip("torch")
    expected = torch.tensor([2**62], dtype=torch.int64)
    result = probe.difference_metrics(expected + 1, expected)
    assert not result["exact"] and result["different_elements"] == 1
    assert "relative_rms" not in result


def test_nonfinite_metrics_remain_json_serializable():
    torch = pytest.importorskip("torch")
    result = probe.difference_metrics(torch.tensor([float("nan")]), torch.tensor([1.0]))
    assert not result["finite"] and not result["exact"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("position,width,expected", [
    (0, 3, [[-1, -1, -1], [-1, -1, 99]]),
    (1, 3, [[-1, -1, 10], [-1, 10, 99]]),
    (4, 3, [[20, 30, 40], [30, 40, 99]]),
    (4, 0, [[], []]),
])
def test_ple_candidate_context_is_ordered_and_left_padded(position, width, expected):
    torch = pytest.importorskip("torch")
    contexts = torch.empty((2, width), dtype=torch.int64)
    probe.host_contexts(torch.tensor([10, 20, 30, 40]), position,
                        torch.tensor([99, 100]), contexts, -1)
    assert contexts.tolist() == expected


def test_ple_staging_rejects_incomplete_history():
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="complete host history"):
        probe.host_contexts(torch.tensor([10]), 2, torch.tensor([20, 30]),
                            torch.empty((2, 3), dtype=torch.int64), -1)


def test_committed_metrics_exclude_unreachable_tail(monkeypatch):
    window = state_window(monkeypatch)
    saved = window.capture()
    window.views["kv_page"][:, :, 7].fill_(1)
    window.views["cmp_page"][:, 1].fill_(1)
    assert all(row["exact"] for row in window.metrics(saved, committed_end=7).values())
    assert not window.metrics(saved, committed_end=8)["kv_page"]["exact"]


def test_activation_trace_requires_matching_stages():
    with pytest.raises(RuntimeError, match="stages differ"):
        probe.activation_comparison({"00/input": None}, {"00/output": None})


def test_activation_snapshot_owns_storage(monkeypatch):
    torch = pytest.importorskip("torch")
    source = torch.tensor([[1.0, 2.0]])
    monkeypatch.setattr(probe, "_ACTIVATIONS", {("captured", 1): {"00/input": source}})
    saved = probe.activation_snapshot("captured", 1)
    source.fill_(3)
    assert saved["00/input"].tolist() == [[1.0, 2.0]]
    with pytest.raises(RuntimeError, match="missing its forward"):
        probe.activation_snapshot("eager", 2)


@pytest.mark.parametrize("fused,rows,expected_calls", [
    (True, 2, [1, 1]), (False, 2, [2]), (True, 1, [1]), (True, 3, [3]),
])
def test_serial_dense_policy_is_limited_to_two_token_verification(monkeypatch, fused, rows, expected_calls):
    torch = pytest.importorskip("torch")
    import torch.nn.functional as functional
    import freetoken.core as core

    original = functional.linear
    calls = []

    def observe(input, weight, bias=None):
        calls.append(input.shape[0])
        return original(input, weight, bias)

    # Register the original attribute with monkeypatch so the diagnostic wrapper
    # is removed after this test as well as the observer.
    monkeypatch.setattr(functional, "linear", observe)
    monkeypatch.setattr(core, "get_global_ctx", lambda: SimpleNamespace(batch=SimpleNamespace(mtp_fused=fused)))
    probe.install_serial_linear()
    inputs = torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3)
    weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    bias = torch.tensor([2.0, 3.0])
    actual = functional.linear(input=inputs, weight=weight, bias=bias)
    assert torch.equal(actual, original(inputs, weight, bias))
    assert calls == expected_calls
