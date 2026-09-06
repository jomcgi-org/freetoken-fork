"""Hermetic checks for bounded request scheduling and complete-workload timing."""

import importlib.util
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest


path = Path(__file__).parents[1] / "bench/concurrent-prefill-wall.py"
spec = importlib.util.spec_from_file_location("concurrent_client", path)
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)


def case(ordinal, *, warmup=False):
    return dict(ordinal=ordinal, warmup=warmup, kind="json", prompt=str(ordinal),
                expected={"r00": 1}, max_tokens=512)


def response(**updates):
    return dict(dict(wall_s=7.0, ttft_s=2.0, decode_s=5.0, text='{"r00": 1}',
                     usage=dict(prompt_tokens=1700, completion_tokens=2),
                     finish_reason="stop", completed=True, response_checks_passed=True), **updates)


def test_requests_overlap_and_never_exceed_client_limit():
    barrier = threading.Barrier(3, timeout=3)
    lock = threading.Lock()
    active = peak = 0
    seen = []

    def send(item):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            seen.append(item["ordinal"])
        try:
            barrier.wait()
            return response()
        finally:
            with lock:
                active -= 1

    rows, summary = client.run_group([case(i) for i in range(6)], 3, send)
    assert peak == 3 and active == 0
    assert sorted(seen) == list(range(6))
    assert sorted(r["ordinal"] for r in rows) == list(range(6))
    assert summary["request_errors"] == 0
    assert summary["completed"] == 6
    assert summary["all_response_checks_passed"]


def test_finished_request_refills_while_an_earlier_request_is_pending():
    next_started = threading.Event()
    events = []

    def send(item):
        ordinal = item["ordinal"]
        events.append(("start", ordinal))
        if ordinal == 0:
            assert next_started.wait(3), "client stalled behind an earlier request"
        if ordinal == 2:
            next_started.set()
        events.append(("finish", ordinal))
        return response()

    _, summary = client.run_group([case(i) for i in range(3)], 2, send)
    assert summary["request_errors"] == 0
    assert events.index(("start", 2)) < events.index(("finish", 0))


def test_workload_wall_includes_recording_and_excludes_outer_io_snapshots():
    now = [0.0]
    observations = []

    def send(item):
        now[0] += 7.0
        return response()

    def emit(row):
        now[0] += 3.0

    def snapshot():
        value = now[0]
        observations.append(value)
        now[0] += 100.0
        return value

    rows, summary = client.run_group([case(0), case(1)], 1, send,
                                     emit=emit, snapshot=snapshot, clock=lambda: now[0])
    assert summary["wall_s"] == 20.0
    assert summary["summed_request_wall_s"] == 14.0
    assert [row["client_active_s"] for row in rows] == [7.0, 7.0]
    assert observations == [0.0, 120.0]
    assert summary["process_io"] == dict(before=0.0, after=120.0)
    assert summary["checked_requests_per_s"] == 0.1
    assert summary["checked_completion_tokens_per_s"] == 0.2


def test_request_failure_is_retained_and_does_not_drop_remaining_work():
    seen = []

    def send(item):
        seen.append(item["ordinal"])
        if item["ordinal"] == 1:
            raise TimeoutError("injected stream timeout")
        return response()

    rows, summary = client.run_group([case(i) for i in range(5)], 2, send)
    assert sorted(seen) == list(range(5))
    failed = next(row for row in rows if row["ordinal"] == 1)
    assert failed["prompt"] == "1"
    assert failed["request_error"] == "TimeoutError: injected stream timeout"
    assert summary["requests"] == 5 and summary["completed"] == 4
    assert summary["request_errors"] == 1
    assert not summary["all_response_checks_passed"]
    assert summary["checked_requests_per_s"] is None
    assert summary["checked_completion_tokens_per_s"] is None


@pytest.mark.parametrize("cases,concurrency", [
    ([], 1), ([case(0)], 0), ([case(0)], -1),
    ([case(0), case(0)], 2), ([case(0, warmup=True), case(1)], 2),
])
def test_invalid_groups_are_rejected_before_any_request(cases, concurrency):
    def unexpected(*args):
        raise AssertionError("invalid groups must not issue requests or sample I/O")

    with pytest.raises(ValueError):
        client.run_group(cases, concurrency, unexpected, snapshot=unexpected)


@pytest.mark.parametrize("text,reason,passed", [
    ('{"r00": 1}', "stop", True),
    ('{"r00": 1, "r00": 1}', "stop", False),
    ('{"r00": true}', "stop", False),
    ('{"r00": 1.0}', "stop", False),
    ('{"r00": 1}', "length", False),
    ('{}', "stop", False),
])
def test_json_requires_complete_exact_values_types_and_multiplicity(text, reason, passed):
    checks = client.sibling("concurrent_test_json", "staged-prefill-long-output.py")
    row = client.complete_case(case(0), lambda item: response(text=text, finish_reason=reason),
                               checks, None)
    assert row["completed"] is passed
    assert row["response_checks_passed"] is passed
    assert row["text"] == text


def test_prose_format_is_separate_from_completion_and_semantics():
    protocol = client.sibling("concurrent_test_prose", "sustained-prefill-wall.py")
    item = dict(case(0), kind="essay", expected=None)
    row = client.complete_case(item, lambda item: response(text="First.\n\nSecond."),
                               None, protocol)
    assert row["completed"]
    assert not row["response_checks_passed"]
    assert row["semantic_quality_scored"] is False


def test_short_prompt_is_retained_but_cannot_qualify_long_prefill():
    row = client.complete_case(case(0), lambda item: response(
        usage=dict(prompt_tokens=10, completion_tokens=2)), None, None)
    assert row["text"] == '{"r00": 1}'
    assert not row["completed"] and not row["response_checks_passed"]
    assert "protocol_error" in row


def test_cli_separates_warmups_preserves_outputs_and_refuses_overwrite(tmp_path, monkeypatch):
    cases = [case(0, warmup=True), case(1, warmup=True), case(2), case(3)]
    protocol = client.sibling("concurrent_test_main_protocol", "sustained-prefill-wall.py")
    protocol.build_cases = lambda *args: cases
    checks = client.sibling("concurrent_test_main_checks", "staged-prefill-long-output.py")
    events = []

    def snapshot(pid):
        assert pid == 42
        events.append("snapshot")
        return dict(pid=pid, starttime_ticks=123, counters=dict(rchar=0, read_bytes=0))

    checks.process_io_snapshot = snapshot

    def request(base_url, payload):
        ordinal = int(payload["messages"][0]["content"])
        events.append(ordinal)
        assert payload["chat_template_kwargs"] == dict(enable_thinking=False)
        assert payload["temperature"] == 0 and payload["max_tokens"] == 512
        return response()

    modules = {"sustained-prefill-wall.py": protocol, "staged-prefill-long-output.py": checks,
               "selective-prefill.py": SimpleNamespace(request=request)}
    monkeypatch.setattr(client, "sibling", lambda name, filename: modules[filename])
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: object())))
    output = tmp_path / "responses.jsonl"
    monkeypatch.setattr(sys, "argv", [str(path), "--tokenizer", "unused", "--output", str(output),
                                      "--mode", "optimized", "--concurrency", "2", "--io-pid", "42"])
    client.main()
    assert events[0] == events[3] == events[4] == events[7] == "snapshot"
    assert set(events[1:3]) == {0, 1} and set(events[5:7]) == {2, 3}
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert sorted(row["ordinal"] for row in rows) == list(range(4))
    assert all(row["client_concurrency"] == 2 and not row["diagnostic_phase_io"] for row in rows)
    assert all("process_io" not in row for row in rows)
    assert json.loads(output.with_suffix(".prompts.json").read_text()) == cases
    summaries = json.loads(output.with_suffix(".summary.json").read_text())
    assert [s["phase"] for s in summaries] == ["warmup", "measured"]
    assert all(s["completed"] == 2 and s["all_response_checks_passed"] for s in summaries)
    assert all(s["process_io"]["delta"] == dict(rchar=0, read_bytes=0) for s in summaries)
    with pytest.raises(SystemExit):
        client.main()
    assert len(events) == 8
