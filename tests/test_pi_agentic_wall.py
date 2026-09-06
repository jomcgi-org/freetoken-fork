"""Hermetic checks for agent completion, failure accounting and task grading."""

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).resolve().parents[1]
client = load("pi_wall", ROOT / "bench/pi-agentic-wall.py")
grader = load("pi_grader", ROOT / "bench/agentic-verify.py")


class CorrectCache:
    def __init__(self, capacity=None):
        if capacity is not None and capacity <= 0:
            raise ValueError("capacity")
        self.capacity = capacity
        self.items = {}

    def put(self, key, value, ttl, now):
        self.items = {k: v for k, v in self.items.items() if v[1] > now and k != key}
        if ttl <= 0:
            return
        self.items[key] = (value, now + ttl)
        if self.capacity is not None:
            while len(self.items) > self.capacity:
                del self.items[next(iter(self.items))]

    def get(self, key, now, default=None):
        item = self.items.pop(key, None)
        if item is None or item[1] <= now:
            return default
        self.items[key] = item
        return item[0]

    def get_or_load(self, key, loader, ttl, now):
        missing = object()
        found = self.get(key, now, missing)
        if found is not missing:
            return found
        value = loader()
        self.put(key, value, ttl, now)
        return value


@pytest.mark.parametrize("stage", [1, 2, 3])
def test_independent_grader_accepts_correct_implementation(stage):
    grader.check(CorrectCache, stage)


def test_initial_public_smoke_pass_is_insufficient_for_grader():
    initial = load("buggy_cache", ROOT / "bench/agentic-fixtures/expiry-cache/cache.py")
    with pytest.raises(AssertionError, match="expiry boundary"):
        grader.check(initial.Cache, 1)


def test_grader_rejects_failure_to_promote_put():
    class Broken(CorrectCache):
        def put(self, key, value, ttl, now):
            old_order = list(self.items)
            super().put(key, value, ttl, now)
            if key in old_order and key in self.items:
                self.items = {k: self.items[k] for k in old_order if k in self.items}

    with pytest.raises(AssertionError, match="put must promote"):
        grader.check(Broken, 2)


def test_grader_rejects_treating_none_as_loader_miss():
    class Broken(CorrectCache):
        def get_or_load(self, key, loader, ttl, now):
            value = self.get(key, now)
            if value is None:
                value = loader()
                self.put(key, value, ttl, now)
            return value

    with pytest.raises(AssertionError, match="cached None"):
        grader.check(Broken, 3)


def test_repairs_and_verifier_time_stay_in_complete_task_wall(monkeypatch, tmp_path):
    now = [0.0]
    monkeypatch.setattr(client.time, "perf_counter", lambda: now[0])
    prompts = []

    class Rpc:
        def prompt(self, message, deadline):
            assert deadline == 100
            prompts.append(message)
            now[0] += 7
            return dict(wall_s=7)

    checks = iter([False, True, True, True])

    def verify(*args):
        now[0] += 2
        return dict(passed=next(checks), output="boundary failure")

    row = client.run_stages(Rpc(), ["fix", "extend", "load"], tmp_path,
                            timeout=100, repairs=1, verifier=verify)
    assert row["passed"] and row["verified_task_wall_s"] == 36
    assert row["repair_prompts"] == 1
    assert "boundary failure" in prompts[1]
    assert prompts[2:] == ["extend", "load"]


def test_exhausted_repairs_are_retained_and_cannot_count_as_success(tmp_path):
    class Rpc:
        def prompt(self, *_):
            return dict(wall_s=1)

    row = client.run_stages(Rpc(), ["a", "b"], tmp_path, timeout=10, repairs=1,
                            verifier=lambda *_: dict(passed=False, output="bad"))
    assert not row["passed"] and row["verified_task_wall_s"] is None
    assert len(row["stages"]) == 1 and len(row["stages"][0]["attempts"]) == 2
    summary = client.summarize([row, dict(passed=True, task_wall_s=10)])
    assert summary["passed"] == 1 and summary["failed"] == 1
    assert summary["checked_tasks_per_hour"] is None
    assert summary["attempted_task_wall_s"] == row["task_wall_s"] + 10


def test_agent_error_is_retained_without_running_grader(tmp_path):
    class Rpc:
        def prompt(self, *_):
            raise TimeoutError("budget")

    def forbidden(*_):
        raise AssertionError("must not grade incomplete prompt")

    row = client.run_stages(Rpc(), ["a"], tmp_path, timeout=10, repairs=1, verifier=forbidden)
    assert not row["passed"] and row["error"] == "TimeoutError: budget"
    assert row["verified_task_wall_s"] is None


def rpc_for_events(tmp_path, events, *, trace=False):
    # A real subprocess/pipe exercises framing and process cleanup without network.
    script = "import json,sys,time\njson.loads(sys.stdin.buffer.readline())\n"
    script += "events = " + repr(events) + "\n"
    script += "for event in events:\n print(json.dumps(event, ensure_ascii=False), flush=True)\n"
    script += "time.sleep(30)\n"
    err = (tmp_path / "stderr").open("wb")
    rpc = client.Rpc([sys.executable, "-u", "-c", script], tmp_path,
                     dict(PATH=os.environ["PATH"]), err, trace=trace)
    return rpc, err


def accepted():
    return dict(type="response", id="1", success=True)


def test_rpc_waits_until_settled_and_keeps_unicode_inside_json(tmp_path):
    events = [accepted(), dict(type="turn_start"), dict(type="agent_end"),
              dict(type="message_update", delta="one\u2028two"),
              dict(type="message_end", message=dict(role="assistant", content="one\u2028two", stopReason="stop")),
              dict(type="agent_settled")]
    rpc, err = rpc_for_events(tmp_path, events)
    try:
        response = rpc.prompt("test", time.perf_counter() + 3)
        assert response["event_end"] == 5
        assert rpc.events[-1]["event"]["type"] == "agent_settled"
        assert rpc.events[-2]["event"]["message"]["content"] == "one\u2028two"
        assert rpc.model_calls == 1
    finally:
        rpc.close()
        err.close()
    assert rpc.process.poll() is not None


@pytest.mark.parametrize("reason", ["length", "error", "aborted"])
def test_rpc_rejects_incomplete_responses(tmp_path, reason):
    events = [accepted(), dict(type="message_end", message=dict(role="assistant", stopReason=reason)),
              dict(type="agent_settled")]
    rpc, err = rpc_for_events(tmp_path, events)
    try:
        with pytest.raises(RuntimeError, match="incomplete"):
            rpc.prompt("test", time.perf_counter() + 3)
    finally:
        rpc.close()
        err.close()


def test_rpc_retains_recovered_truncation_before_success(tmp_path):
    events = [accepted(), dict(type="turn_start"),
              dict(type="message_end", message=dict(role="assistant", stopReason="length")),
              dict(type="turn_start"),
              dict(type="message_end", message=dict(role="assistant", stopReason="stop")),
              dict(type="agent_settled")]
    rpc, err = rpc_for_events(tmp_path, events)
    try:
        rpc.prompt("test", time.perf_counter() + 3)
        assert rpc.model_calls == 2
        assert client.event_metrics(rpc.events)["finish_reasons"] == ["length", "stop"]
    finally:
        rpc.close()
        err.close()


def test_rpc_enforces_model_call_budget(tmp_path):
    rpc, err = rpc_for_events(tmp_path, [accepted(), dict(type="turn_start"), dict(type="turn_start")])
    rpc.max_model_calls = 1
    try:
        with pytest.raises(RuntimeError, match="model-call budget"):
            rpc.prompt("test", time.perf_counter() + 3)
    finally:
        rpc.close()
        err.close()


def test_rpc_timeout_does_not_wait_for_process_exit(tmp_path):
    rpc, err = rpc_for_events(tmp_path, [accepted()])
    try:
        with pytest.raises(TimeoutError):
            rpc.prompt("test", time.perf_counter() + 0.1)
    finally:
        rpc.close()
        err.close()


def test_overlapping_tools_use_union_and_assistant_usage_is_not_duplicated():
    def event(at, kind, **fields):
        return dict(at=at, event=dict(type=kind, **fields))

    msg = dict(role="assistant", usage=dict(input=100, output=20), stopReason="toolUse")
    rows = [event(0, "tool_execution_start", toolCallId="a"),
            event(1, "tool_execution_start", toolCallId="b"),
            event(3, "tool_execution_end", toolCallId="a", isError=False),
            event(4, "tool_execution_end", toolCallId="b", isError=True),
            event(5, "message_end", message=msg), event(6, "agent_end", messages=[msg])]
    metrics = client.event_metrics(rows)
    assert metrics["tool_wall_s"] == 4
    assert metrics["tool_calls"] == 2 and metrics["tool_errors"] == 1
    assert metrics["model_usage"] == [dict(input=100, output=20)]


def test_interrupted_cli_records_failure_and_stops_pi_before_next_session(tmp_path):
    fake = tmp_path / "pi"
    fake.write_text(f"#!{sys.executable}\n" + '''import json,os,sys,time
from pathlib import Path
if "--version" in sys.argv:
    print("0.85.1")
    raise SystemExit(0)
for line in sys.stdin:
    command = json.loads(line)
    if command["type"] == "get_state":
        print(json.dumps(dict(type="response",id=command["id"],success=True,data=dict(model=dict(provider="freetoken",id="qwen3.6-27b")))),flush=True)
    if command["type"] == "prompt":
        Path("ready.pid").write_text(str(os.getpid()))
        print(json.dumps(dict(type="response",id=command["id"],success=True)),flush=True)
        time.sleep(30)
''')
    fake.chmod(0o755)
    output = tmp_path / "output"
    process = subprocess.Popen([sys.executable, str(ROOT / "bench/pi-agentic-wall.py"),
                                "--pi", str(fake), "--output-dir", str(output),
                                "--label", "hermetic-interruption", "--sessions", "2"],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    ready = output / "workspace/ready.pid"
    try:
        deadline = time.monotonic() + 5
        while not ready.exists():
            if process.poll() is not None or time.monotonic() > deadline:
                raise AssertionError("fake Pi did not accept prompt")
            time.sleep(0.01)
        pi_pid = int(ready.read_text())
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == 1, (stdout, stderr)
        row = json.loads((output / "session-1/result.json").read_text())
        assert not row["passed"] and "interrupted by signal" in row["error"]
        assert not (output / "session-2").exists()
        summary = json.loads((output / "summary.json").read_text())
        assert summary["cancelled"] and not summary["completed_schedule"]
        with pytest.raises(ProcessLookupError):
            os.kill(pi_pid, 0)
    finally:
        client.kill_group(process)
