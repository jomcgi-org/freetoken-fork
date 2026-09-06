"""Measure complete, independently verified multi-turn Pi coding sessions.

Only the Python standard library is needed by the harness. Install the pinned
Pi package separately. The supplied endpoint must be a local model server.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import selectors
import shutil
import signal
import subprocess
import sys
import time
from urllib.parse import urlsplit


PI_VERSION = "0.85.1"
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def kill_group(process):
    """Include an agent's still-running bash children in timeout cleanup."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        pass
    # The group can outlive its leader, so always attempt the final kill.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


class Rpc:
    def __init__(self, command, workspace, env, stderr, *, trace=False):
        self.process = subprocess.Popen(command, cwd=workspace, env=env,
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=stderr, start_new_session=True)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.buffer = b""
        self.events = []
        self.trace = trace
        self.counter = 0
        self.model_calls = 0
        self.max_model_calls = 0

    def send(self, kind, **fields):
        self.counter += 1
        command = dict(id=str(self.counter), type=kind, **fields)
        self.process.stdin.write(json.dumps(command).encode() + b"\n")
        self.process.stdin.flush()
        return command["id"]

    def next(self, deadline):
        if time.perf_counter() >= deadline:
            raise TimeoutError("Pi session exceeded wall-time budget")
        while b"\n" not in self.buffer:
            remaining = deadline - time.perf_counter()
            if remaining <= 0 or not self.selector.select(remaining):
                raise TimeoutError("Pi session exceeded wall-time budget")
            data = os.read(self.process.stdout.fileno(), 65536)
            if not data:
                raise RuntimeError(f"Pi closed RPC output (exit {self.process.poll()})")
            self.buffer += data
            if len(self.buffer) > 32 * 1024 * 1024:
                raise RuntimeError("RPC frame exceeded 32 MiB")
        line, self.buffer = self.buffer.split(b"\n", 1)
        event = json.loads(line)
        kind = event.get("type")
        if kind == "turn_start":
            self.model_calls += 1
            if self.max_model_calls and self.model_calls > self.max_model_calls:
                raise RuntimeError("Pi exceeded model-call budget")
        if self.trace or kind != "message_update":
            self.events.append(dict(at=time.perf_counter(), event=event))
        if kind == "extension_ui_request":
            raise RuntimeError("unexpected interactive request in isolated benchmark")
        return event

    def command(self, kind, deadline, **fields):
        request_id = self.send(kind, **fields)
        while True:
            event = self.next(deadline)
            if event.get("type") == "response" and event.get("id") == request_id:
                if not event.get("success"):
                    raise RuntimeError(f"Pi rejected {kind}: {event}")
                return event.get("data")

    def prompt(self, message, deadline):
        start = time.perf_counter()
        first_event = len(self.events)
        request_id = self.send("prompt", message=message)
        accepted = False
        last_stop_reason = None
        while True:
            event = self.next(deadline)
            if event.get("type") == "response" and event.get("id") == request_id:
                if not event.get("success"):
                    raise RuntimeError(f"Pi rejected prompt: {event}")
                accepted = True
            if event.get("type") == "message_end":
                msg = event.get("message", {})
                if msg.get("role") == "assistant":
                    last_stop_reason = msg.get("stopReason")
                    if last_stop_reason in ("error", "aborted"):
                        raise RuntimeError(f"incomplete assistant response: {msg}")
            # agent_end can precede retries/compaction. Only settled ends a prompt.
            if event.get("type") == "agent_settled":
                if not accepted:
                    raise RuntimeError("agent settled before accepting prompt")
                if last_stop_reason != "stop":
                    raise RuntimeError(f"incomplete final assistant response: {last_stop_reason}")
                return dict(prompt=message, wall_s=time.perf_counter() - start,
                            event_start=first_event, event_end=len(self.events))

    def close(self):
        kill_group(self.process)
        self.selector.close()
        self.process.stdin.close()
        self.process.stdout.close()


def verify(workspace, stage, deadline, *, script=None):
    start = time.perf_counter()
    process = subprocess.Popen([sys.executable, "-B", str(script or HERE / "agentic-verify.py"),
                                str(workspace), str(stage)], stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, start_new_session=True)
    try:
        output, _ = process.communicate(timeout=max(0.001, min(10, deadline - start)))
        text = output.decode(errors="replace")
        try:
            result = json.loads(text)
        except ValueError:
            result = dict(passed=False, error="invalid verifier output")
        result["passed"] = process.returncode == 0 and result.get("passed") is True
        return dict(result, exit_code=process.returncode, output=text, wall_s=time.perf_counter() - start)
    except subprocess.TimeoutExpired:
        kill_group(process)
        return dict(passed=False, error="verifier timeout", wall_s=time.perf_counter() - start)


def run_stages(rpc, prompts, workspace, *, timeout, repairs, verifier=verify):
    start = time.perf_counter()
    deadline = start + timeout
    result = dict(passed=False, stages=[], error=None)
    try:
        for stage, prompt in enumerate(prompts, 1):
            row = dict(stage=stage, attempts=[], passed=False)
            result["stages"].append(row)
            for attempt in range(repairs + 1):
                response = rpc.prompt(prompt, deadline)
                verification = verifier(workspace, stage, deadline)
                row["attempts"].append(dict(response=response, verification=verification))
                if verification["passed"]:
                    row["passed"] = True
                    break
                prompt = ("The independent checks failed. Fix the implementation while preserving all "
                          "requirements so far, add regression tests, and run them. Check result:\n" +
                          verification.get("output", verification.get("error", "unknown error")))
            if not row["passed"]:
                break
        result["passed"] = len(result["stages"]) == len(prompts) and all(s["passed"] for s in result["stages"])
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["task_wall_s"] = time.perf_counter() - start
    result["verified_task_wall_s"] = result["task_wall_s"] if result["passed"] else None
    result["repair_prompts"] = sum(max(0, len(s["attempts"]) - 1) for s in result["stages"])
    return result


def summarize(rows):
    total = sum(row["task_wall_s"] for row in rows)
    successes = sum(row["passed"] for row in rows)
    qualified = bool(rows) and successes == len(rows)
    return dict(sessions=len(rows), passed=successes, failed=len(rows) - successes,
                attempted_task_wall_s=total, all_tasks_passed=qualified,
                checked_tasks_per_hour=3600 * successes / total if qualified and total > 0 else None,
                success_fraction=successes / len(rows) if rows else None,
                broad_quality_equivalence=False)


def event_metrics(events):
    """Use completed messages once; tool spans are a union, not a parallel sum."""
    starts = {}
    spans = []
    messages = []
    tool_calls = tool_errors = 0
    for record in events:
        event, at = record["event"], record["at"]
        kind = event.get("type")
        if kind == "tool_execution_start":
            starts[event["toolCallId"]] = at
            tool_calls += 1
        elif kind == "tool_execution_end":
            start = starts.pop(event["toolCallId"], None)
            if start is not None:
                spans.append((start, at))
            tool_errors += bool(event.get("isError"))
        elif kind == "message_end" and event.get("message", {}).get("role") == "assistant":
            messages.append(event["message"])
    tool_wall = 0.0
    end = float("-inf")
    for left, right in sorted(spans):
        tool_wall += max(0, right - max(left, end))
        end = max(end, right)
    return dict(tool_calls=tool_calls, tool_errors=tool_errors, unfinished_tools=len(starts),
                tool_wall_s=tool_wall,
                model_usage=[msg.get("usage") for msg in messages],
                finish_reasons=[msg.get("stopReason") for msg in messages])


def model_config(args):
    return dict(providers=dict(freetoken=dict(
        api="openai-completions", baseUrl=args.base_url, apiKey="local-benchmark",
        compat=dict(supportsStore=False, supportsDeveloperRole=False,
                    supportsReasoningEffort=False, maxTokensField="max_tokens"),
        models=[dict(id=args.model, contextWindow=args.context_tokens, maxTokens=args.max_tokens,
                     reasoning=False, input=["text"], samplingParams=dict(
                         temperature=0, max_tokens=args.max_tokens,
                         chat_template_kwargs=dict(enable_thinking=False)))])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pi", required=True, type=Path, help="Path to pinned Pi executable")
    parser.add_argument("--base-url", default="http://127.0.0.1:8090/v1")
    parser.add_argument("--model", default="qwen3.6-27b")
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory, never overwrites a run")
    parser.add_argument("--label", required=True, help="Server revision/configuration being tested")
    parser.add_argument("--server-metadata", type=Path, help="Captured server revision, flags, binary hashes and cache policy")
    parser.add_argument("--sessions", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=900, help="Seconds per whole task, including repair and verification")
    parser.add_argument("--repairs", type=int, default=1, help="Maximum verifier feedback rounds per stage")
    parser.add_argument("--max-model-calls", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--context-tokens", type=int, default=32768)
    parser.add_argument("--trace", action="store_true", help="Retain streaming deltas; diagnostic run, not a wall-time gate")
    args = parser.parse_args()
    if urlsplit(args.base_url).hostname not in ("127.0.0.1", "localhost", "::1"):
        parser.error("use a loopback local server or SSH tunnel")
    if min(args.sessions, args.timeout, args.max_model_calls, args.max_tokens) <= 0 or args.repairs < 0:
        parser.error("budgets must be positive; repairs may be zero")
    if args.max_tokens >= args.context_tokens:
        parser.error("context window must exceed maximum output")
    args.pi = args.pi.resolve()
    version = subprocess.check_output([str(args.pi), "--version"], text=True, timeout=30).strip()
    if version != PI_VERSION:
        parser.error(f"expected Pi {PI_VERSION}, got {version!r}")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    fixture = HERE / "agentic-fixtures/expiry-cache"
    task = json.loads((fixture / "task.json").read_text())
    verifier_script = root / "verifier.py"
    shutil.copyfile(HERE / "agentic-verify.py", verifier_script)
    config = root / "pi-config"
    config.mkdir()
    write_json(config / "models.json", model_config(args))
    write_json(config / "settings.json", dict(compaction=dict(enabled=False),
                                              retry=dict(enabled=False), quietStartup=True))
    # Do not expose model API credentials, user extensions or shell startup files.
    # This is configuration isolation, not an OS sandbox for the agent's tools.
    env = dict(PATH=os.environ["PATH"], PI_CODING_AGENT_DIR=str(config), PI_OFFLINE="1",
               PI_TELEMETRY="0", TERM="dumb", SHELL="/bin/bash", LANG="en_US.UTF-8")
    command = [str(args.pi), "--mode", "rpc", "--no-session", "--offline",
               "--no-extensions", "--no-skills", "--no-prompt-templates", "--no-themes",
               "--no-context-files", "--no-approve", "--provider", "freetoken",
               "--model", args.model, "--thinking", "off", "--tools", "read,bash,edit,write"]
    metadata = dict(label=args.label, pi_version=version, pi_executable_sha256=sha(args.pi),
                    command=command, client_host=platform.platform(), python=sys.version,
                    trace=args.trace, started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    task=task, model_config=model_config(args),
                    settings=json.loads((config / "settings.json").read_text()),
                    requested_sessions=args.sessions,
                    budgets=dict(timeout=args.timeout, repairs=args.repairs, max_model_calls=args.max_model_calls),
                    sources={str(p.relative_to(HERE)): sha(p) for p in
                             [Path(__file__), HERE / "agentic-verify.py", HERE / "pi/package-lock.json",
                              *sorted(fixture.iterdir())] if p.is_file()},
                    server_metadata=json.loads(args.server_metadata.read_text()) if args.server_metadata else None,
                    cache_note="New Pi session each task; server cache is not flushed. Follow-up history is retained. "
                               "A warm prefix hit is not assumed; Pi cacheRead may be zero when the server does not report it.")
    write_json(root / "metadata.json", metadata)
    rows = []
    workspace = root / "workspace"
    cancelled = False

    def interrupted(signum, _frame):
        nonlocal cancelled
        cancelled = True
        # selectors treats InterruptedError as an interrupted system call and
        # can swallow it. Use a regular exception to retain the cancellation.
        raise RuntimeError(f"benchmark interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    for ordinal in range(1, args.sessions + 1):
        if workspace.exists():
            shutil.rmtree(workspace)
        workspace.mkdir()
        for name in ("cache.py", "test_cache.py"):
            shutil.copyfile(fixture / name, workspace / name)
        session_dir = root / f"session-{ordinal}"
        session_dir.mkdir()
        rpc = None
        launch = time.perf_counter()
        try:
            with (session_dir / "stderr.log").open("wb") as stderr:
                rpc = Rpc(command, workspace, env, stderr, trace=args.trace)
                state = rpc.command("get_state", time.perf_counter() + 30)
                if state.get("model", {}).get("provider") != "freetoken" or state["model"]["id"] != args.model:
                    raise RuntimeError(f"unexpected selected model: {state.get('model')}")
                rpc.max_model_calls = args.max_model_calls
                startup = time.perf_counter() - launch
                row = run_stages(rpc, task["stages"], workspace, timeout=args.timeout, repairs=args.repairs,
                                 verifier=lambda w, s, d: verify(w, s, d, script=verifier_script))
                row.update(ordinal=ordinal, startup_s=startup, model_calls=rpc.model_calls)
                # Retrieval and artifact writes stay outside the task wall clock.
                if row["error"] is None:
                    try:
                        row["session_stats"] = rpc.command("get_session_stats", time.perf_counter() + 10)
                        row["messages"] = rpc.command("get_messages", time.perf_counter() + 10)
                    except Exception as exc:
                        row["artifact_error"] = f"{type(exc).__name__}: {exc}"
                row["events"] = rpc.events
        except Exception as exc:
            row = dict(ordinal=ordinal, passed=False, task_wall_s=time.perf_counter() - launch,
                       verified_task_wall_s=None, stages=[], error=f"{type(exc).__name__}: {exc}",
                       events=rpc.events if rpc else [])
        finally:
            if rpc:
                rpc.close()
        row["trace"] = args.trace
        row["event_metrics"] = event_metrics(row["events"])
        # A client cannot prove server isolation from an endpoint. Keep an
        # explicit operator declaration, without treating a metadata file as proof.
        row["wall_gate_eligible"] = False
        shutil.copytree(workspace, session_dir / "final-workspace", symlinks=True)
        write_json(session_dir / "result.json", row)
        rows.append(row)
        summary = summarize(rows)
        summary["cancelled"] = cancelled
        summary["completed_schedule"] = not cancelled and len(rows) == args.sessions
        summary["wall_gate_eligible"] = False
        summary["qualification_note"] = "Requires paired server-controlled runs and review of server identity, diagnostics, cache state and competing work."
        write_json(root / "summary.json", summary)
        print(json.dumps(dict(ordinal=ordinal, passed=row["passed"], wall_s=row["task_wall_s"],
                              model_calls=row.get("model_calls"), error=row.get("error"))), flush=True)
        if cancelled:
            break
    return 0 if not cancelled and len(rows) == args.sessions and all(row["passed"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
