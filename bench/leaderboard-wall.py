"""Run frozen leaderboard tasks through their original agent loop and graders.

All inputs and results are supplied privately. This client does not manage a GPU
lease or change serving: an exclusive supervisor must own its local endpoint.
"""

import argparse
import asyncio
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace
from urllib.parse import urlsplit


ARMS = (("r1", "off"), ("r2", "on"), ("r3", "on"), ("r4", "off"))


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.chmod(0o600)
    temp.replace(path)


def file_hashes(root):
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("benchmark trees must not contain symlinks")
        if path.is_file() and "__pycache__" not in path.parts:
            result[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def load_inputs(root):
    root = root.resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text())
    actual = file_hashes(root)
    del actual["manifest.json"]
    if actual != manifest["files"]:
        raise ValueError("frozen benchmark bundle changed")
    if not manifest["task_ids"] or len(set(manifest["task_ids"])) != len(manifest["task_ids"]):
        raise ValueError("task selection must be nonempty and unique")
    # Import only after authenticating the supplied source bundle. A fresh
    # process avoids mixing its package with a different installed `bench`.
    if any(name == "bench" or name.startswith("bench.") for name in sys.modules):
        raise RuntimeError("benchmark package was imported before bundle validation")
    sys.path.insert(0, str(root))
    from bench.agent import run_agent_cell
    from bench.cache import cell_key, fixture_hash
    from bench.cli import load_tasks
    from bench.openrouter import OpenRouterClient
    from bench.registry import load_registry
    from bench.verifiers import get_verifier, verifier_source_hash

    model = next(m for m in load_registry(root / "models.yaml") if m.id == manifest["model_id"])
    tasks = {t.id: t for t in load_tasks(root / "tasks")}
    if set(tasks) != set(manifest["task_ids"]):
        raise ValueError("bundle task definitions differ from the frozen selection")
    for task in tasks.values():
        if task.mode != "agentic" or task.agent.exec:
            raise ValueError("this comparison requires the original file-tool agentic protocol")
        params = (f"agentic:{task.agent.max_tokens or model.params.max_tokens}"
                  f":turns={task.agent.max_turns}:exec={task.agent.exec}"
                  f":provider={model.provider}:api_model={model.api_model or model.id}"
                  f":extra={json.dumps(model.extra_body, sort_keys=True)}")
        verifier = json.dumps(dict(kind=task.verifier.kind, args=task.verifier.args,
                                   src=verifier_source_hash(task.verifier.kind)), sort_keys=True)
        key = cell_key(prompt=task.prompt, fixture_hash=fixture_hash(root / "tasks" / task.id / "fixture"),
                       verifier_repr=verifier, model_id=model.id, params_repr=params)
        if key != manifest["historical"][task.id]["content_hash"]:
            raise ValueError("task no longer matches its historical benchmark inputs: " + task.id)
    return SimpleNamespace(root=root, manifest=manifest, model=model, tasks=tasks,
                           run_agent_cell=run_agent_cell, Client=OpenRouterClient,
                           get_verifier=get_verifier)


def qualify(api, output):
    rows = []
    for task_id in api.manifest["task_ids"]:
        task = api.tasks[task_id]
        fixture = api.root / "tasks" / task_id / "fixture"
        verify = api.get_verifier(task.verifier.kind)
        outcomes = {}
        for reference in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                work = Path(directory)
                shutil.copytree(fixture, work, dirs_exist_ok=True)
                if reference:
                    for relative, source in api.manifest["references"][task_id].items():
                        if relative not in task.target_files:
                            raise ValueError("reference patch changes an undeclared target")
                        shutil.copyfile(api.root / source, work / relative)
                result = verify(work, task.verifier.args)
                outcomes["reference" if reference else "buggy"] = dict(
                    passed=result.passed, feedback=result.feedback)
        rows.append(dict(task_id=task_id, checks_passed=not outcomes["buggy"]["passed"]
                         and outcomes["reference"]["passed"], outcomes=outcomes))
        save(output, dict(completed=False, tasks=rows))
    result = dict(completed=True, checks_passed=all(r["checks_passed"] for r in rows),
                  manifest_sha256=hashlib.sha256((api.root / "manifest.json").read_bytes()).hexdigest(),
                  tasks=rows)
    save(output, result)
    return result


async def run_task(api, task_id, client, *, clock=time.perf_counter):
    task, model = api.tasks[task_id], api.model
    fixture = api.root / "tasks" / task_id / "fixture"
    original_files = file_hashes(fixture)
    observation = dict(calls=0, grade_s=0., grade_ran=False, scope_ok=False,
                       changed_files=[], done_called=False)
    calls, edits = [], {}

    async def chat(**kwargs):
        kwargs["model"] = model.api_model or model.id
        if model.extra_body:
            kwargs["extra_body"] = model.extra_body
        observation["calls"] += 1
        result = await client.chat(**kwargs)
        calls.append(dict(latency_ms=result.latency_ms, prompt_tokens=result.prompt_tokens,
                          completion_tokens=result.completion_tokens))
        observation["done_called"] |= any(
            call.get("function", {}).get("name") == "done"
            for call in result.message.get("tool_calls", []) or [])
        return result

    def verify(work, args):
        current = file_hashes(work)
        changed = sorted(name for name in current.keys() | original_files.keys()
                         if current.get(name) != original_files.get(name))
        observation["changed_files"] = changed
        observation["scope_ok"] = set(changed) <= set(task.target_files)
        for name in changed:
            path = work / name
            edits[name] = path.read_text() if path.is_file() else None
        started = clock()
        result = api.get_verifier(task.verifier.kind)(work, args)
        observation["grade_s"] = clock() - started
        observation["grade_ran"] = True
        return result

    started = clock()
    cell = await api.run_agent_cell(
        task_id=task.id, task_version=task.version, model_id=model.id,
        content_hash=api.manifest["historical"][task.id]["content_hash"],
        fixture_dir=fixture, task_prompt=task.prompt, chat=chat, verify=verify,
        verifier_args=task.verifier.args, cost_fn=lambda p, c: 0.,
        max_turns=task.agent.max_turns, max_tokens=task.agent.max_tokens or model.params.max_tokens,
        allow_exec=task.agent.exec)
    elapsed = clock() - started
    return dict(task_id=task_id, task_wall_s=elapsed, model_request_s=cell.total_latency_ms / 1000,
                passed=cell.first_attempt_passed, completion_qualified=bool(
                    cell.first_attempt_passed and observation["grade_ran"]
                    and observation["scope_ok"] and cell.tool_use_ok),
                cell=cell.model_dump(), observation=observation, calls=calls, edits=edits)


async def run_arm(api, output, *, base_url, arm, qualification):
    endpoint = urlsplit(base_url)
    if endpoint.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("owned benchmark endpoint must be local")
    manifest_sha = hashlib.sha256((api.root / "manifest.json").read_bytes()).hexdigest()
    if not qualification.get("completed") or not qualification.get("checks_passed") or (
            qualification.get("manifest_sha256") != manifest_sha):
        raise ValueError("matching fail-to-pass grader qualification is required")
    report = dict(completed=False, arm=arm, mode=dict(ARMS)[arm], manifest_sha256=manifest_sha,
                  records=[], historical=api.manifest["historical"],
                  limitations=["Agent trajectories and output lengths may differ",
                               "Historical metric excludes tool and grader time",
                               "Small task selection does not prove broad quality equivalence"])
    client = api.Client(api_key="local", base_url=base_url, timeout=600.)
    try:
        for task_id in api.manifest["task_ids"]:
            report["active_task"] = task_id
            save(output, report)
            row = await run_task(api, task_id, client)
            row.update(arm=arm, mode=dict(ARMS)[arm])
            report["records"].append(row)
            save(output, report)
        report["completed"] = True
        report["all_tasks_passed"] = all(r["completion_qualified"] for r in report["records"])
    finally:
        await client._client.aclose()
        save(output, report)
    return report


def summarize(reports, task_ids):
    expected = {(arm, task) for arm, _ in ARMS for task in task_ids}
    rows = [row for report in reports for row in report["records"]]
    keys = [(row["arm"], row["task_id"]) for row in rows]
    if (not task_ids or len(task_ids) != len(set(task_ids)) or len(reports) != len(ARMS)
            or len(keys) != len(set(keys)) or set(keys) != expected
            or not all(report["completed"] for report in reports)
            or len({report["manifest_sha256"] for report in reports}) != 1
            or any(row["mode"] != dict(ARMS)[row["arm"]] for row in rows)):
        raise ValueError("comparison requires one complete matched task set in both orders")
    if any(not math.isfinite(r[field]) or r[field] < 0 for r in rows
           for field in ("task_wall_s", "model_request_s")):
        raise ValueError("comparison requires finite nonnegative wall measurements")
    def totals(selected):
        passed = sum(r["completion_qualified"] for r in selected)
        elapsed = sum(r["task_wall_s"] for r in selected)
        return dict(tasks=len(selected), passed=passed, completion_rate=passed / len(selected),
                    task_wall_s=elapsed,
                    successful_tasks_per_hour=passed * 3600 / elapsed if elapsed > 0 else None,
                    model_request_s=sum(r["model_request_s"] for r in selected),
                    calls=sum(r["observation"]["calls"] for r in selected),
                    output_tokens=sum(a["completion_tokens"] for r in selected for a in r["cell"]["attempts"]))
    def compare(selected):
        modes = {mode: totals([r for r in selected if r["mode"] == mode]) for mode in ("off", "on")}
        def reduction(field):
            return (100 * (1 - modes["on"][field] / modes["off"][field])
                    if modes["off"][field] > 0 else None)
        off_rate = modes["off"]["successful_tasks_per_hour"]
        on_rate = modes["on"]["successful_tasks_per_hour"]
        return dict(modes=modes, wall_reduction_percent=reduction("task_wall_s"),
                    model_time_reduction_percent=reduction("model_request_s"),
                    successful_task_throughput_ratio=(on_rate / off_rate
                        if off_rate is not None and off_rate > 0 and on_rate is not None else None))
    return dict(completed=True, all_tasks_passed=all(r["completion_qualified"] for r in rows),
                comparison=compare(rows), orders={
                    name: compare([r for r in rows if r["arm"] in arms])
                    for name, arms in (("A/B", ("r1", "r2")), ("B/A", ("r3", "r4")))},
                tasks={task: compare([r for r in rows if r["task_id"] == task]) for task in task_ids},
                historical=reports[0]["historical"], isolated_runtime_speedup=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("qualify", "run", "summarize"))
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:18090/v1")
    parser.add_argument("--arm", choices=[name for name, _ in ARMS])
    parser.add_argument("--reports", type=Path, nargs="+")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refuse to overwrite a benchmark result")
    sys.dont_write_bytecode = True
    api = load_inputs(args.bundle)
    if args.action == "qualify":
        result = qualify(api, args.output)
        return 0 if result["checks_passed"] else 1
    if args.action == "run":
        if args.arm is None or args.qualification is None:
            parser.error("run requires --arm and --qualification")
        asyncio.run(run_arm(api, args.output, base_url=args.base_url, arm=args.arm,
                            qualification=json.loads(args.qualification.read_text())))
    else:
        if not args.reports:
            parser.error("summarize requires --reports")
        save(args.output, summarize([json.loads(p.read_text()) for p in args.reports], api.manifest["task_ids"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
