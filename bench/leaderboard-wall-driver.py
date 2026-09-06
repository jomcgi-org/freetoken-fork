"""Owned Linux supervisor for the original/selected leaderboard comparison."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time


UNIT = "astra-leaderboard-wall"


def load_guard():
    path = Path(__file__).with_name("pi-decode-prefix-wall-driver.py")
    spec = importlib.util.spec_from_file_location("leaderboard_guard", path)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    guard.ORIGINAL_BASELINE = True
    return guard


def stop_child(child):
    if child is not None and child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=90)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=15)


def require_owned_unit(state, pid):
    if state.get("ActiveState") != "active" or int(state.get("MainPID", 0)) != pid:
        raise RuntimeError("run must be the main process of the owned systemd unit")


def preflight(args, guard):
    if args.output.exists():
        raise RuntimeError("refuse to rerun an existing experiment")
    if guard.live("astra-target-verify-cost"):
        raise RuntimeError("another GPU experiment is live")
    result = guard.preflight()
    tree = Path(__file__).resolve().parents[1]
    if guard.run("git", "-C", str(tree), "status", "--porcelain").stdout.strip():
        raise RuntimeError("benchmark source must be clean")
    result["benchmark_revision"] = guard.run("git", "-C", str(tree), "rev-parse", "HEAD").stdout.strip()
    result["benchmark_tree"] = str(tree)
    result["go_sha256"] = guard.sha(args.go_bin / "go")
    result["manifest_sha256"] = guard.sha(args.bundle / "manifest.json")
    qualification = json.loads(args.qualification.read_text())
    if (not qualification.get("completed") or not qualification.get("checks_passed")
            or qualification.get("manifest_sha256") != result["manifest_sha256"]):
        raise RuntimeError("matching fail-to-pass grading qualification is required")
    # Publicly reconstructed keys are independently compared to the private
    # original cells by the controller before launching this supervisor.
    if not args.keys_verified:
        raise RuntimeError("controller must verify original input keys before launch")
    return result


def qualify_worker(guard, mode, plan, text):
    workers = guard.gpu_pids()
    if len(workers) != 1:
        raise RuntimeError("benchmark must own the only GPU worker")
    worker = workers[0]
    unit = guard.state(UNIT)
    if unit["ControlGroup"] not in Path(f"/proc/{worker}/cgroup").read_text():
        raise RuntimeError("GPU worker escaped the owned unit")
    maps = Path(f"/proc/{worker}/maps").read_text().splitlines()
    for name, identity in plan["identities"][mode]["native_extensions"].items():
        loaded = [line.split()[-1] for line in maps if "/" + name + "." in line]
        if name in ("_cpu_moe", "_ple_uring") and not loaded:
            raise RuntimeError("required native extension is not mapped: " + name)
        if any(path != identity["path"] for path in loaded):
            raise RuntimeError("native extension mapping changed: " + name)
    env = dict(item.split(b"=", 1) for item in Path(f"/proc/{worker}/environ").read_bytes().split(b"\0") if b"=" in item)
    if any(env.get(k.encode()) != v.encode() for k, v in guard.server_env(mode).items()):
        raise RuntimeError("worker environment differs from the qualified runtime")
    for marker in ("moe_collect_stats=False", "moe_step_timing=False", "speculative_mtp='off'",
                   "cache_type='radix'", "kv_ladder='off'", "kv_disk_cache_gib=0.0"):
        if marker not in text:
            raise RuntimeError("serving qualification marker is missing: " + marker)
    return worker, guard.geometry(text)


def run_experiment(args, guard):
    require_owned_unit(guard.state(UNIT), os.getpid())
    plan = preflight(args, guard)
    args.output.mkdir(mode=0o700)
    report = dict(completed=False, plan=plan, arms=[], model_wall_qualified=False)
    path = args.output / "report.json"
    client = Path(__file__).with_name("leaderboard-wall.py")
    python = guard.SRC / ".venv/bin/python"
    server = None

    def save():
        guard.save(path, report)
        path.chmod(0o600)

    def interrupted(signum, frame):
        raise RuntimeError("owned benchmark supervisor interrupted")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    save()
    try:
        guard.stop("freetoken-serve")
        guard.wait_gpu_release()
        for arm, mode in guard.ARMS:
            if guard.identities() != plan["identities"] or guard.sha(guard.SERVICE) != plan["service_sha256"]:
                raise RuntimeError("serving identity changed during the experiment")
            if guard.sha(args.bundle / "manifest.json") != plan["manifest_sha256"]:
                raise RuntimeError("benchmark manifest changed during the experiment")
            if guard.sha(args.go_bin / "go") != plan["go_sha256"]:
                raise RuntimeError("grading toolchain changed during the experiment")
            row = dict(arm=arm, mode=mode, stage="loading model")
            report["arms"].append(row)
            save()
            env = dict(os.environ, **guard.server_env(mode))
            env.pop("CUDA_VISIBLE_DEVICES", None)
            for name in list(env):
                if name.startswith("FREETOKEN_TARGET_VERIFY_") or name == "FREETOKEN_COMPACT_ROLLBACK_CUDA_TEST":
                    del env[name]
            log_path = args.output / (arm + "-server.log")
            with log_path.open("w") as log:
                log_path.chmod(0o600)
                server = subprocess.Popen(guard.server_command(mode), cwd=guard.runtime_tree(mode),
                                          env=env, stdout=log, stderr=subprocess.STDOUT)
                row["server_pid"] = server.pid
                save()
                try:
                    row["health"] = guard.ready(18090)
                    guard.completion(18090)
                    row["completion_ok_verified"] = True
                    worker, geometry = qualify_worker(guard, mode, plan, log_path.read_text())
                    if report["arms"][0].get("geometry", geometry) != geometry:
                        raise RuntimeError("serving geometry differs between arms")
                    row.update(worker=worker, geometry=geometry, stage="running leaderboard tasks",
                               memory_before=guard.memory_snapshot(worker), io_before=guard.io_snapshot(worker))
                    save()
                    client_env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(guard.R / "tmp"),
                                      PATH=str(args.go_bin) + ":" + str(python.parent) + ":/usr/bin:/bin")
                    output = args.output / (arm + ".json")
                    command = [str(python), str(client), "run", "--bundle", str(args.bundle),
                               "--qualification", str(args.qualification), "--arm", arm,
                               "--output", str(output)]
                    started = time.perf_counter()
                    with (args.output / (arm + "-client.log")).open("w") as stream:
                        result = subprocess.run(command, cwd=client.parents[1], env=client_env,
                                                stdout=stream, stderr=subprocess.STDOUT, timeout=3600)
                    row["client_process_wall_s"] = time.perf_counter() - started
                    row["client_returncode"] = result.returncode
                    row.update(memory_after=guard.memory_snapshot(worker), io_after=guard.io_snapshot(worker))
                    if result.returncode != 0:
                        raise RuntimeError("leaderboard client failed; inspect its private log")
                    recorded = json.loads(output.read_text())
                    if not recorded.get("completed"):
                        raise RuntimeError("leaderboard arm did not complete")
                    row.update(stage="completed", all_tasks_passed=recorded["all_tasks_passed"])
                    save()
                finally:
                    stop_child(server)
                    row["server_returncode"] = server.returncode
                    server = None
                    save()
            guard.wait_gpu_release()
        command = [str(python), str(client), "summarize", "--bundle", str(args.bundle),
                   "--output", str(args.output / "comparison.json"), "--reports"]
        command += [str(args.output / (arm + ".json")) for arm, _ in guard.ARMS]
        subprocess.run(command, check=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        report["completed"] = True
        report["all_tasks_passed"] = all(row["all_tasks_passed"] for row in report["arms"])
        # Final recovery and journal auditing are separate qualification gates.
        save()
    except BaseException as exc:
        report["error"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        stop_child(server)
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "run"))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--go-bin", type=Path, required=True)
    parser.add_argument("--keys-verified", action="store_true")
    args = parser.parse_args()
    guard = load_guard()
    if args.action == "preflight":
        if guard.live(UNIT):
            raise RuntimeError("owned benchmark unit is already live")
        result = preflight(args, guard)
        print(json.dumps(dict(qualified=True, benchmark_revision=result["benchmark_revision"])))
    else:
        run_experiment(args, guard)


if __name__ == "__main__":
    main()
