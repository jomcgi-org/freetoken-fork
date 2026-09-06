"""CPU-layer parity and cost with explicit two-input dot dispatch."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refuse to overwrite an executor result")

    import torch
    from freetoken.kernel import _cpu_moe

    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "pair_executor_inputs", root / "tests/moe/test_nvfp4_pair_dot.py")
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    result = dict(completed=False, serving_wall_qualified=False,
                  revision=subprocess.check_output(
                      ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                  native_sha256=hashlib.sha256(Path(_cpu_moe.__file__).read_bytes()).hexdigest(), cases=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for threads in (1, 14):
        for batch in (1, 2, 5):
            executor = fixtures.make_executor(2560, 640, batch, threads=threads)
            for routes in ("disjoint", "shared", "mixed"):
                x, weights, ids = fixtures.executor_inputs(2560, batch, routes)
                io = executor._io_for(batch)
                io["x"].copy_(x)
                io["ids"].copy_(ids)
                io["w"].copy_(weights)
                task = executor._task_for(0, batch)
                executor._ext.run_task(task)
                reference = io["y"].clone()
                case = dict(threads=threads, batch=batch, routes=routes, iterations=100, samples=[])
                for enabled in (False, True, True, False):
                    if not executor._ext.set_nvfp4_pair_dot(enabled):
                        raise RuntimeError("pair executor requires AVX-512 VNNI")
                    for _ in range(5):
                        executor._ext.run_task(task)
                    start = time.perf_counter()
                    for _ in range(case["iterations"]):
                        executor._ext.run_task(task)
                    elapsed = time.perf_counter() - start
                    if not torch.equal(reference.view(torch.int16), io["y"].view(torch.int16)):
                        raise RuntimeError("complete expert output changed")
                    case["samples"].append(dict(enabled=enabled, elapsed_s=elapsed, exact=True))
                result["cases"].append(case)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
            del executor
    result["completed"] = True
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
