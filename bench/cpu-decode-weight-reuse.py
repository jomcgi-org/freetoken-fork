"""Paired resident-weight decode timings; full model wall time remains the gate."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import time

import torch

from freetoken.kernel import _cpu_moe
from freetoken.moe.cpu_executor import CpuMoeExecutor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--threads", type=int, default=14)
    args = parser.parse_args()
    if min(args.repeats, args.threads) < 1:
        parser.error("repeats and threads must be positive")
    if args.output.exists():
        parser.error("refusing to overwrite an earlier benchmark")
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "fixtures", root / "tests/moe/test_cpu_moe_prefill_batch.py",
    )
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    hidden, intermediate, top_k, experts = 2560, 640, 10, 128
    cache = fixtures._make_nvfp4_cache(experts, hidden, intermediate, seed=451)
    executor = CpuMoeExecutor(
        cache, top_k=top_k, activation="silu", apply_router_weight_on_input=False,
        num_threads=args.threads, max_tokens=8, device=torch.device("cpu"),
        prefill_batch="off", step_timing=False,
    )
    extension = executor._ext
    assert extension.decode_weight_reuse_available(), "AVX-512 VNNI is required"
    source = root / "python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp"
    binary = Path(_cpu_moe.__file__).resolve()
    report = dict(
        hidden=hidden, intermediate=intermediate, experts=experts, top_k=top_k,
        threads=args.threads, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        native_path=str(binary), native_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        isa=extension.isa_name(), diagnostics=False, records=[], summaries=[],
        limitation="Synthetic resident weights measure native task time, not model throughput or storage behavior.",
    )
    with args.output.open("x") as out:
        out.write(json.dumps(report, indent=2) + "\n")

    for batch in (1, 2, 4, 8):
        io = executor._io_for(batch)
        task = executor._task_for(0, batch)
        io["x"].copy_(torch.randn(batch, hidden, dtype=torch.bfloat16))
        io["w"].copy_(torch.rand(batch, top_k))
        for active_routes in (1, 4, 10):
            for shared_routes in sorted({0, active_routes // 2, active_routes}):
                ids = torch.full((batch, top_k), -1, dtype=torch.int32)
                for token in range(batch):
                    ids[token, :shared_routes] = torch.arange(shared_routes, dtype=torch.int32)
                    ids[token, shared_routes:active_routes] = (
                        10 + token * 10 + torch.arange(active_routes - shared_routes, dtype=torch.int32)
                    )
                io["ids"].copy_(ids)
                # The exact native task is warmed in both modes before paired timing.
                for enabled in (False, True):
                    extension.set_decode_weight_reuse(enabled)
                    extension.run_task(task)
                samples = {False: [], True: []}
                for repeat in range(args.repeats):
                    outputs = {}
                    pair = []
                    for enabled in ((False, True) if repeat % 2 == 0 else (True, False)):
                        extension.set_decode_weight_reuse(enabled)
                        begin = time.perf_counter_ns()
                        extension.run_task(task)
                        wall_ms = (time.perf_counter_ns() - begin) / 1e6
                        outputs[enabled] = io["y"].clone()
                        samples[enabled].append(wall_ms)
                        row = dict(batch=batch, active_routes=active_routes, shared_routes=shared_routes,
                                   repeat=repeat, reuse=enabled, wall_ms=wall_ms)
                        report["records"].append(row)
                        pair.append(row)
                    equal = torch.equal(outputs[False].view(torch.int16), outputs[True].view(torch.int16))
                    finite = bool(torch.isfinite(outputs[True]).all())
                    for row in pair:
                        row.update(bitwise_equal=equal, finite=finite)
                    if not equal or not finite:
                        report["failure_outputs"] = {str(k): v.float().tolist() for k, v in outputs.items()}
                        args.output.write_text(json.dumps(report, indent=2) + "\n")
                        raise RuntimeError("paired decode outputs differ or are nonfinite")
                before, after = (statistics.median(samples[k]) for k in (False, True))
                summary = dict(batch=batch, active_routes=active_routes, shared_routes=shared_routes,
                               distinct_experts=len(set(ids[ids >= 0].tolist())),
                               reference_median_ms=before, reuse_median_ms=after,
                               wall_reduction_fraction=1 - after / before, bitwise_equal=True)
                report["summaries"].append(summary)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(summary), flush=True)
    report["completed"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
