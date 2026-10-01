#!/usr/bin/env python3
"""Bandwidth micro-benchmark: BF16 vs per-row FP8 GEMV for the Qwen3.8-Flash-Next projections.

Times, per projection shape (read from the checkpoint's config.json) and decode row count M:
  * bf16  -- ``F.linear`` over the BF16 weight (what ships today, cuBLAS GEMV)
  * fp8   -- ``fp8_dense_linear`` (Triton W8A16 GEMV, default launch plan)
  * best  -- with ``--sweep``, the fastest of the fixed candidate plans (what
             ``FREETOKEN_FP8_GEMV_TUNE=1`` would pick) and its config

and reports microseconds, achieved GB/s over the bytes each variant must read, and the speedup.
Each timing is a CUDA-graph replay of ``--inner`` back-to-back launches over a rotating set of
weight copies large enough to defeat the 72 MB L2, so it measures HBM, not cache.

    python bench/fp8_dense_gemv.py                      # M=1, all shapes
    python bench/fp8_dense_gemv.py --m 1 2 16 --sweep   # plus the plan sweep
    python bench/fp8_dense_gemv.py --shapes gdn.in_proj lm_head --json out.json

Needs a free GPU. Run it with the server stopped (or at least idle): other traffic skews GB/s.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from freetoken.kernel.triton.fp8_dense_linear import (  # noqa: E402
    candidate_plans,
    fp8_dense_gemv,
    fp8_dense_linear,
    plan_gemv,
    quantize_rowwise_fp8,
)
from freetoken.models.dense_fp8 import projection_shapes  # noqa: E402

DEFAULT_MODEL = "/var/lib/longhorn/nvme-02/freetoken/models/flash-e2m1.ftw"
L2_FLUSH_BYTES = 256 << 20  # > the 4090's 72 MB L2, with margin


def time_graph(fn, copies: int, inner: int, reps: int) -> float:
    """Median microseconds per call of ``fn(i)`` (i cycles through ``copies`` weight sets)."""
    for i in range(copies):  # warmup + compile outside capture
        fn(i)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for j in range(inner):
            fn(j % copies)
    graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1000.0 / inner)
    times.sort()
    return times[len(times) // 2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="checkpoint dir with config.json")
    ap.add_argument("--m", type=int, nargs="+", default=[1], help="decode row counts (<= 32)")
    ap.add_argument("--shapes", nargs="*", help="subset of projection names (default: all)")
    ap.add_argument("--sweep", action="store_true", help="also time every candidate GEMV plan")
    ap.add_argument("--inner", type=int, default=20, help="launches per graph replay")
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--json", help="write the rows to this file")
    args = ap.parse_args()

    with open(os.path.join(args.model, "config.json")) as f:
        shapes = projection_shapes(json.load(f)["text_config"])
    names = args.shapes or list(shapes)
    props = torch.cuda.get_device_properties(0)
    print(f"{props.name}: {props.multi_processor_count} SMs; torch {torch.__version__}")
    print(f"{'shape':28s} {'N':>7s} {'K':>6s} {'M':>3s} | {'bf16 us':>8s} {'GB/s':>6s} | "
          f"{'fp8 us':>8s} {'GB/s':>6s} {'speedup':>7s}" + (" | best us  config" if args.sweep else ""))
    rows = []
    for name in names:
        n, k = shapes[name]
        bf16_bytes, fp8_bytes = 2 * n * k, n * k + 4 * n
        copies = max(2, -(-L2_FLUSH_BYTES // bf16_bytes))
        w = [torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.03 for _ in range(copies)]
        q = [quantize_rowwise_fp8(t) for t in w]
        for m in args.m:
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            t_bf16 = time_graph(lambda i: F.linear(x, w[i]), copies, args.inner, args.reps)
            t_fp8 = time_graph(
                lambda i: fp8_dense_linear(x, q[i][0], q[i][1]), copies, args.inner, args.reps
            )
            row = {
                "shape": name, "N": n, "K": k, "M": m, "bf16_us": t_bf16, "fp8_us": t_fp8,
                "bf16_gbps": bf16_bytes / t_bf16 / 1e3, "fp8_gbps": fp8_bytes / t_fp8 / 1e3,
                "speedup": t_bf16 / t_fp8, "plan": str(plan_gemv(m, n, k, sm_count=props.multi_processor_count)),
            }
            line = (f"{name:28s} {n:7d} {k:6d} {m:3d} | {t_bf16:8.1f} {row['bf16_gbps']:6.0f} | "
                    f"{t_fp8:8.1f} {row['fp8_gbps']:6.0f} {row['speedup']:6.2f}x")
            if args.sweep:
                best = min(
                    (
                        (time_graph(lambda i, p=p: fp8_dense_gemv(x, q[i][0], q[i][1], p),
                                    copies, args.inner, 7), p)
                        for p in candidate_plans(m, n, k, sm_count=props.multi_processor_count)
                    ),
                    key=lambda t: t[0],
                )
                row["best_us"], row["best_plan"] = best[0], str(best[1])
                line += (f" | {best[0]:7.1f}  bn={best[1].block_n} bk={best[1].block_k} "
                         f"split={best[1].split_k} w={best[1].num_warps}")
            print(line, flush=True)
            rows.append(row)
        del w, q
        torch.cuda.empty_cache()

    if len(args.m) and 1 in args.m:
        step = [r for r in rows if r["M"] == 1 and r["shape"] != "lm_head"]
        print("\nM=1 only: per-shape speedups above; the layer counts per step are in the PR "
              "(qkv x12, o_proj x12, in_proj x36, out_proj x36, shared x48, hc x96+2, lm_head x1).")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
