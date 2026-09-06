"""Explicit CPU-only cost probe; does not change serving dispatch."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cpu", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refuse to overwrite a probe result")
    os.sched_setaffinity(0, {args.cpu})

    import torch
    from freetoken.kernel import _cpu_moe

    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    result = dict(completed=False, serving_wall_qualified=False, cpu=args.cpu,
                  revision=subprocess.check_output(
                      ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                  native_sha256=hashlib.sha256(Path(_cpu_moe.__file__).read_bytes()).hexdigest(),
                  prefetch_blocks=os.environ.get("FREETOKEN_CPU_MOE_PF_BLOCKS"), cases=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Small tiles, complete expert matrices and a weight pool larger than LLC.
    for hidden, rows, iterations in [(2560, 32, 1000), (640, 32, 1000),
                                      (2560, 1280, 50), (640, 2560, 50),
                                      (2560, 131072, 2)]:
        rng = torch.Generator().manual_seed(4090 + hidden + rows)
        packed = torch.randint(0, 256, (rows, hidden // 2), dtype=torch.uint8, generator=rng)
        scales = torch.randint(0, 127, (rows, hidden // 16), dtype=torch.uint8, generator=rng)
        globals_ = (torch.rand(rows, generator=rng) * .04).half().float()
        acts = torch.randint(-127, 128, (2, hidden), dtype=torch.int8, generator=rng)
        act_scales = torch.rand(2, hidden // 16, generator=rng) * .03
        case = dict(hidden=hidden, rows=rows, iterations=iterations, samples=[])
        for pair_first in (False, True, True, False):
            measured = _cpu_moe.nvfp4_pair_dot_probe(
                packed, scales, globals_, acts, act_scales, iterations, pair_first)
            exact = torch.equal(measured["paired"].view(torch.int32),
                                measured["singles"].view(torch.int32))
            if not exact or not torch.isfinite(measured["singles"]).all():
                raise RuntimeError("pair dot differs from ordinary decode")
            case["samples"].append(dict(pair_first=pair_first, exact=exact,
                                        paired_s=measured["paired_s"], singles_s=measured["singles_s"]))
        result["cases"].append(case)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    result["completed"] = True
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
