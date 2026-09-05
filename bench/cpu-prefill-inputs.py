"""CPU-only paired NVFP4 batch timings with input reuse on and off.

Run from a source checkout with pytest installed and the CPU extension rebuilt.
Synthetic resident weights isolate CPU work; model wall time is the acceptance
measurement. No diagnostic counters or internal timers are enabled.
"""

import argparse
import hashlib
import importlib.util
import json
import statistics
import time
from pathlib import Path

import torch

from freetoken.moe.cpu_executor import CpuMoeExecutor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--threads", type=int, default=14)
    parser.add_argument("--experts", type=int, default=128)
    args = parser.parse_args()
    if args.repeats < 1 or args.threads < 1 or args.experts < 10:
        parser.error("positive repeats/threads and at least ten experts are required")
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "fixtures", root / "tests/moe/test_cpu_moe_prefill_batch.py",
    )
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    hidden, intermediate, top_k, capacity = 2560, 640, 10, 2048
    cache = fixtures._make_nvfp4_cache(args.experts, hidden, intermediate, seed=4090)
    executor = CpuMoeExecutor(
        cache, top_k=top_k, activation="silu", apply_router_weight_on_input=False,
        num_threads=args.threads, max_tokens=1, max_prefill_tokens=capacity,
        device=torch.device("cpu"), prefill_batch="on", step_timing=False,
    )
    assert executor._prefill_batch_enabled
    extension = executor._ext
    before = extension.prefill_batch_buffer_bytes()
    source = root / "python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp"
    record = dict(
        hidden=hidden, intermediate=intermediate, top_k=top_k,
        threads=args.threads, experts=args.experts, buffer_bytes=before,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), requests=[],
    )
    for tokens in (64, 512, capacity):
        x = torch.randn(tokens, hidden, dtype=torch.bfloat16)
        weights = torch.rand(tokens, top_k)
        y = torch.empty_like(x)
        for routes_per_token in (1, 4, 10):
            ids = torch.randint(0, args.experts, (tokens, top_k), dtype=torch.int32)
            ids[:, :top_k - routes_per_token] = -1
            native_args = (0, tokens, x.data_ptr(), ids.data_ptr(), weights.data_ptr(), y.data_ptr())
            for enabled in (False, True):
                extension.set_prefill_input_reuse(enabled)
                extension.run_prefill_batch_sync(*native_args)
            samples = {False: [], True: []}
            for repeat in range(args.repeats):
                outputs = {}
                for enabled in ((False, True) if repeat % 2 == 0 else (True, False)):
                    extension.set_prefill_input_reuse(enabled)
                    start = time.perf_counter_ns()
                    rows, gemms = extension.run_prefill_batch_sync(*native_args)
                    elapsed_ms = (time.perf_counter_ns() - start) / 1e6
                    assert rows == tokens * routes_per_token
                    samples[enabled].append(elapsed_ms)
                    outputs[enabled] = y.clone()
                    record["requests"].append(dict(
                        tokens=tokens, routes_per_token=routes_per_token,
                        repeat=repeat, reuse=enabled, wall_ms=elapsed_ms,
                        rows=rows, gemms=gemms,
                    ))
                assert torch.isfinite(outputs[True]).all()
                assert torch.equal(outputs[False].view(torch.int16), outputs[True].view(torch.int16))
            assert extension.prefill_batch_buffer_bytes() == before
            print(json.dumps(dict(
                tokens=tokens, routes_per_token=routes_per_token,
                reference_median_ms=statistics.median(samples[False]),
                reuse_median_ms=statistics.median(samples[True]), bitwise_equal=True,
            )), flush=True)
            args.output.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
