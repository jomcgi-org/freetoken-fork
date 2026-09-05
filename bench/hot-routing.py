"""Kernel-only HOT routing benchmark at Qwen/4090 geometry; use wall time to accept.

CUDA events are confined to this diagnostic benchmark. The measured operation
includes copying the original route IDs back before their in-place remapping.
"""

import argparse
import json
import statistics

import torch

from freetoken.moe import offload_kernels as kernels
from freetoken.moe.offload_cache import OffloadMoeCache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cache = OffloadMoeCache(
        num_layers=1, num_experts=512, cache_size=3753, device=torch.device("cuda"),
    )
    rng = torch.Generator().manual_seed(4090)
    hot = torch.randperm(512, generator=rng)[:82].cuda()
    slots = torch.arange(1024, 1106, device="cuda", dtype=torch.int32)
    cache.hot_row_for_expert[0, hot] = torch.arange(82, device="cuda", dtype=torch.int32)
    cache.slot_for_id[0, hot] = slots
    cache.id_of_slot[slots.long()] = hot.to(torch.int32)
    cache.hot_adapt_enabled = True
    cache._hot_decay_factor = 0.9996534864594093
    cache.collect_stats = False
    rows = []
    for count in (10, 40, 64, 160, 640, 2560, 5120, 20480):
        raw = torch.randint(0, 512, (count,), generator=rng, dtype=torch.int32).cuda()
        ids = torch.empty_like(raw)
        for parallel in (False, True):
            kernels._PARALLEL_HOT_ROUTING = parallel

            def step():
                ids.copy_(raw)
                kernels.ensure_experts_hot(cache, 0, ids, route_weight=0.1)

            for _ in range(3):
                step()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(10):
                    step()
            samples = []
            for _ in range(9):
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
                graph.replay()
                end.record()
                end.synchronize()
                samples.append(begin.elapsed_time(end) * 1000 / 10)
            row = dict(routes=count, parallel=parallel, median_us=statistics.median(samples))
            rows.append(row)
            print(json.dumps(row), flush=True)
    with open(args.output, "w") as out:
        json.dump(rows, out, indent=2)
        out.write("\n")


if __name__ == "__main__":
    main()
