"""Compare direct pinned NVFP4 gathers with the existing bounded staging path.

Synthetic resident banks isolate transfer cost. This does not measure model
throughput, RAM reclamation, cold storage latency or a new placement policy.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import time

import torch


HIDDEN, INTERMEDIATE, EXPERTS, CAPACITY = 2560, 640, 512, 40
STRIDE = 73  # coprime to 512: every source row is visited before repeating
MISSES = (0, 1, 2, 5, 10, 20, 40)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, report):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def make_sources(directory):
    from freetoken.moe.benchbw import _cpu_moe_bank_sources

    torch.manual_seed(451)
    pinned = _cpu_moe_bank_sources('nvfp4', HIDDEN, INTERMEDIATE, EXPERTS)
    mapped = {}
    for name, source in pinned.items():
        if name.endswith('_packed'):
            source.random_(0, 256)
        elif name.endswith('_scale'):
            source.fill_(0x38)  # finite unit e4m3 scale
        else:
            values = (torch.arange(EXPERTS) % 7).float() * 0.125 + 0.5
            source.copy_(values[:, None].expand_as(source))
        path = directory / (name + '.bin')
        with path.open('xb') as output:
            output.write(memoryview(source.numpy()).cast('B'))
            output.flush()
            os.fsync(output.fileno())
        mapped[name] = torch.from_file(str(path), shared=False, size=source.numel(),
                                      dtype=source.dtype).reshape(source.shape)
        # Populate page-table entries before measurement. The file remains
        # mapped and resident; disk faults are outside this transfer-cost probe.
        mapped[name].view(torch.uint8).reshape(-1)[::4096].sum().item()
    return pinned, mapped


def make_cache(sources, *, staged, threads):
    from freetoken.moe.cpu_executor import CpuMoeExecutor
    from freetoken.moe.offload_cache import OffloadMoeCache

    cache = OffloadMoeCache(
        # The production cache retains its full-layer allocation floor. Only
        # CAPACITY destination rows participate in this decode-copy probe.
        num_layers=1, num_experts=EXPERTS, cache_size=EXPERTS,
        device=torch.device('cuda'), quant_format='nvfp4', decode_target='gpu',
        prefill_overlap=False, moe_disk_decode='gpufetch' if staged else 'cpu',
        collect_stats=False,
    )
    cache.set_bank_sources({name: [source] for name, source in sources.items()},
                           layer_residency=['disk' if staged else 'pinned'])
    executor = None
    if staged:
        executor = CpuMoeExecutor(
            cache, top_k=10, activation='silu', apply_router_weight_on_input=False,
            num_threads=threads, max_tokens=4, device=torch.device('cuda'),
            disk_lookahead=False, step_timing=False, moe_cpu_willneed='off',
            prefill_coalesce=False, prefill_batch='off',
        )
        cache.set_cpu_executor(executor)
        cache.init_disk_gpufetch(executor, max_tokens=4, top_k=10)
        assert cache._gpufetch_capacity == CAPACITY and cache._gpufetch_fused_ok
    else:
        assert cache._copy_fused_ok
    cache._pending_src_layer = 0
    cache._pending_whole_layer = False
    cache.evict_slots[:CAPACITY].copy_(torch.arange(CAPACITY - 1, -1, -1,
                                                  device='cuda', dtype=torch.int32))
    cache.src_indices.zero_()
    cache.num_indices.zero_()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(3):
            cache.copy_missing()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        # Both paths receive the same rotating GPU-resident miss IDs. The
        # staging path additionally performs its real D2H/coordinator/H2D work.
        cache.src_indices[:CAPACITY].add_(STRIDE).remainder_(EXPERTS)
        cache.copy_missing()
    return cache, executor, graph


def prepare(cache, missing, offset):
    ids = (torch.arange(CAPACITY, dtype=torch.int32) * 17 + offset) % EXPERTS
    cache.src_indices[:CAPACITY].copy_(ids)
    cache.num_indices.fill_(missing)
    for _sources, destination in cache.banks:
        destination.zero_()
    torch.cuda.synchronize()
    return ids


def check_bytes(cache, pinned, missing, initial_ids, steps):
    expected_ids = (initial_ids + steps * STRIDE) % EXPERTS
    assert torch.equal(cache.src_indices[:CAPACITY].cpu(), expected_ids), 'source sequence changed'
    destination_ids = torch.arange(CAPACITY - 1, CAPACITY - missing - 1, -1)
    for name, destination in cache.bank_caches.items():
        expected = torch.zeros((CAPACITY, *destination.shape[1:]), dtype=destination.dtype)
        if missing:
            expected.index_copy_(0, destination_ids,
                                 pinned[name].index_select(0, expected_ids[:missing].long()))
        actual = destination[:CAPACITY].cpu()
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)), (
            'copied or untouched weight bytes differ', name, missing)
        assert torch.count_nonzero(destination[CAPACITY:].view(torch.uint8)).item() == 0, (
            'copy wrote beyond the destination window', name, missing)


def run(args, report):
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.kernel import _cpu_moe
    from freetoken.moe import cpu_executor, offload_cache

    gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                   '--format=csv,noheader,nounits'], text=True).strip()
    if {int(pid) for pid in gpu.splitlines() if pid.strip()} - {os.getpid()}:
        raise RuntimeError('GPU is occupied; run under an exclusive supervisor with serving recovery')
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    if try_get_tp_info() is None:
        set_tp_info(0, 1)
    runtime_root = Path(offload_cache.__file__).resolve().parents[3]
    dirty = subprocess.check_output(['git', '-C', str(runtime_root), 'status', '--porcelain'], text=True)
    assert not dirty.strip(), 'runtime worktree is dirty'
    report.update(runtime_revision=subprocess.check_output(
        ['git', '-C', str(runtime_root), 'rev-parse', 'HEAD'], text=True).strip(),
        runtime_root=str(runtime_root), native_path=str(Path(_cpu_moe.__file__).resolve()),
        native_sha256=sha(_cpu_moe.__file__), controller_sha256=sha(__file__),
        sources={name: sha(path) for name, path in {
            'offload_cache.py': offload_cache.__file__, 'cpu_executor.py': cpu_executor.__file__,
            'cpu_moe_ext.cpp': runtime_root / 'python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp',
        }.items()})
    with tempfile.TemporaryDirectory(prefix='astra-gpu-source-cost-', dir=args.scratch_dir) as directory:
        pinned, mapped = make_sources(Path(directory))
        rigs = {'pinned': make_cache(pinned, staged=False, threads=args.threads),
                'staged': make_cache(mapped, staged=True, threads=args.threads)}
        executor = rigs['staged'][1]
        report.update(
            source_bytes_per_mode=sum(t.numel() * t.element_size() for t in pinned.values()),
            bytes_per_expert=sum(t[0].numel() * t.element_size() for t in pinned.values()),
            pinned_ring_bytes=sum(t.numel() * t.element_size() for t in rigs['staged'][0]._gpufetch_staging),
            native_transport='memops' if executor._gpufetch_tasks[0][1] is not None else 'host_callback',
        )
        save(args.output, report)
        for missing in MISSES:
            for cache, _executor, graph in rigs.values():
                initial = prepare(cache, missing, 0)
                for _ in range(args.warmup):
                    graph.replay()
                torch.cuda.synchronize()
                check_bytes(cache, pinned, missing, initial, args.warmup)
            samples = {'pinned': [], 'staged': []}
            for repeat in range(args.repeats):
                order = ('pinned', 'staged') if repeat % 2 == 0 else ('staged', 'pinned')
                for mode in order:
                    cache, native, graph = rigs[mode]
                    initial = prepare(cache, missing, repeat * args.steps * STRIDE)
                    if native is not None:
                        native._ext.gpufetch_stats(True)
                    started = time.perf_counter()
                    for _ in range(args.steps):
                        graph.replay()
                    torch.cuda.synchronize()
                    wall = time.perf_counter() - started
                    row = dict(missing=missing, repeat=repeat, order=list(order), mode=mode,
                               steps=args.steps, wall_s=wall, wall_ms_per_copy=wall * 1000 / args.steps,
                               bytes_verified=False)
                    report['records'].append(row)
                    save(args.output, report)
                    check_bytes(cache, pinned, missing, initial, args.steps)
                    if native is not None:
                        fills, calls, _timing = native._ext.gpufetch_stats(True)
                        assert native._ext.gpufetch_error_code() == 0, 'native staging error'
                        assert fills == missing * args.steps and calls == args.steps, 'unexpected staging work'
                    row['bytes_verified'] = True
                    samples[mode].append(row['wall_ms_per_copy'])
                    save(args.output, report)
            direct, staged = (statistics.median(samples[m]) for m in ('pinned', 'staged'))
            summary = dict(missing=missing, pinned_median_ms=direct, staged_median_ms=staged,
                           added_ms_per_copy=staged - direct, bytes_verified=True)
            report['summaries'].append(summary)
            save(args.output, report)
            print(json.dumps(summary), flush=True)
        assert sha(_cpu_moe.__file__) == report['native_sha256'], 'native binary changed during measurement'
        report['completed'] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scratch-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--warmup', type=int, default=64)
    parser.add_argument('--repeats', type=int, default=8)
    parser.add_argument('--threads', type=int, default=14)
    args = parser.parse_args()
    if min(args.steps, args.warmup, args.repeats, args.threads) < 1 or args.repeats % 2:
        parser.error('counts must be positive and repeats even to balance both orders')
    if not args.scratch_dir.is_dir():
        parser.error('scratch directory must already exist')
    report = dict(completed=False, hidden=HIDDEN, intermediate=INTERMEDIATE, experts=EXPERTS,
                  capacity=CAPACITY, gpu_cache_rows=EXPERTS, misses=list(MISSES),
                  steps=args.steps, warmup=args.warmup,
                  repeats=args.repeats, threads=args.threads, cuda_graph=True,
                  new_gpu_timing_events=False, records=[], summaries=[],
                  limitation='Resident synthetic transfer cost only. No model throughput, cold-I/O, '
                             'RAM-reclamation or placement-quality claim.')
    # Exclusive creation retains every attempt instead of overwriting failed runs.
    with args.output.open('x') as output:
        output.write(json.dumps(report, indent=2) + '\n')
    try:
        run(args, report)
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        save(args.output, report)


if __name__ == '__main__':
    main()
