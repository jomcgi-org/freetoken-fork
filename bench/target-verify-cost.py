"""Explicit, destructive-to-the-test-worker target-only K=1 cost probe.

Install the import hook in an isolated worker with external serving recovery.
This measures components, never serving throughput. The worker exits after the
report because repeated forwards alter cache history and adaptation counters.
"""

import copy
import importlib.machinery
import json
import os
from pathlib import Path
import statistics
import sys
import time


OUTPUT_ENV = "FREETOKEN_TARGET_VERIFY_COST_DIR"
MODES = ("graph_one", "graph_two", "snapshot", "accept", "reject")


def eligible_position(position, *, page_size, ratio, remaining):
    """Leave room for three adjacent windows within an already allocated page."""
    return (
        ratio >= 2 and page_size % ratio == 0 and remaining >= 5
        and position % ratio == ratio - 2
        and position // page_size == (position + 3) // page_size
    )


def summarize(records):
    """Keep correctness independent of component costs and never claim a speedup."""
    groups = {}
    for record in records:
        if not record["warmup"]:
            groups.setdefault(record["case"], {}).setdefault(record["mode"], []).append(record)
    result = {}
    for case, modes in groups.items():
        complete = set(modes) == set(MODES)
        valid = complete and all(r["checks_passed"] for r in records if r["case"] == case)
        medians = {name: statistics.median(r["wall_s"] for r in rows)
                   for name, rows in modes.items()}
        row = dict(complete=complete, checks_passed=valid, median_component_s=medians,
                   model_wall_qualified=False)
        if valid:
            one, accepted, rejected = (medians[k] for k in ("graph_one", "accept", "reject"))
            denominator = rejected - accepted + one
            # p*A + (1-p)*R < (1+p)*D. Proposer and scheduler work is additional.
            row["break_even_acceptance_excluding_proposer"] = (
                (rejected - one) / denominator if denominator > 0 else None
            )
        result[case] = row
    return result


def save(directory, report):
    path = directory / (str(os.getpid()) + ".json")
    temporary = path.with_suffix(".tmp")
    with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as out:
        json.dump(report, out, indent=2, allow_nan=False)
        out.write("\n")
    temporary.replace(path)


def state_views(engine, req):
    from freetoken.spec_decode import request_state_views

    raw = request_state_views(engine.linear_state_pool, engine.kv_cache, req)
    views = {k: v for k, v in raw.items() if k not in ("slot", "slot_states")}
    views.update({"slot/" + k: v for k, v in raw.get("slot_states", {}).items()})
    return views


def bytes_equal(left, right):
    import torch

    return (left.shape == right.shape and left.dtype == right.dtype
            and torch.equal(left.contiguous().view(torch.uint8),
                            right.contiguous().view(torch.uint8)))


class StateWindow:
    """Reset trials fully; compare only state reachable at the committed length.

    KV and compressed rows beyond that length may contain rejected writes. They
    are deliberately NOT restored on the measured rejection path. Following the
    rejection with another ordinary decode checks that those rows are overwritten.
    """

    def __init__(self, engine, req, position):
        self.engine, self.req = engine, req
        self.page_size = engine.config.page_size
        self.ratio = engine.kv_cache.index_ratio
        self.logical_page = position // self.page_size
        self.base = self.logical_page * self.page_size
        page_row = engine.page_table[req.table_idx, self.base:self.base + self.page_size]
        physical = page_row.cpu().tolist()
        if (len(physical) != self.page_size or physical[0] % self.page_size
                or physical != list(range(physical[0], physical[0] + self.page_size))):
            raise RuntimeError("probe window is not in a fully allocated contiguous page")
        page = physical[0] // self.page_size
        if page >= engine.num_pages:
            raise RuntimeError("probe must not use the dummy KV page")
        self.locations = page_row[position - self.base:position - self.base + 2].clone()
        self.views = state_views(engine, req)
        self.views["kv_page"] = engine.kv_cache._kv_buffer.select(2, page)
        begin = physical[0] // self.ratio
        end = begin + self.page_size // self.ratio
        self.views["cmp_page"] = engine.kv_cache._cmp_k_buffer[:, begin:end]
        scratch = engine.kv_cache.cmp_scratch_base + req.table_idx
        self.views["cmp_scratch"] = engine.kv_cache._cmp_k_buffer[:, scratch]

    def capture(self):
        return {name: tensor.clone() for name, tensor in self.views.items()}

    def reset(self, snapshot):
        for name, tensor in self.views.items():
            tensor.copy_(snapshot[name])

    def compare(self, expected, *, committed_end):
        mismatches = []
        for name, actual in self.views.items():
            wanted = expected[name]
            if name == "cmp_scratch":
                continue  # A sink, never an attention input.
            if name == "kv_page":
                count = committed_end - self.base
                actual, wanted = actual[:, :, :count], wanted[:, :, :count]
            elif name == "cmp_page":
                count = (committed_end - self.base) // self.ratio
                actual, wanted = actual[:, :count], wanted[:, :count]
            if not bytes_equal(actual, wanted):
                mismatches.append(name)
        return mismatches


def run_window(engine, source_batch, position, seed, host_prefix, *, repeats, warmup, report, directory):
    import torch
    from freetoken.attention.linear import build_fla_metadata
    from freetoken.core import Batch
    from freetoken.spec_decode import (
        configure_mtp_decode_step, configure_mtp_fused_step, greedy_accept_prefix,
        restore_verify_state, snapshot_verify_state,
    )

    req = copy.copy(source_batch.reqs[0])
    batch = Batch(reqs=[req], phase="decode")
    batch.padded_reqs = [req]
    batch.linear_table_idx = source_batch.linear_table_idx.clone()
    batch.active_table_idx = source_batch.active_table_idx.clone()
    batch.mtp_original_cached_len = position
    batch.mtp_original_device_len = position + 1
    positions = torch.arange(position, position + 2, device=engine.device, dtype=torch.int32)
    window = StateWindow(engine, req, position)
    initial = window.capture()
    verify_ids = torch.cat((seed.reshape(1), seed.reshape(1))).to(torch.int32)
    host_seed = seed.cpu().reshape(1).to(host_prefix.dtype)
    req.input_ids = torch.cat((host_prefix, host_seed, host_seed))

    def forward(step=None):
        if step is None:
            configure_mtp_fused_step(batch, verify_ids, positions, window.locations)
        else:
            configure_mtp_decode_step(batch, verify_ids, positions, window.locations, step)
        batch.fla_metadata = build_fla_metadata(batch, engine.device)
        engine.attn_backend.prepare_metadata(batch)
        engine.cpu_moe_executor.begin_decode_step()
        with engine.ctx.forward_batch(batch):
            if step is None:
                logits = engine.model.forward(select_last=False)
            else:
                if not engine.graph_runner.can_use_cuda_graph(batch):
                    raise RuntimeError("ordinary CUDA graph baseline is unavailable")
                logits = engine.graph_runner.replay(batch)
        engine.cpu_moe_executor.raise_if_unhealthy()
        return torch.argmax(logits, dim=-1).to(torch.int32)

    # Derive an actual accepted candidate, and independent one/two-step state references.
    first = forward(0)
    expected_one = window.capture()
    verify_ids[1].copy_(first[0])
    req.input_ids[-1:].copy_(first.cpu().to(req.input_ids.dtype))
    second = forward(1)
    expected_two = window.capture()
    good_ids = verify_ids.clone()
    good_history = req.input_ids.clone()
    expected_tokens = torch.cat((first, second))
    wrong_id = (int(first.item()) + 1) % engine.config.model_config.vocab_size
    pool, kv = engine.linear_state_pool, engine.kv_cache
    case = str(position)

    def execute(mode):
        if mode == "graph_one":
            return forward(0), None
        if mode == "graph_two":
            return torch.cat((forward(0), forward(1))), None
        saved = snapshot_verify_state(pool, kv, req)
        if mode == "snapshot":
            return saved, None
        targets = forward()
        accepted, matched = greedy_accept_prefix(verify_ids[1:], targets)
        if mode == "reject":
            # Deliberately leave future KV/index writes in place, as production does.
            restore_verify_state(pool, kv, req, saved)
            return forward(0), matched
        return accepted, matched

    for repeat in range(-warmup, repeats):
        order = MODES if repeat % 2 == 0 else tuple(reversed(MODES))
        for mode in order:
            window.reset(initial)
            verify_ids.copy_(good_ids)
            req.input_ids.copy_(good_history)
            if mode == "reject":
                verify_ids[1] = wrong_id
                req.input_ids[-1] = wrong_id
            # Reset and diagnostic copies are outside the timed component window.
            engine.stream.synchronize()
            started = time.perf_counter()
            output, matched = execute(mode)
            engine.stream.synchronize()
            elapsed = time.perf_counter() - started
            mismatches = []
            if mode == "snapshot":
                mismatches = window.compare(initial, committed_end=position)
                snapshot_views = state_views(engine, req)
                flat = {k: v for k, v in output.items() if k not in ("slot", "slot_states")}
                flat.update({"slot/" + k: v for k, v in output.get("slot_states", {}).items()})
                mismatches += ["snapshot/" + k for k in snapshot_views
                               if not bytes_equal(snapshot_views[k], flat[k])]
            else:
                count = 2 if mode in ("graph_two", "accept") else 1
                if not bytes_equal(output, expected_tokens[:count]):
                    mismatches.append("greedy_tokens")
                mismatches += window.compare(expected_two if count == 2 else expected_one,
                                             committed_end=position + count)
                if mode in ("accept", "reject") and matched != (mode == "accept"):
                    mismatches.append("proposal_match")
                if mode == "reject":
                    verify_ids.copy_(good_ids)
                    req.input_ids.copy_(good_history)
                    continued = forward(1)
                    if not bytes_equal(continued, second):
                        mismatches.append("post_reject_token")
                    mismatches += ["post_reject/" + k for k in
                                   window.compare(expected_two, committed_end=position + 2)]
            row = dict(case=case, position=position, compression_offset=position % window.ratio,
                       mode=mode, repeat=repeat,
                       warmup=repeat < 0, wall_s=elapsed, matched=matched,
                       checks_passed=not mismatches, mismatches=mismatches)
            report["records"].append(row)
            save(directory, report)
    # Advance only through the independent ordinary-graph reference for the next case.
    window.reset(expected_one)
    engine.stream.synchronize()
    return first, torch.cat((host_prefix, host_seed))


def probe(engine, batch, directory, *, repeats=4, warmup=1):
    import torch
    from freetoken.kernel import _cpu_moe
    import hashlib
    import subprocess

    report = dict(diagnostic_only=True, model_wall_qualified=False, completed=False,
                  records=[], pid=os.getpid(), speculative_mtp=engine.config.speculative_mtp,
                  graph_sizes=sorted(engine.graph_runner.graph_map),
                  cpu_max_tokens=engine.cpu_moe_executor.max_tokens,
                  ring_capacity=engine.kv_cache.ring_capacity,
                  index_ratio=engine.kv_cache.index_ratio,
                  native_sha256=hashlib.sha256(Path(_cpu_moe.__file__).read_bytes()).hexdigest(),
                  limitations=["Component costs exclude proposer and scheduler work",
                               "Repeated windows warm expert and file caches",
                               "All arms reserve CPU rows for two tokens and a speculative QSA ring",
                               "Exact local state checks are not broad quality evaluation",
                               "No service continuation is permitted after the probe"])
    try:
        root = Path(__file__).resolve().parents[1]
        runtime_root = Path(sys.modules[type(engine).__module__].__file__).resolve().parents[3]
        if runtime_root != root:
            raise RuntimeError("probe and runtime must come from the same worktree")
        report["runtime_root"] = str(runtime_root)
        report["source_revision"] = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        if subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip():
            raise RuntimeError("probe source worktree is dirty")
        engine.stream.synchronize()
        req = batch.reqs[0]
        position = req.cached_len
        seed = batch.input_ids[:1].clone()
        prefix = req.input_ids[:position].clone()
        if len(prefix) != position:
            raise RuntimeError("host history does not reach probe position")
        for offset in range(3):
            seed, prefix = run_window(engine, batch, position + offset, seed, prefix,
                                      repeats=repeats, warmup=warmup,
                                      report=report, directory=directory)
        report["summary"] = summarize(report["records"])
        report["checks_passed"] = all(r["checks_passed"] for r in report["records"])
        report["completed"] = True
    except BaseException as exc:
        report["error"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        save(directory, report)


def install(engine_class):
    """Provision the diagnostic before Engine initializes CUDA; no draft head."""
    import torch
    from freetoken.kvcache.qsa_pool import QSAKVCache
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    if not os.environ.get(OUTPUT_ENV):
        raise RuntimeError("explicit private probe output directory is required")
    if torch.cuda.is_initialized():
        raise RuntimeError("install the probe before Engine initializes CUDA")
    directory = Path(os.environ[OUTPUT_ENV])
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    original_cpu_init = CpuMoeExecutor.__init__
    original_ring_capacity = QSAKVCache.ring_capacity_for
    original_init = engine_class.__init__
    original_forward = engine_class.forward_batch

    def cpu_init(executor, *args, **kwargs):
        if kwargs.get("max_tokens") != 1:
            raise RuntimeError("probe requires the capacity-one CPU decode configuration")
        kwargs["max_tokens"] = 2
        original_cpu_init(executor, *args, **kwargs)

    def ring_capacity(cls, index_ratio, num_speculative_tokens=0):
        return original_ring_capacity(index_ratio, max(1, num_speculative_tokens))

    def initialize(engine, config):
        if (config.speculative_mtp != "off" or config.max_running_req != 1
                or config.moe_step_timing or config.moe_collect_stats):
            raise RuntimeError("probe requires capacity one, MTP off and diagnostics off")
        original_init(engine, config)
        if (not isinstance(engine.kv_cache, QSAKVCache)
                or engine.cpu_moe_executor is None or engine.linear_state_pool is None
                or sorted(engine.graph_runner.graph_map) != [1]
                or getattr(engine.model, "mtp", None) is not None):
            raise RuntimeError("probe requires Qwen hybrid state and ordinary graph size one")

    def forward(engine, batch, args):
        if (batch.is_decode and batch.size == 1 and batch.padded_size == 1
                and not batch.lazy_restore_pending):
            req = batch.reqs[0]
            if (req.decode_batch_idx >= 16 and req.sampling_params.is_greedy
                    and req.sampling_params.guided_decoding is None
                    and eligible_position(req.cached_len, page_size=engine.config.page_size,
                                          ratio=engine.kv_cache.index_ratio, remaining=req.remain_len)):
                with torch.inference_mode():
                    probe(engine, batch, directory)
                # This request is intentionally aborted. Its repeated expert/cache
                # history must never escape into a serving benchmark or user result.
                raise SystemExit(0)
        return original_forward(engine, batch, args)

    CpuMoeExecutor.__init__ = cpu_init
    QSAKVCache.ring_capacity_for = classmethod(ring_capacity)
    engine_class.__init__ = initialize
    engine_class.forward_batch = forward


def install_import_hook():
    """Explicit sitecustomize entry point, safe in frontend and spawned workers."""
    if not os.environ.get(OUTPUT_ENV):
        raise RuntimeError("explicit private probe output directory is required")
    target = "freetoken.engine.engine"
    if target in sys.modules:
        raise RuntimeError("install the import hook before importing Engine")

    class Loader:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def create_module(self, spec):
            return self.wrapped.create_module(spec)

        def exec_module(self, module):
            self.wrapped.exec_module(module)
            install(module.Engine)

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

    class Finder:
        def find_spec(self, fullname, path=None, target=None):
            if fullname != "freetoken.engine.engine":
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is not None:
                spec.loader = Loader(spec.loader)
                sys.meta_path.remove(self)
            return spec

    sys.meta_path.insert(0, Finder())
