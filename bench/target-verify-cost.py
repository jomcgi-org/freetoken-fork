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
GRAPH_ENV = "FREETOKEN_TARGET_VERIFY_GRAPH"
TRACE_ENV = "FREETOKEN_TARGET_VERIFY_LAYER_TRACE"
SERIAL_LINEAR_ENV = "FREETOKEN_TARGET_VERIFY_SERIAL_LINEAR"
CHECKPOINT_ENV = "FREETOKEN_TARGET_VERIFY_SEED_CHECKPOINT"
WIDTH_ENV = "FREETOKEN_TARGET_VERIFY_WIDTH"
COMPACT_ENV = "FREETOKEN_TARGET_VERIFY_COMPACT_ROLLBACK"
PAIR_ENV = "FREETOKEN_TARGET_VERIFY_CPU_PAIR_COMPARE"
MODES = ("graph_one", "graph_two", "snapshot", "accept", "reject")
GRAPH_MODES = ("accept_graph", "reject_graph")
CHECKPOINT_MODES = ("accept_checkpoint", "reject_checkpoint")
_ACTIVATIONS = {}
_CHECKPOINT_API = None
_MULTI_API = None
_COMPACT_API = None


def verification_width():
    width = int(os.environ.get(WIDTH_ENV, "2"))
    if width not in (2, 3, 5):
        raise ValueError("target verification width must be two, three or five")
    return width


def install_serial_linear(width=2):
    """Keep ordinary dense row reduction order inside the two-token diagnostic."""
    import torch
    import torch.nn.functional as functional
    from freetoken.core import get_global_ctx

    original = functional.linear

    def linear(input, weight, bias=None):
        if input.ndim == 2 and input.shape[0] == width:
            try:
                batch = get_global_ctx().batch
            except AssertionError:
                batch = None
            if getattr(batch, "mtp_fused", False):
                return torch.cat([original(input[i:i + 1], weight, bias)
                                  for i in range(width)], dim=0)
        return original(input, weight, bias)

    functional.linear = linear


def eligible_position(position, *, page_size, ratio, remaining):
    """Leave room for three adjacent windows within an already allocated page."""
    return (
        ratio >= 2 and page_size % ratio == 0 and remaining >= 5
        and position % ratio == ratio - 2
        and position // page_size == (position + 3) // page_size
    )


def summarize(records, *, graph_enabled=False, checkpoint_enabled=False):
    """Keep correctness independent of component costs and never claim a speedup."""
    groups = {}
    for record in records:
        if not record["warmup"]:
            groups.setdefault(record["case"], {}).setdefault(record["mode"], []).append(record)
    result = {}
    for case, modes in groups.items():
        complete = set(modes) == set(MODES + (GRAPH_MODES if graph_enabled else ())
                                    + (CHECKPOINT_MODES if checkpoint_enabled else ()))
        valid = complete and all(r["checks_passed"] for r in records if r["case"] == case)
        medians = {name: statistics.median(r["wall_s"] for r in rows)
                   for name, rows in modes.items()}
        row = dict(complete=complete, checks_passed=valid, median_component_s=medians,
                   model_wall_qualified=False)
        if valid:
            paths = [("", "accept", "reject")]
            if graph_enabled:
                paths.append(("graph_", "accept_graph", "reject_graph"))
            if checkpoint_enabled:
                paths.append(("checkpoint_", "accept_checkpoint", "reject_checkpoint"))
            for prefix, accept_mode, reject_mode in paths:
                one, accepted, rejected = (
                    medians[k] for k in ("graph_one", accept_mode, reject_mode))
                denominator = rejected - accepted + one
                # p*A + (1-p)*R < (1+p)*D. Proposer and scheduler work is additional.
                row[prefix + "break_even_acceptance_excluding_proposer"] = (
                    (rejected - one) / denominator if denominator > 0 else None
                )
        result[case] = row
    return result


def numerical_mismatches(checks, *, graph_enabled=False, checkpoint_enabled=False):
    """Include untimed reference comparisons in component qualification."""
    prefixes = ["eager_vs_sequential"]
    if graph_enabled:
        prefixes.append("graph_vs_eager")
    if checkpoint_enabled:
        prefixes.append("checkpoint_vs_eager")
    mismatches = []
    for prefix in prefixes:
        if not checks[prefix + "_logits"]["exact"]:
            mismatches.append(prefix + "/logits")
        mismatches.extend(prefix + "/" + name
                          for name, metric in checks[prefix + "_state"].items()
                          if not metric["exact"])
    if graph_enabled and not checks["graph_vs_eager_tokens_equal"]:
        mismatches.append("graph_vs_eager/tokens")
    if checkpoint_enabled:
        if not checks["checkpoint_vs_eager_tokens_equal"]:
            mismatches.append("checkpoint_vs_eager/tokens")
        mismatches.extend("checkpoint_seed/" + name
                          for name, metric in checks["checkpoint_seed_state"].items()
                          if not metric["exact"])
    return mismatches


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


def difference_metrics(actual, expected):
    """Describe discrepancies without choosing a tolerance or relaxing qualification."""
    import torch

    result = dict(dtype=str(actual.dtype), elements=actual.numel(),
                  exact=bytes_equal(actual, expected))
    if not actual.is_floating_point():
        # Integer PLE history must never pass through a lossy float conversion.
        result["different_elements"] = int(torch.count_nonzero(actual != expected).item())
        return result
    left, right = actual.float(), expected.float()
    finite = bool(torch.isfinite(left).all().item() and torch.isfinite(right).all().item())
    result["finite"] = finite
    if not finite or not actual.numel():
        return result
    delta = left - right
    rms = float(delta.square().mean().sqrt().item())
    reference_rms = float(right.square().mean().sqrt().item())
    result.update(different_elements=int(torch.count_nonzero(left != right).item()),
                  max_abs=float(delta.abs().max().item()), rms=rms,
                  relative_rms=rms / reference_rms if reference_rms else None)
    return result


def host_contexts(history, position, ids, contexts, boundary):
    """Fill token-ordered PLE contexts from known inputs, including left padding."""
    if (ids.numel() not in (2, 3, 5) or contexts.shape[0] != ids.numel()
            or history.numel() < position):
        raise ValueError("PLE staging requires matching width and complete host history")
    width = contexts.shape[1]
    contexts.fill_(boundary)
    count = min(width, position)
    if count:
        contexts[0, width - count:].copy_(history[position - count:position])
    if width:
        for step in range(1, ids.numel()):
            contexts[step, :-1].copy_(contexts[step - 1, 1:])
            contexts[step, -1].copy_(ids[step - 1])


def install_layer_trace():
    """Explicitly add activation copies to graph/eager decode, never to wall runs."""
    import torch
    from freetoken.models.qwen4_exp.model import Qwen4ExpDecoderLayer

    original = Qwen4ExpDecoderLayer.forward

    def forward(layer, hidden, batch):
        if not batch.is_decode:
            return original(layer, hidden, batch)
        mode = "captured" if torch.cuda.is_current_stream_capturing() else "eager"
        records = _ACTIVATIONS.setdefault((mode, hidden.shape[0]), {})

        def note(stage, value):
            records[f"{layer._layer_id:02d}/{stage}"] = value.detach().clone()

        # Same operation order as Qwen4ExpDecoderLayer.forward. The only added
        # operations copy activations to diagnostic-owned tensors.
        note("input", hidden)
        if layer.ple is not None:
            hidden = hidden + layer.ple.forward(hidden, batch)
            note("after_ple", hidden)
        block_input, inject = layer.attn_hyper_connection.mix(hidden)
        note("attn_input", block_input)
        if layer._is_linear:
            block_output = layer.linear_attn.forward(block_input)
        else:
            block_output = layer.self_attn.forward(block_input, batch)
        note("attn_output", block_output)
        hidden = layer.attn_hyper_connection.combine(hidden, block_output, inject)
        note("after_attention", hidden)
        block_input, inject = layer.mlp_hyper_connection.mix(hidden)
        note("mlp_input", block_input)
        block_output = layer.mlp.forward(block_input)
        note("mlp_output", block_output)
        hidden = layer.mlp_hyper_connection.combine(hidden, block_output, inject)
        note("output", hidden)
        return hidden

    Qwen4ExpDecoderLayer.forward = forward


def activation_snapshot(mode, width):
    values = _ACTIVATIONS.get((mode, width))
    if not values:
        raise RuntimeError("activation trace is missing its forward")
    return {name: value.clone() for name, value in values.items()}


def activation_comparison(actual, expected):
    if actual.keys() != expected.keys():
        raise RuntimeError("activation trace stages differ")
    return {name: difference_metrics(value, expected[name]) for name, value in actual.items()}


def install_graph_support(width=2):
    """Diagnostic-only adaptations of the existing feat/mtp-graphs PLE work.

    Retain serial width-one GDN updates. Only provision PLE's second row and make
    the two constant PLE indptrs capture-safe. Ordinary decode delegates unchanged.
    """
    import torch
    from freetoken.models.qwen4_exp import ple
    from freetoken.models.qwen4_exp.ple_uring import UringTable

    original_uring = UringTable.__init__
    original_hash = ple.NGramEmbedding.snapshot_host_hash_constants
    original_metadata = ple.build_ple_metadata
    original_conv = ple.PLELayer._short_conv
    indptrs = {}

    def indptr(device, width):
        key = (device, width)
        if key not in indptrs:
            indptrs[key] = torch.arange(0, width + 1, width, device=device, dtype=torch.int32)
        return indptrs[key]

    def uring(table, *args, **kwargs):
        kwargs["max_decode_batch_size"] = max(width, kwargs["max_decode_batch_size"])
        original_uring(table, *args, **kwargs)

    def hash_constants(embedding, max_batch_size=None):
        original_hash(embedding, max(width, max_batch_size or 0))

    def metadata(batch, args, device, context_pool=None):
        if not getattr(batch, "mtp_fused", False):
            return original_metadata(batch, args, device, context_pool)
        if context_pool is None:
            context_pool = ple._ngram_context_pool()
        slots = batch.linear_table_idx.long()
        if slots.numel() != 1 or batch.input_ids.numel() != width:
            raise ValueError("fused probe requires one request and matching token width")
        return ple.PLEMetadata(input_ids=batch.input_ids, cu_seqlens=indptr(device, width),
                               seq_lens=(width,), ngram_context=context_pool.index_select(0, slots).long(),
                               state_slots=slots, fresh_slots=None, is_decode=False, mtp_fused=True)

    def short_conv(layer, x, meta, states):
        if not meta.mtp_fused:
            return original_conv(layer, x, meta, states)
        pieces = []
        for step in range(width):
            one = ple.PLEMetadata(input_ids=meta.input_ids[step:step + 1],
                                   cu_seqlens=indptr(meta.cu_seqlens.device, 1), seq_lens=(1,),
                                   ngram_context=meta.ngram_context, state_slots=meta.state_slots,
                                   fresh_slots=None, is_decode=True)
            pieces.append(layer._decode_conv(x[step:step + 1], one, states))
        return torch.cat(pieces, dim=0)

    UringTable.__init__ = uring
    ple.NGramEmbedding.snapshot_host_hash_constants = hash_constants
    ple.build_ple_metadata = metadata
    ple.PLELayer._short_conv = short_conv


class FusedGraph:
    """A dedicated graph for one diagnostic window, with dynamic token inputs.

    Positions, request slots and page mapping remain fixed for this window. This
    does not implement a scheduler or generalize the graph to other requests.
    """

    def __init__(self, engine, source):
        import torch
        from freetoken.attention.linear import build_fla_metadata

        self.engine = engine
        self.width = width = source.input_ids.numel()
        self.batch = batch = copy.copy(source)
        batch.input_ids = source.input_ids.clone()
        batch.positions = source.positions.clone()
        batch.out_loc = source.out_loc.clone()
        batch.linear_table_idx = source.linear_table_idx.clone()
        batch.active_table_idx = source.active_table_idx.clone()
        batch.fla_metadata = build_fla_metadata(batch, engine.device)
        # The existing eager QSA builder gives each token independent addressing
        # tensors. Keep those tensors alive and fixed throughout this graph's use.
        engine.attn_backend.prepare_metadata(batch)
        args = engine.model._config.qwen4_args
        self.ids = torch.empty(width, dtype=torch.int64, pin_memory=True)
        self.contexts = torch.empty((width, args.ngram_size - 1), dtype=torch.int64, pin_memory=True)
        self.boundary = args.ngram_boundary_token_id
        self.copy_done = torch.cuda.Event()
        self.backends = getattr(engine.model, "_ple_disk_decode", ())
        if not self.backends:
            raise RuntimeError("verification graph probe requires staged PLE backends")
        self.logits = torch.empty((width, engine.config.model_config.vocab_size),
                                  device=engine.device, dtype=torch.float32)
        self.graph = torch.cuda.CUDAGraph()
        self._prepare()
        with engine.ctx.forward_batch(batch):
            self.logits.copy_(engine.model.forward(select_last=False))
        engine.model.finish_cuda_graph_replay(record_event=True)
        engine.stream.synchronize()
        self._prepare()
        with engine.ctx.forward_batch(batch):
            with torch.cuda.graph(self.graph, stream=engine.stream):
                self.logits.copy_(engine.model.forward(select_last=False))
        engine.model.finish_cuda_graph_replay(record_event=False)
        engine.graph_runner._reset_moe_offload_cache()
        engine.stream.synchronize()

    def _prepare(self):
        import torch

        self.ids.copy_(self.batch.input_ids, non_blocking=True)
        self.copy_done.record(self.engine.stream)
        self.copy_done.synchronize()
        req = self.batch.reqs[0]
        host_contexts(req.input_ids, req.cached_len, self.ids, self.contexts, self.boundary)
        for ple, backend in self.backends:
            rows = ple.ple_embedding.host_decode_row_ids(self.contexts, self.ids)
            backend.prepare_decode(rows)
            if getattr(backend, "_decode_shape", None) != torch.Size(rows.shape):
                raise RuntimeError("verification PLE staging did not complete")

    def replay(self, batch):
        for name in ("input_ids", "positions", "out_loc", "linear_table_idx", "active_table_idx"):
            if getattr(batch, name).shape != getattr(self.batch, name).shape:
                raise RuntimeError("verification graph shape changed")
            getattr(self.batch, name).copy_(getattr(batch, name))
        # Restage the address/length inputs as a serving graph would. The scatter
        # plans are captured operations derived from these persistent inputs.
        for destination, source in zip(self.batch.mtp_qsa_metadata, batch.mtp_qsa_metadata):
            for name in ("block_table", "seq_lens", "ring_slots", "token_to_req", "cu_seqlens"):
                getattr(destination, name).copy_(getattr(source, name))
        self._prepare()
        self.graph.replay()
        self.engine.model.finish_cuda_graph_replay(record_event=True)
        return self.logits

    def close(self):
        self.engine.stream.synchronize()
        self.graph.reset()


class StateWindow:
    """Reset trials fully; compare only state reachable at the committed length.

    KV and compressed rows beyond that length may contain rejected writes. They
    are deliberately NOT restored on the measured rejection path. Following the
    rejection with another ordinary decode checks that those rows are overwritten.
    """

    def __init__(self, engine, req, position, *, width=2):
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
        self.locations = page_row[position - self.base:position - self.base + width].clone()
        if self.locations.numel() != width:
            raise RuntimeError("verification window extends beyond its allocated page")
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

    def committed_pairs(self, expected, committed_end):
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
            yield name, actual, wanted

    def compare(self, expected, *, committed_end):
        return [name for name, actual, wanted in self.committed_pairs(expected, committed_end)
                if not bytes_equal(actual, wanted)]

    def metrics(self, expected, *, committed_end):
        return {name: difference_metrics(actual, wanted)
                for name, actual, wanted in self.committed_pairs(expected, committed_end)}


def run_window(engine, source_batch, position, seed, host_prefix, *, repeats, warmup, report,
               directory, graph_enabled=False, trace_only=False, checkpoint_enabled=False):
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
    request_rows = source_batch.active_table_idx.clone()
    fused_rows = request_rows.repeat(2)
    batch.active_table_idx = request_rows
    batch.mtp_original_cached_len = position
    batch.mtp_original_device_len = position + 1
    positions = torch.arange(position, position + 2, device=engine.device, dtype=torch.int32)
    window = StateWindow(engine, req, position)
    initial = window.capture()
    verify_ids = torch.cat((seed.reshape(1), seed.reshape(1))).to(torch.int32)
    host_seed = seed.cpu().reshape(1).to(host_prefix.dtype)
    req.input_ids = torch.cat((host_prefix, host_seed, host_seed))

    captured_graph = None
    checkpoint_graph = None
    checkpoint = None

    def forward(step=None, *, captured=False, with_logits=False, ordinary_eager=False,
                seed_checkpoint=False):
        if step is None:
            configure_mtp_fused_step(batch, verify_ids, positions, window.locations)
            batch.active_table_idx = fused_rows
        else:
            configure_mtp_decode_step(batch, verify_ids, positions, window.locations, step)
            batch.active_table_idx = request_rows
        batch.fla_metadata = build_fla_metadata(batch, engine.device)
        engine.attn_backend.prepare_metadata(batch)
        engine.cpu_moe_executor.begin_decode_step()
        with engine.ctx.forward_batch(batch):
            if captured:
                graph = checkpoint_graph if seed_checkpoint else captured_graph
                if step is not None or graph is None:
                    raise RuntimeError("dedicated verification graph is unavailable")
                logits = graph.replay(batch)
            elif step is None or ordinary_eager:
                logits = engine.model.forward(select_last=False)
            else:
                if not engine.graph_runner.can_use_cuda_graph(batch):
                    raise RuntimeError("ordinary CUDA graph baseline is unavailable")
                logits = engine.graph_runner.replay(batch)
        engine.cpu_moe_executor.raise_if_unhealthy()
        tokens = torch.argmax(logits, dim=-1).to(torch.int32)
        return (tokens, logits.float().clone()) if with_logits else tokens

    # Derive an actual accepted candidate, and independent one/two-step state references.
    first, first_logits = forward(0, with_logits=True)
    expected_one = window.capture()
    first_trace = activation_snapshot("captured", 1) if trace_only else None
    verify_ids[1].copy_(first[0])
    req.input_ids[-1:].copy_(first.cpu().to(req.input_ids.dtype))
    second, second_logits = forward(1, with_logits=True)
    expected_two = window.capture()
    if trace_only:
        second_trace = activation_snapshot("captured", 1)
        sequential_trace = {name: torch.cat((value, second_trace[name]))
                            for name, value in first_trace.items()}
    good_ids = verify_ids.clone()
    good_history = req.input_ids.clone()
    expected_tokens = torch.cat((first, second))
    wrong_id = (int(first.item()) + 1) % engine.config.model_config.vocab_size
    pool, kv = engine.linear_state_pool, engine.kv_cache
    case = str(position)
    single_diagnostic = None
    if trace_only:
        window.reset(initial)
        single_tokens, single_logits = forward(0, ordinary_eager=True, with_logits=True)
        single_diagnostic = dict(
            tokens_equal=bytes_equal(single_tokens, first),
            logits=difference_metrics(single_logits, first_logits),
            state=window.metrics(expected_one, committed_end=position + 1),
            activations=activation_comparison(activation_snapshot("eager", 1), first_trace))
    window.reset(initial)
    eager_tokens, eager_logits = forward(with_logits=True)
    expected_eager = window.capture()
    diagnostic = dict(position=position, eager_vs_sequential_logits=difference_metrics(
        eager_logits, torch.cat((first_logits, second_logits))),
        eager_vs_sequential_state=window.metrics(expected_two, committed_end=position + 2))
    report.setdefault("numerical_checks", {})[case] = diagnostic
    if trace_only:
        diagnostic["eager_one_vs_graph_one"] = single_diagnostic
        diagnostic["fused_vs_sequential_activations"] = activation_comparison(
            activation_snapshot("eager", 2), sequential_trace)
        save(directory, report)
        window.reset(expected_one)
        engine.stream.synchronize()
        return first, torch.cat((host_prefix, host_seed))
    if graph_enabled:
        report["stage"] = "capturing verification graph at " + case
        save(directory, report)
        window.reset(initial)
        engine.stream.synchronize()
        captured_graph = FusedGraph(engine, batch)
        window.reset(initial)
        graph_tokens, graph_logits = forward(captured=True, with_logits=True)
        diagnostic["graph_vs_eager_logits"] = difference_metrics(graph_logits, eager_logits)
        diagnostic["graph_vs_eager_state"] = window.metrics(expected_eager, committed_end=position + 2)
        diagnostic["graph_vs_eager_tokens_equal"] = bytes_equal(graph_tokens, eager_tokens)
        report["stage"] = "measuring verification graph at " + case
        save(directory, report)

    if checkpoint_enabled:
        report["stage"] = "capturing first-token checkpoint graph at " + case
        save(directory, report)
        window.reset(initial)
        engine.stream.synchronize()
        checkpoint = _CHECKPOINT_API["SeedCheckpoint"].from_engine(
            engine, req, state_views(engine, req))
        with _CHECKPOINT_API["capture_context"](checkpoint):
            checkpoint_graph = FusedGraph(engine, batch)
        window.reset(initial)
        checkpoint_tokens, checkpoint_logits = forward(
            captured=True, seed_checkpoint=True, with_logits=True)
        diagnostic["checkpoint_vs_eager_logits"] = difference_metrics(checkpoint_logits, eager_logits)
        diagnostic["checkpoint_vs_eager_state"] = window.metrics(expected_eager, committed_end=position + 2)
        diagnostic["checkpoint_vs_eager_tokens_equal"] = bytes_equal(checkpoint_tokens, eager_tokens)
        diagnostic["checkpoint_seed_state"] = {
            name: difference_metrics(value, expected_one[name]) for name, value in checkpoint.saved.items()}
        report["stage"] = "measuring first-token checkpoint graph at " + case
        save(directory, report)

    reference_mismatches = numerical_mismatches(
        diagnostic, graph_enabled=graph_enabled, checkpoint_enabled=checkpoint_enabled)

    def execute(mode):
        if mode == "graph_one":
            return forward(0), None
        if mode == "graph_two":
            return torch.cat((forward(0), forward(1))), None
        if mode in CHECKPOINT_MODES:
            targets = forward(captured=True, seed_checkpoint=True)
            accepted, matched = greedy_accept_prefix(verify_ids[1:], targets)
            if mode == "reject_checkpoint":
                checkpoint.restore()
            return accepted, matched
        saved = snapshot_verify_state(pool, kv, req)
        if mode == "snapshot":
            return saved, None
        targets = forward(captured=mode in GRAPH_MODES)
        accepted, matched = greedy_accept_prefix(verify_ids[1:], targets)
        if mode.startswith("reject"):
            # Deliberately leave future KV/index writes in place, as production does.
            restore_verify_state(pool, kv, req, saved)
            return forward(0), matched
        return accepted, matched

    for repeat in range(-warmup, repeats):
        modes = (MODES + (GRAPH_MODES if graph_enabled else ())
                 + (CHECKPOINT_MODES if checkpoint_enabled else ()))
        order = modes if repeat % 2 == 0 else tuple(reversed(modes))
        for mode in order:
            window.reset(initial)
            verify_ids.copy_(good_ids)
            req.input_ids.copy_(good_history)
            if mode.startswith("reject"):
                verify_ids[1] = wrong_id
                req.input_ids[-1] = wrong_id
            # Reset and diagnostic copies are outside the timed component window.
            engine.stream.synchronize()
            started = time.perf_counter()
            output, matched = execute(mode)
            engine.stream.synchronize()
            elapsed = time.perf_counter() - started
            mismatches = list(reference_mismatches)
            if mode == "snapshot":
                mismatches += window.compare(initial, committed_end=position)
                snapshot_views = state_views(engine, req)
                flat = {k: v for k, v in output.items() if k not in ("slot", "slot_states")}
                flat.update({"slot/" + k: v for k, v in output.get("slot_states", {}).items()})
                mismatches += ["snapshot/" + k for k in snapshot_views
                               if not bytes_equal(snapshot_views[k], flat[k])]
            else:
                count = 2 if mode == "graph_two" or mode.startswith("accept") else 1
                if not bytes_equal(output, expected_tokens[:count]):
                    mismatches.append("greedy_tokens")
                mismatches += window.compare(expected_two if count == 2 else expected_one,
                                             committed_end=position + count)
                if mode.startswith(("accept", "reject")) and matched != mode.startswith("accept"):
                    mismatches.append("proposal_match")
                if mode == "accept_graph":
                    mismatches += ["graph_vs_eager/" + k for k in
                                   window.compare(expected_eager, committed_end=position + 2)]
                if mode in CHECKPOINT_MODES:
                    mismatches += ["checkpoint_seed/" + name for name, value in checkpoint.saved.items()
                                   if not bytes_equal(value, expected_one[name])]
                if mode.startswith("reject"):
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
    if captured_graph is not None:
        captured_graph.close()
    if checkpoint_graph is not None:
        checkpoint_graph.close()
    return first, torch.cat((host_prefix, host_seed))


def probe(engine, batch, directory, *, repeats=4, warmup=1):
    width = verification_width()
    if width > 2:
        return _MULTI_API["probe"](engine, batch, directory, width=width, base=globals(),
                                    seed_api=_CHECKPOINT_API, repeats=repeats, warmup=warmup,
                                    pair_compare=os.environ.get(PAIR_ENV) == "1")
    import torch
    from freetoken.kernel import _cpu_moe
    import hashlib
    import subprocess

    graph_enabled = os.environ.get(GRAPH_ENV) == "1"
    trace_only = os.environ.get(TRACE_ENV) == "1"
    checkpoint_enabled = os.environ.get(CHECKPOINT_ENV) == "1"
    report = dict(diagnostic_only=True, model_wall_qualified=False, completed=False,
                  graph_enabled=graph_enabled, trace_only=trace_only,
                  checkpoint_enabled=checkpoint_enabled,
                  serial_linear=os.environ.get(SERIAL_LINEAR_ENV) == "1",
                  component_timings_usable=not trace_only,
                  records=[], pid=os.getpid(), speculative_mtp=engine.config.speculative_mtp,
                  graph_sizes=sorted(engine.graph_runner.graph_map),
                  cpu_max_tokens=engine.cpu_moe_executor.max_tokens,
                  ring_capacity=engine.kv_cache.ring_capacity,
                  index_ratio=engine.kv_cache.index_ratio,
                  native_sha256=hashlib.sha256(Path(_cpu_moe.__file__).read_bytes()).hexdigest(),
                  limitations=["Component costs exclude proposer and scheduler work",
                               "Repeated windows warm expert and file caches",
                               "All arms reserve CPU rows for two tokens and a speculative QSA ring",
                               "Graph mode also reserves two PLE staging/hash rows in every arm",
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
                                      report=report, directory=directory, graph_enabled=graph_enabled,
                                      trace_only=trace_only, checkpoint_enabled=checkpoint_enabled)
        if not trace_only:
            report["summary"] = summarize(report["records"], graph_enabled=graph_enabled,
                                          checkpoint_enabled=checkpoint_enabled)
            report["checks_passed"] = all(r["checks_passed"] for r in report["records"])
        report["completed"] = True
    except BaseException as exc:
        report["error"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        save(directory, report)


def install(engine_class):
    """Provision the diagnostic before Engine initializes CUDA; no draft head."""
    global _CHECKPOINT_API, _MULTI_API, _COMPACT_API
    import torch
    from freetoken.kvcache.qsa_pool import QSAKVCache
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    if not os.environ.get(OUTPUT_ENV):
        raise RuntimeError("explicit private probe output directory is required")
    if torch.cuda.is_initialized():
        raise RuntimeError("install the probe before Engine initializes CUDA")
    width = verification_width()
    if os.environ.get(PAIR_ENV) == "1" and width == 2:
        raise RuntimeError("CPU pair comparison requires the wider target graph diagnostic")
    if os.environ.get(COMPACT_ENV) == "1" and width == 2:
        raise RuntimeError("compact rollback comparison requires a wider target graph")
    if width > 2 and (os.environ.get(GRAPH_ENV) != "1" or os.environ.get(SERIAL_LINEAR_ENV) != "1"
                      or os.environ.get(CHECKPOINT_ENV) != "1" or os.environ.get(TRACE_ENV) == "1"):
        raise RuntimeError("wider verification requires graphs, serial linears and checkpoints, with tracing off")
    if os.environ.get(SERIAL_LINEAR_ENV) == "1":
        install_serial_linear(width)
    if os.environ.get(TRACE_ENV) == "1":
        if os.environ.get(GRAPH_ENV) == "1":
            raise RuntimeError("run activation tracing separately from graph cost measurement")
        install_layer_trace()
    if os.environ.get(GRAPH_ENV) == "1":
        install_graph_support(width)
    if os.environ.get(CHECKPOINT_ENV) == "1":
        if (os.environ.get(GRAPH_ENV) != "1" or os.environ.get(SERIAL_LINEAR_ENV) != "1"
                or os.environ.get(TRACE_ENV) == "1"):
            raise RuntimeError("seed checkpoint requires graph and serial linear modes, with tracing off")
        import runpy
        _CHECKPOINT_API = runpy.run_path(str(Path(__file__).with_name("target_seed_checkpoint.py")))
        _CHECKPOINT_API["install"]()
    if width > 2:
        import runpy
        _MULTI_API = runpy.run_path(str(Path(__file__).with_name("target_multitoken.py")))
        _MULTI_API["install"](width)
    if os.environ.get(COMPACT_ENV) == "1":
        import runpy
        _COMPACT_API = runpy.run_path(str(Path(__file__).with_name("target_compact_rollback.py")))
        _COMPACT_API["Checkpoint"] = _COMPACT_API["make_checkpoint_type"](_CHECKPOINT_API["SeedCheckpoint"])
    directory = Path(os.environ[OUTPUT_ENV])
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    original_cpu_init = CpuMoeExecutor.__init__
    original_ring_capacity = QSAKVCache.ring_capacity_for
    original_init = engine_class.__init__
    original_forward = engine_class.forward_batch

    def cpu_init(executor, *args, **kwargs):
        if kwargs.get("max_tokens") != 1:
            raise RuntimeError("probe requires the capacity-one CPU decode configuration")
        kwargs["max_tokens"] = width
        original_cpu_init(executor, *args, **kwargs)
        if os.environ.get(PAIR_ENV) == "1":
            setter = getattr(executor._ext, "set_nvfp4_pair_dot", None)
            if setter is None or not setter(True) or not setter(False):
                raise RuntimeError("CPU pair comparison requires the qualified AVX-512 NVFP4 native kernel")

    def ring_capacity(cls, index_ratio, num_speculative_tokens=0):
        return original_ring_capacity(index_ratio, max(width - 1, num_speculative_tokens))

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
                                          ratio=engine.kv_cache.index_ratio, remaining=req.remain_len)
                    and (width == 2 or _MULTI_API["eligible_window"](
                        req.cached_len, engine.config.page_size, width, req.remain_len))):
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
