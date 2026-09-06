"""Explicit wider target-verification diagnostic with retained prefix state."""

import copy
from contextlib import nullcontext
from dataclasses import replace
import math
from pathlib import Path
import statistics
from types import SimpleNamespace


def eligible_window(position, page_size, width, remaining):
    """Four adjacent windows, including their final target row, fit one page."""
    return remaining >= width + 3 and position // page_size == (position + width + 2) // page_size


def modes(width, compact_enabled=False, pair_compare=False):
    outcomes = ("accept",) + tuple(f"reject_{i}" for i in range(width - 1))
    ordinary = (("graph_one", "graph_all") + outcomes
                + (tuple("compact_" + name for name in outcomes) if compact_enabled else ()))
    return ordinary + (tuple("pair_" + name for name in ordinary) if pair_compare else ())


def trial_order(width, compact_enabled, pair_compare, repeat):
    ordinary = modes(width, compact_enabled)
    ordered = ordinary if repeat % 2 == 0 else tuple(reversed(ordinary))
    if not pair_compare:
        return ordered
    prefixes = ("", "pair_") if repeat % 2 == 0 else ("pair_", "")
    return tuple(prefix + mode for mode in ordered for prefix in prefixes)


def set_cpu_pair(engine, enabled):
    setter = getattr(engine.cpu_moe_executor._ext, "set_nvfp4_pair_dot", None)
    if setter is None:
        raise RuntimeError("CPU pair comparison requires its native setter")
    # Graph-replayed CPU tasks must have finished before changing their dispatch.
    engine.stream.synchronize()
    if not setter(enabled):
        raise RuntimeError("CPU pair comparison requires AVX-512 NVFP4")


def summarize(records, width, compact_enabled=False, pair_compare=False):
    cases = {}
    for row in records:
        cases.setdefault(row["case"], []).append(row)
    result = {}
    for case, rows in cases.items():
        values = {}
        for row in rows:
            if not row["warmup"]:
                values.setdefault(row["mode"], []).append(row["wall_s"])
        medians = {name: statistics.median(costs) for name, costs in values.items()}
        valid = (set(values) == set(modes(width, compact_enabled, pair_compare))
                 and all(row["checks_passed"] and math.isfinite(row["wall_s"])
                         and row["wall_s"] > 0 for row in rows)
                 and (not pair_compare or all(
                     row.get("cpu_pair_enabled") is row["mode"].startswith("pair_") for row in rows)))
        result[case] = dict(checks_passed=valid, median_component_s=medians,
                            model_wall_qualified=False)
        if valid:
            result[case]["component_cost_over_one"] = {
                name: cost / medians["graph_one"] for name, cost in medians.items()}
            if pair_compare:
                result[case]["cpu_pair_reduction_percent"] = {
                    name: 100 * (1 - medians["pair_" + name] / medians[name])
                    for name in modes(width, compact_enabled)}
    return result


def numerical_failures(value, path=""):
    if not isinstance(value, dict):
        return []
    if "exact" in value:
        return [] if value["exact"] else [path]
    errors = []
    for name, child in value.items():
        child_path = path + "/" + name
        if name.endswith("tokens_equal") and not child:
            errors.append(child_path)
        else:
            errors.extend(numerical_failures(child, child_path))
    return errors


def graph_reuse_qualified(records, captures, compact_enabled):
    cases = {}
    for row in records:
        cases.setdefault(row["case"], set()).add(row.get("graph_reused"))
    expected = {"full": 1, "compact": 1 if compact_enabled else 0}
    return captures == expected and list(cases.values()) == [{False}, {True}, {True}, {True}]


def close_graphs(graphs, stream):
    for name in ("graph", "compact_graph"):
        if graphs.get(name) is not None:
            graphs[name].close()
    if graphs.get("compact_checkpoint") is not None:
        graphs["compact_checkpoint"].close(stream)
    graphs.clear()


def configure_fused(batch, ids, positions, locations, width):
    if ids.numel() != width or positions.numel() != width or locations.numel() != width:
        raise ValueError("fused inputs must match the configured verification width")
    req = batch.reqs[0]
    req.cached_len = batch.mtp_original_cached_len
    req.device_len = batch.mtp_original_device_len + width - 1
    batch.input_ids, batch.positions, batch.out_loc = ids, positions, locations
    batch.phase, batch.mtp_fused = "decode", True


def install(width):
    """Keep width-one GDN/QSA math while the outer expert work sees all rows."""
    import torch
    from freetoken.core import get_global_ctx
    from freetoken.models.qwen4_exp.gdn import Qwen4ExpGatedDeltaNet
    from freetoken.attention.qsa_sparse import QSASparseAttnBackend, QSASparseMetadata

    original_gdn = Qwen4ExpGatedDeltaNet.forward
    original_metadata = QSASparseAttnBackend.prepare_metadata
    pin = dict(device="cpu", dtype=torch.int32, pin_memory=True)

    def gdn(layer, hidden):
        batch = get_global_ctx().batch
        if not getattr(batch, "mtp_fused", False):
            return original_gdn(layer, hidden)
        if hidden.shape[0] != width:
            raise ValueError("GDN input width differs from verification width")
        # GDN consumes hidden rows and width-one FLA metadata, not token addresses.
        # Restore the outer classification even when a primitive raises.
        batch.mtp_fused = False
        try:
            return torch.cat([original_gdn(layer, hidden[i:i + 1]) for i in range(width)], dim=0)
        finally:
            batch.mtp_fused = True

    def metadata(backend, batch):
        if not getattr(batch, "mtp_fused", False):
            return original_metadata(backend, batch)
        if batch.input_ids.numel() != width or len(batch.reqs) != 1:
            raise ValueError("QSA metadata requires one matching-width request")
        start = int(batch.mtp_original_device_len)
        batch.attn_metadata = QSASparseMetadata(
            is_decode=True, last_indices=torch.full((1,), width - 1, device=backend.device, dtype=torch.int32),
            qo_indptr_cpu=torch.tensor([0, width], **pin),
            kv_len_cpu=torch.tensor([start + width - 1], **pin))
        steps = []
        for step in range(width):
            one = QSASparseMetadata(
                is_decode=True, last_indices=torch.zeros(1, device=backend.device, dtype=torch.int32),
                qo_indptr_cpu=torch.tensor([0, 1], **pin),
                kv_len_cpu=torch.tensor([start + step], **pin))
            backend._snapshot_decode(one, batch)
            steps.append(one)
        batch.mtp_qsa_metadata = tuple(steps)

    def sparse(backend, q, k, v, index, layer_id, batch):
        if any(t.shape[0] != width for t in (q, k, v, index.k, index.q)):
            raise ValueError("QSA inputs differ from verification width")
        if len(batch.mtp_qsa_metadata) != width:
            raise ValueError("QSA metadata does not cover every target position")
        outputs = []
        for step in range(width):
            one = SimpleNamespace(
                padded_reqs=batch.padded_reqs, reqs=batch.reqs,
                lazy_restore_pending=getattr(batch, "lazy_restore_pending", False),
                positions=batch.positions[step:step + 1], out_loc=batch.out_loc[step:step + 1],
                attn_metadata=batch.mtp_qsa_metadata[step])
            one_index = replace(index, q=index.q[step:step + 1], k=index.k[step:step + 1])
            outputs.append(backend._qsa_forward_one(q[step:step + 1], k[step:step + 1],
                                                  v[step:step + 1], one_index, layer_id, one))
        return torch.cat(outputs, dim=0)

    Qwen4ExpGatedDeltaNet.forward = gdn
    QSASparseAttnBackend.prepare_metadata = metadata
    QSASparseAttnBackend._qsa_forward_mtp_k1 = sparse


def run_window(engine, source, position, seed, host_prefix, *, width, base, seed_api,
               report, directory, repeats, warmup, compact_type=None, pair_compare=False,
               relocatable_state=False, graphs=None, relocation=None, control_logits=None):
    import torch
    from freetoken.attention.linear import build_fla_metadata
    from freetoken.core import Batch
    from freetoken.spec_decode import configure_mtp_decode_step, greedy_accept_prefix

    equal, metrics = base["bytes_equal"], base["difference_metrics"]
    if pair_compare:
        set_cpu_pair(engine, False)
    req = copy.copy(source.reqs[0])
    batch = Batch(reqs=[req], phase="decode")
    batch.padded_reqs = [req]
    batch.linear_table_idx = source.linear_table_idx.clone()
    request_rows = source.active_table_idx.clone()
    fused_rows = request_rows.repeat(width)
    batch.active_table_idx = request_rows
    batch.mtp_original_cached_len, batch.mtp_original_device_len = position, position + 1
    positions = torch.arange(position, position + width, device=engine.device, dtype=torch.int32)
    window = base["StateWindow"](engine, req, position, width=width,
                                 allow_reserved_page=relocation is not None)
    initial = window.capture()
    neighbours = relocation.neighbours() if relocation is not None else {}
    # Inactive-slot checks are diagnostic work outside the forward timer.
    # Host snapshots also keep their comparison temporaries out of VRAM.
    neighbour_state = {name: value.detach().to("cpu", copy=True) for name, value in neighbours.items()}

    def neighbour_errors():
        return ["neighbour/" + name for name, value in neighbours.items()
                if not equal(value.cpu(), neighbour_state[name])]
    ids = seed.reshape(1).repeat(width).to(torch.int32)
    host_seed = seed.cpu().reshape(1).to(host_prefix.dtype)
    req.input_ids = torch.cat((host_prefix, host_seed.repeat(width)))
    if graphs is not None and not relocatable_state:
        raise ValueError("graph reuse requires explicit relocatable state bindings")
    graph = graphs.get("graph") if graphs is not None else None
    checkpoint = graphs.get("checkpoint") if graphs is not None else None
    compact_graph = graphs.get("compact_graph") if graphs is not None else None
    compact_checkpoint = graphs.get("compact_checkpoint") if graphs is not None else None
    graph_reused = graph is not None

    def forward(step=None, *, captured=False, with_logits=False, compact=False):
        if step is None:
            configure_fused(batch, ids, positions, window.locations, width)
            batch.active_table_idx = fused_rows
        else:
            configure_mtp_decode_step(batch, ids, positions, window.locations, step)
            batch.active_table_idx = request_rows
        batch.fla_metadata = build_fla_metadata(batch, engine.device)
        engine.attn_backend.prepare_metadata(batch)
        engine.cpu_moe_executor.begin_decode_step()
        with engine.ctx.forward_batch(batch):
            if captured:
                selected_graph = compact_graph if compact else graph
                if selected_graph is None or step is not None:
                    raise RuntimeError("verification graph is unavailable")
                logits = selected_graph.replay(batch)
            elif step is None:
                logits = engine.model.forward(select_last=False)
            else:
                if not engine.graph_runner.can_use_cuda_graph(batch):
                    raise RuntimeError("ordinary graph is unavailable")
                logits = engine.graph_runner.replay(batch)
        engine.cpu_moe_executor.raise_if_unhealthy()
        tokens = torch.argmax(logits, dim=-1).to(torch.int32)
        return (tokens, logits.float().clone()) if with_logits else tokens

    tokens, logits, states = [], [], []
    for step in range(width):
        token, logit = forward(step, with_logits=True)
        tokens.append(token)
        logits.append(logit)
        states.append(window.capture())
        if step + 1 < width:
            ids[step + 1].copy_(token[0])
            req.input_ids[position + step + 1].copy_(token.cpu()[0].to(req.input_ids.dtype))
    reference_tokens, reference_logits = torch.cat(tokens), torch.cat(logits)
    good_ids, good_history = ids.clone(), req.input_ids.clone()
    case = str(position)

    window.reset(initial)
    eager_tokens, eager_logits = forward(with_logits=True)
    eager_state = window.capture()
    checks = dict(eager_logits=metrics(eager_logits, reference_logits),
                  eager_tokens_equal=equal(eager_tokens, reference_tokens),
                  eager_state=window.metrics(states[-1], committed_end=position + width))
    if control_logits is not None:
        checks["original_mapping_seed_logits"] = metrics(reference_logits[:1], control_logits)
    report.setdefault("numerical_checks", {})[case] = checks
    report["stage"] = "capturing wider checkpoint graph at " + case
    base["save"](directory, report)
    window.reset(initial)
    engine.stream.synchronize()
    if graph is None:
        checkpoint = seed_api["SeedCheckpoint"].from_engine(
            engine, req, base["state_views"](engine, req), width=width)
        with seed_api["capture_context"](checkpoint):
            graph = base["FusedGraph"](engine, batch,
                                       state_checkpoint=checkpoint if relocatable_state else None)
        report.setdefault("graph_captures", {"full": 0, "compact": 0})["full"] += 1
        if graphs is not None:
            graphs.update(graph=graph, checkpoint=checkpoint)
    window.reset(initial)
    graph_tokens, graph_logits = forward(captured=True, with_logits=True)
    checks.update(graph_logits=metrics(graph_logits, reference_logits),
                  graph_tokens_equal=equal(graph_tokens, reference_tokens),
                  graph_state=window.metrics(states[-1], committed_end=position + width),
                  graph_vs_eager_logits=metrics(graph_logits, eager_logits),
                  graph_vs_eager_state=window.metrics(eager_state, committed_end=position + width),
                  prefix_state={str(length): {name: metrics(value, states[length - 1][name])
                                             for name, value in prefix.items()}
                                for length, prefix in enumerate(checkpoint.prefixes, 1)})
    if compact_type is not None:
        report["stage"] = "capturing compact target and rollback graphs at " + case
        base["save"](directory, report)
        window.reset(initial)
        engine.stream.synchronize()
        def allocator_state():
            return dict(allocated_bytes=torch.cuda.memory_allocated(engine.device),
                        reserved_bytes=torch.cuda.memory_reserved(engine.device))

        allocator = dict(before_compact=allocator_state())
        capture_compact = compact_graph is None
        if capture_compact:
            compact_checkpoint = compact_type.from_engine(
                engine, req, base["state_views"](engine, req), width=width)
            with seed_api["capture_context"](compact_checkpoint):
                compact_graph = base["FusedGraph"](
                    engine, batch, state_checkpoint=compact_checkpoint if relocatable_state else None)
            report.setdefault("graph_captures", {"full": 0, "compact": 0})["compact"] += 1
            if graphs is not None:
                graphs.update(compact_graph=compact_graph, compact_checkpoint=compact_checkpoint)
        allocator["after_compact_target"] = allocator_state()
        if capture_compact:
            compact_checkpoint.capture_restore_graphs(engine.stream)
        allocator["after_rollback_graphs"] = allocator_state()
        window.reset(initial)
        compact_tokens, compact_logits = forward(captured=True, compact=True, with_logits=True)
        compact_checks = dict(tokens_equal=equal(compact_tokens, reference_tokens),
                              logits=metrics(compact_logits, reference_logits),
                              state=window.metrics(states[-1], committed_end=position + width),
                              restored_prefix_state={})
        for length in range(1, width):
            compact_checkpoint.restore(length)
            # Compare all state, including reconstructed recurrence, not only
            # the smaller tensors physically retained in compact.prefixes.
            compact_checks["restored_prefix_state"][str(length)] = window.metrics(
                states[length - 1], committed_end=position + length)
        checks["compact"] = compact_checks
        report.setdefault("checkpoint_storage", {})[case] = dict(
            full_tensor_bytes=checkpoint.owned_tensor_bytes(),
            compact_tensor_bytes=compact_checkpoint.owned_tensor_bytes(),
            rollback_graphs=len(compact_checkpoint.restore_graphs),
            excludes_graph_pool_allocations=True,
            process_allocator_snapshots=allocator)
    if pair_compare:
        set_cpu_pair(engine, True)
        pair_checks = {}
        variants = [("eager", False, False), ("graph", True, False)]
        if compact_type is not None:
            variants.append(("compact", True, True))
        for name, captured, is_compact in variants:
            window.reset(initial)
            actual_tokens, actual_logits = forward(
                captured=captured, compact=is_compact, with_logits=True)
            checked = dict(tokens_equal=equal(actual_tokens, reference_tokens),
                           logits=metrics(actual_logits, reference_logits),
                           state=window.metrics(states[-1], committed_end=position + width))
            if captured:
                selected = compact_checkpoint if is_compact else checkpoint
                checked["restored_prefix_state"] = {}
                for length in range(1, width):
                    selected.restore(length)
                    checked["restored_prefix_state"][str(length)] = window.metrics(
                        states[length - 1], committed_end=position + length)
            pair_checks[name] = checked
        checks["cpu_pair"] = pair_checks
        set_cpu_pair(engine, False)
    if relocation is not None:
        checks["neighbour_state"] = {name: metrics(value.cpu(), neighbour_state[name])
                                     for name, value in neighbours.items()}
    errors = numerical_failures(checks)
    report["stage"] = "measuring wider checkpoint graph at " + case
    base["save"](directory, report)

    for repeat in range(-warmup, repeats):
        order = trial_order(width, compact_type is not None, pair_compare, repeat)
        for mode in order:
            paired = mode.startswith("pair_")
            if pair_compare:
                set_cpu_pair(engine, paired)
            plain_mode = mode.removeprefix("pair_")
            is_compact = plain_mode.startswith("compact_")
            outcome = plain_mode.removeprefix("compact_")
            selected_checkpoint = compact_checkpoint if is_compact else checkpoint
            window.reset(initial)
            ids.copy_(good_ids)
            req.input_ids.copy_(good_history)
            expected_match = width - 1
            if outcome.startswith("reject_"):
                expected_match = int(outcome.split("_")[1])
                wrong = (int(good_ids[expected_match + 1].item()) + 1) % engine.config.model_config.vocab_size
                ids[expected_match + 1] = wrong
                req.input_ids[position + expected_match + 1] = wrong
            engine.stream.synchronize()
            import time
            started = time.perf_counter()
            matched = None
            if outcome == "graph_one":
                output = forward(0)
                count = 1
            elif outcome == "graph_all":
                output = torch.cat([forward(step) for step in range(width)])
                count = width
            else:
                target = forward(captured=True, compact=is_compact)
                output, matched = greedy_accept_prefix(ids[1:], target)
                count = matched + 1
                if matched < width - 1:
                    selected_checkpoint.restore(count)
            engine.stream.synchronize()
            elapsed = time.perf_counter() - started
            mismatches = list(errors)
            if not equal(output, reference_tokens[:count]):
                mismatches.append("output_tokens")
            mismatches += window.compare(states[count - 1], committed_end=position + count)
            if outcome not in ("graph_one", "graph_all"):
                if matched != expected_match:
                    mismatches.append("accepted_prefix_length")
                for length in range(1, min(count, width - 1) + 1):
                    if is_compact:
                        selected_checkpoint.restore(length)
                        mismatches += [f"restored_prefix/{length}/" + name for name in
                                       window.compare(states[length - 1], committed_end=position + length)]
                    else:
                        mismatches += [f"prefix/{length}/" + name
                                       for name, value in checkpoint.prefixes[length - 1].items()
                                       if not equal(value, states[length - 1][name])]
                if count < width:
                    ids.copy_(good_ids)
                    req.input_ids.copy_(good_history)
                    continued = forward(count)
                    if not equal(continued, reference_tokens[count:count + 1]):
                        mismatches.append("post_reject_token")
                    mismatches += ["post_reject/" + name for name in
                                   window.compare(states[count], committed_end=position + count + 1)]
            neighbour_mismatches = neighbour_errors()
            mismatches += neighbour_mismatches
            report["records"].append(dict(case=case, mode=mode, repeat=repeat, warmup=repeat < 0,
                                           wall_s=elapsed, matched=matched, output_tokens=count,
                                           cpu_pair_enabled=paired,
                                           graph_reused=graph_reused,
                                           request_table=req.table_idx,
                                           linear_slot=req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx,
                                           physical_pages=list(window.physical_pages),
                                           neighbours_unchanged=not neighbour_mismatches,
                                           checks_passed=not mismatches, mismatches=mismatches))
            base["save"](directory, report)
    window.reset(states[0])
    engine.stream.synchronize()
    if pair_compare:
        set_cpu_pair(engine, False)
    if graphs is None:
        close_graphs(dict(graph=graph, compact_graph=compact_graph,
                          compact_checkpoint=compact_checkpoint), engine.stream)
    return tokens[0], torch.cat((host_prefix, host_seed))


def original_seed_logits(engine, source, base):
    """Check the next ordinary token before the diagnostic relocates any pages."""
    from freetoken.attention.linear import build_fla_metadata

    batch = copy.copy(source)
    batch.reqs = [copy.copy(source.reqs[0])]
    batch.padded_reqs = batch.reqs
    window = base["StateWindow"](engine, batch.reqs[0], batch.reqs[0].cached_len, width=1)
    initial = window.capture()
    batch.fla_metadata = build_fla_metadata(batch, engine.device)
    engine.attn_backend.prepare_metadata(batch)
    engine.cpu_moe_executor.begin_decode_step()
    try:
        with engine.ctx.forward_batch(batch):
            if not engine.graph_runner.can_use_cuda_graph(batch):
                raise RuntimeError("ordinary control graph is unavailable")
            result = engine.graph_runner.replay(batch).float().clone()
        engine.cpu_moe_executor.raise_if_unhealthy()
        return result
    finally:
        window.reset(initial)
        engine.stream.synchronize()


def probe(engine, batch, directory, *, width, base, seed_api, repeats=4, warmup=1,
          pair_compare=False, relocatable_state=False, boundary_relocation=False):
    import hashlib
    import subprocess
    import sys
    from freetoken.kernel import _cpu_moe

    root = Path(__file__).resolve().parents[1]
    compact_type = base["_COMPACT_API"]["Checkpoint"] if base["_COMPACT_API"] else None
    graphs = {} if relocatable_state else None
    report = dict(completed=False, diagnostic_only=True, model_wall_qualified=False,
                  width=width, draft_tokens=width - 1, graph_enabled=True, trace_only=False,
                  checkpoint_enabled=True, serial_linear=True, records=[],
                  compact_enabled=compact_type is not None,
                  cpu_pair_compare=pair_compare,
                  relocatable_state=relocatable_state,
                  boundary_relocation=boundary_relocation,
                  cpu_max_tokens=engine.cpu_moe_executor.max_tokens,
                  ring_capacity=engine.kv_cache.ring_capacity,
                  native_sha256=hashlib.sha256(Path(_cpu_moe.__file__).read_bytes()).hexdigest(),
                  limitations=["No proposer or serving scheduler", "Repeated windows warm expert/file caches",
                               "All arms reserve wider CPU, PLE and QSA workspace",
                               "Exact local state checks are not broad quality evaluation"])
    try:
        runtime_root = Path(sys.modules[type(engine).__module__].__file__).resolve().parents[3]
        if runtime_root != root or report["cpu_max_tokens"] != width:
            raise RuntimeError("wider probe runtime or workspace differs from its source")
        report["source_revision"] = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        if subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip():
            raise RuntimeError("wider probe source is dirty")
        req = batch.reqs[0]
        position = req.cached_len
        seed, prefix = batch.input_ids[:1].clone(), req.input_ids[:position].clone()
        relocation_api = base.get("_RELOCATION_API") if boundary_relocation else None
        if boundary_relocation and (relocation_api is None or not relocatable_state):
            raise RuntimeError("boundary relocation requires its explicit helper and state bindings")
        eligible = (relocation_api["eligible_boundary"] if boundary_relocation else eligible_window)
        if len(prefix) != position or not eligible(position, engine.config.page_size, width, req.remain_len):
            raise RuntimeError("wider probe needs complete history and a fully allocated window")
        control = original_seed_logits(engine, batch, base) if boundary_relocation else None
        lease = (relocation_api["RelocationLease"](engine, batch, position, width, base["state_views"])
                 if boundary_relocation else nullcontext())
        with lease as relocation:
            if relocation is not None:
                report["relocation_layout"] = dict(relocation.layout)
            for offset in range(4):
                selected = relocation.select(offset) if relocation is not None else batch
                seed, prefix = run_window(engine, selected, position + offset, seed, prefix,
                                          width=width, base=base, seed_api=seed_api, report=report,
                                          directory=directory, repeats=repeats, warmup=warmup,
                                          compact_type=compact_type, pair_compare=pair_compare,
                                          relocatable_state=relocatable_state, graphs=graphs,
                                          relocation=relocation, control_logits=control if offset == 0 else None)
        report["summary"] = summarize(report["records"], width, compact_type is not None, pair_compare)
        report["checks_passed"] = (len(report["summary"]) == 4
                                   and all(case["checks_passed"] for case in report["summary"].values()))
        if relocatable_state:
            report["graph_reuse_qualified"] = graph_reuse_qualified(
                report["records"], report.get("graph_captures"), compact_type is not None)
            report["checks_passed"] &= report["graph_reuse_qualified"]
        if boundary_relocation:
            report["boundary_relocation_qualified"] = relocation_api["qualify"](
                report["records"], report.get("relocation_layout"), width)
            report["checks_passed"] &= report["boundary_relocation_qualified"]
        report["completed"] = True
    except BaseException as exc:
        report["error"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        try:
            if graphs is not None:
                close_graphs(graphs, engine.stream)
        finally:
            base["save"](directory, report)
