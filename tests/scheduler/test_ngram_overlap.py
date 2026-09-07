"""Drive real scheduling and host drains with CPU pages and controlled forwards."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from freetoken.core import Batch
from freetoken.message import AbortBackendMsg
from freetoken.scheduler.prefill import ChunkedReq
from freetoken.scheduler.scheduler import ForwardInput, Scheduler
from freetoken.verification.runtime import NgramTarget
from tests.scheduler.test_abort_inflight_prefill import _launch_req, _setup


def rig(*, phase="decode", matching=True, cls=None):
    pool, cm, tm, dm, pm, sent, sched = _setup()
    suffix = list(range(7)) + [42]
    history = suffix + [31, 32, 33, 34] + suffix[:-1]
    if not matching:
        history = list(range(100, 119))
    kwargs = {} if cls is None else {"cls": cls}
    req = _launch_req(pool, cm, tm, torch.tensor(history, dtype=torch.int32),
                      output_len=32, **kwargs)
    dm.filter_reqs([req])
    calls = []
    sched.config.speculative_ngram = "on"
    sched.config.speculative_mtp = "off"
    sched._forward_iter = 0
    sched._pending_rebuild = None
    sched.prefill_budget = 64
    sched._kv_ladder_has_drainable_waiter = lambda: False
    sched._drain_kv_ladder_waiting = lambda: calls.append("idle")
    sched._queue_stats = lambda: (0, {}, 0.0)
    sched.receive_msg = lambda blocking: []
    sched.engine_stream_ctx = nullcontext()
    sched.stream = SimpleNamespace(wait_stream=lambda stream: calls.append("scheduler_wait"))
    engine_stream = SimpleNamespace(wait_stream=lambda stream: calls.append("engine_wait"))
    engine = sched.engine = SimpleNamespace(
        stream=engine_stream, moe_offload_cache=None, config=SimpleNamespace(ngram_debug=False),
        graph_runner=SimpleNamespace(pad_batch=lambda batch: setattr(batch, "padded_reqs", batch.reqs)),
        resolve_mtp_timing=lambda batch: None,
    )
    target = engine.ngram_target = NgramTarget(engine)
    target.checkpoint = SimpleNamespace(restore=lambda count: calls.append(("restore", count)))
    original_release = target.release

    def release(batch):
        calls.append("release")
        original_release(batch)

    target.release = release
    sched._restore_linear_states = lambda batch: None
    sched._build_forward_input = lambda batch: ForwardInput(batch, None, None, None)
    allocate = cm.allocate_paged

    def allocate_paged(reqs):
        assert all(r.table_idx >= 0 for r in reqs)
        calls.append("allocate")
        allocate(reqs)

    cm.allocate_paged = allocate_paged
    append = req.append_host

    def append_host(tokens):
        calls.append(("append", tokens.tolist()))
        append(tokens)

    req.append_host = append_host
    free = sched._free_req_resources

    def free_req(request, **kwargs):
        calls.append("free")
        free(request, **kwargs)

    sched._free_req_resources = free_req
    tokens = torch.tensor([-999], dtype=torch.int32)

    def fence():
        calls.append("pending_ready")
        tokens.fill_(42)

    prior = Batch(reqs=[req], phase=phase)
    prior.padded_reqs = [req]
    last = (ForwardInput(prior, None, None, None), (None, tokens, SimpleNamespace(synchronize=fence)))

    def forward(data):
        batch = data.batch
        assert target.owner is None
        if getattr(batch, "ngram_verify", False):
            assert req.input_ids.numel() == batch.mtp_original_device_len
            calls.append("verify")
            target.owner = batch
            req.cached_len = batch.mtp_original_cached_len + 2
            req.device_len = batch.mtp_original_device_len + 2
            batch.generated_tokens = 2
            output = torch.tensor([31, 32], dtype=torch.int32)
        else:
            calls.append("ordinary")
            req.complete_one()
            batch.generated_tokens = 1
            output = torch.tensor([80], dtype=torch.int32)
        dm.filter_reqs([req])
        return None, output, SimpleNamespace(synchronize=lambda: calls.append("output_ready"))

    sched._forward = forward
    return SimpleNamespace(sched=sched, req=req, calls=calls, sent=sent, last=last,
                           target=target, cm=cm, dm=dm, forward=forward)


@pytest.mark.parametrize("phase", ["decode", "prefill"])
def test_candidate_drains_before_reservation_and_releases_before_next_launch(phase):
    f = rig(phase=phase)
    assert f.sched.overlap_loop(f.last) is None
    assert f.calls.index("pending_ready") < f.calls.index(("append", [42]))
    assert f.calls.index(("append", [42])) < f.calls.index("allocate") < f.calls.index("verify")
    assert f.calls.index(("append", [32])) < f.calls.index("release")
    assert [m.next_token for m in f.sent] == [42, 31, 32]
    assert f.target.owner is None and f.sched._last_data is None
    ongoing = f.sched.overlap_loop(None)
    assert ongoing is not None
    assert f.calls.index("release") < len(f.calls) - 1
    f.sched._process_last_data(ongoing)
    assert [m.next_token for m in f.sent] == [42, 31, 32, 80]
    f.cm.check_integrity()


@pytest.mark.parametrize("reason", ["no_match", "backoff", "sampling", "grammar", "lazy", "budget", "off"])
def test_ordinary_fallback_still_launches_before_prior_host_drain(reason):
    f = rig(matching=reason != "no_match")
    if reason == "backoff": f.req._ngram_retry_at = f.req.device_len + 16
    elif reason == "sampling": f.req.sampling_params.temperature = 1.0
    elif reason == "grammar": f.req.guided_state = object()
    elif reason == "lazy": f.last[0].batch.lazy_restore_pending = True
    elif reason == "budget": f.req.max_device_len = f.req.device_len + 4
    elif reason == "off": f.sched.config.speculative_ngram = "off"
    if reason == "grammar": f.req.guided_state = SimpleNamespace(terminated=False)
    ongoing = f.sched.overlap_loop(f.last)
    assert ongoing is not None and "verify" not in f.calls
    assert f.calls.index("ordinary") < f.calls.index(("append", [42]))
    assert [m.next_token for m in f.sent] == [42]
    if reason in ("backoff", "sampling", "grammar", "lazy", "budget", "off"):
        assert f.calls.index("ordinary") < f.calls.index("pending_ready")
    f.sched._process_last_data(ongoing)
    f.cm.check_integrity()


@pytest.mark.parametrize("stop", ["eos", "string"])
def test_pending_terminal_token_is_freed_before_any_speculative_allocation(stop):
    f = rig()
    if stop == "eos": f.sched.eos_token_ids = {42}
    else:
        f.req.sampling_params.stop_strs = ["STOP"]
        f.sched._match_stop_str = lambda req: "STOP"
    assert f.sched.overlap_loop(f.last) is None
    assert "allocate" not in f.calls and "verify" not in f.calls
    assert f.calls.count("free") == 1 and f.req.table_idx == -1
    assert len(f.sent) == 1 and f.sent[0].finished and f.sent[0].finish_reason == "stop"
    f.cm.check_integrity()


def test_new_tool_anchor_is_rechecked_after_pending_host_drain():
    f = rig()
    f.sched.toolcall_anchor_id = 42
    ongoing = f.sched.overlap_loop(f.last)
    assert f.req.toolcall_anchor_len == 20
    assert "verify" not in f.calls and ongoing is not None
    assert f.calls.index(("append", [42])) < f.calls.index("ordinary")
    f.sched._process_last_data(ongoing)
    f.cm.check_integrity()


def test_abort_arrival_marks_then_frees_at_drain_without_a_forward():
    f = rig()
    f.sched.receive_msg = lambda blocking: [AbortBackendMsg(uid=f.req.uid)]
    assert f.sched.overlap_loop(f.last) is None
    assert "allocate" not in f.calls and "verify" not in f.calls
    assert f.calls.index("pending_ready") < f.calls.index("free")
    assert f.calls.count("free") == 1 and f.req.table_idx == -1
    assert len(f.sent) == 1 and f.sent[0].error == "request aborted"
    f.cm.check_integrity()


def test_chunked_prefill_never_looks_ahead_into_its_non_output_token():
    f = rig(phase="prefill", cls=ChunkedReq)
    assert not f.sched._ngram_needs_drain(f.last)
    assert f.calls == []


def test_speculative_stop_rolls_back_and_releases_in_the_same_iteration():
    f = rig()
    f.req.sampling_params.stop_strs = ["STOP"]
    f.sched._match_stop_str = lambda req: "STOP" if req.input_ids[-1].item() == 31 else None
    assert f.sched.overlap_loop(f.last) is None
    assert f.calls.index(("restore", 1)) < f.calls.index("free") < f.calls.index("release")
    assert f.target.owner is None
    assert [m.next_token for m in f.sent] == [42, 31] and f.sent[-1].finished
    f.cm.check_integrity()


def test_forward_oom_preserves_prior_drain_and_releases_target_before_error_flush():
    f = rig()

    def oom(data):
        f.target.owner = data.batch
        raise torch.OutOfMemoryError("synthetic target failure")

    def recover(data, error):
        f.sched._restore_failed_request_lengths(data)
        assert f.target.owner is None
        f.calls.append("recover")
        return None

    f.sched._forward = oom
    f.sched._recover_forward_oom = recover
    assert f.sched.overlap_loop(f.last) is None
    assert [m.next_token for m in f.sent] == [42]
    assert f.calls.index(("append", [42])) < f.calls.index("recover")


@pytest.mark.parametrize("disable,mtp", [(False, "off"), (True, "off"), (False, "on")])
def test_ngram_uses_overlap_unless_the_explicit_serial_controls_require_it(monkeypatch, disable, mtp):
    from freetoken.env import ENV

    f = rig()
    f.sched.config.speculative_mtp = mtp
    monkeypatch.setattr(ENV.DISABLE_OVERLAP_SCHEDULING, "value", disable)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: f.sched.stream)

    def stop(mode):
        raise RuntimeError(mode)

    f.sched.normal_loop = lambda: stop("serial")
    f.sched.overlap_loop = lambda data: stop("overlap")
    with pytest.raises(RuntimeError, match="serial" if disable or mtp == "on" else "overlap"):
        f.sched.run_forever()
