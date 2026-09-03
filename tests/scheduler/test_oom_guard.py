"""GPU-free request-level OOM recovery tests.

The forward, CUDA allocator calls, streams, managers, and cache cleanup are all stubs.
No Engine is initialized and no CUDA device is required.
"""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import freetoken.scheduler.scheduler as scheduler_module
from freetoken.message import AbortBackendMsg, ErrorReplyMsg
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.scheduler import ForwardInput, Scheduler


def _batch(phase: str, reqs: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(
        phase=phase,
        is_prefill=phase == "prefill",
        is_decode=phase == "decode",
        reqs=reqs,
    )


def _forward_input(batch: SimpleNamespace) -> ForwardInput:
    return ForwardInput(batch=batch, sample_args=None, input_tuple=None, write_tuple=None)


def _req(uid: int, admission_order: int, cached_len: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        uid=uid,
        admission_order=admission_order,
        table_idx=uid,
        cached_len=cached_len,
    )


def _scheduler(monkeypatch):
    sched = Scheduler.__new__(Scheduler)
    calls = SimpleNamespace(
        cache=[], tables=[], prefill_aborts=[], decode_removes=[], empty_cache=0, sync=0,
    )

    def cache_req(req, *, finished, failed=False):
        calls.cache.append((req.uid, finished, failed))

    def free_table(table_idx):
        calls.tables.append(table_idx)

    def abort_prefill(uid):
        calls.prefill_aborts.append(uid)

    def remove_decode(req):
        calls.decode_removes.append(req.uid)

    def synchronize():
        calls.sync += 1

    def empty_cache():
        calls.empty_cache += 1

    sched.cache_manager = SimpleNamespace(cache_req=cache_req)
    sched.table_manager = SimpleNamespace(free=free_table)
    sched.prefill_manager = SimpleNamespace(
        runnable=True, pending_list=[], abort_req=abort_prefill
    )
    sched.decode_manager = SimpleNamespace(
        runnable=False,
        running_reqs=[],
        remove_req=remove_decode,
        abort_req=lambda uid: None,
    )
    sched.status_reporter = SimpleNamespace(oom_aborts=0, client_aborts=0)
    sched.engine = SimpleNamespace(stream=SimpleNamespace(synchronize=synchronize))
    sched.device = torch.device("cuda")
    sched.finished_reqs = set()
    sched._pending_oom_errors = {}
    sched._pending_abort_acks = set()
    sched._abort_tombstones = {}
    sched._completed_uids = {}
    sched._last_data = None
    sched._pending_rebuild = None
    sent: list[ErrorReplyMsg] = []
    sched.send_result = lambda replies: sent.extend(replies)
    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    monkeypatch.setattr(torch, "empty", lambda *args, **kwargs: object())
    return sched, calls, sent


def test_prefill_oom_aborts_batch_frees_resources_and_next_forward_runs(monkeypatch):
    sched, calls, sent = _scheduler(monkeypatch)
    reqs = [_req(10, 1), _req(11, 2)]
    current = _forward_input(_batch("prefill", reqs))
    forward_calls = 0

    def oom(_forward_input):
        nonlocal forward_calls
        forward_calls += 1
        raise torch.OutOfMemoryError("prefill allocation failed")

    sched._forward = oom
    processed = []
    sched.receive_msg = lambda blocking: []
    sched._schedule_next_batch = lambda: current
    sched._restore_linear_states = lambda _batch: None
    sched._process_last_data = processed.append
    Scheduler.normal_loop(sched)

    assert processed == [None]
    assert forward_calls == 1
    assert calls.prefill_aborts == [10, 11]
    assert calls.decode_removes == [10, 11]
    assert calls.cache == [(10, True, True), (11, True, True)]
    assert calls.tables == [10, 11]
    assert all(req.table_idx == -1 for req in reqs)
    assert calls.empty_cache == 1 and calls.sync == 1
    assert sched.status_reporter.oom_aborts == 2

    assert [reply.uid for reply in sent] == [10, 11]
    for reply in sent:
        assert reply.status_code == 503
        assert reply.code == "server_out_of_memory"
        assert "shorten prompt / lower max_tokens" in reply.error
        assert not hasattr(reply, "finish_reason")

    sched._forward = lambda _forward_input: "ok"
    next_input = _forward_input(_batch("prefill", [_req(12, 3)]))
    sched._schedule_next_batch = lambda: next_input
    Scheduler.normal_loop(sched)
    assert processed[-1] == (next_input, "ok")


def test_decode_oom_aborts_youngest_and_retry_recovers(monkeypatch):
    sched, calls, sent = _scheduler(monkeypatch)
    older, younger = _req(20, 4), _req(21, 5)
    current = _forward_input(_batch("decode", [older, younger]))
    seen: list[list[int]] = []

    def forward(value):
        uids = [req.uid for req in value.batch.reqs]
        seen.append(uids)
        if len(seen) == 1:
            raise torch.cuda.OutOfMemoryError("decode graph OOM")
        return "retried"

    sched._forward = forward
    sched._prepare_decode_retry = lambda reqs: _forward_input(_batch("decode", reqs))
    result = Scheduler._forward_with_oom_guard(sched, current)

    assert seen == [[20, 21], [20]]
    assert result is not None and result[0].batch.reqs == [older]
    assert result[1] == "retried"
    assert calls.decode_removes == [21]
    assert calls.cache == [(21, True, True)]
    assert younger.table_idx == -1 and older.table_idx == 20
    assert calls.empty_cache == 1 and sched.status_reporter.oom_aborts == 1
    Scheduler._flush_oom_errors(sched)
    assert [reply.uid for reply in sent] == [21]


def test_decode_oom_retry_failure_aborts_remainder_and_loop_continues(monkeypatch):
    sched, calls, sent = _scheduler(monkeypatch)
    older, younger = _req(30, 6), _req(31, 7)
    current = _forward_input(_batch("decode", [older, younger]))
    seen: list[list[int]] = []

    def oom(value):
        seen.append([req.uid for req in value.batch.reqs])
        raise torch.OutOfMemoryError("still out of memory")

    sched._forward = oom
    sched._prepare_decode_retry = lambda reqs: _forward_input(_batch("decode", reqs))
    assert Scheduler._forward_with_oom_guard(sched, current) is None

    assert seen == [[30, 31], [30]]
    assert calls.decode_removes == [31, 30]
    assert calls.cache == [(31, True, True), (30, True, True)]
    assert older.table_idx == younger.table_idx == -1
    assert calls.empty_cache == 2 and calls.sync == 2
    assert sched.status_reporter.oom_aborts == 2
    Scheduler._flush_oom_errors(sched)
    assert [reply.uid for reply in sent] == [31, 30]

    sched._forward = lambda _forward_input: "alive"
    next_input = _forward_input(_batch("decode", [_req(32, 8)]))
    assert Scheduler._forward_with_oom_guard(sched, next_input) == (next_input, "alive")


def test_poisoned_cuda_context_logs_loudly_and_exits_nonzero(monkeypatch):
    sched, calls, _sent = _scheduler(monkeypatch)
    req = _req(40, 9)
    current = _forward_input(_batch("prefill", [req]))
    logs: list[str] = []

    class Logger:
        def warning_rank0(self, message, *args):
            logs.append(message % args)

        def critical_rank0(self, message, *args):
            logs.append(message % args)

    monkeypatch.setattr(scheduler_module, "logger", Logger())
    sched._forward = lambda _value: (_ for _ in ()).throw(
        torch.OutOfMemoryError("forward OOM")
    )
    monkeypatch.setattr(
        torch,
        "empty",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            torch.cuda.OutOfMemoryError("CUDA context is unusable")
        ),
    )

    with pytest.raises(SystemExit) as exc_info:
        Scheduler._forward_with_oom_guard(sched, current)

    assert exc_info.value.code != 0
    assert calls.cache == [(40, True, True)]
    assert calls.empty_cache == 1
    assert any("CUDA CONTEXT CORRUPTION" in message for message in logs)


def test_decode_status_line_includes_cumulative_oom_aborts():
    logs: list[str] = []
    reporter = scheduler_module.SchedulerStatusReporter(
        log=logs.append, clock=lambda: 1.0, decode_log_interval=1
    )
    reporter.record_oom_aborts(3)
    reporter.report_batch(
        _batch("decode", [_req(50, 10)]),
        running_reqs=1,
        queue_reqs=0,
        kv_used_pages=1,
        kv_total_pages=2,
        page_size=1,
    )
    assert "oom_aborts: 3" in logs[-1]


def test_failed_cache_cleanup_discards_all_pools_without_prefix_commit():
    calls = SimpleNamespace(unlock=[], kv=[], swa=[], gdn=[], profiles=[])
    manager = SimpleNamespace(
        page_size=1,
        page_table=torch.tensor([[100, 101, 102, 103, 104, 105]], dtype=torch.int32),
        swa_paged=True,
        is_hybrid=True,
        unlock=lambda handle: calls.unlock.append(handle),
        _free=lambda indices: calls.kv.append(indices.tolist()),
        _free_swa=lambda indices: calls.swa.append(indices.tolist()),
        _free_req_slots=lambda req: calls.gdn.append(req.uid),
        abort_pending_expert_profile=lambda uid: calls.profiles.append(uid),
    )
    handle = SimpleNamespace(cached_len=2)
    req = SimpleNamespace(uid=60, table_idx=0, device_len=5, cache_handle=handle)

    CacheManager._discard_failed_req(manager, req)

    assert calls.unlock == [handle]
    assert calls.kv == [[102, 103, 104]]
    assert calls.swa == [[102, 103, 104]]
    assert calls.gdn == [60]
    assert calls.profiles == [60]


def test_client_abort_drops_queued_request_before_admission(monkeypatch):
    sched, calls, _sent = _scheduler(monkeypatch)
    pending = SimpleNamespace(uid=70, chunked_req=None)
    sched.prefill_manager.pending_list = [pending]
    logs = []
    monkeypatch.setattr(
        scheduler_module,
        "logger",
        SimpleNamespace(
            warning_rank0=lambda message, *args: logs.append(message % args),
            debug_rank0=lambda *_args: None,
        ),
    )

    def abort_prefill(uid):
        calls.prefill_aborts.append(uid)
        sched.prefill_manager.pending_list = [
            req for req in sched.prefill_manager.pending_list if req.uid != uid
        ]
        return None

    sched.prefill_manager.abort_req = abort_prefill

    Scheduler._process_one_msg(
        sched, AbortBackendMsg(uid=70, client_disconnected=True)
    )

    assert sched.prefill_manager.pending_list == []
    assert calls.prefill_aborts == [70]
    assert calls.cache == [] and calls.tables == []
    assert sched.status_reporter.client_aborts == 1
    assert 70 in sched._pending_abort_acks
    assert logs == [
        "Client abort request_id=70, phase=queued, tokens_processed=0"
    ]


def test_client_abort_of_inflight_request_releases_owned_slots(monkeypatch):
    sched, calls, _sent = _scheduler(monkeypatch)
    req = type("ReqDouble", (), {})()
    req.uid = 71
    req.admission_order = 11
    req.table_idx = 71
    req.cached_len = 123
    req.device_len = 124
    req.aborted = False
    sched.decode_manager.running_reqs = [req]
    logs = []
    monkeypatch.setattr(
        scheduler_module,
        "logger",
        SimpleNamespace(
            warning_rank0=lambda message, *args: logs.append(message % args),
            debug_rank0=lambda *_args: None,
        ),
    )

    def abort_decode(uid):
        for running in list(sched.decode_manager.running_reqs):
            if running.uid == uid:
                sched.decode_manager.running_reqs.remove(running)
                return running
        return None

    sched.decode_manager.abort_req = abort_decode
    batch = _batch("prefill", [req])
    copy_done = SimpleNamespace(synchronize=lambda: None)
    sched._last_data = (
        SimpleNamespace(batch=batch),
        (None, torch.tensor([], dtype=torch.int32), copy_done),
    )
    sched.cache_manager.lazy_free_region = nullcontext
    sched._kv_usage_pages = lambda: (0, 0)
    sched._mamba_slot_usage = lambda: None
    sched._swa_token_usage = lambda: None
    sched._gpu_mem_bytes = lambda: 0
    sched.status_reporter.report_batch = lambda *args, **kwargs: None
    sched.config = SimpleNamespace(page_size=1)

    Scheduler._process_one_msg(
        sched, AbortBackendMsg(uid=71, client_disconnected=True)
    )

    assert req.aborted
    assert req.table_idx == 71
    assert calls.cache == [] and calls.tables == []

    Scheduler._process_last_data(sched, sched._last_data)

    # failed=True selects the OOM guard's discard path: release every request-owned
    # KV/GDN allocation without inserting a prefix or parking a session profile.
    assert calls.cache == [(71, True, True)]
    assert calls.tables == [71]
    assert req.table_idx == -1
    assert req.device_len == 123
    assert sched.decode_manager.running_reqs == []
    assert sched.status_reporter.client_aborts == 1
    assert logs == [
        "Client abort request_id=71, phase=prefill, tokens_processed=123"
    ]


def test_client_abort_does_not_abort_completed_request_during_output_drain(monkeypatch):
    sched, calls, _sent = _scheduler(monkeypatch)
    req = _req(72, 12, cached_len=128)
    req.table_idx = -1
    sched.finished_reqs = [req]

    def unexpected_abort(_uid):
        raise AssertionError("completed request must not reach manager abort paths")

    sched.prefill_manager.abort_req = unexpected_abort
    sched.decode_manager.abort_req = unexpected_abort

    Scheduler._process_one_msg(
        sched, AbortBackendMsg(uid=72, client_disconnected=True)
    )

    assert calls.cache == [] and calls.tables == []
    assert sched.status_reporter.client_aborts == 0
    assert 72 not in sched._abort_tombstones
    assert 72 not in sched._pending_abort_acks
