"""Scheduler idle-hook routing without constructing an engine."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from freetoken.scheduler.io import SchedulerIOMixin
from freetoken.scheduler.scheduler import Scheduler
from freetoken.utils.mp import ZmqPullQueue, ZmqSubQueue


class _Queue:
    def __init__(self, empty: bool) -> None:
        self._empty = empty
        self.waits = []

    def empty(self) -> bool:
        return self._empty

    def wait_for_item(self, timeout_seconds: float) -> bool:
        self.waits.append(timeout_seconds)
        return not self._empty


@pytest.mark.parametrize("queue_type", [ZmqPullQueue, ZmqSubQueue])
def test_receive_queue_wait_for_item_uses_bounded_socket_poll(queue_type):
    calls = []

    class Socket:
        def poll(self, *, timeout):
            calls.append(timeout)
            return 1

    queue = object.__new__(queue_type)
    queue.socket = Socket()

    assert queue.wait_for_item(0.1234)
    assert calls == [124]


def test_scheduler_idle_hook_uses_the_tokenizer_receive_queue(monkeypatch):
    pending = []
    tokenizer_queue = _Queue(False)
    scheduler = SimpleNamespace(
        cache_manager=SimpleNamespace(check_integrity=lambda: None),
        engine=SimpleNamespace(
            moe_offload_cache=SimpleNamespace(
                hot_adapt_while_idle=lambda request_pending, wait_for_request: (
                    pending.append(request_pending()),
                    wait_for_request(0.25),
                )
            )
        ),
        config=SimpleNamespace(tp_info=SimpleNamespace(size=1)),
        _recv_from_tokenizer=tokenizer_queue,
    )
    monkeypatch.setattr(
        "freetoken.scheduler.scheduler.logger.info_rank0", lambda _message: None
    )

    Scheduler.run_when_idle(scheduler)

    assert pending == [True]
    assert tokenizer_queue.waits == [0.25]


def test_scheduler_idle_hook_is_disabled_for_tensor_parallelism(monkeypatch):
    calls = []
    scheduler = SimpleNamespace(
        cache_manager=SimpleNamespace(check_integrity=lambda: None),
        engine=SimpleNamespace(
            moe_offload_cache=SimpleNamespace(
                hot_adapt_while_idle=lambda *_args: calls.append("idle")
            )
        ),
        config=SimpleNamespace(
            tp_info=SimpleNamespace(size=2, is_primary=lambda: True)
        ),
        _recv_from_tokenizer=_Queue(True),
    )
    monkeypatch.setattr(
        "freetoken.scheduler.scheduler.logger.info_rank0", calls.append
    )

    Scheduler.run_when_idle(scheduler)
    Scheduler.run_when_idle(scheduler)

    assert calls.count("idle") == 0
    assert sum("tensor parallel size is 2" in call for call in calls) == 1


def test_offline_receive_never_calls_run_when_idle():
    calls = []

    class OfflineIO(SchedulerIOMixin):
        def run_when_idle(self):
            calls.append("idle")

        def offline_receive_msg(self, blocking=False):
            calls.append(("offline", blocking))
            return []

    offline = object.__new__(OfflineIO)
    config = SimpleNamespace(
        offline_mode=True,
        tp_info=SimpleNamespace(is_primary=lambda: True),
    )
    SchedulerIOMixin.__init__(offline, config, tp_cpu_group=None)

    assert offline.receive_msg(blocking=True) == []
    assert calls == [("offline", True)]
