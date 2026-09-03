"""Scheduler-worker signal ordering for HOT plan persistence."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from freetoken.server import launch
from freetoken.server.launch import _scheduler_sigterm_handler


def test_sigterm_quiets_periodic_writes_before_existing_shutdown_drain():
    events = []

    class Cache:
        def request_hot_plan_stop(self):
            events.append("stop")

        def shutdown_hot_adaptation(self):
            events.extend(("drain", "write"))

    cache = Cache()

    class Scheduler:
        engine = SimpleNamespace(moe_offload_cache=cache)

        def shutdown(self):
            self.engine.moe_offload_cache.shutdown_hot_adaptation()

    scheduler = Scheduler()
    handler = _scheduler_sigterm_handler(scheduler)

    with pytest.raises(KeyboardInterrupt):
        handler(None, None)
    assert handler.stop_event.is_set()
    handler(None, None)
    scheduler.shutdown()

    assert events == ["stop", "drain", "write"]


def test_scheduler_sigterm_scope_installs_and_restores_handler(monkeypatch):
    scheduler = object()
    installed = object()
    previous = object()
    calls = []

    monkeypatch.setattr(
        launch, "_scheduler_sigterm_handler", lambda value: installed
    )

    def fake_signal(signum, handler):
        calls.append((signum, handler))
        return previous

    monkeypatch.setattr(launch.signal, "signal", fake_signal)

    with launch._scheduler_sigterm_handler_installed(scheduler):
        assert calls == [(launch.signal.SIGTERM, installed)]

    assert calls == [
        (launch.signal.SIGTERM, installed),
        (launch.signal.SIGTERM, previous),
    ]
