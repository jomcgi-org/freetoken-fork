from __future__ import annotations

from types import SimpleNamespace

from freetoken.scheduler.utils import (
    effective_priority,
    order_pending_requests,
    priority_queue_stats,
)


def _req(uid: int, priority: int, arrival_time: float):
    return SimpleNamespace(uid=uid, priority=priority, arrival_time=arrival_time)


def test_waiting_queue_orders_by_priority_then_arrival():
    pending = [
        _req(1, priority=1, arrival_time=1.0),
        _req(2, priority=3, arrival_time=3.0),
        _req(3, priority=3, arrival_time=2.0),
    ]

    ordered = order_pending_requests(pending, now=10.0, aging_seconds=0)

    assert [req.uid for req in ordered] == [3, 2, 1]


def test_all_default_priority_preserves_existing_fifo_order():
    pending = [
        _req(1, priority=0, arrival_time=3.0),
        _req(2, priority=0, arrival_time=1.0),
    ]

    ordered = order_pending_requests(pending, now=100.0, aging_seconds=10.0)

    assert ordered == pending


def test_effective_priority_ages_once_per_complete_interval():
    assert effective_priority(2, arrival_time=10.0, now=39.9, aging_seconds=30.0) == 2
    assert effective_priority(2, arrival_time=10.0, now=40.0, aging_seconds=30.0) == 3
    assert effective_priority(2, arrival_time=10.0, now=100.0, aging_seconds=30.0) == 5
    assert effective_priority(2, arrival_time=10.0, now=100.0, aging_seconds=0) == 2


def test_queue_stats_report_requested_priority_bands_and_oldest_wait():
    pending = [
        _req(1, priority=-1, arrival_time=90.0),
        _req(2, priority=0, arrival_time=80.0),
        _req(3, priority=4, arrival_time=95.0),
    ]

    bands, max_wait = priority_queue_stats(pending, now=100.0)

    assert bands == {"negative": 1, "zero": 1, "positive": 1}
    assert max_wait == 20.0


def test_aging_bounds_starvation_under_continuous_high_priority_arrivals():
    low = _req(1, priority=0, arrival_time=0.0)
    aging_seconds = 10.0

    for uid, now in enumerate((1.0, 11.0, 21.0), start=2):
        high = _req(uid, priority=3, arrival_time=now)
        assert order_pending_requests(
            [low, high], now=now, aging_seconds=aging_seconds
        )[0] is high

    high = _req(5, priority=3, arrival_time=30.0)
    assert order_pending_requests(
        [low, high], now=30.0, aging_seconds=aging_seconds
    )[0] is low
