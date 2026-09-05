from types import SimpleNamespace

import pytest
from freetoken.prefill_timing import PrefillTimings, begin_prefill


class Event:
    def __init__(self, enable_timing):
        assert enable_timing
        self.ms = 250

    def record(self, stream):
        self.stream = stream

    def elapsed_time(self, ended):
        assert self.stream == ended.stream
        return self.ms


def marks():
    req = SimpleNamespace(uid=7, extend_len=64, device_len=128)
    batch = SimpleNamespace(reqs=[req])
    result = begin_prefill(batch, Event, "stream", clock=lambda: 10)
    result[1].record("stream")
    req.extend_len = 1
    req.device_len = 129
    return result


def test_event_duration_not_poll_delay_and_snapshot_before_mutation():
    timing = PrefillTimings(clock=lambda: 99)
    timing.complete(marks())
    chunk = timing.snapshot()["chunks"][0]
    assert chunk["elapsed_ms"] == 250
    assert chunk["tokens_per_second"] == 256
    assert chunk["dispatched_at_s"] == 10
    assert chunk["observed_at_s"] == 99
    assert chunk["requests"] == [{"uid": 7, "tokens": 64, "completed_tokens": 128}]
    assert timing.snapshot()["chunks"] == timing.snapshot()["chunks"]


def test_bounded_replay_with_monotonic_sequence():
    timing = PrefillTimings(capacity=2)
    for _ in range(3):
        timing.complete(marks())
    assert [c["sequence"] for c in timing.snapshot()["chunks"]] == [2, 3]


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf")])
def test_invalid_duration_is_not_a_rate(duration):
    timing = PrefillTimings()
    sample = marks()
    sample[0].ms = duration
    timing.complete(sample)
    assert timing.snapshot()["chunks"] == []
