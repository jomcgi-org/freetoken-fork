from __future__ import annotations

from types import SimpleNamespace

from freetoken.moe.cpu_executor import CpuMoeExecutor, _StepTimingEvents


class _Mark:
    def __init__(self, milliseconds: float):
        self.milliseconds = milliseconds

    def elapsed_time(self, other: "_Mark") -> float:
        return other.milliseconds - self.milliseconds


def test_step_timing_breakdown_aggregates_native_rows_and_d2h():
    executor = CpuMoeExecutor.__new__(CpuMoeExecutor)
    executor._step_timing = True
    executor._step_timing_events = {
        (2, 4): _StepTimingEvents(
            _Mark(0), _Mark(1), _Mark(1.25), _Mark(2), _Mark(3), _Mark(4)
        ),
        (7, 4): _StepTimingEvents(
            _Mark(5), _Mark(6), _Mark(6.5), _Mark(7), _Mark(8), _Mark(9)
        ),
    }
    executor._ext = SimpleNamespace(
        step_timing_snapshot_and_reset=lambda: {
            "2": {
                "wake_us": 10,
                "groups_us": 2,
                "gil_us": 3,
                "precb_us": 4,
                "notify_us": 1,
                "coord_pre_us": 2,
                "coord_post_us": 3,
                "compute_us": 100,
                "signal_us": 5,
                "tasks": 1,
                "experts": 4,
                "bytes": 4096,
            },
            7: {
                "wake_us": 20,
                "groups_us": 4,
                "gil_us": 5,
                "precb_us": 6,
                "notify_us": 5,
                "coord_pre_us": 4,
                "coord_post_us": 5,
                "compute_us": 200,
                "signal_us": 6,
                "tasks": 2,
                "experts": 8,
                "bytes": 8192,
            },
        }
    )

    result = executor.step_timing_breakdown(bs=4)

    assert result["per_layer"][2] == {
        "wake_us": 10.0,
        "groups_us": 2.0,
        "gil_us": 3.0,
        "precb_us": 4.0,
        "notify_us": 1.0,
        "coord_pre_us": 2.0,
        "coord_post_us": 3.0,
        "h2d_us": 1000.0,
        "compute_us": 100.0,
        "signal_us": 5.0,
        "tasks": 1,
        "experts": 4,
        "bytes": 4096,
    }
    assert result["total"] == {
        "wake_us": 30.0,
        "groups_us": 6.0,
        "gil_us": 8.0,
        "precb_us": 10.0,
        "notify_us": 6.0,
        "coord_pre_us": 6.0,
        "coord_post_us": 8.0,
        "h2d_us": 2000.0,
        "compute_us": 300.0,
        "signal_us": 11.0,
        "total_tasks": 3,
        "total_experts": 12,
        "total_bytes": 12_288,
    }
    assert result["submit_d2h_us"] == {
        "per_layer": {2: 250.0, 7: 500.0},
        "total": 750.0,
    }


def test_step_timing_breakdown_preserves_native_wake_decomposition():
    executor = CpuMoeExecutor.__new__(CpuMoeExecutor)
    executor._step_timing = True
    executor._step_timing_events = {}
    executor._ext = SimpleNamespace(
        step_timing_snapshot_and_reset=lambda: {
            3: {
                "wake_us": 17,
                "groups_us": 2,
                "gil_us": 4,
                "precb_us": 8,
                "notify_us": 3,
                "tasks": 1,
            }
        }
    )

    row = executor.step_timing_breakdown()["per_layer"][3]

    assert row["groups_us"] + row["gil_us"] + row["precb_us"] + row["notify_us"] == 17
    assert row["wake_us"] == 17


def test_pre_run_callback_mode_is_forwarded_to_native_executor():
    calls = []
    executor = CpuMoeExecutor.__new__(CpuMoeExecutor)
    executor._ext = SimpleNamespace(set_pre_run_callback_mode=calls.append)

    executor._configure_pre_run_callback_mode("before")
    executor._configure_pre_run_callback_mode("after")

    assert calls == [1]


def test_step_timing_breakdown_is_zero_and_does_not_call_native_when_off():
    executor = CpuMoeExecutor.__new__(CpuMoeExecutor)
    executor._step_timing = False
    executor._ext = SimpleNamespace(
        step_timing_snapshot_and_reset=lambda: (_ for _ in ()).throw(AssertionError())
    )

    assert executor.step_timing_breakdown() == {
        "per_layer": {},
        "total": {
            "wake_us": 0.0,
            "groups_us": 0.0,
            "gil_us": 0.0,
            "precb_us": 0.0,
            "notify_us": 0.0,
            "coord_pre_us": 0.0,
            "coord_post_us": 0.0,
            "h2d_us": 0.0,
            "compute_us": 0.0,
            "signal_us": 0.0,
            "total_tasks": 0,
            "total_experts": 0,
            "total_bytes": 0,
        },
        "submit_d2h_us": {"per_layer": {}, "total": 0.0},
    }
