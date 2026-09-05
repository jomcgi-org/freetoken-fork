"""Bounded prefill chunk telemetry, resolved at the existing output fence.

CUDA event elapsed time includes stream execution, transfers and host dispatch
gaps between the marks. It excludes admission, tokenization and transport. Host
timestamps describe dispatch and observation, not precise GPU event timestamps.
"""

import math
import time
from collections import deque


def begin_prefill(batch, event_factory, stream, clock=time.monotonic):
    requests = [
        {"uid": r.uid, "tokens": r.extend_len, "completed_tokens": r.device_len}
        for r in batch.reqs
    ]
    started = event_factory(enable_timing=True)
    ended = event_factory(enable_timing=True)
    dispatched_at = clock()
    started.record(stream)
    return started, ended, dispatched_at, requests


class PrefillTimings:
    def __init__(self, capacity=256, clock=time.monotonic):
        self.chunks = deque(maxlen=capacity)
        self.sequence = 0
        self.clock = clock

    def complete(self, marks):
        # The caller has already synchronized the sampled-output fence. Do not
        # synchronize here, or charge the next overlapped batch to this batch.
        started, ended, dispatched_at, requests = marks
        elapsed_ms = started.elapsed_time(ended)
        tokens = sum(r["tokens"] for r in requests)
        if not math.isfinite(elapsed_ms) or elapsed_ms <= 0 or tokens <= 0:
            return
        self.sequence += 1
        self.chunks.append(
            {
                "sequence": self.sequence,
                "dispatched_at_s": dispatched_at,
                "observed_at_s": self.clock(),
                "elapsed_ms": elapsed_ms,
                "tokens": tokens,
                "tokens_per_second": tokens * 1000 / elapsed_ms,
                "requests": requests,
            }
        )

    def snapshot(self):
        return {"clock_s": self.clock(), "chunks": list(self.chunks)}
