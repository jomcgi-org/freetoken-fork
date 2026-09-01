from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Sequence, TypeVar

import torch

if TYPE_CHECKING:
    from freetoken.core import SamplingParams

    from .prefill import ChunkedReq


@dataclass
class PendingReq:
    uid: int
    input_ids: torch.Tensor
    sampling_params: SamplingParams
    chunked_req: ChunkedReq | None = None
    mm_embeds: torch.Tensor | None = None
    priority: int = 0
    arrival_time: float = field(default_factory=time.monotonic)
    expert_profile: Any | None = None

    @property
    def input_len(self) -> int:
        return len(self.input_ids)

    @property
    def output_len(self) -> int:
        return self.sampling_params.max_tokens


@dataclass
class ScheduleResult:
    reqs: List[PendingReq]
    output_indices: List[torch.Tensor]


_PriorityReq = TypeVar("_PriorityReq", bound=PendingReq)


def effective_priority(
    priority: int,
    arrival_time: float,
    now: float,
    aging_seconds: float,
) -> int:
    """Requested priority plus one point per complete aging interval waited."""
    if aging_seconds <= 0:
        return priority
    waited = max(0.0, now - arrival_time)
    return priority + int(waited // aging_seconds)


def order_pending_requests(
    pending: Sequence[_PriorityReq],
    *,
    now: float,
    aging_seconds: float,
) -> List[_PriorityReq]:
    """Order by effective priority descending, then arrival ascending.

    The all-zero fast path is also a compatibility guarantee: requests that do not use
    the extension retain the queue's existing FIFO order byte-for-byte, including ties
    introduced by multiple tokenizer workers.
    """
    if not any(req.priority != 0 for req in pending):
        return list(pending)
    return sorted(
        pending,
        key=lambda req: (
            -effective_priority(req.priority, req.arrival_time, now, aging_seconds),
            req.arrival_time,
        ),
    )


def priority_queue_stats(
    pending: Sequence[PendingReq], *, now: float
) -> tuple[dict[str, int], float]:
    """Current queue depth in requested-priority bands and oldest wait."""
    bands = {"negative": 0, "zero": 0, "positive": 0}
    max_wait = 0.0
    for req in pending:
        band = "negative" if req.priority < 0 else "positive" if req.priority > 0 else "zero"
        bands[band] += 1
        max_wait = max(max_wait, max(0.0, now - req.arrival_time))
    return bands, max_wait
