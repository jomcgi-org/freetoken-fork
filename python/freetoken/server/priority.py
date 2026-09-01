from __future__ import annotations

from typing import Any


PRIORITY_HEADER = "x-request-priority"


def resolve_request_priority(body_priority: int = 0, request: Any | None = None) -> int:
    """Resolve FreeToken's OpenAI-compatible priority extension.

    The HTTP header wins when present. Keeping this at the adapter boundary means all
    downstream message types carry one already-resolved integer.
    """
    if request is None:
        return body_priority
    value = request.headers.get(PRIORITY_HEADER)
    if value is None:
        return body_priority
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{PRIORITY_HEADER} must be an integer, got {value!r}") from exc
