"""Diagnostic CUDA-event spans for layer-major prefill (off unless enabled).

``span(name)`` brackets the enclosed GPU work with timing events while a collection
is active. Spans may nest; sums per name then include nested spans.
"""

from __future__ import annotations

import contextlib

import torch

_active: list | None = None


def enable() -> None:
    global _active
    _active = []


def collect() -> list:
    """Stop collecting and return ``[(name, start_event, end_event), ...]``."""
    global _active
    spans, _active = _active or [], None
    return spans


@contextlib.contextmanager
def span(name: str):
    if _active is None:
        yield
        return
    start = torch.cuda.Event(enable_timing=True)
    start.record()
    try:
        yield
    finally:
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        if _active is not None:
            _active.append((name, start, end))
