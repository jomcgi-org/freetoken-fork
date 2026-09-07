"""Compatibility entry point for the shared target checkpoint implementation."""

from freetoken.verification import checkpoint as _implementation
from freetoken.verification.checkpoint import (
    SeedCheckpoint, SlotStateBindings, capture_context, install,
)

def __getattr__(name):
    return getattr(_implementation, name)
