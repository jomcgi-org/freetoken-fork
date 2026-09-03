from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import SchedulerConfig
    from .scheduler import Scheduler

__all__ = ["Scheduler", "SchedulerConfig"]


def __getattr__(name: str):
    if name == "Scheduler":
        from .scheduler import Scheduler

        return Scheduler
    if name == "SchedulerConfig":
        from .config import SchedulerConfig

        return SchedulerConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
