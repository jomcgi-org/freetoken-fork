#!/usr/bin/env python3
"""Sum HOT adapter ticks from a server journal and judge whether a hot-set A/B arm is valid.

The scheduler logs hot_adapt_ticks_{prefill,decode,idle} as deltas since the previous
stats line, so an arm total is the sum over every line. A hot-set knob (prefill weight,
capacity policy, history split, aim) only changes what the adapter aims at, so an arm
in which the adapter never ticked measures the stale starting plan and reads neutral
(issue #23). Usage: adapt-ticks-check.py JOURNAL [--require] [--min-ticks N]
"""

import argparse
import re
import sys

_TICKS = re.compile(r"hot_adapt_ticks_(prefill|decode|idle): (\d+)")
_INTERVAL = re.compile(r"hot_adapt_interval: (\d+)")
_MODE = re.compile(r"MoE HOT adaptation intervals: mode=(\S+?),.*?current_interval=(\d+)")


def summarize(text):
    """Return per-boundary tick sums plus the startup mode and last logged interval."""
    ticks = {"prefill": 0, "decode": 0, "idle": 0}
    for kind, value in _TICKS.findall(text):
        ticks[kind] += int(value)
    mode = _MODE.search(text)
    intervals = _INTERVAL.findall(text)
    return {
        "ticks": ticks,
        "mode": mode.group(1) if mode else None,
        "startup_interval": int(mode.group(2)) if mode else None,
        "last_interval": int(intervals[-1]) if intervals else None,
    }


def verdict(summary, min_ticks):
    """Non-idle ticks (prefill, decode, and the post-prefill tick) are the ones an arm drives."""
    ticks = summary["ticks"]
    return "VALID" if ticks["prefill"] + ticks["decode"] >= min_ticks else "INVALID"


def render(summary, min_ticks):
    ticks = summary["ticks"]
    state = verdict(summary, min_ticks)
    line = (
        f"adapt_ticks_prefill={ticks['prefill']} adapt_ticks_decode={ticks['decode']} "
        f"adapt_ticks_idle={ticks['idle']} mode={summary['mode']} "
        f"startup_interval={summary['startup_interval']} last_interval={summary['last_interval']}"
    )
    if state == "INVALID":
        line += (
            f" ARM INVALID: fewer than {min_ticks} non-idle adapt ticks, so a hot-set knob "
            "cannot act (lower --moe-hot-adapt-interval-steps or lengthen the arm; "
            "see issue #23)"
        )
    return line


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("journal")
    parser.add_argument("--min-ticks", type=int, default=3)
    parser.add_argument("--require", action="store_true", help="exit 2 when the arm is INVALID")
    args = parser.parse_args(argv)
    with open(args.journal, errors="ignore") as handle:
        summary = summarize(handle.read())
    print(render(summary, args.min_ticks))
    if args.require and verdict(summary, args.min_ticks) == "INVALID":
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
