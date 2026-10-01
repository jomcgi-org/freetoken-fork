#!/usr/bin/env python3
"""Count HOT adapter ticks (per-tick log lines) in a server journal and judge whether a hot-set A/B arm is valid.

Ticks are counted per boundary from "MoE HOT adaptation tick" lines (fallback: the
hot_adapt_ticks_{prefill,decode,idle} stats, logged as deltas since the previous stats
line, summed over every line). A hot-set knob (prefill weight,
capacity policy, history split, aim) only changes what the adapter aims at, so an arm
in which the adapter never ticked measures the stale starting plan and reads neutral
(issue #23). Usage: adapt-ticks-check.py JOURNAL [--require] [--min-ticks N]
"""

import argparse
import re
import sys
import time

_TICKS = re.compile(r"hot_adapt_ticks_(prefill|decode|idle): (\d+)")
_INTERVAL = re.compile(r"hot_adapt_interval: (\d+)")
# The production server logs one line per tick (the hot_adapt_ticks_* stats above are not logged):
#   MoE HOT adaptation tick token=775, boundary=decode: decayed_hot_pair_rate=77.06%, ticks=1, ...
_TICK_LINE = re.compile(r"MoE HOT adaptation tick .*?boundary=(\w+): decayed_hot_pair_rate=([\d.]+)%")
_STAMP = re.compile(r"(\d{4}-\d\d-\d\d)\|(\d\d:\d\d:\d\d)\|")
_BATCH_RATE = re.compile(r"(?<!decayed_)(?<!profiled )hot_pair_rate[:=] ?([\d.]+)%")
_MODE = re.compile(r"MoE HOT adaptation intervals: mode=(\S+?),.*?current_interval=(\d+)")


def tick_events(text):
    """[(epoch or None, boundary, decayed_hot_pair_rate %)] for every per-tick log line (local time stamps)."""
    events = []
    for line in text.splitlines():
        m = _TICK_LINE.search(line)
        if not m:
            continue
        st = _STAMP.search(line)
        epoch = time.mktime(time.strptime(f"{st.group(1)} {st.group(2)}", "%Y-%m-%d %H:%M:%S")) if st else None
        events.append((epoch, m.group(1), float(m.group(2))))
    return events


def summarize(text):
    """Return per-boundary tick counts plus the startup mode, last logged interval and hot pair rates.

    Ticks are counted from the per-tick log lines; the older hot_adapt_ticks_* stats deltas are the fallback."""
    ticks = {"prefill": 0, "decode": 0, "idle": 0}
    events = tick_events(text)
    if events:
        for _, boundary, _ in events:
            if boundary in ticks:
                ticks[boundary] += 1
    else:
        for kind, value in _TICKS.findall(text):
            ticks[kind] += int(value)
    rates = [r for _, _, r in events] or [float(v) for v in _BATCH_RATE.findall(text)]
    mode = _MODE.search(text)
    intervals = _INTERVAL.findall(text)
    return {
        "ticks": ticks,
        "mode": mode.group(1) if mode else None,
        "startup_interval": int(mode.group(2)) if mode else None,
        "last_interval": int(intervals[-1]) if intervals else None,
        "hot_pair_rate": {"n": len(rates), "mean": sum(rates) / len(rates), "min": min(rates), "max": max(rates),
                          "last": rates[-1]} if rates else None,
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
