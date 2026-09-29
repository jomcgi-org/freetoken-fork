#!/usr/bin/env python3
"""Summarize one BFCL run, or diff which entries pass between two runs.

  compare.py RUN               per-category pass counts
  compare.py BASE CANDIDATE    also lists entries that flipped; exits 1 when the
                               candidate loses more than --max-lost entries

A run is a BFCL project root written by run.sh. BFCL's score files list only
failing entries, so an entry passes when it has a result and no score record.
Scores at temperature 0 still move by an entry or two between builds with
harmless numeric differences; a real regression shows up as a cluster of new
failures, often one error type.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys


def _jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _error_type(rec: dict) -> str:
    # AST categories carry error_type beside a list of messages; multi-turn nests it
    # in an error dict.
    error = rec.get("error")
    if rec.get("error_type"):
        return rec["error_type"]
    if isinstance(error, dict) and error.get("error_type"):
        return error["error_type"]
    return "?"


def load(root: str) -> tuple[dict[str, str], dict[str, str]]:
    """Return ({id: category}, {failed id: error type}) for one run."""
    ran: dict[str, str] = {}
    for path in glob.glob(os.path.join(root, "result", "*", "*", "BFCL_v4_*_result.json")):
        category = os.path.basename(path)[len("BFCL_v4_"):-len("_result.json")]
        for rec in _jsonl(path):
            ran[rec["id"]] = category
    if not ran:
        raise SystemExit(f"{root}: no BFCL results")
    failed: dict[str, str] = {}
    for path in glob.glob(os.path.join(root, "score", "*", "*", "BFCL_v4_*_score.json")):
        for rec in _jsonl(path)[1:]:  # line 1 is the category summary
            failed[rec["id"]] = _error_type(rec)
    return ran, failed


def summary(root: str) -> None:
    ran, failed = load(root)
    by_cat = collections.Counter(ran.values())
    fails = collections.Counter(ran[i] for i in failed if i in ran)
    wall = ""
    if os.path.exists(os.path.join(root, "wall.txt")):
        wall = open(os.path.join(root, "wall.txt")).read().strip()
    print(f"{root}  {wall}")
    for cat in sorted(by_cat):
        n = by_cat[cat]
        print(f"  {cat:<28} {n - fails[cat]:>3}/{n}")
    total = len(ran)
    print(f"  {'total':<28} {total - len(failed):>3}/{total}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("base")
    ap.add_argument("candidate", nargs="?")
    ap.add_argument("--max-lost", type=int, default=2,
                    help="net entries the candidate may lose before failing")
    args = ap.parse_args()
    summary(args.base)
    if not args.candidate:
        return 0
    summary(args.candidate)
    base_ran, base_failed = load(args.base)
    cand_ran, cand_failed = load(args.candidate)
    common = sorted(set(base_ran) & set(cand_ran))
    if len(common) != len(base_ran) or len(common) != len(cand_ran):
        print(f"note: runs share {len(common)} of {len(base_ran)}/{len(cand_ran)} entries; "
              f"comparing the shared ones")
    lost = [i for i in common if i in cand_failed and i not in base_failed]
    gained = [i for i in common if i in base_failed and i not in cand_failed]
    print(f"\nnewly failing ({len(lost)}):")
    for i in lost:
        print(f"  {i:<32} {cand_failed[i]}")
    print(f"newly passing ({len(gained)}):")
    for i in gained:
        print(f"  {i}")
    net = len(lost) - len(gained)
    print(f"\nnet {-net:+d} entries")
    return 1 if net > args.max_lost else 0


if __name__ == "__main__":
    sys.exit(main())
