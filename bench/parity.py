#!/usr/bin/env python3
"""Model-output parity gate: did this build change what the model computes?

Replays a fixed set of greedy requests against a running FreeToken server and
compares the outputs with a baseline recorded from a trusted build. The cases
are shaped like the box's real workload: a coding-agent session with tool
definitions, tool calls and real source files as tool results, cut at several
context lengths so prefill crosses chunk and layer-group boundaries, plus the
short prompts from quality.sh. Each case can be replayed more than once so the
later passes exercise prefix-cache and restore paths.

The baseline stores the exact request bodies, so ``compare`` sends
byte-identical requests whatever checkout it runs from.

When the server returns logprobs, the comparison is token-level: the first
divergent token, the baseline's top-1/top-2 margin at that position (a flip at
a near-tie is plausible numeric noise; a flip where the baseline was confident
is a bug), and the logprob drift over the matched prefix. Without logprobs it
falls back to exact text matching.

Usage:
  python bench/parity.py record  --base-url http://127.0.0.1:8090/v1 --out parity-base.json
  python bench/parity.py compare --base-url http://127.0.0.1:8090/v1 --baseline parity-base.json

``compare`` exits non-zero when any case fails. Record the baseline with the
same model, checkpoint and KV dtype as the candidate; serving knobs (threads,
budgets, chunk size) may differ, which is the point.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

FORMAT = "freetoken-parity-v1"
TOP_LOGPROBS = 5
CHARS_PER_TOKEN = 3.3  # rough, for sizing cuts; actual prompt tokens are recorded
READ_CHARS = 24000  # largest single Read result, about 7k tokens

SYSTEM_PROMPT = """You are an autonomous coding agent working inside a git checkout of an \
inference engine. You investigate by calling tools, one call at a time, and read \
code before proposing changes. Keep reasoning brief. When you have enough \
information, reply with a short diagnosis and the exact change you would make.

Rules:
- Prefer reading the relevant file over guessing.
- Never invent file contents; quote only what a tool returned.
- Use Grep to locate symbols before reading large files.
- Do not run destructive commands."""

TOOLS = [
    {"type": "function", "function": {
        "name": "Read",
        "description": "Read a file from the repository. Returns its contents.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Repository-relative path."}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "Grep",
        "description": "Search the repository for a regular expression.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string", "description": "Directory to search."}},
            "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "Bash",
        "description": "Run a shell command in the repository root.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "Edit",
        "description": "Replace an exact string in a file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "old": {"type": "string"},
            "new": {"type": "string"}}, "required": ["path", "old", "new"]}}},
]

TASK = ("A user reports that after a server restart, the first long coding-agent request "
        "sometimes produces different output than before the restart, even at "
        "temperature 0. Find where cached prefix state is restored and whether any "
        "state could be stale or mismatched. Start by reading the relevant files.")

# Read in this order at record time; later files only land in the longer cuts.
SESSION_FILES = [
    "python/freetoken/server/request_logger.py",
    "python/freetoken/kvcache/disk_prefix_cache.py",
    "python/freetoken/scheduler/cache.py",
    "python/freetoken/server/generation.py",
    "python/freetoken/scheduler/scheduler.py",
    "python/freetoken/engine/engine.py",
    "python/freetoken/moe/offload_cache.py",
]

SHORT_PROMPTS = [
    ("arith", "Compute step by step, showing each partial sum: 17 + 28 + 45 + 96 + 133 = "),
    ("recall", "Remember this codeword: ZEPHYR-9142. Now count from one to twenty, "
               "then repeat the codeword exactly. "),
    ("reason", "A farmer has 17 sheep. All but 9 run away. How many sheep does the "
               "farmer have left? Think step by step. "),
]


def _post(base_url: str, payload: dict, timeout: float) -> tuple[dict, float]:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", body,
                                 {"content-type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.load(resp)
    return out, time.time() - t0


def _model_id(base_url: str) -> str:
    with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=10) as resp:
        return json.load(resp)["data"][0]["id"]


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _session_messages(cuts: list[int]) -> list[tuple[int, list[dict]]]:
    """Build one agent session from real repo files; return a snapshot per cut.

    Files are read in line-range chunks, as an agent pages through large files, and
    the read that crosses a cut is trimmed to land on it. Each snapshot ends on a
    tool result, so the model's next turn is an agent step (reasoning plus a tool
    call or an answer)."""
    root = _repo_root()
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": TASK}]
    chars = sum(len(m["content"]) for m in messages)
    snapshots: list[tuple[int, list[dict]]] = []
    pending = sorted(cuts)
    calls = 0
    for rel in SESSION_FILES:
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        line = 0
        while line < len(lines) and pending:
            room = int(pending[0] * CHARS_PER_TOKEN) - chars
            chunk, size = [], 0
            while line + len(chunk) < len(lines) and size < min(room, READ_CHARS):
                chunk.append(lines[line + len(chunk)])
                size += len(chunk[-1])
            args = {"path": rel, "offset": line + 1, "limit": len(chunk)}
            call_id = f"call_{calls:02d}"
            calls += 1
            messages.append({"role": "assistant", "content": "", "tool_calls": [{
                "id": call_id, "type": "function",
                "function": {"name": "Read", "arguments": json.dumps(args)}}]})
            messages.append({"role": "tool", "tool_call_id": call_id,
                             "content": "".join(chunk)})
            chars += size + 80
            line += len(chunk)
            while pending and chars >= pending[0] * CHARS_PER_TOKEN:
                snapshots.append((pending.pop(0), json.loads(json.dumps(messages))))
    for cut in pending:  # repo smaller than asked: keep whatever the session reached
        snapshots.append((cut, json.loads(json.dumps(messages))))
    return snapshots


def build_cases(model: str, cuts: list[int], max_tokens: int, logprobs: bool) -> list[dict]:
    extra = {"logprobs": True, "top_logprobs": TOP_LOGPROBS} if logprobs else {}
    cases = []
    for name, prompt in SHORT_PROMPTS:
        cases.append({"name": f"short-{name}", "request": {
            "model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, **extra}})
    seen = set()
    for cut, messages in _session_messages(cuts):
        key = len(messages)
        if key in seen:  # two cuts landed on the same snapshot
            continue
        seen.add(key)
        cases.append({"name": f"agent-{cut // 1000}k", "request": {
            "model": model, "messages": messages, "tools": TOOLS,
            "max_tokens": max_tokens, "temperature": 0, **extra}})
    return cases


def _extract(out: dict) -> dict:
    choice = out["choices"][0]
    msg = choice.get("message") or {}
    calls = [{"name": c["function"]["name"], "arguments": c["function"]["arguments"]}
             for c in msg.get("tool_calls") or []]
    lp = choice.get("logprobs") or {}
    tokens = [{"token": t["token"], "logprob": t["logprob"],
               "top": [[x["token"], x["logprob"]] for x in t.get("top_logprobs") or []]}
              for t in lp.get("content") or []]
    return {
        "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
        "content": msg.get("content") or "",
        "tool_calls": calls,
        "finish_reason": choice.get("finish_reason"),
        "usage": out.get("usage") or {},
        "tokens": tokens or None,
    }


def _text(result: dict) -> str:
    return (result["reasoning"] + "\x00" + result["content"] + "\x00"
            + json.dumps(result["tool_calls"], sort_keys=True))


def run_case(base_url: str, case: dict, timeout: float) -> dict:
    out, wall = _post(base_url, case["request"], timeout)
    result = _extract(out)
    result["wall_s"] = round(wall, 2)
    return result


def _git_rev() -> str:
    try:
        return subprocess.run(["git", "-C", _repo_root(), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _supports_logprobs(base_url: str, model: str, timeout: float) -> bool:
    out, _ = _post(base_url, {"model": model, "max_tokens": 2, "temperature": 0,
                              "logprobs": True, "top_logprobs": 2,
                              "messages": [{"role": "user", "content": "hi"}]}, timeout)
    return bool((out["choices"][0].get("logprobs") or {}).get("content"))


def cmd_record(args) -> int:
    model = args.model or _model_id(args.base_url)
    logprobs = _supports_logprobs(args.base_url, model, args.timeout)
    cuts = [int(c) for c in args.cuts.split(",")]
    cases = build_cases(model, cuts, args.max_tokens, logprobs)
    print(f"recording {len(cases)} cases from {args.base_url} "
          f"(model={model}, logprobs={'yes' if logprobs else 'no'})", file=sys.stderr)
    for case in cases:
        case["result"] = run_case(args.base_url, case, args.timeout)
        r = case["result"]
        print(f"  {case['name']:<14} prompt={r['usage'].get('prompt_tokens', '?'):>6} "
              f"out={r['usage'].get('completion_tokens', '?'):>4} {r['wall_s']:>7.1f}s",
              file=sys.stderr)
    doc = {"format": FORMAT, "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "git_rev": _git_rev(), "base_url": args.base_url, "model": model,
           "logprobs": logprobs, "cases": cases}
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=1)
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


def _margin(tok: dict) -> float | None:
    top = tok["top"]
    return top[0][1] - top[1][1] if len(top) >= 2 else None


def compare_result(base: dict, cand: dict, args) -> dict:
    """Classify one replay against its baseline."""
    verdict = {"match": _text(base) == _text(cand)}
    bt, ct = base.get("tokens"), cand.get("tokens")
    if bt and ct:
        n = min(len(bt), len(ct))
        div = next((i for i in range(n) if bt[i]["token"] != ct[i]["token"]),
                   None if len(bt) == len(ct) else n)
        upto = n if div is None else div
        deltas = [abs(bt[i]["logprob"] - ct[i]["logprob"]) for i in range(upto)]
        verdict.update({
            "tokens": len(bt),
            "diverge_at": div,
            "mean_dlogprob": round(sum(deltas) / len(deltas), 4) if deltas else 0.0,
            "max_dlogprob": round(max(deltas), 4) if deltas else 0.0,
        })
        margin = _margin(bt[div]) if div is not None and div < len(bt) else None
        verdict["base_margin"] = None if margin is None else round(margin, 3)
        confident_flip = div is not None and (margin is None or margin > args.tie_margin)
        drift = verdict["mean_dlogprob"] > args.max_mean_drift
        verdict["ok"] = not confident_flip and not drift
        if confident_flip:
            verdict["why"] = "confident flip" if margin is not None else "length/finish change"
        elif drift:
            verdict["why"] = "logprob drift"
        elif div is not None:
            verdict["why"] = "near-tie flip"
    else:
        verdict["ok"] = verdict["match"]
        if not verdict["match"]:
            a, b = _text(base), _text(cand)
            verdict["diverge_char"] = next(
                (i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b)))
            verdict["why"] = "text differs"
    return verdict


def cmd_compare(args) -> int:
    with open(args.baseline) as f:
        doc = json.load(f)
    if doc.get("format") != FORMAT:
        raise SystemExit(f"{args.baseline}: not a {FORMAT} baseline")
    live = args.model or _model_id(args.base_url)
    if live != doc["model"] and not args.model:
        print(f"note: server model id {live!r} differs from baseline {doc['model']!r}; "
              f"sending the baseline's id", file=sys.stderr)
    if doc["logprobs"] and not _supports_logprobs(args.base_url, doc["model"], args.timeout):
        print("note: baseline has logprobs but this server returns none; "
              "falling back to exact text matching", file=sys.stderr)
    print(f"baseline {doc['git_rev']} ({doc['recorded_at']}) vs candidate {_git_rev()}, "
          f"{len(doc['cases'])} cases x {args.passes} passes")
    print(f"{'case':<14} {'pass':>4} {'prompt':>7} {'wall':>7}  verdict")
    failures = 0
    report = []
    for p in range(1, args.passes + 1):
        for case in doc["cases"]:
            cand = run_case(args.base_url, case, args.timeout)
            v = compare_result(case["result"], cand, args)
            failures += not v["ok"]
            report.append({"case": case["name"], "pass": p, "verdict": v, "result": cand})
            if "diverge_at" in v:
                detail = ("identical" if v["diverge_at"] is None
                          else f"diverge@{v['diverge_at']}/{v['tokens']} margin={v['base_margin']}")
                detail += f" mean|dlp|={v['mean_dlogprob']} max={v['max_dlogprob']}"
            else:
                detail = "identical" if v["match"] else f"diverge@char{v['diverge_char']}"
            status = "ok  " if v["ok"] else "FAIL"
            why = f" ({v['why']})" if v.get("why") else ""
            print(f"{case['name']:<14} {p:>4} {cand['usage'].get('prompt_tokens', '?'):>7} "
                  f"{cand['wall_s']:>6.1f}s  {status} {detail}{why}", flush=True)
    total = len(report)
    print(f"\n{total - failures}/{total} ok" + (f", {failures} FAILED" if failures else ""))
    if args.report:
        with open(args.report, "w") as f:
            json.dump({"baseline": args.baseline, "candidate_rev": _git_rev(),
                       "failures": failures, "replays": report}, f, indent=1)
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("record", "compare"):
        p = sub.add_parser(name)
        p.add_argument("--base-url", default="http://127.0.0.1:8090/v1")
        p.add_argument("--model", help="model id to send (default: the server's)")
        p.add_argument("--timeout", type=float, default=1800.0)
    rec = sub.choices["record"]
    rec.add_argument("--out", required=True)
    rec.add_argument("--cuts", default="4000,16000,40000,60000",
                     help="agent-session context lengths in tokens (approximate)")
    rec.add_argument("--max-tokens", type=int, default=160)
    cmp_ = sub.choices["compare"]
    cmp_.add_argument("--baseline", required=True)
    cmp_.add_argument("--passes", type=int, default=2,
                      help="replay each case this many times (later passes hit the prefix cache)")
    cmp_.add_argument("--tie-margin", type=float, default=0.1,
                      help="a flip is tolerated when the baseline's top-1/top-2 gap is at most "
                           "this many nats")
    cmp_.add_argument("--max-mean-drift", type=float, default=0.05,
                      help="fail when mean |dlogprob| over the matched prefix exceeds this")
    cmp_.add_argument("--report", help="write per-replay details as JSON")
    args = ap.parse_args()
    try:
        return cmd_record(args) if args.cmd == "record" else cmd_compare(args)
    except urllib.error.URLError as e:
        print(f"request failed: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
