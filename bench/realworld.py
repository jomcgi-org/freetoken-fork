#!/usr/bin/env python3
"""Real-world scenario benchmark: the acceptance gate for serving changes.

Runs against any OpenAI-compatible FreeToken server and prints a scorecard.
Scenarios mirror the box's actual workload: multi-turn chat, coding-agent
resume (optionally across a server restart), interactive latency under a
long-prefill contender, and structured-output validity.

Usage:
  python bench/realworld.py --base-url http://127.0.0.1:8090/v1 --model qwen3.6-27b
  python bench/realworld.py ... --restart-cmd "sudo systemctl restart freetoken-serve"
  python bench/realworld.py ... --skip contention,structured

The scorecard is append-friendly: pipe to `tee -a results/realworld.txt`
and diff runs across builds.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

REPO_CONTEXT_WORDS = 1500  # ~10k tokens of synthetic "repo" context


def _post(base_url: str, payload: dict, timeout: float = 900.0) -> tuple[dict, float, float]:
    """POST a chat completion (streaming) and return (final, ttft_s, total_s)."""
    payload = dict(payload)
    payload["stream"] = True
    payload["stream_options"] = {"include_usage": True}
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions", body,
        {"content-type": "application/json"},
    )
    t0 = time.time()
    ttft = None
    usage = {}
    content_len = 0
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                if delta.get("content") or delta.get("reasoning_content"):
                    content_len += len(delta.get("content") or "")
                    if ttft is None:
                        ttft = time.time() - t0
    total = time.time() - t0
    return {"usage": usage, "content_len": content_len}, (ttft if ttft is not None else total), total


def _wait_ready(base_url: str, model: str, deadline_s: float = 1800.0) -> None:
    t0 = time.time()
    payload = {"model": model, "messages": [{"role": "user", "content": "hi"}],
               "max_tokens": 2, "temperature": 0}
    while time.time() - t0 < deadline_s:
        try:
            _post(base_url, payload, timeout=60)
            return
        except (urllib.error.URLError, OSError, TimeoutError):
            time.sleep(10)
    raise RuntimeError("server never became ready")


def _fmt(xs: list[float]) -> str:
    if not xs:
        return "n/a"
    p50 = statistics.median(xs)
    worst = max(xs)
    return f"p50={p50:.2f}s worst={worst:.2f}s"


def scenario_conversation(base_url: str, model: str, turns: int = 10) -> dict:
    """Growing multi-turn chat; measures per-turn TTFT (prefix reuse shows here)."""
    messages = [{"role": "system", "content":
                 "You are a concise assistant helping debug a home Kubernetes lab. "
                 "Answer briefly and remember details from earlier in the chat."}]
    ttfts, rates = [], []
    for turn in range(turns):
        messages.append({"role": "user", "content":
                         f"Turn {turn + 1}: name one more plausible cause of a pod stuck in "
                         f"Pending and how to confirm it. Do not repeat earlier causes."})
        out, ttft, total = _post(base_url, {
            "model": model, "messages": messages, "max_tokens": 120, "temperature": 0})
        reply_tokens = out["usage"].get("completion_tokens", 0)
        messages.append({"role": "assistant", "content": f"(cause {turn + 1} recorded)"})
        ttfts.append(ttft)
        if total > ttft and reply_tokens:
            rates.append(reply_tokens / (total - ttft))
    return {"ttft_first": ttfts[0], "ttft_rest": ttfts[1:], "decode_rates": rates}


def scenario_agent_resume(base_url: str, model: str, restart_cmd: str | None) -> dict:
    """Big-context agent session; optional mid-scenario restart tests disk restore."""
    context = ("def handler(request):\n    return route(request)\n# module "
               .join(f"file_{i} " for i in range(REPO_CONTEXT_WORDS)))
    messages = [{"role": "system", "content":
                 "You are a coding agent working in this repository:\n" + context},
                {"role": "user", "content": "Summarize what this codebase does in one sentence."}]
    _, cold_ttft, _ = _post(base_url, {
        "model": model, "messages": messages, "max_tokens": 60, "temperature": 0})
    messages.append({"role": "assistant", "content": "It routes requests to handlers."})
    messages.append({"role": "user", "content": "Name one refactor you would do first."})
    _, warm_ttft, _ = _post(base_url, {
        "model": model, "messages": messages, "max_tokens": 60, "temperature": 0})
    result = {"cold_ttft": cold_ttft, "warm_followup_ttft": warm_ttft}
    if restart_cmd:
        subprocess.run(restart_cmd, shell=True, check=True)
        _wait_ready(base_url, model)
        messages.append({"role": "assistant", "content": "Extract the routing table."})
        messages.append({"role": "user", "content": "And the second refactor?"})
        _, resume_ttft, _ = _post(base_url, {
            "model": model, "messages": messages, "max_tokens": 60, "temperature": 0})
        result["post_restart_resume_ttft"] = resume_ttft
    return result


def scenario_contention(base_url: str, model: str) -> dict:
    """Interactive TTFT while a long prefill grinds; priority header exercised."""
    filler = "background document word " * 2000
    long_payload = {"model": model, "max_tokens": 30, "temperature": 0,
                    "messages": [{"role": "user", "content": filler + " Summarize in one line."}]}
    quick_payload = {"model": model, "max_tokens": 40, "temperature": 0,
                     "priority": 10,
                     "messages": [{"role": "user", "content": "Quick: what is 2+2?"}]}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        long_f = pool.submit(_post, base_url, long_payload)
        time.sleep(3.0)  # let the long prefill occupy the engine
        t0 = time.time()
        _, quick_ttft, _ = _post(base_url, quick_payload)
        interactive_wait = time.time() - t0
        long_f.result()
    return {"interactive_ttft_under_load": quick_ttft,
            "interactive_total_under_load": interactive_wait}


def scenario_structured(base_url: str, model: str) -> dict:
    """JSON validity with response_format (no-op skip until guided decoding lands)."""
    payload = {"model": model, "max_tokens": 120, "temperature": 0,
               "response_format": {"type": "json_object"},
               "messages": [{"role": "user", "content":
                             "Return a JSON object with keys name and port for an ssh service."}]}
    valid = 0
    n = 5
    for _ in range(n):
        try:
            out, _, _ = _post(base_url, payload)
        except (urllib.error.URLError, OSError):
            return {"skipped": "server rejected response_format (guided decoding absent)"}
        # content_len only proves output; validity needs the text, so re-request unstreamed
        body = json.dumps({**payload, "stream": False}).encode()
        req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", body,
                                     {"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            text = json.load(resp)["choices"][0]["message"]["content"]
        try:
            json.loads(text)
            valid += 1
        except json.JSONDecodeError:
            pass
    return {"json_valid": f"{valid}/{n}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8090/v1")
    ap.add_argument("--model", default="qwen3.6-27b")
    ap.add_argument("--restart-cmd", default=None,
                    help="command that restarts the server (enables the resume-restart leg)")
    ap.add_argument("--skip", default="", help="comma list: conversation,agent,contention,structured")
    args = ap.parse_args()
    skip = set(filter(None, args.skip.split(",")))

    print(f"# realworld scorecard  {time.strftime('%Y-%m-%d %H:%M:%S')}  {args.base_url}")
    _wait_ready(args.base_url, args.model)

    if "conversation" not in skip:
        c = scenario_conversation(args.base_url, args.model)
        print(f"conversation: first-turn ttft={c['ttft_first']:.2f}s | "
              f"follow-ups {_fmt(c['ttft_rest'])} | decode {_fmt(c['decode_rates'])} tok/s")
    if "agent" not in skip:
        a = scenario_agent_resume(args.base_url, args.model, args.restart_cmd)
        line = (f"agent-resume: cold ttft={a['cold_ttft']:.2f}s | "
                f"warm follow-up ttft={a['warm_followup_ttft']:.2f}s")
        if "post_restart_resume_ttft" in a:
            line += f" | post-restart resume ttft={a['post_restart_resume_ttft']:.2f}s"
        print(line)
    if "contention" not in skip:
        k = scenario_contention(args.base_url, args.model)
        print(f"contention: interactive ttft under long prefill={k['interactive_ttft_under_load']:.2f}s")
    if "structured" not in skip:
        s = scenario_structured(args.base_url, args.model)
        print(f"structured: {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
