"""Fixed workload for bench/ab.py: optional cache-disjoint warmup, then greedy decodes of fixed
length (--workload default: essays + doc; mixed-thinking: 12 alternating thinking/plain requests). Around each measured request it records server major faults, NVMe bytes read and
wall time. Usage: ab-client.py OUT_JSON [--workload NAME] [--warmup] [--port 18090] [--max-tokens 1000]
[--request-cap SECONDS]. Emits one JSON line per event (start/done) so the runner can watch progress.

Warmup prompts share no leading text with any measured prompt (checked at import by
check_disjoint), so a warm arm cannot hit a measured request's prefix-cache entry."""
import argparse, json, subprocess, sys, time, urllib.request
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"
DOC = (DOCS / "4090-performance.md").read_text()
WARM_DOC = (DOCS / "prefill-depth-benchmark.md").read_text()
MEASURED = [("essay1", "Write a long technical essay on how GPUs schedule warps."),
            ("essay2", "Write a long history of the printing press."),
            ("doc", "Here is a document:\n\n" + DOC * 4 + "\n\nSummarize it at length, section by section."),
            ("essay1b", "Write a long technical essay on how GPUs schedule warps.")]
WARMUP = [("warm1", "Explain how tide tables are computed for a harbour, in detail."),
          ("warm2", "Describe the life cycle of a star from nebula to remnant, at length."),
          ("warm3", "Reference material B follows.\n\n" + WARM_DOC * 2 + "\n\nList every benchmark it reports.")]
TOPICS = ["how a mechanical watch keeps time", "the causes of the 1929 stock market crash",
          "how vaccines train the immune system", "the architecture of Roman aqueducts",
          "how public-key cryptography works", "why the sky is blue and sunsets are red",
          "the history of the steam locomotive", "how bees communicate the location of food",
          "the rules and strategy of chess openings", "how lithium-ion batteries degrade",
          "the formation of the Himalayas", "how compilers optimise loops"]
ROUNDS = 6
MIXED = [(f"r{i // 2 + 1}-{'think' if i % 2 == 0 else 'plain'}", f"Explain {t}, step by step.", i % 2 == 0)
         for i, t in enumerate(TOPICS)]  # (name, prompt, thinking); round r is requests 2r-2 and 2r-1
PREFIX_CHARS = 24


def check_disjoint(warm, measured, n=PREFIX_CHARS):
    heads = {p[:n] for _, p, *_ in measured}
    clash = [name for name, p in warm if p[:n] in heads]
    if clash:
        raise SystemExit(f"warmup prompts share a prefix with measured prompts: {clash}")


def pids(port):
    out = subprocess.run(["pgrep", "-f", f"port {port}"], capture_output=True, text=True).stdout.split()
    return [int(p) for p in out]


def majflt(port):
    tot = 0
    for p in pids(port):
        for task in Path(f"/proc/{p}/task").glob("*"):
            try: tot += int((task / "stat").read_text().rsplit(")", 1)[1].split()[9])
            except Exception: pass
    return tot


def nvme_sectors():
    tot = 0
    for d in Path("/sys/block").glob("nvme*"):
        tot += int((d / "stat").read_text().split()[2])
    return tot


def emit(**kw): print(json.dumps(kw), flush=True)


def ask(args, name, content, thinking=False, stream=False):
    body = {"model": "qwen3.6-27b", "messages": [{"role": "user", "content": content}],
            "max_tokens": args.max_tokens, "temperature": 0, "seed": 0,
            "chat_template_kwargs": {"enable_thinking": thinking}}
    if stream:
        body.update(stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(f"http://127.0.0.1:{args.port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    emit(event="start", name=name)
    f0, s0, t0 = majflt(args.port), nvme_sectors(), time.time()
    extra = {}
    with urllib.request.urlopen(req, timeout=args.request_cap) as r:
        if not stream:
            tokens = json.load(r)["usage"]["completion_tokens"]
        else:
            tokens, reasoning, answer, usage = 0, 0, 0, None
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:") or line.endswith("[DONE]"):
                    continue
                chunk = json.loads(line[5:])
                usage = chunk.get("usage") or usage
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta", {})
                    reasoning += bool(delta.get("reasoning_content") or delta.get("reasoning"))
                    answer += bool(delta.get("content"))
            details = (usage or {}).get("completion_tokens_details") or {}
            if usage and "reasoning_tokens" in details:  # exact split from usage
                tokens = usage["completion_tokens"]
                extra = dict(reasoning_tokens=details["reasoning_tokens"],
                             content_tokens=tokens - details["reasoning_tokens"], token_counts="usage")
            else:  # one chunk is about one token
                tokens = usage["completion_tokens"] if usage else reasoning + answer
                extra = dict(reasoning_tokens=reasoning, content_tokens=answer, token_counts="chunks")
    t1 = time.time()
    return dict(name=name, wall=t1 - t0, tokens=tokens, tok_s=tokens / (t1 - t0), t_start=t0, t_end=t1,
                majflt=majflt(args.port) - f0, nvme_gib=(nvme_sectors() - s0) * 512 / 2**30, **extra)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--workload", choices=["default", "mixed-thinking"], default="default")
    ap.add_argument("--warmup", action="store_true")
    ap.add_argument("--port", type=int, default=18090)
    ap.add_argument("--max-tokens", type=int, default=None, help="default 1000, or 600 for mixed-thinking")
    ap.add_argument("--request-cap", type=float, default=1800)
    args = ap.parse_args(argv)
    measured = MEASURED if args.workload == "default" else MIXED
    args.max_tokens = args.max_tokens or (1000 if args.workload == "default" else 600)
    check_disjoint(WARMUP, measured)
    if args.warmup:
        for name, prompt in WARMUP:
            emit(event="done", warmup=True, **ask(args, name, prompt))
    rows = []
    for name, prompt, *flag in measured:
        if flag:
            r = ask(args, name, prompt, thinking=flag[0], stream=True)
            r.update(thinking=flag[0], round=int(name[1:].split("-")[0]))
        else:
            r = ask(args, name, prompt)
        rows.append(r)
        emit(event="done", **r)
    Path(args.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
