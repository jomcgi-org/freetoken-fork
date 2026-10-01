"""Fixed workload for bench/ab.py: optional cache-disjoint warmup, then four greedy decodes of
fixed length. Around each measured request it records server major faults, NVMe bytes read and
wall time. Usage: ab-client.py OUT_JSON [--warmup] [--port 18090] [--max-tokens 1000]
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
PREFIX_CHARS = 24


def check_disjoint(warm, measured, n=PREFIX_CHARS):
    heads = {p[:n] for _, p in measured}
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


def ask(args, name, content):
    body = json.dumps({"model": "qwen3.6-27b", "messages": [{"role": "user", "content": content}],
                       "max_tokens": args.max_tokens, "temperature": 0, "seed": 0,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{args.port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    emit(event="start", name=name)
    f0, s0, t0 = majflt(args.port), nvme_sectors(), time.time()
    with urllib.request.urlopen(req, timeout=args.request_cap) as r: rsp = json.load(r)
    wall = time.time() - t0
    tokens = rsp["usage"]["completion_tokens"]
    return dict(name=name, wall=wall, tokens=tokens, tok_s=tokens / wall,
                majflt=majflt(args.port) - f0, nvme_gib=(nvme_sectors() - s0) * 512 / 2**30)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--warmup", action="store_true")
    ap.add_argument("--port", type=int, default=18090)
    ap.add_argument("--max-tokens", type=int, default=1000)
    ap.add_argument("--request-cap", type=float, default=1800)
    args = ap.parse_args(argv)
    check_disjoint(WARMUP, MEASURED)
    if args.warmup:
        for name, prompt in WARMUP:
            emit(event="done", warmup=True, **ask(args, name, prompt))
    rows = []
    for name, prompt in MEASURED:
        r = ask(args, name, prompt); rows.append(r); emit(event="done", **r)
    Path(args.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
