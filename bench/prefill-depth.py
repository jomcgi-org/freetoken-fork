"""Measure real-source cold prefill, cached repeat, and subsequent decode.

Run against an externally selected serving configuration. Prepare the manifest
once and reuse it in every arm. Start each arm with an empty prefix cache.
Results retain full responses and failures; source copying is a narrow fidelity
check, not a general model quality score.
"""

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def prepare(root, tokenizer_path, depths, runs):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    files = sorted((root / "docs").glob("*.md")) + sorted(
        (root / "python/freetoken").rglob("*.py")
    )
    corpus = "\n\n".join(
        f"FILE {path.relative_to(root)}\n{path.read_text()}" for path in files
    )
    ids = tokenizer.encode(corpus, add_special_tokens=False)
    cases = []
    for depth in depths:
        for run in range(runs):
            # Unique prefix per case defeats reuse across depths and repetitions.
            nonce = sha(f"node4-depth-v1/{depth}/{run}")[:24]
            expected = {f"r{i}": 1000 + (i * 719 + run * 193) % 8000 for i in range(8)}
            records = json.dumps(expected)
            instruction = (
                "\nThe source above is background. Copy only this JSON object exactly, "
                "with no markdown or explanation:\n" + records
            )
            count = depth - 180
            text = f"Measurement {nonce}.\n" + tokenizer.decode(ids[:count]) + instruction
            messages = [dict(role="user", content=text)]
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
            )
            if isinstance(rendered, Mapping):
                rendered = rendered["input_ids"]
            if not depth - 256 <= len(rendered) <= depth:
                raise ValueError(f"unexpected prompt size {len(rendered)} for depth {depth}")
            if len(ids) < count:
                raise ValueError("source corpus is shorter than requested depth")
            cases.append(dict(depth=depth, run=run, messages=messages,
                              expected=expected, rendered_tokens=len(rendered)))
    return dict(version=1, corpus_sha256=sha(corpus), cases=cases)


def request(base_url, messages, max_tokens):
    payload = dict(model="qwen3.6-27b", messages=messages, temperature=0,
                   max_tokens=max_tokens, stream=True,
                   stream_options=dict(include_usage=True),
                   chat_template_kwargs=dict(enable_thinking=False))
    encoded = json.dumps(payload).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/chat/completions",
                                 data=encoded, headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    first = None
    last = None
    chunks = []
    usage = None
    finish = None
    done = False
    error = None
    try:
        with urllib.request.urlopen(req, timeout=1800) as response:
            for raw in response:
                if not raw.startswith(b"data:"):
                    continue
                data = raw[5:].strip()
                if data == b"[DONE]":
                    done = True
                    break
                chunk = json.loads(data)
                chunks.append(chunk)
                if chunk.get("error"):
                    raise RuntimeError(chunk["error"])
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta", {})
                    if delta.get("content") or delta.get("reasoning_content"):
                        last = time.perf_counter() - started
                        if first is None:
                            first = last
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    wall = time.perf_counter() - started
    text = "".join(c.get("delta", {}).get("content", "") or ""
                   for chunk in chunks for c in chunk.get("choices", []))
    reasoning = "".join(c.get("delta", {}).get("reasoning_content", "") or ""
                        for chunk in chunks for c in chunk.get("choices", []))
    cached = (usage or {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
    tokens = (usage or {}).get("completion_tokens", 0)
    return dict(request_sha256=hashlib.sha256(encoded).hexdigest(), text=text,
                reasoning=reasoning, usage=usage, cached_tokens=cached,
                ttft_s=first, wall_s=wall, finish_reason=finish, done=done, error=error,
                decode_tokens_per_s=((tokens - 1) / (last - first)
                                     if first is not None and last > first and tokens > 1 else None),
                chunks=chunks)


def score(row, expected):
    try:
        pairs = json.loads(row["text"], object_pairs_hook=list)
        valid = (pairs == list(expected.items())
                 and all(type(value) is int for _, value in pairs))
    except (ValueError, TypeError):
        valid = False
    return bool(valid and row["done"] and row["finish_reason"] == "stop"
                and row["usage"] and not row["error"] and not row["reasoning"])


def measure(args):
    manifest = json.loads(args.manifest.read_text())
    with args.output.open("x") as out:
        def save(row):
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(json.dumps({k: v for k, v in row.items()
                              if k not in ("chunks", "text", "reasoning")}), flush=True)

        for case in manifest["cases"]:
            for phase in ("cold", "repeat"):
                row = request(args.base_url, case["messages"], 192)
                row.update(arm=args.arm, depth=case["depth"], run=case["run"], phase=phase,
                           manifest_sha256=sha(args.manifest.read_text()))
                row["passed"] = score(row, case["expected"])
                row["cold_valid"] = phase != "cold" or row["cached_tokens"] == 0
                save(row)
                if row["error"]:
                    raise RuntimeError(row["error"])
            # A distinct, long enough completion measures decode after the document.
            expected = {f"r{i:02}": 10000 + (7919 * i) % 89999 for i in range(32)}
            prompt = (f"Decode check {case['depth']}/{case['run']}. "
                      "Copy this JSON object with two-space indentation, no markdown "
                      "or explanation, preserving order and integer values:\n" + json.dumps(expected))
            row = request(args.base_url, [dict(role="user", content=prompt)], 768)
            row.update(arm=args.arm, depth=case["depth"], run=case["run"], phase="post_decode")
            row["passed"] = score(row, expected)
            save(row)
            if row["error"]:
                raise RuntimeError(row["error"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--source", type=Path, required=True)
    prep.add_argument("--tokenizer", required=True)
    prep.add_argument("--depths", type=int, nargs="+", default=[8000, 32000, 100000])
    prep.add_argument("--runs", type=int, default=3)
    prep.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("measure")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--arm", required=True)
    run.add_argument("--base-url", default="http://127.0.0.1:18090")
    args = parser.parse_args()
    if args.command == "prepare":
        manifest = prepare(args.source, args.tokenizer, args.depths, args.runs)
        with args.output.open("x") as out:
            json.dump(manifest, out)
    else:
        measure(args)


if __name__ == "__main__":
    main()
