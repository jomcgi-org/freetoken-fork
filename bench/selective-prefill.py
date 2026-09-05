"""Matched client timings for a server running either prefill transfer policy.

Run each arm in a fresh server with the same HOT plan and adaptation disabled.
Prompt nonces are deterministic and differ within an arm, so all arms receive
identical input without reusing a measured prompt's prefix. Disk KV reuse must
be disabled. This reports client TTFT, not the idle-sensitive scheduler rate.
"""

import argparse
import hashlib
import json
import random
import time
import urllib.request
from pathlib import Path


def request(base_url, payload):
    payload = dict(payload, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(
        base_url + "/v1/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    first = None
    chunks, usage = [], {}
    with urllib.request.urlopen(req, timeout=300) as response:
        for raw in response:
            if not raw.startswith(b"data: "):
                continue
            data = raw[6:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(event["error"])
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                text = delta.get("content") or delta.get("reasoning_content") or ""
                if text:
                    first = first or time.perf_counter()
                    chunks.append(text)
    end = time.perf_counter()
    if first is None or not usage:
        raise RuntimeError("stream returned no generated text or usage")
    return {
        "ttft_s": first - start, "wall_s": end - start,
        "decode_s": end - first, "text": "".join(chunks), "usage": usage,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18090")
    parser.add_argument("--model", default="qwen3.6-27b")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    source = Path(__file__).parents[1] / "python/freetoken/moe/offload_cache.py"
    document = tokenizer.encode(source.read_text(), add_special_tokens=False)
    cases = [("warmup", 64, -2), ("warmup", 512, -1)]
    measured = [("prefill", size, rep) for rep in range(args.repeats) for size in (64, 128, 256, 512, 2048)]
    random.Random(4090).shuffle(measured)
    cases.extend(measured)
    cases.extend(("decode", 64, rep) for rep in range(3))
    with args.output.open("w") as out:
        for kind, size, rep in cases:
            nonce = hashlib.sha256(f"{kind}/{size}/{rep}".encode()).hexdigest()[:20]
            if kind == "decode":
                prompt = f"{nonce}. Write a detailed 400-word essay on the history of the Roman Republic, part {rep}."
                max_tokens = 192
            else:
                prefix = f"{nonce}. Read this source excerpt and answer with only the word OK.\n"
                head = tokenizer.encode(prefix, add_special_tokens=False)
                prompt = prefix + tokenizer.decode(document[:max(0, size - len(head))])
                max_tokens = 4
            result = request(args.base_url, {
                "model": args.model, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens, "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
            })
            row = dict(kind=kind, size=size, repeat=rep, **result)
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(json.dumps({key: value for key, value in row.items() if key != "text"}), flush=True)


if __name__ == "__main__":
    main()
