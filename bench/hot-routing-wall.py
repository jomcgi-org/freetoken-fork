"""Paired non-debug HOT routing timings in one server with fixed HOT placement.

Use a naive KV cache, disabled HOT adaptation, and no disk KV reuse. The server
must read the eight-byte control file at begin_prefill and set
offload_kernels._PARALLEL_HOT_ROUTING accordingly. This control hook is confined
to the benchmark server; it records no timers, counters, or profiling data.
"""

import argparse
import hashlib
import importlib.util
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18090")
    parser.add_argument("--model", default="qwen3.6-27b")
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--decode-repeats", type=int, default=3)
    parser.add_argument("--order-offset", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    spec = importlib.util.spec_from_file_location(
        "client", Path(__file__).with_name("selective-prefill.py")
    )
    client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    source = Path(__file__).parents[1] / "python/freetoken/moe/offload_cache.py"
    document = tokenizer.encode(source.read_text(), add_special_tokens=False)

    def one(out, kind, size, rep, parallel):
        with args.control.open("r+b", buffering=0) as control:
            control.write(int(parallel).to_bytes(8, "little"))
        nonce = hashlib.sha256(f"{kind}/{size}/{rep}".encode()).hexdigest()[:20]
        if kind == "decode":
            prompt = f"{nonce}. Write a detailed 400-word essay on the history of the Roman Republic, part {rep}."
            max_tokens = 192
        else:
            prefix = f"{nonce}. Read this source excerpt and answer with only the word OK.\n"
            head = tokenizer.encode(prefix, add_special_tokens=False)
            prompt = prefix + tokenizer.decode(document[:max(0, size - len(head))])
            max_tokens = 4
        result = client.request(args.base_url, {
            "model": args.model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        })
        row = dict(kind=kind, size=size, repeat=rep, parallel=parallel, **result)
        out.write(json.dumps(row) + "\n")
        out.flush()
        print(json.dumps({k: v for k, v in row.items() if k != "text"}), flush=True)

    with args.output.open("w") as out:
        for parallel in (False, True):
            for size in (64, 512, 2048):
                one(out, "warmup", size, -1, parallel)
        cases = [("prefill", size, rep) for rep in range(args.repeats) for size in (64, 512, 2048)]
        cases.extend(("decode", 64, rep) for rep in range(args.decode_repeats))
        random.Random(4090).shuffle(cases)
        # Balance first/second position within each workload. Using the global
        # shuffled case index can put every decode control second, where expert
        # cache warming from the identical first request dominates its timing.
        for kind, size, rep in cases:
            for parallel in (
                (False, True) if (rep + args.order_offset) % 2 == 0 else (True, False)
            ):
                one(out, kind, size, rep, parallel)


if __name__ == "__main__":
    main()
