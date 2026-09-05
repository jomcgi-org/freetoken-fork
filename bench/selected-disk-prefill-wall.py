"""Compare selected DISK staging with CPU prefill using client wall time.

The benchmark server reads the control file at each prefill boundary. Zero
selects CPU execution; one selects exact routed-union GPU staging for chunks
of at least 512 tokens. Both arms allocate the same staging ring, use a naive
KV cache, and disable adaptation, prefix reuse, and diagnostic collection.
Reverse --order-offset between starts to balance each individual prompt.
"""

import argparse
import hashlib
import importlib.util
import json
import random
from pathlib import Path


def load_sibling(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18090")
    parser.add_argument("--model", default="qwen3.6-27b")
    parser.add_argument("--order-offset", type=int, choices=(0, 1), required=True)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--prefill-sizes", type=int, nargs="+", default=[64, 512, 2048])
    args = parser.parse_args()
    from transformers import AutoTokenizer

    client = load_sibling("wall_client", "selective-prefill.py")
    fidelity = load_sibling("fidelity", "moe-fidelity.py")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    source = Path(__file__).parents[1] / "python/freetoken/moe/offload_cache.py"
    document = tokenizer.encode(source.read_text(), add_special_tokens=False)

    def order(repeat):
        return (False, True) if (repeat + args.order_offset) % 2 == 0 else (True, False)

    def prefill_prompt(size, repeat, kind):
        nonce = hashlib.sha256(f"{kind}/{size}/{repeat}".encode()).hexdigest()[:20]
        prefix = f"{nonce}. Read this source excerpt and answer with only the word OK.\n"
        head = tokenizer.encode(prefix, add_special_tokens=False)
        return prefix + tokenizer.decode(document[:max(0, size - len(head))])

    def one(out, selected, kind, size, repeat, prompt, max_tokens, expected=None):
        with args.control.open("r+b", buffering=0) as control:
            control.write(int(selected).to_bytes(8, "little"))
        result = client.request(args.base_url, {
            "model": args.model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        })
        row = dict(
            kind=kind, size=size, repeat=repeat, mode="selected" if selected else "cpu",
            order_offset=args.order_offset, prompt=prompt, **result,
        )
        if expected is not None:
            row.update(expected=expected, passed=row["text"].strip() == expected)
            if row["usage"]["prompt_tokens"] < 512:
                raise RuntimeError("fidelity prompt did not exercise the GPU staging policy")
        out.write(json.dumps(row) + "\n")
        out.flush()
        print(json.dumps({key: value for key, value in row.items() if key != "prompt"}), flush=True)

    with args.output.open("w") as out:
        for selected in order(0):
            for size in args.prefill_sizes:
                one(out, selected, "warmup", size, -1,
                    prefill_prompt(size, -1, "warmup"), 4)
        cases = [("prefill", size, rep) for rep in range(args.repeats) for size in args.prefill_sizes]
        cases.extend(("multichunk", 4096, rep) for rep in range(2))
        cases.extend(("decode", 64, rep) for rep in range(2))
        random.Random(4090).shuffle(cases)
        for kind, size, repeat in cases:
            if kind == "decode":
                prompt = f"staging-{repeat}. Write a detailed 400-word essay on the history of the Roman Republic, part {repeat}."
                max_tokens = 192
            else:
                prompt = prefill_prompt(size, repeat, kind)
                max_tokens = 4
            for selected in order(repeat):
                one(out, selected, kind, size, repeat, prompt, max_tokens)

        background = tokenizer.decode(document[:1800])
        for index, (name, question, expected) in enumerate(fidelity.cases()):
            # The previous short checks usually stayed on CPU in both arms.
            # A bounded background puts every check above the GPU threshold.
            prompt = question if name == "long_retrieval" else (
                "The following source excerpt is background only. Answer the question after it.\n"
                f"<background>\n{background}\n</background>\nQuestion: {question}"
            )
            for selected in order(index):
                one(out, selected, "fidelity_" + name, 0, index, prompt, 64, expected)


if __name__ == "__main__":
    main()
