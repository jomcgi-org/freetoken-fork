"""Measure whole responses after long CPU or staged prefill, with diagnostics off.

Use the same threshold-only A/B selector as selected-disk-prefill-wall.py.
Both server starts retain the same placement and disable KV reuse. Reverse
--order-offset between starts. Warmups generate complete responses too, so
first-use decode pages do not all fall into the first measured arm.

Optional --io-pid snapshots Linux process I/O outside the client timing
interval. Run the client on the server host and pass its GPU worker PID.
These counters include all worker I/O, including PLE, and are not an expert
traffic attribution or a measurement of device bandwidth.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

import importlib.util


def sibling(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def process_io_snapshot(pid, proc_root=Path("/proc")):
    base = proc_root / str(pid)
    # comm can contain spaces and parentheses. Field 22 follows the last ')'.
    fields = (base / "stat").read_text().rpartition(")")[2].split()
    counters = {
        key: int(value)
        for key, value in (line.split(":", 1) for line in (base / "io").read_text().splitlines())
    }
    return dict(pid=pid, starttime_ticks=int(fields[19]), counters=counters)


def process_io_delta(before, after):
    if (before["pid"], before["starttime_ticks"]) != (after["pid"], after["starttime_ticks"]):
        raise RuntimeError("I/O worker identity changed during the request")
    return {key: value - before["counters"][key] for key, value in after["counters"].items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18090")
    parser.add_argument("--model", default="qwen3.6-27b")
    parser.add_argument("--order-offset", type=int, choices=(0, 1), required=True)
    parser.add_argument("--mode-labels", nargs=2, default=("cpu", "selected"),
                        metavar=("CONTROL_0", "CONTROL_1"))
    parser.add_argument("--only-mode", type=int, choices=(0, 1),
                        help="Keep one policy for a whole server start, including warmup.")
    parser.add_argument("--io-pid", type=int,
                        help="Optional local Linux GPU worker PID for process I/O snapshots.")
    args = parser.parse_args()
    if args.io_pid is not None and args.io_pid <= 0:
        parser.error("--io-pid must be positive")
    from transformers import AutoTokenizer

    client = sibling("wall_client", "selective-prefill.py")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    source = Path(__file__).parents[1] / "python/freetoken/moe/offload_cache.py"
    document = tokenizer.encode(source.read_text(), add_special_tokens=False)

    def order(repeat):
        if args.only_mode is not None:
            return (bool(args.only_mode),)
        return (False, True) if (repeat + args.order_offset) % 2 == 0 else (True, False)

    def prompt(kind, repeat):
        nonce = hashlib.sha256(f"long-output/{kind}/{repeat}".encode()).hexdigest()[:20]
        if kind == "json":
            expected = {f"r{i:02}": (24531 + 7919 * i) % 99999 for i in range(32)}
            records = "\n".join(f"{key} = {value}" for key, value in expected.items())
            text = (
                f"{nonce}. The following source is background only:\n<background>\n"
                + tokenizer.decode(document[:1400]) + "\n</background>\n"
                "Copy every record below into one JSON object, in the given key order. "
                "Use integer values. Include all 32 records exactly once, with no extra keys. "
                "Output only the complete JSON object, without markdown or explanation.\n" + records
            )
            return text, expected, 384
        text = (
            f"{nonce}. Read this source excerpt:\n<source>\n"
            + tokenizer.decode(document[:1800]) + "\n</source>\n"
            "Explain how an expert cache can move model weights among GPU memory, RAM, "
            "and disk while preserving the router's choices. Write three detailed paragraphs "
            "covering ownership, asynchronous copies, and numerical behavior."
        )
        return text, None, 192

    def one(out, selected, kind, repeat, warmup):
        text, expected, max_tokens = prompt(kind, repeat)
        with args.control.open("r+b", buffering=0) as control:
            control.write(int(selected).to_bytes(8, "little"))
        io_before = process_io_snapshot(args.io_pid) if args.io_pid is not None else None
        result = client.request(args.base_url, dict(
            model=args.model, messages=[dict(role="user", content=text)],
            max_tokens=max_tokens, temperature=0,
            chat_template_kwargs=dict(enable_thinking=False),
        ))
        io_after = process_io_snapshot(args.io_pid) if args.io_pid is not None else None
        row = dict(kind=kind, repeat=repeat, warmup=warmup,
                   mode=args.mode_labels[int(selected)], order_offset=args.order_offset,
                   prompt=text, **result)
        if io_before is not None:
            row["process_io"] = dict(before=io_before, after=io_after,
                                     delta=process_io_delta(io_before, io_after))
        if result["usage"]["prompt_tokens"] < 1024:
            raise RuntimeError("long output prompt did not exercise staged prefill")
        if expected is not None:
            try:
                parsed = json.loads(result["text"])
            except ValueError:
                parsed = None
            row.update(expected=expected, passed=parsed == expected and list(parsed) == list(expected),
                       correct_records=sum(isinstance(parsed, dict) and parsed.get(k) == v
                                           for k, v in expected.items()))
        out.write(json.dumps(row) + "\n")
        out.flush()
        print(json.dumps({k: v for k, v in row.items() if k != "prompt"}), flush=True)

    with args.output.open("w") as out:
        for selected in order(0):
            for kind in ("json", "essay"):
                one(out, selected, kind, -1, True)
        cases = [(kind, repeat) for kind in ("json", "essay") for repeat in range(2)]
        random.Random(4090).shuffle(cases)
        for kind, repeat in cases:
            for selected in order(repeat):
                one(out, selected, kind, repeat, False)


if __name__ == "__main__":
    main()
