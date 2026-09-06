"""Measure successive complete responses without resetting serving state.

The external driver selects one serving policy for the entire process. This
client prepares every prompt before warmup, then sends balanced JSON/prose
blocks with no intentional idle period. HOT adaptation remains active. Keep
KV reuse and diagnostic instrumentation off when comparing the prefill paths.
Prose completion and formatting checks do not score factual correctness.
"""

import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path


PROSE_REFERENCE = """Reference specification:
The router selects expert IDs and routing weights independently of cache
residency. Every selected contribution must execute; a cache hit changes
where weights are obtained, never which experts the router selects.
Original packed checkpoint weights and their scale bytes are authoritative.
Transfers add no quantization. Fully resident file rows may be read through
the page cache; other rows may use direct I/O. Residency hints can become
stale, so they never replace the actual required file read.
Sparse GPU scratch must not advertise uncopied rows as valid experts.
Protected HOT rows can be reused only when their published owner is valid.
A reusable pinned buffer cannot be overwritten until its previous DMA copy
has completed. GPU computation must observe completed weight transfers.
CPU and GPU execution can differ in floating-point rounding. Identical
checkpoint bytes and routing do not promise identical generated text across
those two execution paths. Placement changes must not bias routing toward
HOT experts or drop selected contributions."""

SOURCE_FILES = (
    "python/freetoken/moe/offload_cache.py",
    "python/freetoken/moe/hot_adapt.py",
    "python/freetoken/moe/disk_prefill_staging.py",
)


def sibling(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def schedule(warmup_pairs, blocks):
    if warmup_pairs < 1 or blocks < 1:
        raise ValueError("at least one warmup pair and measured block are required")
    rows = []
    for block in range(-warmup_pairs, blocks):
        kinds = ("json", "essay") if block % 2 == 0 else ("essay", "json")
        for kind in kinds:
            rows.append(dict(ordinal=len(rows), block=block, kind=kind, warmup=block < 0))
    return rows


def prose_checks(text, finish_reason):
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    completed = finish_reason == "stop" and bool(text.strip())
    return dict(
        completed=completed, paragraphs=len(paragraphs), words=len(text.split()),
        prose_format_passed=completed and len(paragraphs) == 3,
        semantic_quality_scored=False,
    )


def build_cases(tokenizer, root, warmup_pairs=2, blocks=6):
    backgrounds = {
        name: tokenizer.decode(tokenizer.encode((root / name).read_text(), add_special_tokens=False)[:1400])
        for name in SOURCE_FILES
    }
    cases = []
    for row in schedule(warmup_pairs, blocks):
        block, kind = row["block"], row["kind"]
        source = SOURCE_FILES[block % len(SOURCE_FILES)]
        nonce = hashlib.sha256(f"sustained/{block}/{kind}".encode()).hexdigest()[:20]
        background = f"{nonce}. Background code excerpt:\n<background>\n{backgrounds[source]}\n</background>\n"
        if kind == "json":
            expected = {f"r{i:02}": (24531 + 7919 * i + 104729 * block) % 99999 for i in range(32)}
            records = "\n".join(f"{key} = {value}" for key, value in expected.items())
            prompt = (
                background + "The code is background only. Copy every record below into one JSON "
                "object, in the given key order. Use integer values. Include all 32 records "
                "exactly once, with no extra keys. Output only the complete JSON object, "
                "without markdown or explanation.\n" + records
            )
            cap = 512
        else:
            expected = None
            prompt = (
                background + PROSE_REFERENCE + "\n\nUsing the reference specification as the "
                "authority for factual claims, explain this design in exactly three paragraphs. "
                "Use two or three sentences per paragraph. Cover routing and weight preservation "
                "first, asynchronous copies and ownership second, and numerical behavior third. "
                "Use no headings or lists."
            )
            cap = 1024
        cases.append(dict(row, source_file=source, prompt=prompt, expected=expected, max_tokens=cap))
    return cases


def measure_request(stream, checks, base_url, payload, pid, *, phase_io=False):
    """Optionally split worker I/O at client-observed first generated text.

    This boundary includes network delay and may include queued generation.
    It is not an exact prefill/decode boundary or expert-only attribution.
    """
    before = checks.process_io_snapshot(pid)
    observer = {"observe_first_text": lambda: checks.process_io_snapshot(pid)} if phase_io else {}
    result = stream.request(base_url, payload, **observer)
    after = checks.process_io_snapshot(pid)
    process_io = dict(before=before, after=after, delta=checks.process_io_delta(before, after))
    if phase_io:
        first = result.pop("first_text_observation")
        process_io.update(
            first_text=first,
            before_first_text_delta=checks.process_io_delta(before, first),
            after_first_text_delta=checks.process_io_delta(first, after),
        )
    return result, process_io


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("baseline", "optimized"))
    parser.add_argument("--base-url", default="http://127.0.0.1:18090")
    parser.add_argument("--model", default="qwen3.6-27b")
    parser.add_argument("--warmup-pairs", type=int, default=2)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--io-pid", required=True, type=int)
    parser.add_argument(
        "--phase-io", action="store_true",
        help="diagnostic I/O snapshot at first text; keep off for wall-time comparisons",
    )
    args = parser.parse_args()
    if args.io_pid <= 0 or args.warmup_pairs < 1 or args.blocks < 1:
        parser.error("I/O PID, warmup pairs, and blocks must be positive")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    root = Path(__file__).parents[1]
    cases = build_cases(tokenizer, root, args.warmup_pairs, args.blocks)
    # The manifest is prepared before the first warmup and is identical in both
    # policies. No tokenization, source reads, or control toggles occur per request.
    args.output.with_suffix(".prompts.json").write_text(json.dumps(cases, indent=2) + "\n")
    client = sibling("sustained_stream", "selective-prefill.py")
    checks = sibling("sustained_checks", "staged-prefill-long-output.py")
    cumulative = dict(prompt_tokens=0, completion_tokens=0, wall_s=0.0)
    with args.output.open("w") as out:
        for case in cases:
            result, process_io = measure_request(client, checks, args.base_url, dict(
                model=args.model, messages=[dict(role="user", content=case["prompt"])],
                max_tokens=case["max_tokens"], temperature=0,
                chat_template_kwargs=dict(enable_thinking=False),
            ), args.io_pid, phase_io=args.phase_io)
            if result["usage"]["prompt_tokens"] < 1024:
                raise RuntimeError("sustained prompt must exercise long prefill")
            for key in ("prompt_tokens", "completion_tokens"):
                cumulative[key] += result["usage"][key]
            cumulative["wall_s"] += result["wall_s"]
            row = dict(case, mode=args.mode, **result, cumulative=dict(cumulative), diagnostic_phase_io=args.phase_io)
            row["process_io"] = process_io
            if case["kind"] == "json":
                row.update(checks.score_json_response(result["text"], case["expected"]))
                row["completed"] = result["finish_reason"] == "stop" and row["passed"]
            else:
                row.update(prose_checks(result["text"], result["finish_reason"]))
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(json.dumps({k: v for k, v in row.items() if k not in ("prompt", "text", "process_io", "expected")}), flush=True)


if __name__ == "__main__":
    main()
