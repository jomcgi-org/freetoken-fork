"""Measure a fixed complete-response workload at bounded client concurrency.

Total workload wall time includes client submission, completion processing,
and output recording. Per-request streaming latency is reported separately.
Warmups finish before measurement. Whole-worker I/O is sampled only around
each group because concurrent requests cannot own separate worker counters.
"""

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import importlib.util
import json
from pathlib import Path
import time


def sibling(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def complete_case(case, request, checks, protocol):
    result = request(case)
    if result["usage"]["prompt_tokens"] < 1024:
        return dict(result, completed=False, response_checks_passed=False,
                    protocol_error="prompt did not exercise long prefill")
    if case["kind"] == "json":
        result.update(checks.score_json_response(result["text"], case["expected"]))
        result["completed"] = result["finish_reason"] == "stop" and result["passed"]
        result["response_checks_passed"] = result["completed"]
    else:
        result.update(protocol.prose_checks(result["text"], result["finish_reason"]))
        result["response_checks_passed"] = result["prose_format_passed"]
    return result


def run_group(cases, concurrency, send, *, emit=None, snapshot=None, clock=time.perf_counter):
    """Continuously refill at most ``concurrency`` requests and retain failures."""
    if concurrency < 1 or not cases:
        raise ValueError("concurrency and request count must be positive")
    if len({case["ordinal"] for case in cases}) != len(cases):
        raise ValueError("request ordinals must be unique")
    if len({case["warmup"] for case in cases}) != 1:
        raise ValueError("warmups and measured requests must be separate groups")
    before = snapshot() if snapshot is not None else None
    started = clock()

    def invoke(case, submitted):
        active = clock()
        result = {}
        try:
            result = send(case)
        except Exception as error:
            result = dict(completed=False, response_checks_passed=False,
                          request_error=f"{type(error).__name__}: {error}")
        finished = clock()
        return dict(case, **result, diagnostic_phase_io=False,
                    submitted_at_s=submitted-started, active_at_s=active-started,
                    finished_at_s=finished-started, client_queue_s=active-submitted,
                    client_active_s=finished-active)

    rows = []
    remaining = iter(cases)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        pending = set()

        def submit_next():
            case = next(remaining, None)
            if case is not None:
                pending.add(pool.submit(invoke, case, clock()))

        for _ in range(min(concurrency, len(cases))):
            submit_next()
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                row = future.result()
                rows.append(row)
                if emit is not None:
                    emit(row)
                submit_next()
    wall = clock() - started
    after = snapshot() if snapshot is not None else None
    checked = all(row.get("response_checks_passed", False) for row in rows)
    tokens = sum(row.get("usage", {}).get("completion_tokens", 0) for row in rows)
    summary = dict(
        phase="warmup" if cases[0]["warmup"] else "measured", concurrency=concurrency,
        requests=len(rows), completed=sum(row.get("completed", False) for row in rows),
        request_errors=sum("request_error" in row for row in rows),
        all_response_checks_passed=checked, semantic_quality_scored=False,
        wall_s=wall, summed_request_wall_s=sum(row.get("wall_s", 0) for row in rows),
        completion_tokens=tokens,
        checked_requests_per_s=len(rows)/wall if checked and wall > 0 else None,
        checked_completion_tokens_per_s=tokens/wall if checked and wall > 0 else None,
    )
    if snapshot is not None:
        summary["process_io"] = dict(before=before, after=after)
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("baseline", "optimized"))
    parser.add_argument("--concurrency", required=True, type=int)
    parser.add_argument("--base-url", default="http://127.0.0.1:18090")
    parser.add_argument("--model", default="qwen3.6-27b")
    parser.add_argument("--warmup-pairs", type=int, default=2)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--io-pid", required=True, type=int)
    args = parser.parse_args()
    if min(args.concurrency, args.warmup_pairs, args.blocks, args.io_pid) < 1:
        parser.error("concurrency, warmup pairs, blocks, and I/O PID must be positive")
    manifest_path = args.output.with_suffix(".prompts.json")
    summary_path = args.output.with_suffix(".summary.json")
    if any(path.exists() for path in (args.output, manifest_path, summary_path)):
        parser.error("refusing to overwrite benchmark records")
    from transformers import AutoTokenizer

    protocol = sibling("concurrent_protocol", "sustained-prefill-wall.py")
    stream = sibling("concurrent_stream", "selective-prefill.py")
    checks = sibling("concurrent_checks", "staged-prefill-long-output.py")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    cases = protocol.build_cases(tokenizer, Path(__file__).parents[1], args.warmup_pairs, args.blocks)
    with manifest_path.open("x") as out:
        out.write(json.dumps(cases, indent=2) + "\n")

    def request(case):
        return stream.request(args.base_url, dict(
            model=args.model, messages=[dict(role="user", content=case["prompt"])],
            max_tokens=case["max_tokens"], temperature=0,
            chat_template_kwargs=dict(enable_thinking=False),
        ))

    summaries = []
    with args.output.open("x") as out:
        def emit(row):
            record = dict(row, mode=args.mode, client_concurrency=args.concurrency)
            out.write(json.dumps(record) + "\n")
            out.flush()
            print(json.dumps({k: v for k, v in record.items()
                              if k not in ("prompt", "text", "expected")}), flush=True)

        for warmup in (True, False):
            group = [case for case in cases if case["warmup"] == warmup]
            _, summary = run_group(
                group, args.concurrency,
                lambda case: complete_case(case, request, checks, protocol), emit=emit,
                snapshot=lambda: checks.process_io_snapshot(args.io_pid),
            )
            io = summary["process_io"]
            io["delta"] = checks.process_io_delta(io["before"], io["after"])
            summaries.append(dict(summary, mode=args.mode))
    with summary_path.open("x") as out:
        out.write(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps(summaries), flush=True)
    if not all(summary["all_response_checks_passed"] for summary in summaries):
        raise SystemExit("one or more responses failed completion or output checks")


if __name__ == "__main__":
    main()
