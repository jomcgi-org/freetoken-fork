"""Repeatable client-side prefill probe for an OpenAI-compatible server."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


WORDS_PER_TOKEN = 0.75
REQUEST_TIMEOUT_S = 1800.0

_WORDS = (
    "amber anchor apple archive arrow atlas autumn bamboo beacon birch blue border "
    "brass breeze bridge brook cable cedar circle cloud cobalt compass copper coral "
    "crane crystal delta drift dune echo elm ember falcon field flint forest frost "
    "garden glass gold granite green harbor hazel hill horizon indigo island ivory "
    "jade juniper lantern leaf linen maple marble meadow mercury mesa moss nickel "
    "north oak ocean olive orbit paper pearl pine pixel plain plum quartz rain raven "
    "reed ridge river rose sail scarlet shadow silver slate snow solar sparrow spring "
    "stone summit teal timber trail umber valley violet water willow wind winter zinc"
).split()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_PREFILL_RATE_RE = re.compile(
    r"input throughput \(token/s\)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
_NEW_TOKENS_RE = re.compile(r"#new-token\s*[:=]\s*(\d+)", re.IGNORECASE)
_CACHED_TOKENS_RE = re.compile(r"#cached-token\s*[:=]\s*(\d+)", re.IGNORECASE)
_HOT_ROUTE_RE = re.compile(
    r"prefill_hot_route_frac\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)(%)?",
    re.IGNORECASE,
)
_CPU_EXPERTS_RE = re.compile(r"prefill_cpu_experts\s*[:=]\s*(\d+)", re.IGNORECASE)
_PREFILL_PATHS_RE = re.compile(
    r"prefill_paths\s*[:=]\s*(.*?)"
    r"(?=,\s*[A-Za-z_][A-Za-z0-9_ /#().-]*\s*:|$)",
    re.IGNORECASE,
)
_GIT_DESCRIBE_RE = re.compile(
    r"(?:git[ _-]?describe|serving tree)\s*[:=]\s*([^\s,]+)",
    re.IGNORECASE,
)


class ProbeError(RuntimeError):
    pass


@dataclass(frozen=True)
class Prompt:
    text: str
    target_tokens: int
    sized_tokens: int | None
    sizing: str


@dataclass(frozen=True)
class StreamMeasurement:
    ttft_s: float
    wall_s: float
    first_delta_kind: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    prefill_tokens_per_s: float


class LogWindow:
    """Read only complete log lines appended after this object is created."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.offset = self.path.stat().st_size if self.path.exists() else 0
        self.start_offset = self.offset
        self.initial_git_describe = self._read_initial_git_describe()
        self._pending = b""
        self._discard_first_partial = False
        if self.offset:
            with self.path.open("rb") as handle:
                handle.seek(self.offset - 1)
                self._discard_first_partial = handle.read(1) != b"\n"

    def _read_initial_git_describe(self) -> str | None:
        """Read metadata from the prefix without exposing old prefill records."""
        if not self.offset:
            return None
        found = None
        read_bytes = 0
        with self.path.open("rb") as handle:
            for raw_line in handle:
                read_bytes += len(raw_line)
                if read_bytes > self.offset:
                    break
                match = _GIT_DESCRIBE_RE.search(raw_line.decode("utf-8", "replace"))
                if match:
                    found = match.group(1)
        return found

    def read_lines(self) -> list[str]:
        if not self.path.exists():
            return []
        size = self.path.stat().st_size
        if size < self.offset:
            self.offset = 0
            self._pending = b""
            self._discard_first_partial = False
        with self.path.open("rb") as handle:
            handle.seek(self.offset)
            chunk = handle.read()
        self.offset += len(chunk)
        data = self._pending + chunk
        parts = data.split(b"\n")
        self._pending = parts.pop()
        if self._discard_first_partial and parts:
            parts.pop(0)
            self._discard_first_partial = False
        return [part.decode("utf-8", "replace") for part in parts]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _rng(seed: int, sequence: int, nonce: str | None) -> random.Random:
    material = f"freetoken-prefill:{seed}:{sequence}:{nonce or ''}".encode()
    derived = int.from_bytes(hashlib.sha256(material).digest()[:16], "big")
    return random.Random(derived)


def _prompt_words(
    seed: int,
    sequence: int,
    count: int,
    invocation_nonce: str | None,
) -> list[str]:
    rng = _rng(seed, sequence, invocation_nonce)
    marker_material = f"{seed}:{sequence}:{invocation_nonce or ''}".encode()
    marker = hashlib.sha256(marker_material).hexdigest()[:16]
    words = [f"probe_{marker}"]
    words.extend(rng.choice(_WORDS) for _ in range(max(0, count - 1)))
    return words


def _encode(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    return list(encoded)


def _decode_prefix(tokenizer: Any, ids: list[int]) -> str:
    try:
        return tokenizer.decode(
            ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(ids)


def _token_sized_text(
    tokenizer: Any,
    target_tokens: int,
    seed: int,
    sequence: int,
    invocation_nonce: str | None,
) -> str:
    word_count = max(32, math.ceil(target_tokens * 1.5))
    for _ in range(8):
        source = " ".join(_prompt_words(seed, sequence, word_count, invocation_nonce))
        ids = _encode(tokenizer, source)
        if len(ids) >= target_tokens:
            text = _decode_prefix(tokenizer, ids[:target_tokens])
            for _ in range(4):
                actual_ids = _encode(tokenizer, text)
                if len(actual_ids) == target_tokens:
                    return text
                if len(actual_ids) > target_tokens:
                    text = _decode_prefix(tokenizer, actual_ids[:target_tokens])
                else:
                    suffix = " " + " ".join(
                        _prompt_words(
                            seed,
                            sequence,
                            target_tokens - len(actual_ids) + 8,
                            invocation_nonce,
                        )
                    )
                    text = _decode_prefix(
                        tokenizer,
                        _encode(tokenizer, text + suffix)[:target_tokens],
                    )
            break
        word_count *= 2
    raise ProbeError(
        f"tokenizer could not construct an exact {target_tokens}-token prompt"
    )


def make_prompt(
    target_tokens: int,
    seed: int,
    sequence: int,
    *,
    tokenizer: Any | None,
    words_per_token: float = WORDS_PER_TOKEN,
    nonce: str | None = None,
) -> Prompt:
    if tokenizer is not None:
        text = _token_sized_text(tokenizer, target_tokens, seed, sequence, nonce)
        return Prompt(
            text=text,
            target_tokens=target_tokens,
            sized_tokens=len(_encode(tokenizer, text)),
            sizing="tokenizer",
        )
    word_count = max(1, math.ceil(target_tokens * words_per_token))
    return Prompt(
        text=" ".join(_prompt_words(seed, sequence, word_count, nonce)),
        target_tokens=target_tokens,
        sized_tokens=None,
        sizing=f"word estimate ({words_per_token:g} words/token)",
    )


def load_tokenizer(source: str) -> tuple[Any | None, str | None]:
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(source), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def measure_stream(
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int,
    *,
    opener: Callable[..., Any] = urllib.request.urlopen,
    clock: Callable[[], float] = time.perf_counter,
    timeout: float = REQUEST_TIMEOUT_S,
) -> StreamMeasurement:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
    )
    started = clock()
    first_token_at: float | None = None
    first_delta_kind: str | None = None
    usage: dict[str, Any] | None = None
    try:
        response = opener(request, timeout=timeout)
        with response:
            for raw in response:
                line = raw.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[len(b"data:") :].strip()
                if data == b"[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("error"):
                    raise ProbeError(f"server stream error: {chunk['error']}")
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta") or {}
                    if first_token_at is None:
                        for kind in ("content", "reasoning_content"):
                            if delta.get(kind):
                                first_token_at = clock()
                                first_delta_kind = kind
                                break
    except urllib.error.HTTPError as exc:
        detail = exc.read(1000).decode("utf-8", "replace")
        raise ProbeError(f"request failed with HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ProbeError(f"request failed: {exc.reason}") from exc
    ended = clock()
    if first_token_at is None or first_delta_kind is None:
        raise ProbeError("stream ended without a content or reasoning_content delta")
    if usage is None:
        raise ProbeError(
            "stream ended without usage; the server must support include_usage"
        )
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    details = usage.get("prompt_tokens_details") or {}
    cached_tokens = int(details.get("cached_tokens", 0) or 0)
    ttft_s = first_token_at - started
    if ttft_s <= 0:
        raise ProbeError(f"non-positive TTFT measured: {ttft_s}")
    return StreamMeasurement(
        ttft_s=ttft_s,
        wall_s=ended - started,
        first_delta_kind=first_delta_kind,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=cached_tokens,
        prefill_tokens_per_s=prompt_tokens / ttft_s,
    )


def parse_log_lines(lines: Iterable[str]) -> dict[str, Any]:
    prefill: list[dict[str, Any]] = []
    pending_split: dict[str, Any] = {}
    git_describe: str | None = None
    for raw_line in lines:
        line = _ANSI_RE.sub("", raw_line)
        is_prefill_line = "Prefill batch" in line
        git_match = _GIT_DESCRIBE_RE.search(line)
        if git_match:
            git_describe = git_match.group(1)

        split: dict[str, Any] = {}
        hot_match = _HOT_ROUTE_RE.search(line)
        if hot_match:
            hot = float(hot_match.group(1))
            split["prefill_hot_route_frac"] = hot / 100.0 if hot_match.group(2) else hot
        cpu_match = _CPU_EXPERTS_RE.search(line)
        if cpu_match:
            split["prefill_cpu_experts"] = int(cpu_match.group(1))
        paths_match = _PREFILL_PATHS_RE.search(line)
        if paths_match:
            split["prefill_paths"] = paths_match.group(1).strip()

        rate_match = _PREFILL_RATE_RE.search(line) if is_prefill_line else None
        if rate_match:
            record: dict[str, Any] = {
                "input_tokens_per_s": float(rate_match.group(1)),
                "line": line.strip(),
            }
            new_match = _NEW_TOKENS_RE.search(line)
            cached_match = _CACHED_TOKENS_RE.search(line)
            if new_match:
                record["new_tokens"] = int(new_match.group(1))
            if cached_match:
                record["cached_tokens"] = int(cached_match.group(1))
            record.update(pending_split)
            record.update(split)
            pending_split = {}
            prefill.append(record)
        elif split and is_prefill_line:
            if prefill:
                prefill[-1].update(split)
            else:
                pending_split.update(split)
    return {"prefill": prefill, "git_describe": git_describe}


def server_prefill_rate(records: list[dict[str, Any]]) -> float | None:
    timed_tokens = [
        (float(row["new_tokens"]), float(row["input_tokens_per_s"]))
        for row in records
        if row.get("new_tokens", 0) > 0 and row.get("input_tokens_per_s", 0) > 0
    ]
    if timed_tokens:
        tokens = sum(item[0] for item in timed_tokens)
        seconds = sum(item[0] / item[1] for item in timed_tokens)
        return tokens / seconds if seconds > 0 else None
    rates = [float(row["input_tokens_per_s"]) for row in records]
    return statistics.median(rates) if rates else None


def _run_record(
    prompt: Prompt,
    measurement: StreamMeasurement,
    log_lines: list[str],
) -> dict[str, Any]:
    parsed = parse_log_lines(log_lines)
    records = parsed["prefill"]
    latest = records[-1] if records else {}
    server_cached_tokens = (
        sum(
            int(record["cached_tokens"])
            for record in records
            if "cached_tokens" in record
        )
        if any("cached_tokens" in record for record in records)
        else None
    )
    return {
        "prompt": prompt.text,
        "prompt_sha256": hashlib.sha256(prompt.text.encode("utf-8")).hexdigest(),
        "target_tokens": prompt.target_tokens,
        "sized_tokens": prompt.sized_tokens,
        "sizing": prompt.sizing,
        **asdict(measurement),
        "server_cached_tokens": server_cached_tokens,
        "server_prefill_tokens_per_s": server_prefill_rate(records),
        "prefill_hot_route_frac": latest.get("prefill_hot_route_frac"),
        "prefill_cpu_experts": latest.get("prefill_cpu_experts"),
        "prefill_paths": latest.get("prefill_paths"),
        "server_prefill_records": records,
        "serving_git_describe": parsed["git_describe"],
    }


def _format_run(
    number: int, row: dict[str, Any], *, use_server_cache: bool = False
) -> str:
    engine = row["server_prefill_tokens_per_s"]
    engine_text = f"{engine:.2f}" if engine is not None else "n/a"
    cached_tokens = (
        row.get("server_cached_tokens") if use_server_cache else row["cached_tokens"]
    )
    cached_text = "n/a" if cached_tokens is None else f"{cached_tokens} tok"
    if use_server_cache:
        cached_text += " (engine log)"
    fields = [
        f"run {number}",
        f"prompt={row['prompt_tokens']} tok",
        f"ttft={row['ttft_s']:.3f}s",
        f"prefill={row['prefill_tokens_per_s']:.2f} tok/s",
        f"engine={engine_text} tok/s",
        f"wall={row['wall_s']:.3f}s",
        f"completion={row['completion_tokens']} tok",
        f"cached={cached_text}",
    ]
    hot = row.get("prefill_hot_route_frac")
    if hot is not None:
        fields.append(f"prefill_hot_route_frac={hot:.2%}")
    cpu = row.get("prefill_cpu_experts")
    if cpu is not None:
        fields.append(f"prefill_cpu_experts={cpu}")
    paths = row.get("prefill_paths")
    if paths is not None:
        fields.append(f"prefill_paths={paths}")
    return "  ".join(fields)


def build_parser(prog: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument(
        "--base-url", required=True, help="server origin or v1 base URL"
    )
    parser.add_argument("--model", required=True, help="served model name")
    parser.add_argument(
        "--tokenizer",
        help="tokenizer path or name; defaults to --model",
    )
    parser.add_argument("--warmup-tokens", type=_positive_int, default=1500)
    parser.add_argument("--tokens", type=_positive_int, default=2000)
    parser.add_argument("--runs", type=_positive_int, default=3)
    parser.add_argument("--max-tokens", type=_positive_int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--nonce",
        help="reproduce a non-repeat invocation with the previously printed nonce",
    )
    parser.add_argument(
        "--repeat",
        action="store_true",
        help="repeat one prompt to exercise the warm prefix and session-cache path",
    )
    parser.add_argument(
        "--log", type=Path, help="server log to tail from its current byte offset"
    )
    parser.add_argument(
        "--json", type=Path, dest="json_path", help="write full results as JSON"
    )
    return parser


def main(argv: list[str] | None = None, prog: str = "freetoken.probe_prefill") -> int:
    parser = build_parser(prog)
    args = parser.parse_args(argv)
    if args.repeat and args.nonce is not None:
        parser.error("--nonce cannot be used with --repeat")
    if args.log is not None and not args.log.is_file():
        parser.error(f"--log is not a file: {args.log}")

    invocation_nonce = None if args.repeat else (args.nonce or os.urandom(16).hex())
    print(
        f"nonce={invocation_nonce if invocation_nonce is not None else 'none'}",
        flush=True,
    )

    endpoint = chat_completions_url(args.base_url)
    log_window = LogWindow(args.log) if args.log is not None else None
    tokenizer_source = args.tokenizer or args.model
    tokenizer, tokenizer_error = load_tokenizer(tokenizer_source)
    if tokenizer is None:
        print(
            f"warning: could not load tokenizer from {tokenizer_source!r}: {tokenizer_error}; "
            f"using {WORDS_PER_TOKEN:g} words/token estimate",
            file=sys.stderr,
        )

    def execute(target_tokens: int, sequence: int) -> dict[str, Any]:
        prompt = make_prompt(
            target_tokens,
            args.seed,
            sequence,
            tokenizer=tokenizer,
            nonce=invocation_nonce,
        )
        measurement = measure_stream(
            endpoint,
            args.model,
            prompt.text,
            args.max_tokens,
        )
        lines = log_window.read_lines() if log_window is not None else []
        return _run_record(prompt, measurement, lines)

    try:
        warmup_sequence = 0 if args.repeat else -1
        warmup = execute(args.warmup_tokens, warmup_sequence)
        print(
            f"warmup complete: prompt={warmup['prompt_tokens']} tok "
            f"ttft={warmup['ttft_s']:.3f}s",
            file=sys.stderr,
        )
        rows = []
        cache_path = "warm" if args.repeat else "cold"
        for index in range(args.runs):
            sequence = 0 if args.repeat else index
            row = execute(args.tokens, sequence)
            rows.append(row)
            print(
                _format_run(index + 1, row, use_server_cache=log_window is not None),
                flush=True,
            )
            server_cached = row.get("server_cached_tokens")
            if cache_path == "cold" and server_cached:
                print(
                    f"WARNING: run {index + 1} is labeled cold but the engine log "
                    f"reported {server_cached} cached tokens",
                    file=sys.stderr,
                    flush=True,
                )
    except (OSError, ValueError, json.JSONDecodeError, ProbeError) as exc:
        print(f"{prog}: error: {exc}", file=sys.stderr)
        return 1

    rates = [row["prefill_tokens_per_s"] for row in rows]
    summary = {
        "median_prefill_tokens_per_s": statistics.median(rates),
        "min_prefill_tokens_per_s": min(rates),
        "max_prefill_tokens_per_s": max(rates),
        "runs": args.runs,
        "target_tokens": args.tokens,
        "cache_path": cache_path,
    }
    print(
        f"prefill tok/s median={summary['median_prefill_tokens_per_s']:.2f} "
        f"min={summary['min_prefill_tokens_per_s']:.2f} "
        f"max={summary['max_prefill_tokens_per_s']:.2f} "
        f"({args.runs} runs, {args.tokens} tokens, {cache_path})"
    )

    if args.json_path is not None:
        serving_describe = next(
            (
                row["serving_git_describe"]
                for row in [warmup, *rows]
                if row["serving_git_describe"] is not None
            ),
            None,
        )
        if serving_describe is None and log_window is not None:
            serving_describe = log_window.initial_git_describe
        document = {
            "schema_version": 1,
            "inputs": {
                "base_url": args.base_url,
                "endpoint": endpoint,
                "model": args.model,
                "tokenizer": args.tokenizer,
                "tokenizer_source": tokenizer_source,
                "tokenizer_loaded": tokenizer is not None,
                "tokenizer_error": tokenizer_error,
                "words_per_token": WORDS_PER_TOKEN if tokenizer is None else None,
                "warmup_tokens": args.warmup_tokens,
                "tokens": args.tokens,
                "runs": args.runs,
                "max_tokens": args.max_tokens,
                "temperature": 0,
                "enable_thinking": False,
                "request_timeout_s": REQUEST_TIMEOUT_S,
                "seed": args.seed,
                "nonce": invocation_nonce,
                "repeat": args.repeat,
                "log": str(args.log) if args.log is not None else None,
                "log_start_offset": log_window.start_offset
                if log_window is not None
                else None,
            },
            "serving_git_describe": serving_describe,
            "warmup": warmup,
            "runs": rows,
            "summary": summary,
        }
        try:
            with args.json_path.open("w", encoding="utf-8") as handle:
                json.dump(document, handle, indent=2, sort_keys=True)
                handle.write("\n")
        except OSError as exc:
            print(f"{prog}: error writing {args.json_path}: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
