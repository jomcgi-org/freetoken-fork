from __future__ import annotations

import json
import math

import pytest

from freetoken import probe_prefill
from freetoken.probe_prefill import (
    LogWindow,
    WORDS_PER_TOKEN,
    make_prompt,
    measure_stream,
    parse_log_lines,
    server_prefill_rate,
)


class WordTokenizer:
    """A reversible one-token-per-word reference tokenizer."""

    def __init__(self):
        self._ids: dict[str, int] = {}
        self._words: dict[int, str] = {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        ids = []
        for word in text.split():
            if word not in self._ids:
                token_id = len(self._ids) + 1
                self._ids[word] = token_id
                self._words[token_id] = word
            ids.append(self._ids[word])
        return ids

    def decode(self, ids: list[int], **kwargs: object) -> str:
        return " ".join(self._words[token_id] for token_id in ids)


def test_prompts_are_deterministic_and_unique_per_sequence():
    first = make_prompt(40, 1234, 0, tokenizer=None)
    again = make_prompt(40, 1234, 0, tokenizer=None)
    next_run = make_prompt(40, 1234, 1, tokenizer=None)

    assert first == again
    assert first.text != next_run.text


def test_prompt_nonce_changes_invocations_and_can_be_replayed():
    first = make_prompt(40, 1234, 0, tokenizer=None, nonce="invocation-a")
    replay = make_prompt(40, 1234, 0, tokenizer=None, nonce="invocation-a")
    separate_invocation = make_prompt(40, 1234, 0, tokenizer=None, nonce="invocation-b")

    assert first == replay
    assert first.text != separate_invocation.text


def test_tokenizer_sizes_prompt_in_tokens():
    tokenizer = WordTokenizer()
    prompt = make_prompt(37, 9, 2, tokenizer=tokenizer)

    assert prompt.sizing == "tokenizer"
    assert prompt.sized_tokens == 37
    assert len(tokenizer.encode(prompt.text)) == 37


def test_word_fallback_uses_stated_estimate():
    prompt = make_prompt(10, 9, 2, tokenizer=None)

    assert len(prompt.text.split()) == math.ceil(10 * WORDS_PER_TOKEN)
    assert prompt.sized_tokens is None
    assert f"{WORDS_PER_TOKEN:g} words/token" in prompt.sizing


class FakeStream:
    def __init__(self, lines: list[bytes]):
        self.lines = lines

    def __enter__(self) -> "FakeStream":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def __iter__(self):
        return iter(self.lines)


def _event(value: dict) -> bytes:
    return b"data: " + json.dumps(value).encode() + b"\n"


def test_stream_ttft_and_prefill_arithmetic_uses_first_token_delta():
    response = FakeStream(
        [
            _event({"choices": [{"delta": {"role": "assistant", "content": ""}}]}),
            _event({"choices": [{"delta": {"reasoning_content": "hidden"}}]}),
            _event({"choices": [{"delta": {"content": "hello"}}]}),
            _event(
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 2000,
                        "completion_tokens": 7,
                        "prompt_tokens_details": {"cached_tokens": 125},
                    },
                }
            ),
            b"data: [DONE]\n",
        ]
    )
    opened = {}

    def opener(request, timeout):
        opened["request"] = request
        opened["timeout"] = timeout
        return response

    stamps = iter([10.0, 12.5, 15.0])
    result = measure_stream(
        "http://127.0.0.1:8090/v1/chat/completions",
        "model",
        "prompt",
        32,
        opener=opener,
        clock=lambda: next(stamps),
    )

    assert result.ttft_s == pytest.approx(2.5)
    assert result.wall_s == pytest.approx(5.0)
    assert result.first_delta_kind == "reasoning_content"
    assert result.prompt_tokens == 2000
    assert result.completion_tokens == 7
    assert result.cached_tokens == 125
    assert result.prefill_tokens_per_s == pytest.approx(800.0)
    sent = json.loads(opened["request"].data)
    assert sent["stream_options"] == {"include_usage": True}
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}


def test_log_parser_reads_prefill_and_split_stats_fixture_lines():
    # This is the field order emitted by scheduler/status.py::_report_prefill.
    prefill_line = (
        "[2026-09-02 10:14:03] Prefill batch, #new-seq: 1, #new-token: 2000, "
        "#cached-token: 0, token usage: 0.02, #running-req: 1, #queue-req: 0, "
        "client_aborts: 0, input throughput (token/s): 63.80"
    )
    # This is the appended disk split format emitted by scheduler/scheduler.py.
    split_stats_line = (
        "[2026-09-02 10:14:04] Prefill batch, #new-seq: 1, #new-token: 1500, "
        "#cached-token: 0, token usage: 0.02, #running-req: 1, #queue-req: 0, "
        "client_aborts: 0, input throughput (token/s): 56.40, disk prefetch calls: 42, "
        "disk pages requested: 84, disk major faults: 0, "
        "disk major faults/decode step: 0.00, moe_prefill_coalesce_experts: 17, "
        "moe_prefill_coalesce_ms: 2.1, prefill_hot_route_frac: 62.50%, "
        "prefill_cpu_experts: 17, disk distinct_experts/step: 17.00, "
        "disk dedup_ratio: 1.25, protected_experts: 61, "
        "prefill_paths: gpu=61 cpu=17"
    )
    parsed = parse_log_lines(
        [
            "serving tree: v0.9.1-43-g8230fdb",
            prefill_line,
            split_stats_line,
        ]
    )

    assert parsed["git_describe"] == "v0.9.1-43-g8230fdb"
    assert len(parsed["prefill"]) == 2
    assert parsed["prefill"][0]["new_tokens"] == 2000
    assert parsed["prefill"][0]["input_tokens_per_s"] == pytest.approx(63.8)
    row = parsed["prefill"][1]
    assert row["new_tokens"] == 1500
    assert row["cached_tokens"] == 0
    assert row["input_tokens_per_s"] == pytest.approx(56.4)
    assert row["prefill_hot_route_frac"] == pytest.approx(0.625)
    assert row["prefill_cpu_experts"] == 17
    assert row["prefill_paths"] == "gpu=61 cpu=17"


def test_log_parser_does_not_overwrite_prefill_split_stats_from_decode_line():
    prefill_line = (
        "[2026-09-02 10:14:04] Prefill batch, #new-token: 1500, #cached-token: 0, "
        "input throughput (token/s): 56.40, prefill_hot_route_frac: 7.09%, "
        "prefill_cpu_experts: 17, prefill_paths: gpu=61 cpu=17"
    )
    decode_line = (
        "[2026-09-02 10:14:05] Decode batch, throughput (token/s): 4.20, "
        "prefill_hot_route_frac: 0.00%, prefill_cpu_experts: 0, "
        "prefill_paths: gpu=0 cpu=0"
    )

    row = parse_log_lines([prefill_line, decode_line])["prefill"][0]

    assert row["prefill_hot_route_frac"] == pytest.approx(0.0709)
    assert row["prefill_cpu_experts"] == 17
    assert row["prefill_paths"] == "gpu=61 cpu=17"


def test_main_uses_engine_cached_tokens_and_records_nonce(
    monkeypatch, tmp_path, capsys
):
    log = tmp_path / "server.log"
    log.touch()
    output = tmp_path / "probe.json"
    request_number = 0

    def fake_measure_stream(*args, **kwargs):
        nonlocal request_number
        cached = 0 if request_number == 0 else 1408
        request_number += 1
        with log.open("a", encoding="utf-8") as handle:
            handle.write(
                "Prefill batch, #new-token: 592, "
                f"#cached-token: {cached}, input throughput (token/s): 60.00\n"
            )
        return probe_prefill.StreamMeasurement(
            ttft_s=2.0,
            wall_s=3.0,
            first_delta_kind="content",
            prompt_tokens=2000,
            completion_tokens=1,
            cached_tokens=0,
            prefill_tokens_per_s=1000.0,
        )

    monkeypatch.setattr(probe_prefill, "load_tokenizer", lambda source: (None, "test"))
    monkeypatch.setattr(probe_prefill, "measure_stream", fake_measure_stream)

    result = probe_prefill.main(
        [
            "--base-url",
            "http://127.0.0.1:8090",
            "--model",
            "model",
            "--runs",
            "1",
            "--nonce",
            "replay-me",
            "--log",
            str(log),
            "--json",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    document = json.loads(output.read_text())
    assert result == 0
    assert "nonce=replay-me" in captured.out
    assert "cached=1408 tok (engine log)" in captured.out
    assert "WARNING: run 1 is labeled cold" in captured.err
    assert document["inputs"]["nonce"] == "replay-me"
    assert document["runs"][0]["cached_tokens"] == 0
    assert document["runs"][0]["server_cached_tokens"] == 1408


def test_server_prefill_rate_combines_chunked_prefill_by_elapsed_time():
    records = [
        {"new_tokens": 1000, "input_tokens_per_s": 50.0},
        {"new_tokens": 1000, "input_tokens_per_s": 100.0},
    ]

    assert server_prefill_rate(records) == pytest.approx(2000 / 30)


def test_log_window_starts_at_byte_offset_and_buffers_partial_lines(tmp_path):
    log = tmp_path / "server.log"
    log.write_bytes(b"serving tree: v0.9.1-43-g8230fdb\nold partial")
    window = LogWindow(log)

    with log.open("ab") as handle:
        handle.write(b" completion\nnew line one\nnew line")
    assert window.read_lines() == ["new line one"]

    with log.open("ab") as handle:
        handle.write(b" two\n")
    assert window.read_lines() == ["new line two"]
    assert window.start_offset == len(b"serving tree: v0.9.1-43-g8230fdb\nold partial")
    assert window.initial_git_describe == "v0.9.1-43-g8230fdb"
