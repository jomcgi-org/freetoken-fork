"""Protocol checks for the sustained client, with no model or network access."""

import importlib.util
import json
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest


path = Path(__file__).parents[1] / "bench/sustained-prefill-wall.py"
spec = importlib.util.spec_from_file_location("sustained_client", path)
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)


def test_sustained_schedule_keeps_warmups_out_of_balanced_measurements():
    rows = client.schedule(2, 6)
    assert len(rows) == 16
    assert all(r["warmup"] for r in rows[:4])
    assert not any(r["warmup"] for r in rows[4:])
    for block in range(6):
        pair = [r for r in rows if r["block"] == block]
        assert {r["kind"] for r in pair} == {"json", "essay"}
    assert rows == client.schedule(2, 6)


@pytest.mark.parametrize("warmups,blocks", [(0, 6), (2, 0), (-1, 1)])
def test_sustained_schedule_rejects_empty_groups(warmups, blocks):
    with pytest.raises(ValueError):
        client.schedule(warmups, blocks)


@pytest.mark.parametrize("text,reason,completed,formatted", [
    ("First.\n\nSecond.\n\nThird.", "stop", True, True),
    ("First.\n\nSecond.\n\nThird.", "length", False, False),
    ("First.\n\nSecond.", "stop", True, False),
    ("First.", None, False, False),
    ("  \n", "stop", False, False),
])
def test_prose_format_checks_never_treat_truncation_as_completion(text, reason, completed, formatted):
    result = client.prose_checks(text, reason)
    assert result["completed"] is completed
    assert result["prose_format_passed"] is formatted
    assert result["semantic_quality_scored"] is False


def test_prompt_manifest_is_reproducible_and_varies_within_the_run(tmp_path):
    class Tokenizer:
        def encode(self, text, **kwargs):
            return list(text)

        def decode(self, tokens):
            return "".join(tokens)

    for i, name in enumerate(client.SOURCE_FILES):
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"source{i}\n" * 300)
    cases = client.build_cases(Tokenizer(), tmp_path)
    assert cases == client.build_cases(Tokenizer(), tmp_path)
    assert cases == client.build_cases(Tokenizer(), tmp_path, mixed_prefill=False)
    assert len({c["prompt"] for c in cases}) == len(cases)
    assert {c["source_file"] for c in cases} == set(client.SOURCE_FILES)
    objects = [c["expected"] for c in cases if c["kind"] == "json"]
    assert all(list(obj) == [f"r{i:02}" for i in range(32)] for obj in objects)
    assert len({tuple(obj.values()) for obj in objects}) == len(objects)
    assert all(client.PROSE_REFERENCE in c["prompt"] for c in cases if c["kind"] == "essay")

    mixed = client.build_cases(Tokenizer(), tmp_path, mixed_prefill=True)
    assert mixed == client.build_cases(Tokenizer(), tmp_path, mixed_prefill=True)
    measured = [c for c in mixed if not c["warmup"]]
    assert Counter((c["kind"], c["prefill_band"]) for c in measured) == {
        (kind, band): 3 for kind in ("json", "essay") for band in ("short", "long")
    }
    assert len({(c["source_file"], c["kind"], c["prefill_band"]) for c in measured}) == 12
    assert Counter(c["prefill_band"] for c in mixed if c["warmup"]) == {"short": 2, "long": 2}
    for original, case in zip(cases, mixed):
        assert original["expected"] == case["expected"]
        assert original["max_tokens"] == case["max_tokens"]
        excerpt = case["prompt"].split("<background>\n", 1)[1].split("\n</background>", 1)[0]
        assert len(excerpt) == (128 if case["prefill_band"] == "short" else 1400)
        if case["prefill_band"] == "long":
            assert case["prompt"] == original["prompt"]


@pytest.mark.parametrize("tokens,band,valid", [
    (1, "short", False), (2, "short", True), (1023, "short", True),
    (1024, "short", False), (1023, "long", False), (1024, "long", True),
])
def test_prefill_band_uses_actual_prompt_tokens(tokens, band, valid):
    assert client.prefill_band_matches(tokens, band) is valid


def test_unknown_prefill_band_is_rejected():
    with pytest.raises(ValueError, match="unknown prefill band"):
        client.prefill_band_matches(1024, "unknown")


@pytest.mark.parametrize("tokens,valid", [(511, True), (1024, False)])
def test_mixed_client_retains_response_when_actual_band_is_wrong(tmp_path, monkeypatch, tokens, valid):
    output = tmp_path / "responses.jsonl"
    monkeypatch.setattr(sys, "argv", [str(path), "--tokenizer", "unused", "--output", str(output),
                                    "--mode", "baseline", "--io-pid", "42", "--mixed-prefill"])
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: object())))
    case = dict(ordinal=0, block=0, kind="json", warmup=False, prefill_band="short",
                source_file="unused", prompt="Copy the record.", expected={"r00": 1}, max_tokens=512)

    def prepare(*args, **kwargs):
        assert kwargs["mixed_prefill"] is True
        return [case]

    monkeypatch.setattr(client, "build_cases", prepare)
    result = dict(text='{"r00":1}', finish_reason="stop", wall_s=1.0, ttft_s=0.2, decode_s=0.8,
                  usage=dict(prompt_tokens=tokens, completion_tokens=7))
    monkeypatch.setattr(client, "measure_request", lambda *a, **k: (dict(result), {}))
    client.main()
    row = json.loads(output.read_text())
    assert row["prefill_band_valid"] is valid and row["completed"] is valid
    assert row["passed"] is True
    assert row["text"] == result["text"] and row["wall_s"] == 1.0


@pytest.fixture
def streamed_worker_io(monkeypatch):
    stream = client.sibling("phase_stream", "selective-prefill.py")
    checks = client.sibling("phase_checks", "staged-prefill-long-output.py")
    state = dict(clock=10.0, phase=0, first_field="content", first_identity=123)
    snapshots = []

    def snapshot(pid):
        phase = state["phase"]
        snapshots.append(phase)
        if phase == 1:
            # Diagnostic overhead must remain visible in client wall time.
            state["clock"] += 2
        return dict(
            pid=pid, starttime_ticks=state["first_identity"] if phase == 1 else 123,
            counters=dict(rchar=(0, 100, 180)[phase], read_bytes=(0, 40, 70)[phase]),
        )

    def event(value):
        return b"data: " + json.dumps(value).encode() + b"\n\n"

    def events():
        yield b": keepalive\n"
        yield event({"choices": [{"delta": {"role": "assistant", "content": ""}}]})
        state.update(clock=13.0, phase=1)
        yield event({"choices": [{"delta": {state["first_field"]: "Hello"}}]})
        state["clock"] += 7
        state["phase"] = 2
        yield event({"choices": [{"delta": {"content": " world"}}]})
        yield event({"choices": [{"delta": {}, "finish_reason": "stop"}]})
        yield event({"choices": [], "usage": {"prompt_tokens": 1700, "completion_tokens": 2}})
        yield b"data: [DONE]\n\n"

    @contextmanager
    def response(req, timeout):
        assert json.loads(req.data)["stream_options"] == {"include_usage": True}
        yield events()

    monkeypatch.setattr(stream.urllib.request, "urlopen", response)
    monkeypatch.setattr(stream, "time", SimpleNamespace(perf_counter=lambda: state["clock"]))
    monkeypatch.setattr(checks, "process_io_snapshot", snapshot)
    return stream, checks, state, snapshots


@pytest.mark.parametrize("phase_io", [False, True])
@pytest.mark.parametrize("first_field", ["content", "reasoning_content"])
def test_phase_io_observes_first_text_once_and_retains_its_cost(streamed_worker_io, phase_io, first_field):
    stream, checks, state, snapshots = streamed_worker_io
    state["first_field"] = first_field
    options = {"phase_io": True} if phase_io else {}
    result, io = client.measure_request(stream, checks, "http://unused", {}, 42, **options)
    assert result["text"] == "Hello world"
    assert result["finish_reason"] == "stop"
    assert result["usage"]["completion_tokens"] == 2
    assert result["ttft_s"] == 3
    assert result["wall_s"] == (12 if phase_io else 10)
    assert result["decode_s"] == (9 if phase_io else 7)
    assert snapshots == ([0, 1, 2] if phase_io else [0, 2])
    assert io["delta"] == {"rchar": 180, "read_bytes": 70}
    assert "first_text_observation" not in result
    if phase_io:
        assert io["before_first_text_delta"] == {"rchar": 100, "read_bytes": 40}
        assert io["after_first_text_delta"] == {"rchar": 80, "read_bytes": 30}
    else:
        assert "first_text" not in io
        assert "before_first_text_delta" not in io
        assert "after_first_text_delta" not in io


def test_phase_io_rejects_identity_change_even_when_outer_snapshots_match(streamed_worker_io):
    stream, checks, state, _ = streamed_worker_io
    state["first_identity"] = 456
    with pytest.raises(RuntimeError, match="worker identity changed"):
        client.measure_request(stream, checks, "http://unused", {}, 42, phase_io=True)
