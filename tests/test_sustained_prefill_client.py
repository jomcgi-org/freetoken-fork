"""Protocol checks for the sustained client, with no model or network access."""

import importlib.util
from pathlib import Path

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
    assert len({c["prompt"] for c in cases}) == len(cases)
    assert {c["source_file"] for c in cases} == set(client.SOURCE_FILES)
    objects = [c["expected"] for c in cases if c["kind"] == "json"]
    assert all(list(obj) == [f"r{i:02}" for i in range(32)] for obj in objects)
    assert len({tuple(obj.values()) for obj in objects}) == len(objects)
    assert all(client.PROSE_REFERENCE in c["prompt"] for c in cases if c["kind"] == "essay")
