"""Host-only checks for the hot-set A/B tick validator (issue #23)."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location(
    "adapt_ticks_check", ROOT / "bench" / "adapt-ticks-check.py"
)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)

STARTUP = (
    "MoE HOT adaptation intervals: mode=auto, aim=phase, unit=routed_tokens, "
    "fill_ticks=4, fill_interval=500, steady_interval=1000, current_interval=1000, idle=off"
)


def stats(prefill, decode, idle, interval=1000):
    return (
        f"hot_adapt_interval: {interval}, hot_adapt_ticks_prefill: {prefill}, "
        f"hot_adapt_prefill_run_swaps: 0, hot_adapt_ticks_decode: {decode}, "
        f"hot_adapt_ticks_idle: {idle}, disk lookahead_hit_rate: 0.0"
    )


def test_ticks_are_summed_across_delta_lines():
    text = "\n".join([STARTUP, stats(1, 2, 0), stats(0, 3, 1, 150)])
    summary = check.summarize(text)
    assert summary["ticks"] == {"prefill": 1, "decode": 5, "idle": 1}
    assert summary["mode"] == "auto"
    assert summary["startup_interval"] == 1000
    assert summary["last_interval"] == 150


def test_arm_with_no_ticks_is_invalid_and_exits_nonzero(tmp_path):
    journal = tmp_path / "j.log"
    journal.write_text("\n".join([STARTUP, stats(0, 0, 0)]))
    summary = check.summarize(journal.read_text())
    assert check.verdict(summary, 3) == "INVALID"
    assert "ARM INVALID" in check.render(summary, 3)
    assert check.main([str(journal), "--require"]) == 2
    assert check.main([str(journal)]) == 0


def test_idle_ticks_alone_do_not_validate_an_arm():
    summary = check.summarize(stats(0, 0, 5))
    assert check.verdict(summary, 3) == "INVALID"


def test_enough_non_idle_ticks_is_valid(tmp_path):
    journal = tmp_path / "j.log"
    journal.write_text(stats(1, 4, 0, 150))
    assert check.main([str(journal), "--require", "--min-ticks", "3"]) == 0
    assert "ARM INVALID" not in check.render(check.summarize(journal.read_text()), 3)


def test_empty_journal_is_invalid():
    summary = check.summarize("")
    assert summary["mode"] is None
    assert check.verdict(summary, 1) == "INVALID"
