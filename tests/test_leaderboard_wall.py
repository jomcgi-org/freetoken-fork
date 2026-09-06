import asyncio
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location(
    "leaderboard_wall", Path(__file__).parents[1] / "bench/leaderboard-wall.py")
wall = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wall)


def reports():
    return [dict(completed=True, manifest_sha256="same", historical={}, records=[
        dict(arm=arm, mode=mode, task_id=task, task_wall_s=10. if mode == "off" else 8.,
             model_request_s=6., completion_qualified=True, observation=dict(calls=2),
             cell=dict(attempts=[dict(completion_tokens=50)])) for task in ("one", "two")])
        for arm, mode in wall.ARMS]


def test_failed_attempts_remain_in_both_orders_and_totals():
    data = reports()
    data[1]["records"][0].update(completion_qualified=False, task_wall_s=30.)
    result = wall.summarize(data, ["one", "two"])
    assert not result["all_tasks_passed"]
    on = result["comparison"]["modes"]["on"]
    assert on["tasks"] == 4 and on["passed"] == 3
    assert on["task_wall_s"] == 54.
    assert result["comparison"]["wall_reduction_percent"] < 0
    assert result["orders"]["B/A"]["wall_reduction_percent"] > 0


@pytest.mark.parametrize("change", ["missing", "duplicate", "unfinished", "manifest", "mode"])
def test_incomplete_or_incomparable_runs_cannot_produce_a_speedup(change):
    data = reports()
    if change == "missing":
        data[0]["records"].pop()
    elif change == "duplicate":
        data[0]["records"].append(copy.deepcopy(data[0]["records"][0]))
    elif change == "unfinished":
        data[0]["completed"] = False
    elif change == "manifest":
        data[0]["manifest_sha256"] = "different"
    else:
        data[0]["records"][0]["mode"] = "on"
    with pytest.raises(ValueError, match="matched task set"):
        wall.summarize(data, ["one", "two"])


def test_transport_failures_with_no_recorded_model_time_still_summarize():
    data = reports()
    for report in data:
        for row in report["records"]:
            row.update(completion_qualified=False, model_request_s=0.)
    result = wall.summarize(data, ["one", "two"])
    assert not result["all_tasks_passed"]
    assert result["comparison"]["model_time_reduction_percent"] is None


@pytest.mark.parametrize("duration", [-1., float("nan"), float("inf")])
def test_invalid_measurements_are_rejected(duration):
    data = reports()
    data[0]["records"][0]["task_wall_s"] = duration
    with pytest.raises(ValueError, match="finite nonnegative"):
        wall.summarize(data, ["one", "two"])


@pytest.mark.parametrize("extra_file", [False, True])
def test_original_loop_budgets_and_total_wall_include_grading(tmp_path, extra_file):
    fixture = tmp_path / "tasks/example/fixture"
    fixture.mkdir(parents=True)
    (fixture / "code.py").write_text("buggy\n")
    now = [0.]
    task = SimpleNamespace(id="example", version="v1", prompt="original prompt", target_files=["code.py"],
                           agent=SimpleNamespace(max_turns=20, max_tokens=None, exec=False),
                           verifier=SimpleNamespace(kind="command", args={"unchanged": True}))
    model = SimpleNamespace(id="recorded", api_model="served", extra_body={"flag": True},
                            params=SimpleNamespace(max_tokens=16384))
    cell_dict = dict(attempts=[dict(completion_tokens=50)])
    cell = SimpleNamespace(total_latency_ms=2000, first_attempt_passed=True, tool_use_ok=True,
                           model_dump=lambda: cell_dict)

    async def chat(**kwargs):
        assert kwargs == dict(model="served", extra_body={"flag": True})
        now[0] += 2.
        return SimpleNamespace(latency_ms=2000, prompt_tokens=100, completion_tokens=50,
                               message={"tool_calls": [{"function": {"name": "done"}}]})

    def grade(work, args):
        assert args == {"unchanged": True}
        now[0] += 1.
        return SimpleNamespace(passed=True)

    async def loop(**kwargs):
        assert kwargs["task_prompt"] == "original prompt"
        assert kwargs["max_tokens"] == 16384 and kwargs["max_turns"] == 20
        assert kwargs["allow_exec"] is False
        await kwargs["chat"](model="recorded")
        work = tmp_path / "work"
        work.mkdir()
        (work / "code.py").write_text("fixed\n")
        if extra_file:
            (work / "unrelated.py").write_text("outside requested scope\n")
        kwargs["verify"](work, kwargs["verifier_args"])
        now[0] += 4.
        return cell

    api = SimpleNamespace(root=tmp_path, tasks={"example": task}, model=model,
                          manifest={"historical": {"example": {"content_hash": "frozen"}}},
                          run_agent_cell=loop, get_verifier=lambda kind: grade)
    row = asyncio.run(wall.run_task(api, "example", SimpleNamespace(chat=chat), clock=lambda: now[0]))
    assert row["task_wall_s"] == 7.
    assert row["model_request_s"] == 2.
    assert row["observation"]["grade_s"] == 1.
    assert row["passed"]
    assert row["completion_qualified"] is not extra_file
    assert row["edits"]["code.py"] == "fixed\n"


def test_bundle_authentication_precedes_importing_its_code(tmp_path):
    (tmp_path / "manifest.json").write_text('{"files": {}}')
    (tmp_path / "unexpected.py").write_text("raise RuntimeError('must not execute')\n")
    with pytest.raises(ValueError, match="bundle changed"):
        wall.load_inputs(tmp_path)


def test_symlink_in_fixture_is_rejected(tmp_path):
    (tmp_path / "link").symlink_to(tmp_path.parent)
    with pytest.raises(ValueError, match="symlinks"):
        wall.file_hashes(tmp_path)
