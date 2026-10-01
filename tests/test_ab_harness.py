"""Host-only checks for the A/B harness bench/ab.py (issue #101): no GPU, no systemd, no root."""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests" / "bench_fixtures" / "60-root-runtime.conf"


@pytest.fixture(autouse=True)
def hermetic_trees(monkeypatch, tmp_path):
    """Worktree dirs of earlier real runs (freetoken-bench/a, b) must not change what a dry run prints."""
    monkeypatch.setattr(ab, "BENCH_TREES", tmp_path / "freetoken-bench")


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "bench" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ab = load("ab", "ab.py")
client = load("ab_client", "ab-client.py")


def test_execstart_dropin_is_parsed_not_hardcoded():
    script, model, profile, args = ab.parse_execstart(FIXTURE.read_text())
    assert script.endswith("scripts/serve-qwen-flash-4090.sh")
    assert model.endswith("flash-e2m1.ftw") and profile.endswith("layer-profile-v3.json")
    assert args == ["--kv-reserve-tokens", "100352", "--kv-disk-cache-gib", "500", "--max-extend-length", "8192"]


def test_execstart_last_nonempty_wins_and_continuations():
    text = (
        "[Service]\nExecStart=/bin/true\nExecStart=\n"
        "ExecStart=/bin/bash /x/scripts/serve-qwen-flash-4090.sh /m /p \\\n  --a 1 \\\n  --b 'two words'\n"
    )
    assert ab.parse_execstart(text) == ("/x/scripts/serve-qwen-flash-4090.sh", "/m", "/p", ["--a", "1", "--b", "two words"])
    with pytest.raises(ValueError):
        ab.parse_execstart("[Service]\nExecStart=\n")
    with pytest.raises(ValueError):
        ab.parse_execstart("ExecStart=/bin/true\n")


def test_set_flag_replaces_or_appends():
    assert ab.set_flag(["--port", "8090", "--x"], "--port", 18090) == ["--port", "18090", "--x"]
    assert ab.set_flag(["--port=8090"], "--port", 1) == ["--port", "1"]
    assert ab.set_flag(["--x"], "--port", 1) == ["--x", "--port", "1"]


def test_server_args_override_port_cache_dir_and_adapt_interval():
    prod = ["--kv-disk-cache-dir", "/prod/cache", "--port", "8090", "--kv-reserve-tokens", "1"]
    args = ab.server_args(prod, ["--extra", "1"], "/tmp/arm", "150")
    assert args[args.index("--port") + 1] == "18090" and args.count("--port") == 1
    assert args[args.index("--kv-disk-cache-dir") + 1] == "/tmp/arm" and args.count("--kv-disk-cache-dir") == 1
    assert args[args.index(ab.ADAPT_FLAG) + 1] == "150"
    assert args[-2:] == ["--extra", "1"]
    assert ab.ADAPT_FLAG not in ab.server_args(prod, [], "/tmp/arm", "keep")
    own = ab.server_args(prod, [ab.ADAPT_FLAG, "7"], "/tmp/arm", "150")
    assert own.count(ab.ADAPT_FLAG) == 1 and own[own.index(ab.ADAPT_FLAG) + 1] == "7"


def test_balanced_order():
    assert ab.balanced_order(["A", "B"], 1) == list("ABBA")
    assert ab.balanced_order(["A", "B", "C"], 1) == list("ABCCBA")
    assert ab.balanced_order(["A", "B"], 2) == list("ABBABAAB")
    order = ab.balanced_order(["A", "B", "C"], 3)
    assert all(order.count(n) == 6 for n in "ABC")


def test_compare_verdicts():
    base, near, far = ab.stats([10, 12]), ab.stats([11, 13]), ab.stats([20, 21])
    assert ab.compare(base, near, True)["verdict"] == "neutral (within spread)"  # delta 1 <= spread 2
    assert ab.compare(base, far, True)["verdict"] == "worse"
    assert ab.compare(base, far, False)["verdict"] == "better"
    assert ab.compare(far, base, True)["verdict"] == "better"
    assert ab.compare(ab.stats([10]), ab.stats([20]), True)["verdict"] == "no spread (n<2)"
    wide = ab.stats([10, 30])  # the larger arm's spread decides
    assert ab.compare(wide, ab.stats([21, 22]), True)["verdict"] == "neutral (within spread)"
    assert ab.compare(None, near, True) is None


def fake_run(arm, wall, tok_s, gib, ticks, status="OK"):
    tasks = [{"name": t, "wall": wall / 4, "tokens": int(tok_s * wall / 4), "nvme_gib": gib / 4} for t in ["essay1", "essay2", "doc", "essay1b"]]
    return {"arm": arm, "status": status, "client": tasks, "nvme_gib": gib, "pgmajfault": 5,
            "adapt": {"ticks": {"prefill": ticks, "decode": 0, "idle": 0}}}


def test_summary_flags_cache_state_and_excludes_failed():
    runs = [fake_run("a", 100, 5, 16, 4), fake_run("a", 104, 5, 17, 4),
            fake_run("b", 101, 5, 31, 4), fake_run("b", 103, 5, 30, 4), fake_run("b", 0, 0, 0, 0, "FAILED")]
    summary = ab.build_summary(runs, ["a", "b"])
    assert summary["arms"]["b"]["n_ok"] == 2 and summary["arms"]["b"]["n_failed"] == 1
    assert summary["deltas"]["b"]["total_wall"]["verdict"] == "neutral (within spread)"
    assert any("NVMe read differs" in w for w in summary["warnings"])
    meta = {"stamp": "t", "cache_state": "cold", "plan_mode": "backup", "adapt_interval": "150", "repeats": 1,
            "order_labels": ["a", "b"], "arms": {"a": {"sha": "1" * 40}, "b": {"sha": "2" * 40}}}
    report = ab.render_report(meta, runs, summary)
    assert "neutral (within spread)" in report and "WARNING: NVMe read differs" in report
    json.dumps(summary)


def test_cpu_temp_from_sensors():
    doc = {"coretemp-isa-0000": {"Package id 0": {"temp1_input": 61.0}, "Core 0": {"temp2_input": 58.0}},
           "nvme-pci-0100": {"Composite": {"temp1_input": 99.0}}}
    assert ab.cpu_temp_from_sensors(doc) == 61.0
    assert ab.cpu_temp_from_sensors({"nvme-pci-0100": {}}) is None


def test_client_warmup_is_prefix_disjoint():
    client.check_disjoint(client.WARMUP, client.MEASURED)
    with pytest.raises(SystemExit):
        client.check_disjoint([("w", client.MEASURED[0][1])], client.MEASURED)


def test_dry_run_end_to_end_with_mocked_subprocess(monkeypatch, capsys, tmp_path):
    calls = []

    def fake(cmd, *a, **k):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "d" * 40 + "\n", "")

    monkeypatch.setattr(ab.subprocess, "run", fake)
    rc = ab.main(["--dry-run", "--dropin", str(FIXTURE), "--results-dir", str(tmp_path / "res"),
                  "--arm", "base=main", "--arm", "cand=main", "--env", "cand:FREETOKEN_CPU_MOE_DATAFLOW=1",
                  "--cache-state", "warm"])
    out = capsys.readouterr().out
    assert rc == 0
    assert not (tmp_path / "res").exists()  # dry run writes nothing
    assert all(c[0] in ("git", "systemctl") for c in calls)  # only read-only probes really ran
    launches = [l for l in out.splitlines() if l.startswith("+ systemd-run")]
    assert [l.split("--unit=ft-ab-")[1].split()[0].rsplit("-", 2)[1] for l in launches] == ["base", "cand", "cand", "base"]
    assert all("--uid=jomcgi --collect" in l and "--port 18090" in l for l in launches)
    assert all("serve-qwen-flash-4090.sh" in l and "--max-extend-length 8192" in l for l in launches)
    assert "--setenv=FREETOKEN_CPU_MOE_DATAFLOW=1" in launches[1] and "FREETOKEN_CPU_MOE_DATAFLOW" not in launches[0]
    assert all("freetoken-bench/" in l and "ab-prefix-cache-" in l for l in launches)
    assert out.count("echo 3 > /proc/sys/vm/drop_caches") == 0  # warm run does not drop caches
    assert out.count("ab-client.py") == 4 and out.count("--warmup") == 4
    lines = out.splitlines()
    stop = lines.index("+ systemctl stop freetoken-serve")
    start = lines.index("+ systemctl start freetoken-serve")
    assert stop < lines.index(launches[0]) and lines.index(launches[-1]) < start


def test_production_restarted_when_an_arm_is_interrupted(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(ab.subprocess, "run", lambda cmd, *a, **k: subprocess.CompletedProcess(cmd, 0, "d" * 40, ""))

    def boom(self, arm, slot):
        raise ab.Interrupted("SIGTERM")

    monkeypatch.setattr(ab.Runner, "run_arm", boom)
    rc = ab.main(["--dry-run", "--dropin", str(FIXTURE), "--results-dir", str(tmp_path / "r"),
                  "--arm", "a=main", "--arm", "b=main", "--cache-state", "cold"])
    out = capsys.readouterr().out
    assert rc == 130 and "+ systemctl start freetoken-serve" in out


def test_results_dir_is_created_owned_by_the_client_user(monkeypatch, tmp_path):
    chowned = []
    monkeypatch.setattr(ab.pwd, "getpwnam", lambda n: type("P", (), {"pw_uid": 1000, "pw_gid": 1001})())
    monkeypatch.setattr(ab.os, "chown", lambda path, uid, gid: chowned.append((Path(path), uid, gid)))
    args = ab.make_parser().parse_args(["--results-dir", str(tmp_path / "a" / "res"), "--arm", "a=x", "--arm", "b=x"])
    runner = ab.Runner(args, ab.parse_arms(args), ab.parse_execstart(FIXTURE.read_text()))
    runner.ensure_results_dir()
    assert (tmp_path / "a" / "res").is_dir() and chowned == [(tmp_path / "a" / "res", 1000, 1001)]


def test_gpu_apps_parse_and_foreign_detection(monkeypatch):
    text = "1234, python, 22000 MiB\n77, /usr/bin/python3, 512 MiB\nbad line\n"
    assert ab.parse_gpu_apps(text) == [(1234, "python", 22000), (77, "/usr/bin/python3", 512)]
    monkeypatch.setattr(ab, "in_unit", lambda pid, unit: pid == 1234)
    monkeypatch.setattr(ab.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, text, ""))
    assert ab.foreign_gpu_apps("ft-ab-x") == [(77, "/usr/bin/python3", 512)]
    assert len(ab.foreign_gpu_apps()) == 2


def test_contaminated_runs_are_excluded_from_stats():
    runs = [fake_run("a", 100, 5, 16, 4), fake_run("a", 300, 2, 16, 4, "CONTAMINATED"), fake_run("b", 101, 5, 16, 4)]
    summary = ab.build_summary(runs, ["a", "b"])
    assert summary["arms"]["a"]["n_ok"] == 1 and summary["arms"]["a"]["n_contaminated"] == 1
    assert summary["arms"]["a"]["metrics"]["total_wall"]["mean"] == 100
    assert any("contaminated" in w for w in summary["warnings"])


def mixed_run(arm, think_tok_s, plain_tok_s):
    rows = []
    for rnd in range(1, 7):
        for thinking, rate in ((True, think_tok_s), (False, plain_tok_s)):
            rate = rate / 10 if rnd == 1 else rate  # warm-up round must not count
            rows.append({"name": f"r{rnd}-{'think' if thinking else 'plain'}", "thinking": thinking, "round": rnd,
                         "wall": 600 / rate, "tokens": 600, "nvme_gib": 0.5, "hot_pair_rate": 60.0 if thinking else 70.0})
    return {"arm": arm, "status": "OK", "client": rows, "nvme_gib": 6.0, "pgmajfault": 1,
            "adapt": {"ticks": {"prefill": 2, "decode": 9, "idle": 0}, "hot_pair_rate": {"mean": 65.0}}}


def test_mixed_thinking_phase_metrics_and_report():
    runs = [mixed_run("a", 20, 25), mixed_run("a", 21, 26), mixed_run("b", 20, 25), mixed_run("b", 22, 27)]
    m = ab.run_metrics(runs[0])
    assert m["phase.thinking.tok_s"] == pytest.approx(20) and m["phase.plain.tok_s"] == pytest.approx(25)
    assert m["phase.thinking.hot_pair"] == 60.0 and m["wall.r2-plain"] == pytest.approx(24)
    summary = ab.build_summary(runs, ["a", "b"])
    meta = {"stamp": "t", "cache_state": "cold", "plan_mode": "backup", "adapt_interval": "150", "repeats": 1,
            "workload": "mixed-thinking", "order_labels": ["a", "b"], "arms": {"a": {"sha": "1" * 40}, "b": {"sha": "2" * 40}}}
    report = ab.render_report(meta, runs, summary)
    assert "## Per phase" in report and "## Wall s per task" in report and "## NVMe GiB per task" in report
    assert "r3-think" in report


def test_hot_pair_rate_is_attached_per_request_by_timestamp():
    adapt = load("adapt_ticks_check2", "adapt-ticks-check.py")
    text = (ROOT / "tests" / "bench_fixtures" / "journal-ticks.log").read_text()
    events = adapt.tick_events(text)
    t = events[0][0]
    rows = [{"name": "x", "t_start": t - 2, "t_end": t + 8, "wall": 10}, {"name": "y", "t_start": t + 3600, "t_end": t + 3700}]
    ab.attach_hot_pair_rates(rows, text, adapt)
    assert rows[0]["hot_pair_rate"] == pytest.approx((66.55 + 54.88) / 2)  # ticks at +0 s and +7 s
    assert rows[1]["hot_pair_rate"] is None


def test_client_mixed_workload_shape_and_disjoint_warmup():
    assert len(client.MIXED) == 12 and [m[2] for m in client.MIXED] == [True, False] * 6
    assert len({m[1] for m in client.MIXED}) == 12
    client.check_disjoint(client.WARMUP, client.MIXED)


def test_dry_run_mixed_thinking_end_to_end(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(ab.subprocess, "run", lambda cmd, *a, **k: subprocess.CompletedProcess(cmd, 0, "d" * 40, ""))
    rc = ab.main(["--dry-run", "--dropin", str(FIXTURE), "--results-dir", str(tmp_path / "r"), "--workload", "mixed-thinking",
                  "--arm", "a=main", "--arm", "b=main"])
    out = capsys.readouterr().out
    assert rc == 0 and out.count("--workload mixed-thinking") == 4 and out.count("GPU guard") == 4
    assert "--max-tokens" not in out


def test_per_arm_model_serves_that_model_and_restores_its_plan(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(ab.subprocess, "run", lambda cmd, *a, **k: subprocess.CompletedProcess(cmd, 0, "d" * 40 + "\n", ""))
    rc = ab.main(["--dry-run", "--dropin", str(FIXTURE), "--results-dir", str(tmp_path / "res"),
                  "--arm", "base=main", "--arm", "mtp=main", "--model", "mtp:/models/other-mtp.ftw",
                  "--flags", "mtp:--speculative-mtp on"])
    out = capsys.readouterr().out
    assert rc == 0
    launches = [l for l in out.splitlines() if l.startswith("+ systemd-run")]
    base = [l for l in launches if "-base-" in l]
    mtp = [l for l in launches if "-mtp-" in l]
    assert all("/models/other-mtp.ftw" in l and "--speculative-mtp on" in l for l in mtp)
    assert not any("/models/other-mtp.ftw" in l for l in base)
    assert "/models/other-mtp.ftw/freetoken_hot_plan.json   # same starting plan" in out
    assert "/models/other-mtp.ftw/freetoken_hot_plan.json   # other model's own plan" in out


def test_model_flag_requires_a_declared_arm():
    with pytest.raises(SystemExit):
        ab.main(["--dry-run", "--dropin", str(FIXTURE), "--arm", "a=main", "--arm", "b=main", "--model", "c:/x"])


def test_long_doc_workload_is_one_growing_conversation(monkeypatch, tmp_path):
    seen = []

    def fake_ask(args, name, content, thinking=False, stream=False, messages=None, max_tokens=None):
        seen.append((name, [m["role"] for m in messages], max_tokens))
        return dict(name=name, wall=1.0, tokens=10, tok_s=10.0, _text=f"answer {name}")

    monkeypatch.setattr(client, "ask", fake_ask)
    out = tmp_path / "rows.json"
    client.main([str(out), "--workload", "long-doc"])
    assert [s[0] for s in seen] == ["doc", "t1", "t2", "t3"]
    assert seen[0][1] == ["user"] and seen[0][2] == 300
    assert seen[3][1] == ["user", "assistant", "user", "assistant", "user", "assistant", "user"]
    rows = json.loads(out.read_text())
    assert all("_text" not in r for r in rows)
    assert client.LONG_DOC_CHARS <= len(client.long_doc()) <= client.LONG_DOC_CHARS + 100
