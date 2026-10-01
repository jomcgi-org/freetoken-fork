#!/usr/bin/env python3
"""Fresh-process, cache-disjoint, noise-reporting A/B harness (issue #101). Run as root via sudo.

Each arm is a name + git revision (or an existing tree) + env vars + extra server flags. Arms run
one at a time, in balanced order (ABBA; ABCCBA for three arms, ...), each in a fresh server
process started exactly like production (scripts/serve-qwen-flash-4090.sh with the arguments
parsed from the 60-root-runtime.conf ExecStart, port 18090, empty per-arm KV disk cache). The
production unit is stopped for the whole run and always restarted afterwards. The report shows
per-arm mean and min-max spread next to every delta; a delta inside the spread is "neutral".

  sudo bench/ab.py --dry-run --arm base=main --arm cand=main --env cand:FREETOKEN_X=1
"""
import argparse, fcntl, importlib.util, json, os, pwd, re, shlex, shutil, signal, socket
import statistics, subprocess, sys, threading, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path("/var/lib/longhorn/nvme-02/freetoken")
FORK = Path("/disks/nvme-02/src/freetoken-fork")
BENCH_TREES = Path("/disks/nvme-02/src/freetoken-bench")
VENV = ROOT / "wt-plegather/.venv/bin"
CUDA_HOME = "/usr/local/cuda-13.0"
DROPIN = "/etc/systemd/system/freetoken-serve.service.d/60-root-runtime.conf"
UNIT = "freetoken-serve"
LOCK = ROOT / "results/host.lock"
USER = "jomcgi"
PORT = 18090
PROD_PORT = 8090
SERVE_SCRIPT = "scripts/serve-qwen-flash-4090.sh"
ADAPT_FLAG = "--moe-hot-adapt-interval-steps"
CLIENT = HERE / "ab-client.py"
GIT_RO = ["git", "-c", "safe.directory=*"]  # read-only git as root in jomcgi-owned trees
GPU_QUERY = ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"]


class Interrupted(BaseException):
    """Raised from SIGINT/SIGTERM so every finally block (production restore) runs."""


# ---------------------------------------------------------------- pure helpers (unit-tested)

def parse_execstart(text):
    """Return (script, model, profile, args) from the last non-empty ExecStart= in a unit file."""
    text = re.sub(r"\\\n", " ", text)
    last = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("ExecStart=") and line[len("ExecStart="):].strip():
            last = line[len("ExecStart="):].strip()
    if last is None:
        raise ValueError("no non-empty ExecStart= found")
    words = shlex.split(last.lstrip("-@+:!"))
    idx = next((i for i, w in enumerate(words) if w.endswith(SERVE_SCRIPT)), None)
    if idx is None or len(words) < idx + 3:
        raise ValueError(f"ExecStart does not run {SERVE_SCRIPT} MODEL PROFILE: {last}")
    return words[idx], words[idx + 1], words[idx + 2], words[idx + 3:]


def set_flag(args, flag, value):
    """Return args with `flag value` replaced (also the --flag=value form) or appended."""
    out, done, i = [], False, 0
    while i < len(args):
        a = args[i]
        if a == flag or a.startswith(flag + "="):
            if not done:
                out += [flag, str(value)]
                done = True
            i += 1 if "=" in a else 2
            continue
        out.append(a)
        i += 1
    return out if done else out + [flag, str(value)]


def has_flag(args, flag):
    return any(a == flag or a.startswith(flag + "=") for a in args)


def server_args(prod_args, arm_flags, cache_dir, adapt_interval, port=PORT):
    """Production args with the port and KV disk cache overridden, adapt interval applied, then arm flags."""
    args = set_flag(prod_args, "--port", port)
    args = set_flag(args, "--kv-disk-cache-dir", cache_dir)
    if adapt_interval != "keep" and not has_flag(arm_flags, ADAPT_FLAG):
        args = set_flag(args, ADAPT_FLAG, adapt_interval)
    return args + list(arm_flags)


def balanced_order(names, repeats):
    """ABBA for two arms, ABCCBA for three, ...; repeat k alternates forward-first and reverse-first."""
    order = []
    for k in range(repeats):
        fwd, rev = list(names), list(reversed(names))
        order += (fwd + rev) if k % 2 == 0 else (rev + fwd)
    return order


def spread(values):
    return max(values) - min(values) if values else 0.0


def stats(values):
    if not values:
        return None
    return {"n": len(values), "mean": statistics.fmean(values), "min": min(values), "max": max(values)}


def compare(base, other, lower_is_better):
    """Delta of means (other - base) with a verdict against the larger arm's min-max spread."""
    if not base or not other:
        return None
    delta = other["mean"] - base["mean"]
    pct = delta / base["mean"] * 100 if base["mean"] else None
    if base["n"] < 2 and other["n"] < 2:
        verdict = "no spread (n<2)"
    else:
        sp = max(base["max"] - base["min"], other["max"] - other["min"])
        if abs(delta) <= sp:
            verdict = "neutral (within spread)"
        else:
            verdict = "better" if (delta < 0) == lower_is_better else "worse"
    return {"delta": delta, "pct": pct, "verdict": verdict}


def run_metrics(rec):
    """Flat metric dict for one OK run: total_wall, wall./tok_s./nvme.<task>, nvme_gib, adapt_ticks,
    hot_pair_rate, and for mixed-thinking phase.<thinking|plain>.{tok_s,hot_pair} (round 1 is warm-up)."""
    tasks = {t["name"]: t for t in rec.get("client", [])}
    m = {"total_wall": sum(t["wall"] for t in tasks.values()), "nvme_gib": rec.get("nvme_gib"),
         "pgmajfault": rec.get("pgmajfault")}
    for name, t in tasks.items():
        m[f"wall.{name}"] = t["wall"]
        m[f"tok_s.{name}"] = t["tokens"] / t["wall"] if t["wall"] else 0.0
        m[f"nvme.{name}"] = t.get("nvme_gib")
    for phase, flag in (("thinking", True), ("plain", False)):
        rows = [t for t in tasks.values() if t.get("thinking") is flag and t.get("round", 1) > 1]
        if rows:
            m[f"phase.{phase}.tok_s"] = sum(t["tokens"] for t in rows) / sum(t["wall"] for t in rows)
            rates = [t["hot_pair_rate"] for t in rows if t.get("hot_pair_rate") is not None]
            if rates:
                m[f"phase.{phase}.hot_pair"] = statistics.fmean(rates)
    adapt = rec.get("adapt") or {}
    if adapt.get("ticks"):
        m["adapt_ticks"] = adapt["ticks"]["prefill"] + adapt["ticks"]["decode"]
    if adapt.get("hot_pair_rate"):
        m["hot_pair_rate"] = adapt["hot_pair_rate"]["mean"]
    return m


def attach_hot_pair_rates(rows, journal_text, adapt):
    """Per request: mean decayed_hot_pair_rate of the tick lines logged while it ran (None if no tick fired)."""
    events = [e for e in adapt.tick_events(journal_text) if e[0] is not None]
    for r in rows:
        if "t_start" not in r:
            continue
        # journal stamps have 1 s resolution
        hits = [rate for ts, _, rate in events if int(r["t_start"]) <= ts <= r["t_end"] + 1]
        r["hot_pair_rate"] = sum(hits) / len(hits) if hits else None


def summarize_runs(runs, arm_names):
    """{arm: {n_ok, n_failed, n_invalid, metrics: {metric: stats}}}; only OK runs feed the stats."""
    out = {}
    for name in arm_names:
        mine = [r for r in runs if r["arm"] == name]
        vals = {}
        for r in mine:
            if r["status"] == "OK":
                for k, v in run_metrics(r).items():
                    if v is not None:
                        vals.setdefault(k, []).append(v)
        out[name] = {"n_ok": sum(r["status"] == "OK" for r in mine),
                     "n_failed": sum(r["status"] == "FAILED" for r in mine),
                     "n_invalid": sum(r["status"] == "INVALID" for r in mine),
                     "n_contaminated": sum(r["status"] == "CONTAMINATED" for r in mine),
                     "metrics": {k: stats(v) for k, v in vals.items()}}
    return out


def lower_is_better(metric):
    return "tok_s" not in metric and "hot_pair" not in metric


def build_summary(runs, arm_names):
    per_arm = summarize_runs(runs, arm_names)
    base = arm_names[0]
    deltas = {}
    for other in arm_names[1:]:
        deltas[other] = {}
        for metric in sorted(per_arm[base]["metrics"]):
            c = compare(per_arm[base]["metrics"].get(metric), per_arm[other]["metrics"].get(metric),
                        lower_is_better(metric))
            if c:
                deltas[other][metric] = c
    warnings = []
    for other in arm_names[1:]:
        a, b = per_arm[base]["metrics"].get("nvme_gib"), per_arm[other]["metrics"].get("nvme_gib")
        if a and b and min(a["mean"], b["mean"]) > 0 and max(a["mean"], b["mean"]) / min(a["mean"], b["mean"]) > 1.5:
            warnings.append(f"NVMe read differs {a['mean']:.1f} vs {b['mean']:.1f} GiB between {base} and {other}: "
                            "cache state is not comparable, treat the deltas as suspect")
    for name in arm_names:
        p = per_arm[name]
        if p["n_failed"] or p["n_invalid"] or p["n_contaminated"]:
            warnings.append(f"arm {name}: {p['n_failed']} failed, {p['n_invalid']} invalid and {p['n_contaminated']} "
                            "contaminated (foreign GPU process) run(s) excluded from the statistics")
    return {"arms": per_arm, "baseline": base, "deltas": deltas, "warnings": warnings}


def _f(s, fmt="{:.1f}"):
    if not s:
        return "n/a"
    return f"{fmt.format(s['mean'])} ({fmt.format(s['min'])}-{fmt.format(s['max'])})"


def render_report(meta, runs, summary):
    arms = list(summary["arms"])
    L = [f"# A/B report {meta['stamp']}", "",
         f"cache-state={meta['cache_state']} plan-mode={meta['plan_mode']} adapt-interval={meta['adapt_interval']} "
         f"workload={meta.get('workload', 'default')} repeats={meta['repeats']} order={' '.join(meta['order_labels'])}", ""]
    for w in summary["warnings"]:
        L.append(f"> WARNING: {w}")
    L += ["", "Cells are mean (min-max) over OK runs. Wall in seconds, tok/s per task.", "",
          "## Per arm", "", "| arm | revision | ok/fail/invalid/contaminated | total wall s | NVMe GiB | adapt ticks | major faults |",
          "|---|---|---|---|---|---|---|"]
    for a in arms:
        s, m = summary["arms"][a], summary["arms"][a]["metrics"]
        L.append(f"| {a} | {meta['arms'][a]['sha'][:12]} | {s['n_ok']}/{s['n_failed']}/{s['n_invalid']}/{s['n_contaminated']} | "
                 f"{_f(m.get('total_wall'))} | {_f(m.get('nvme_gib'))} | {_f(m.get('adapt_ticks'), '{:.0f}')} | "
                 f"{_f(m.get('pgmajfault'), '{:.0f}')} |")
    tasks = []
    for r in runs:
        for t in r.get("client", []):
            if t["name"] not in tasks:
                tasks.append(t["name"])

    def task_table(title, key, fmt):
        L.extend(["", title, "", "| arm | " + " | ".join(tasks) + " |", "|---|" + "---|" * len(tasks)])
        for a in arms:
            L.append(f"| {a} | " + " | ".join(_f(summary["arms"][a]["metrics"].get(f"{key}.{t}"), fmt) for t in tasks) + " |")

    task_table("## tok/s per task", "tok_s", "{:.2f}")
    task_table("## Wall s per task (mean (min-max): per-task variance under identical conditions)", "wall", "{:.1f}")
    task_table("## NVMe GiB per task (cache-state check)", "nvme", "{:.2f}")
    if any(k.startswith("phase.") for a in arms for k in summary["arms"][a]["metrics"]):
        L += ["", "## Per phase (mixed-thinking; round 1 excluded as warm-up)", "",
              "| arm | thinking tok/s | plain tok/s | thinking hot pair % | plain hot pair % |", "|---|---|---|---|---|"]
        for a in arms:
            m = summary["arms"][a]["metrics"]
            L.append(f"| {a} | " + " | ".join(_f(m.get(f"phase.{p}.{k}"), "{:.2f}") for k in ("tok_s", "hot_pair")
                                               for p in ("thinking", "plain")) + " |")
        L += ["", "Reasoning vs content tokens per request are in summary.json (`runs[].client[]`)."]
    L += ["", "Mean decayed hot pair rate over all adapt ticks, %: " + ", ".join(
        f"{a}={_f(summary['arms'][a]['metrics'].get('hot_pair_rate'))}" for a in arms)]
    L += ["", f"## Delta vs {summary['baseline']}", "",
          "Delta is other minus baseline on the means; neutral when inside the larger arm's min-max spread.", "",
          "| arm | metric | delta | % | verdict |", "|---|---|---|---|---|"]
    for other, ds in summary["deltas"].items():
        for metric, d in ds.items():
            if metric.startswith(("total_wall", "tok_s.", "wall.", "phase.")) and "hot_pair" not in metric:
                pct = f"{d['pct']:+.1f}%" if d["pct"] is not None else "n/a"
                L.append(f"| {other} | {metric} | {d['delta']:+.2f} | {pct} | {d['verdict']} |")
    L += ["", "## Runs", "", "| # | arm | status | wall s | NVMe GiB | major faults | Cached start GiB | ticks p/d/i | cpu C min-max | note |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(runs, 1):
        t = (r.get("adapt") or {}).get("ticks") or {}
        temp = f"{r['temp_min']:.0f}-{r['temp_max']:.0f}" if r.get("temp_min") is not None else "n/a"
        cached = f"{r['cached_kb_start'] / 2**20:.1f}" if r.get("cached_kb_start") is not None else "n/a"
        nv = f"{r['nvme_gib']:.1f}" if r.get("nvme_gib") is not None else "n/a"
        L.append(f"| {i} | {r['arm']} | {r['status']} | {r.get('arm_wall', 0):.0f} | {nv} | {r.get('pgmajfault', 'n/a')} | "
                 f"{cached} | {t.get('prefill', '-')}/{t.get('decode', '-')}/{t.get('idle', '-')} | {temp} | {r.get('reason', '')} |")
    return "\n".join(L) + "\n"


def parse_arms(args):
    """Arms from --spec JSON and/or --arm NAME=REV, --env NAME:K=V, --flags NAME:'--x y'."""
    arms = {}
    if args.spec:
        for a in json.loads(Path(args.spec).read_text())["arms"]:
            arms[a["name"]] = {"name": a["name"], "rev": a.get("rev") or a.get("tree"), "env": dict(a.get("env", {})),
                               "flags": shlex.split(a["flags"]) if isinstance(a.get("flags"), str) else list(a.get("flags", [])),
                               "model": a.get("model")}
    for spec in args.arm or []:
        name, _, rev = spec.partition("=")
        if not rev:
            raise SystemExit(f"--arm wants NAME=REV, got {spec!r}")
        arms[name] = {"name": name, "rev": rev, "env": {}, "flags": [], "model": None}
    for spec in args.env or []:
        name, _, kv = spec.partition(":")
        k, _, v = kv.partition("=")
        if name not in arms or not k:
            raise SystemExit(f"--env wants ARM:KEY=VALUE for a declared arm, got {spec!r}")
        arms[name]["env"][k] = v
    for spec in args.flags or []:
        name, _, fl = spec.partition(":")
        if name not in arms:
            raise SystemExit(f"--flags wants ARM:'--flag value' for a declared arm, got {spec!r}")
        arms[name]["flags"] += shlex.split(fl)
    for spec in getattr(args, "model", None) or []:
        name, _, path = spec.partition(":")
        if name not in arms or not path:
            raise SystemExit(f"--model wants ARM:/path/to/model.ftw for a declared arm, got {spec!r}")
        arms[name]["model"] = path
    if len(arms) < 2:
        raise SystemExit("need at least two arms (--arm NAME=REV twice, or --spec)")
    for a in arms.values():
        a["is_tree"] = a["rev"].startswith("/") and Path(a["rev"]).is_dir()
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", a["name"]):
            raise SystemExit(f"arm name {a['name']!r} must be [A-Za-z0-9_.-]")
    return list(arms.values())


def cpu_temp_from_sensors(doc):
    """Highest CPU package/core temperature in a `sensors -j` document, or None."""
    best = None
    for chip, body in doc.items():
        if not chip.startswith(("coretemp", "k10temp", "cpu_thermal", "zenpower")) or not isinstance(body, dict):
            continue
        for feat in body.values():
            if isinstance(feat, dict):
                for k, v in feat.items():
                    if k.endswith("_input") and isinstance(v, (int, float)):
                        best = v if best is None else max(best, v)
    return best


# ---------------------------------------------------------------- host access (mocked in tests)

def read_cpu_temp():
    try:
        out = subprocess.run(["sensors", "-j"], capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            t = cpu_temp_from_sensors(json.loads(out.stdout))
            if t is not None:
                return t
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    temps = []
    for z in Path("/sys/class/thermal").glob("thermal_zone*"):
        try:
            if "pkg" in (z / "type").read_text() or "cpu" in (z / "type").read_text():
                temps.append(int((z / "temp").read_text()) / 1000)
        except (OSError, ValueError):
            pass
    return max(temps) if temps else None


def diskstats_sectors(path):
    """Sectors read from the block device holding `path` (falls back to all nvme* disks)."""
    dev = os.stat(path).st_dev
    major, minor = os.major(dev), os.minor(dev)
    for line in Path("/proc/diskstats").read_text().splitlines():
        f = line.split()
        if int(f[0]) == major and int(f[1]) == minor:
            return int(f[5])
    return sum(int((d / "stat").read_text().split()[2]) for d in Path("/sys/block").glob("nvme*"))


def vmstat(key):
    for line in Path("/proc/vmstat").read_text().splitlines():
        if line.startswith(key + " "):
            return int(line.split()[1])


def meminfo_kb(key):
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(key + ":"):
            return int(line.split()[1])


def http_health(port, timeout=3):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as r:
            return json.load(r)
    except Exception:
        return None


def port_listening(port):
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def parse_gpu_apps(text):
    """[(pid, name, MiB)] from `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader`."""
    apps = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3 and parts[0].isdigit():
            mib = re.match(r"\d+", parts[2])
            apps.append((int(parts[0]), parts[1], int(mib.group()) if mib else 0))
    return apps


def in_unit(pid, unit):
    """True when the process belongs to the arm's systemd unit; a vanished process counts as ours (no verdict)."""
    try:
        return f"{unit}.service" in Path(f"/proc/{pid}/cgroup").read_text()
    except OSError:
        return True


def foreign_gpu_apps(unit=None):
    """GPU compute processes that are not part of `unit` (all of them when unit is None)."""
    out = subprocess.run(GPU_QUERY, capture_output=True, text=True, timeout=30)
    if out.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {out.stderr.strip()}")
    return [a for a in parse_gpu_apps(out.stdout) if unit is None or not in_unit(a[0], unit)]


# ---------------------------------------------------------------- runner

class Runner:
    def __init__(self, args, arms, prod):
        self.args, self.arms, self.dry = args, {a["name"]: a for a in arms}, args.dry_run
        self.script, self.model, self.profile, self.prod_args = prod
        self.stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.out = Path(args.results_dir) if args.results_dir else ROOT / f"results/ab-{self.stamp}"
        self.plan = Path(self.model) / "freetoken_hot_plan.json"
        self.plan_bak = self.out / "hot_plan.backup.json"
        self.foreign_plans = {}
        self.runs = []
        self.as_user = ["sudo", "-n", "-u", USER] if (os.geteuid() == 0 or self.dry) else []
        spec = importlib.util.spec_from_file_location("adapt_ticks_check", HERE / "adapt-ticks-check.py")
        self.adapt = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.adapt)

    # every side effect goes through sh/do so --dry-run can print instead of act
    def sh(self, *cmd, readonly=False, out=None, **kw):
        cmd = [str(c) for c in cmd]
        if self.dry and not readonly:
            print("+ " + shlex.join(cmd) + (f" > {out}" if out else ""), flush=True)
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if out:
            with open(out, "w") as f:
                return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, **kw)
        return subprocess.run(cmd, capture_output=kw.pop("capture", True), text=True, **kw)

    def do(self, desc, fn, *a):
        if self.dry:
            print(f"# {desc}", flush=True)
            return None
        return fn(*a)

    def say(self, msg):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

    # ---- trees
    def prepare_tree(self, arm):
        name = arm["name"]
        if arm["is_tree"]:
            tree = Path(arm["rev"])
            r = self.sh(*GIT_RO, "-C", tree, "rev-parse", "HEAD", readonly=True)
            sha = r.stdout.strip() or "unknown"
            marker = None
        else:
            tree = BENCH_TREES / name
            r = self.sh(*GIT_RO, "-C", FORK, "rev-parse", "--verify", f"{arm['rev']}^{{commit}}", readonly=True)
            if r.returncode != 0:
                self.sh(*self.as_user, "git", "-C", FORK, "fetch", "-q", "origin", arm["rev"])
                r = self.sh(*GIT_RO, "-C", FORK, "rev-parse", "--verify", f"{arm['rev']}^{{commit}}", readonly=True)
            sha = r.stdout.strip() if r.returncode == 0 else f"<{arm['rev']} unresolved: dry-run only>"
            if r.returncode != 0 and not self.dry:
                raise SystemExit(f"cannot resolve revision {arm['rev']!r} in {FORK}")
            marker = tree / ".ab-built"
            registered = tree.exists() and str(tree) in self.sh(*GIT_RO, "-C", FORK, "worktree", "list", "--porcelain", readonly=True).stdout
            if tree.exists() and not registered:
                raise SystemExit(f"{tree} exists but is not a worktree of {FORK}; remove it or rename the arm")
            if registered:
                head = self.sh(*GIT_RO, "-C", tree, "rev-parse", "HEAD", readonly=True).stdout.strip()
                if head != sha:
                    self.sh(*self.as_user, "git", "-C", tree, "checkout", "-q", "-f", "--detach", sha)
            else:
                self.sh(*self.as_user, "git", "-C", FORK, "worktree", "add", "-q", "--detach", tree, sha)
        has_so = any((tree / "python").glob("**/*.so")) if tree.exists() else False
        built = marker is None or (marker.exists() and marker.read_text().strip() == sha)
        arm.update(tree=str(tree), sha=sha)
        if has_so and built:
            self.say(f"{name}: tree {tree} at {sha[:12]}, extensions already built")
            return
        env = {"CUDA_HOME": CUDA_HOME, "PATH": f"{VENV}:{CUDA_HOME}/bin:/usr/bin:/bin", "HOME": f"/home/{USER}"}
        self.say(f"{name}: building native extensions in {tree}")
        r = self.sh(*self.as_user, "env", *[f"{k}={v}" for k, v in env.items()], str(VENV / "python"), "setup.py",
                    "build_ext", "--inplace", out=self.out / f"build-{name}.log", cwd=tree if tree.exists() else None)
        if r.returncode != 0:
            raise SystemExit(f"build failed for {name}; see {self.out}/build-{name}.log")
        if marker and not self.dry:
            marker.write_text(sha + "\n")

    # ---- production + plan
    def stop_production(self):
        self.sh("systemctl", "stop", UNIT)
        if not self.dry:
            for _ in range(45):
                if not port_listening(PROD_PORT):
                    break
                time.sleep(2)

    def restore_production(self):
        self.restore_plan()
        self.sh("systemctl", "start", UNIT)
        if self.dry:
            print(f"# poll http://127.0.0.1:{PROD_PORT}/health until ok (<= 300 s) and print it", flush=True)
            return
        health = None
        for _ in range(75):
            health = http_health(PROD_PORT)
            if health and health.get("status") == "ok":
                break
            time.sleep(4)
        self.say(f"production {UNIT}: {self.sh('systemctl', 'is-active', UNIT, readonly=True).stdout.strip()} "
                 f"/health={json.dumps(health)}")

    def copy_plan(self, src, dst):
        shutil.copy2(src, dst)
        st = os.stat(src)
        os.chown(dst, st.st_uid, st.st_gid)

    def restore_plan(self):
        for plan, orig in self.foreign_plans.items():
            if self.dry:
                print(f"+ cp -p {orig} {plan}   # other model's own plan", flush=True)
            elif orig.exists():
                self.copy_plan(orig, plan)
        if self.dry:
            print(f"+ cp -p {self.plan_bak} {self.plan}   # if a backup exists", flush=True)
        elif self.plan_bak.exists():
            self.copy_plan(self.plan_bak, self.plan)
            self.say("hot plan restored from pre-run backup")
        elif self.plan.exists():  # there was no plan before the run, so drop the arm-written one
            self.plan.unlink()

    def arm_model(self, arm):
        return arm.get("model") or self.model

    def reset_plan(self, arm=None):
        """Every arm starts from the same plan: the production one (backup) or none (cold).

        An arm on a different model (--model) gets the production plan copied into that
        model's directory; its own plan is saved once and put back by restore_plan."""
        plan = Path(self.arm_model(arm)) / "freetoken_hot_plan.json" if arm else self.plan
        if plan != self.plan:
            orig = self.out / f"hot_plan.{arm['name']}.orig.json"
            if not self.dry and plan.exists() and not orig.exists():
                shutil.copy2(plan, orig)
            self.foreign_plans[plan] = orig
        if self.args.plan_mode == "cold":
            self.sh("rm", "-f", plan)
        elif self.dry:
            print(f"+ cp -p {self.plan_bak} {plan}   # same starting plan for every arm", flush=True)
        elif self.plan_bak.exists():
            self.copy_plan(self.plan_bak, plan)

    # ---- one arm
    def build_unit_cmd(self, arm, unit, cache_dir):
        tree = arm["tree"]
        sargs = server_args(self.prod_args, arm["flags"], cache_dir, self.args.adapt_interval)
        env = {"CUDA_HOME": CUDA_HOME, "PATH": f"{VENV}:{CUDA_HOME}/bin:/usr/bin:/bin", "PYTHONPATH": f"{tree}/python",
               "TMPDIR": str(ROOT / "tmp"), "FREETOKEN_BIN": str(VENV / "ft"), "HOME": f"/home/{USER}",
               "FREETOKEN_PREFIX_CACHE_DIR": cache_dir, **arm["env"]}
        cmd = ["systemd-run", f"--unit={unit}", f"--uid={USER}", "--collect", f"--working-directory={tree}",
               "-p", "KillMode=control-group", "-p", "TimeoutStopSec=45", "-p", f"RuntimeMaxSec={self.args.arm_cap}",
               *[f"--setenv={k}={v}" for k, v in env.items()],
               "/bin/bash", f"{tree}/{SERVE_SCRIPT}", self.arm_model(arm), self.profile, *sargs]
        return cmd, env

    def run_arm(self, arm, slot):
        name, tag = arm["name"], f"{arm['name']}-{slot:02d}"
        unit = f"ft-ab-{self.stamp.lower()}-{tag}"
        cache_dir = str(ROOT / f"tmp/ab-prefix-cache-{self.stamp}-{tag}")
        rec = {"arm": name, "slot": slot, "rev": arm["rev"], "sha": arm["sha"], "tree": arm["tree"], "unit": unit,
               "env": arm["env"], "cache_dir": cache_dir, "cache_state": self.args.cache_state, "status": "OK"}
        cmd, env = self.build_unit_cmd(arm, unit, cache_dir)
        rec["command"], rec["unit_env"] = cmd, env
        self.say(f"--- run {slot + 1}/{len(self.order)} arm {name} ({unit})")
        if self.dry:
            print(f"# GPU guard: wait up to {self.args.gpu_wait_minutes} min until `{' '.join(GPU_QUERY)}` lists "
                  "no process, else FAILED", flush=True)
        else:
            busy = self.wait_gpu_free()
            if busy:
                rec.update(status="FAILED", gpu_foreign=busy,
                           reason=f"GPU not free after {self.args.gpu_wait_minutes} min: " +
                                  ", ".join(f"pid {p} {n} {m} MiB" for p, n, m in busy))
                self.say(f"{name}: {rec['reason']}")
                (self.out / f"{tag}.run.json").write_text(json.dumps(rec, indent=1))
                self.runs.append(rec)
                return
        self.reset_plan(arm)
        self.do(f"mkdir {cache_dir} (empty, owned by {USER})", self._mkcache, cache_dir)
        if self.args.cache_state == "cold":
            self.sh("sync")
            self.do("echo 3 > /proc/sys/vm/drop_caches", lambda: Path("/proc/sys/vm/drop_caches").write_text("3"))
        elif self.dry:
            print("# warm: client runs the cache-disjoint warmup prompts before the measured ones", flush=True)
        temps, stop, foreign = [], threading.Event(), {}
        pre = None if self.dry else dict(sectors=diskstats_sectors(self.model), maj=vmstat("pgmajfault"),
                                          cached=meminfo_kb("Cached"))
        rec["cached_kb_start"] = pre and pre["cached"]
        t0 = time.time()
        sampler = None
        if not self.dry:
            def sample():
                while not stop.is_set():
                    t = read_cpu_temp()
                    if t is not None:
                        temps.append(t)
                    try:
                        for pid, pname, mib in foreign_gpu_apps(unit):
                            foreign[pid] = {"pid": pid, "name": pname, "mib": max(mib, foreign.get(pid, {}).get("mib", 0))}
                    except (RuntimeError, OSError, subprocess.SubprocessError):
                        pass
                    stop.wait(10)
            sampler = threading.Thread(target=sample, daemon=True)
            sampler.start()
        client_log, client_json = self.out / f"{tag}.client.log", self.out / f"{tag}.client.json"
        client_cmd = ["sudo", "-n", "-u", USER, str(VENV / "python"), str(CLIENT), str(client_json),
                      "--workload", self.args.workload, "--port", str(PORT), "--request-cap", str(self.args.request_cap)]
        if self.args.max_tokens:
            client_cmd += ["--max-tokens", str(self.args.max_tokens)]
        if self.args.cache_state == "warm":
            client_cmd.append("--warmup")
        try:
            self.sh(*cmd, check=True)
            if self.dry:
                print(f"# wait for http://127.0.0.1:{PORT}/health ok (<= {self.args.ready_timeout}s, else FAILED)", flush=True)
                self.sh(*client_cmd, out=client_log)
                print(f"# watchdogs: request cap {self.args.request_cap}s, no progress (client output and journal) "
                      f"{self.args.no_progress_timeout}s -> stop {unit}, mark FAILED, continue", flush=True)
            else:
                self.watch(rec, unit, client_cmd, client_log, client_json)
        except subprocess.CalledProcessError as e:
            rec.update(status="FAILED", reason=f"systemd-run failed: {e}")
        finally:
            stop.set()
            jr = self.out / f"{tag}.journal.log"
            self.sh("journalctl", "-u", unit, "--since", f"@{int(t0)}", "-o", "cat", "--no-pager", out=jr)
            self.sh("systemctl", "stop", unit)
            self.sh("systemctl", "reset-failed", unit)
            if not self.dry:
                for _ in range(30):
                    if not port_listening(PORT):
                        break
                    time.sleep(2)
        if self.dry:
            print(f"# then record NVMe bytes, majflt, Cached, ticks, temperature; + rm -rf {cache_dir}", flush=True)
            return
        rec["arm_wall"] = time.time() - t0
        rec["nvme_bytes"] = (diskstats_sectors(self.model) - pre["sectors"]) * 512
        rec["nvme_gib"] = rec["nvme_bytes"] / 2**30
        rec["pgmajfault"] = vmstat("pgmajfault") - pre["maj"]
        rec["cached_kb_end"] = meminfo_kb("Cached")
        rec["temp_min"], rec["temp_max"] = (min(temps), max(temps)) if temps else (None, None)
        rec["journal"] = str(jr)
        jtext = Path(jr).read_text(errors="ignore")
        rec["adapt"] = self.adapt.summarize(jtext)
        attach_hot_pair_rates(rec.get("client", []), jtext, self.adapt)
        rec["gpu_foreign"] = sorted(foreign.values(), key=lambda f: f["pid"])
        if foreign and rec["status"] in ("OK", "INVALID"):
            rec.update(status="CONTAMINATED", reason="foreign GPU process during the arm: " +
                       ", ".join(f"pid {f['pid']} {f['name']} {f['mib']} MiB" for f in rec["gpu_foreign"]))
        if rec["status"] == "OK" and self.args.hotset_knob and self.adapt.verdict(rec["adapt"], self.args.min_adapt_ticks) == "INVALID":
            rec.update(status="INVALID", reason=f"fewer than {self.args.min_adapt_ticks} non-idle adapt ticks (issue #23)")
        shutil.rmtree(cache_dir, ignore_errors=True)
        (self.out / f"{tag}.run.json").write_text(json.dumps(rec, indent=1))
        self.runs.append(rec)
        self.say(f"{name}: {rec['status']} wall={rec['arm_wall']:.0f}s nvme={rec['nvme_gib']:.1f}GiB "
                 f"ticks={rec['adapt']['ticks']} {rec.get('reason', '')}")

    def ensure_results_dir(self):
        pw = pwd.getpwnam(USER)
        self.out.mkdir(parents=True, exist_ok=True)
        os.chown(self.out, pw.pw_uid, pw.pw_gid)

    def wait_gpu_free(self):
        """Wait up to --gpu-wait-minutes for the GPU to have no compute processes; returns the blockers or []."""
        deadline = time.time() + self.args.gpu_wait_minutes * 60
        while True:
            try:
                apps = foreign_gpu_apps()
            except (RuntimeError, OSError, subprocess.SubprocessError) as e:
                return [(0, f"GPU query failed: {e}", 0)]
            if not apps or time.time() >= deadline:
                return apps
            self.say(f"GPU busy ({apps}); waiting")
            time.sleep(15)

    def _mkcache(self, d):
        pw = pwd.getpwnam(USER)
        os.makedirs(d)
        os.chown(d, pw.pw_uid, pw.pw_gid)

    def watch(self, rec, unit, client_cmd, client_log, client_json):
        a = self.args
        start = time.time()
        while http_health(PORT) is None or http_health(PORT).get("status") != "ok":
            if self.sh("systemctl", "is-active", "--quiet", unit, readonly=True).returncode != 0:
                return rec.update(status="FAILED", reason="server exited before ready")
            if time.time() - start > a.ready_timeout:
                return rec.update(status="FAILED", reason=f"not ready after {a.ready_timeout}s")
            time.sleep(4)
        self.say(f"ready after {time.time() - start:.0f}s")
        log = open(client_log, "w")
        proc = subprocess.Popen(client_cmd, stdout=log, stderr=subprocess.STDOUT)
        last_progress, req_start, size, cursor = time.time(), time.time(), 0, None
        try:
            while proc.poll() is None:
                time.sleep(5)
                now = time.time()
                text = Path(client_log).read_text(errors="ignore")
                if len(text) != size:
                    size, last_progress = len(text), now
                    last = [json.loads(l) for l in text.splitlines() if l.startswith("{")][-1:]
                    if last and last[0].get("event") == "start":
                        req_start = now
                    elif last:
                        req_start = None
                cur = self.sh("journalctl", "-u", unit, "-n", "1", "--show-cursor", "-o", "cat", "--no-pager", readonly=True).stdout
                if cur != cursor:
                    cursor, last_progress = cur, now
                if req_start and now - req_start > a.request_cap:
                    return self._trip(rec, proc, f"request wall cap {a.request_cap}s exceeded")
                if now - last_progress > a.no_progress_timeout:
                    return self._trip(rec, proc, f"no progress for {a.no_progress_timeout}s")
        finally:
            log.close()
        if proc.returncode != 0 or not client_json.exists():
            return rec.update(status="FAILED", reason=f"client exited {proc.returncode}")
        rec["client"] = json.loads(client_json.read_text())

    def _trip(self, rec, proc, why):
        proc.kill()
        proc.wait()
        rec.update(status="FAILED", reason=why)
        self.say(f"WATCHDOG: {why}; retiring arm")

    # ---- whole run
    def go(self):
        names = list(self.arms)
        self.order = balanced_order(names, self.args.repeats)
        meta = {"stamp": self.stamp, "cache_state": self.args.cache_state, "plan_mode": self.args.plan_mode,
                "adapt_interval": self.args.adapt_interval, "workload": self.args.workload, "repeats": self.args.repeats, "order_labels": self.order}
        print(f"results dir: {self.out}\norder: {' '.join(self.order)}", flush=True)
        self.do(f"mkdir -p {self.out} (owned by {USER}: the client runs as {USER} and writes its JSON there)",
                self.ensure_results_dir)
        lock = None
        if self.dry:
            print(f"# flock -xn {LOCK} for the whole run", flush=True)
        else:
            lock = open(LOCK, "a")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SystemExit(f"another run holds {LOCK}")
        try:
            for a in self.arms.values():
                self.prepare_tree(a)
            if not self.dry:
                if self.plan.exists():
                    self.copy_plan(self.plan, self.plan_bak)  # keeps the plan's owner, so restores do too
            else:
                print(f"+ cp -p {self.plan} {self.plan_bak}", flush=True)
            try:
                self.stop_production()
                for slot, name in enumerate(self.order):
                    self.run_arm(self.arms[name], slot)
            finally:
                if not self.dry:  # a second ^C must not abort the production restart
                    signal.signal(signal.SIGINT, signal.SIG_IGN)
                    signal.signal(signal.SIGTERM, signal.SIG_IGN)
                self.restore_production()
        finally:
            if lock:
                lock.close()
        meta["arms"] = {n: {"rev": a["rev"], "sha": a["sha"], "tree": a["tree"], "env": a["env"], "flags": a["flags"]}
                        for n, a in self.arms.items()}
        if self.dry:
            print(f"# would write {self.out}/report.md and summary.json", flush=True)
            return 0
        summary = build_summary(self.runs, names)
        (self.out / "summary.json").write_text(json.dumps({"meta": meta, "summary": summary, "runs": self.runs}, indent=1))
        report = render_report(meta, self.runs, summary)
        (self.out / "report.md").write_text(report)
        print(report)
        return 0 if all(r["status"] == "OK" for r in self.runs) else 1


def make_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec", help='JSON: {"arms": [{"name", "rev"|"tree", "env": {}, "flags": ["--x", "1"]}]}')
    p.add_argument("--arm", action="append", metavar="NAME=REV", help="git revision of freetoken-fork, or an absolute tree path")
    p.add_argument("--env", action="append", metavar="ARM:KEY=VALUE")
    p.add_argument("--flags", action="append", metavar="ARM:'--flag value'")
    p.add_argument("--model", action="append", metavar="ARM:PATH",
                   help="serve this arm from another model directory (e.g. a checkpoint with MTP tensors)")
    p.add_argument("--repeats", type=int, default=1, help="balanced passes (ABBA per pass for two arms); default 1")
    p.add_argument("--cache-state", choices=["cold", "warm"], default="cold")
    p.add_argument("--plan-mode", choices=["backup", "cold"], default="backup",
                   help="backup: every arm starts from the production hot plan; cold: from none")
    p.add_argument("--adapt-interval", default="150", help="--moe-hot-adapt-interval-steps for every arm, or 'keep'")
    p.add_argument("--hotset-knob", action="store_true", help="arm is INVALID unless enough adapt ticks fired")
    p.add_argument("--min-adapt-ticks", type=int, default=3)
    p.add_argument("--dropin", default=DROPIN)
    p.add_argument("--results-dir")
    p.add_argument("--workload", choices=["default", "mixed-thinking"], default="default",
                   help="default: essays + doc, 1000 tokens; mixed-thinking: 6 rounds of thinking/plain, 600 tokens")
    p.add_argument("--gpu-wait-minutes", type=int, default=10,
                   help="before each arm, wait this long for other GPU processes to exit, then fail the arm")
    p.add_argument("--max-tokens", type=int, default=None, help="override the workload's output length")
    p.add_argument("--ready-timeout", type=int, default=900)
    p.add_argument("--request-cap", type=int, default=900, help="per-request wall cap, seconds")
    p.add_argument("--no-progress-timeout", type=int, default=300)
    p.add_argument("--arm-cap", type=int, default=3600, help="systemd RuntimeMaxSec for each server")
    p.add_argument("--dry-run", action="store_true", help="print every command instead of running it")
    return p


def main(argv=None):
    args = make_parser().parse_args(argv)
    arms = parse_arms(args)
    prod = parse_execstart(Path(args.dropin).read_text())
    if not args.dry_run and os.geteuid() != 0:
        raise SystemExit("run as root (sudo)")

    def bail(signum, frame):
        raise Interrupted(signal.Signals(signum).name)
    signal.signal(signal.SIGINT, bail)
    signal.signal(signal.SIGTERM, bail)
    try:
        return Runner(args, arms, prod).go()
    except Interrupted as e:
        print(f"interrupted by {e}; production restored", flush=True)
        return 130


if __name__ == "__main__":
    sys.exit(main())
