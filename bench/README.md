# bench

## A/B harness (`ab.py`)

One stdlib-only runner for A/B screens (issue #101). It replaces the copy-pasted `run-screenN.py`
scripts. Run it as root; it holds `results/host.lock` for the whole run, stops
`freetoken-serve.service`, and always restarts it (also on SIGINT/SIGTERM) and prints `/health`.

Each arm runs in a fresh process, one at a time, in balanced order (`ABBA`, or `ABCCBA` for three
arms; `--repeats N` adds passes). The server command is production's: `scripts/serve-qwen-flash-4090.sh MODEL PROFILE <args>`
from the arm's tree, with the args parsed from the `ExecStart` of
`/etc/systemd/system/freetoken-serve.service.d/60-root-runtime.conf`, port 18090 and a fresh empty
`--kv-disk-cache-dir`. `--adapt-interval` (default 150, `keep` to leave it) makes the hot adapter tick
inside an arm (issue #23); add `--hotset-knob` to mark arms INVALID when too few ticks fired.
`--cache-state cold` drops the page cache before each arm; `warm` runs cache-disjoint warmup prompts
first. `--plan-mode backup|cold` starts every arm from the production hot plan, or from none.
The workload is `ab-client.py` (greedy, fixed seed, fixed length). Always try `--dry-run` first: it
prints every command and changes nothing.

`--workload default` (essays + doc) or `mixed-thinking` (6 rounds of alternating thinking and plain
requests; per-phase tok/s excludes round 1). Before each arm, and every ~10 s during it, the harness
checks `nvidia-smi` for GPU processes outside the arm's unit: it waits `--gpu-wait-minutes` before an
arm, and an arm that saw a foreign process is marked CONTAMINATED and excluded from the statistics.

Results land in `/var/lib/longhorn/nvme-02/freetoken/results/ab-<UTC>/`: `report.md`,
`summary.json`, and per run the command/env, client JSON, journal, NVMe bytes, major faults, adapt
ticks and CPU temperature. A delta inside the larger arm's min-max spread is reported as
"neutral (within spread)"; NVMe GiB and adapt ticks sit beside the deltas so cache-state differences show.

Env-var A/B on one revision (like the cpu-moe stall screen):

```bash
sudo bench/ab.py --arm off=main --arm on=main \
  --env off:FREETOKEN_CPU_MOE_DATAFLOW=0 --env on:FREETOKEN_CPU_MOE_DATAFLOW=1 --repeats 2
```

Two revisions (a branch or sha of jomcgi-org/freetoken-fork, or an absolute tree path), with an extra flag:

```bash
sudo bench/ab.py --arm base=origin/main --arm cand=perf/my-branch \
  --flags cand:"--moe-cpu-threads 8" --cache-state warm --hotset-knob
```

`--spec arms.json` takes `{"arms": [{"name": "base", "rev": "main", "env": {}, "flags": ["--x", "1"]}]}`.

## Hot-set adapt ticks (`adapt-ticks-check.py`)

Sums `hot_adapt_ticks_*` from a server journal and judges whether a hot-set arm is valid
(`adapt-ticks-check.py JOURNAL [--require] [--min-ticks N]`). `ab.py` calls it for every arm.
