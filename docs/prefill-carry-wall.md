# Prefill marker carry wall-time comparison

Compare the pinned prefill-marker carry fix with its parent using the existing
scripted three-turn workload and recovery lease. The arm order is parent, fix,
fix, parent, recorded as off, on, on, off. Each start has one warmup and two
measured conversations. Ordinary greedy model generation supplies every answer.

Run the controller from a clean linked worktree:

```sh
python3 bench/pi-decode-prefix-wall-driver.py \
  --fixed-continuation --prefill-snapshot-carry \
  --run-id astra-pi-agentic-prefill-carry-example \
  --output-dir /tmp/astra-pi-agentic-prefill-carry-example
```

The two runtime revisions are pinned in the controller. Only the scheduler
prefill source changes under `python/`. Both use the same native CPU binary and
source hash, model checkpoint, command line and cache geometry. `PYTHONPATH` is
the only environment difference. Decode snapshots, token traces and invasive
diagnostics remain off in both arms.

The experiment selection travels with every remote action, including the
lease hold and restoration hook. Start/end helpers remain bound to the lease.
Do not alter the controller or runtime source while a comparison is running.

After all four arms finish and restoration is verified:

```sh
python3 bench/pi-agentic-runtime-summary.py --prefill-snapshot-carry \
  /tmp/astra-pi-agentic-prefill-carry-example
```

The summary checks the pinned revisions, matching native identities, runtime
paths, disabled decode snapshots and existing fixed-work requirements. Request
bodies, answer bytes and prompt/output token counts must match across all arms,
including warmups. Any failed task or mismatch suppresses all speedup claims.
First-request and continuation-request totals are retained separately.

Review server journals and restored GPU ownership as well. This synthetic
workload can isolate continuation cost; matching its outputs does not establish
broad model quality equivalence. No performance result is included here.

Preflight requires all serving native extensions, including the io_uring PLE
backend, and records each binary's path, hash and source hash. The two arms
must share those identities. Startup also checks the loaded CPU and PLE maps.
Missing artifacts are rejected before serving is stopped.
