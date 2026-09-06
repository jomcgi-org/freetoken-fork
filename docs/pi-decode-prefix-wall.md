# Decode snapshot wall-time comparison

`bench/pi-decode-prefix-wall-driver.py` compares
`FREETOKEN_DECODE_PREFIX_SNAPSHOT=0` and `1` using the same pinned runtime and
native MoE binary on node-4. It follows the existing Pi runtime driver's
recovery and measurement protocol. The separate frozen driver keeps this
experiment's identity checks independent of the earlier combined-runtime
comparison.

The schedule is off/on/on/off. Each fresh server runs one warmup task and two
measured tasks, giving four measured tasks per flag in both orders. Every task
uses the same three-stage Pi fixture, workspace path, token budgets, sampling,
and independent cumulative checks. Warmups and failures remain in the record.
Missing or failed tasks make the controller exit unsuccessfully even if it
traversed the schedule and restored serving.

Both arms use buffered GPU disk prefill, mmap HOT staging, the same CPU kernel,
3753 expert slots, capacity one, graph batch one, and 65536 FP8 KV tokens.
Radix reuse is enabled; disk prefix caching and persisted HOT plans are off.
Automatic HOT adaptation stays enabled with the same policy. Host page cache
is retained across starts. Command lines must match exactly, and the snapshot
flag must be the only difference in explicit server environment variables.
The driver checks the loaded native module, source identity, startup geometry,
and actual worker environment. MTP and special-token checkpoints must be off.
Engine token tracing and invasive MoE diagnostics are disabled.

Run from this linked worktree with the pinned Pi installation:

```sh
python3 bench/pi-decode-prefix-wall-driver.py \
  --run-id astra-pi-agentic-decode-prefix-wall-20260906 \
  --output-dir /private/tmp/astra-pi-agentic-decode-prefix-wall-20260906 \
  --pi /private/tmp/astra-pi-cli/node_modules/.bin/pi
```

`--preflight` verifies identities and configuration without stopping serving.
A remote lease receives heartbeats every 20 seconds. EOF, a 90-second renewal
timeout, or the four-hour limit triggers restoration of the original server,
including a health check and a real completion. Each task has a 900-second
budget; Pi interruption preserves failure evidence and terminates its process
group. The benchmark uses its own server port and SSH tunnel.

Qualification compares complete attempted task wall time, success, generated
tokens, model calls, tool errors, and repairs. Changes in model trajectory can
change work performed, so task-wall improvements alone cannot isolate engine
speed. Report each order alongside the aggregate. This small coding fixture
does not establish general model quality. Adding the driver makes no speedup
claim; runtime results require separate verification.

Seventeen focused pure-Python checks pass on Linux and macOS. They cover
recovery with an active or stopped original server, retained restoration
failures, heartbeat EOF and timeout, geometry drift, accidental differences
between arms, and missing or failed tasks. These validate the controller;
they do not qualify model performance.
