# Short leaderboard task wall comparison

`bench/leaderboard-wall.py` runs frozen tasks through the original model-bench
agent loop and hidden graders. The frozen input bundle contains the original
Python harness, registry, task definitions, fixtures, published historical rows
and reference fixes. These can be reconstructed directly from the public source
repository on the benchmark node. Private cached result cells stay on the
controller. Compare the reconstructed input keys to those cells before launch.
The bundle manifest checks every file before importing the harness, then
recomputes each recorded input key from the prompt, fixture, verifier, model and
budgets. References are outside the fixture copied into the agent's work directory.

The client preserves the original prompts, file tools, turn and token budgets,
temperature, served model alias and extra request fields. A grading preflight
must fail on each buggy fixture and pass on its reference fix before any model
benchmark can run. Execute grading on Linux with the task's required tools.

Each fresh attempt records the historical sum of model-request latency and an
outer task timer including the agent loop, tools and final grading. Failed tasks
remain in timing totals. The original grade is retained separately from the
completion qualification, which also requires valid tool use and edits confined
to the task's declared target files. Private results retain the final edits so
the work is reviewable after the original loop removes its temporary directory.

Run `qualify`, then `run --arm r1` through `r4` under an exclusive GPU supervisor
that owns the local endpoint and automatically restores original serving. The
arm order is original, selected, selected, original. The supervisor must pin
runtime and native identities, qualify serving geometry, and disable invasive
diagnostics. Source changes, dependency setup and unrelated tests must finish
before measured model requests begin.

`bench/leaderboard-wall-driver.py` implements the Linux supervisor using the
previously validated original/selected runtime guard. Its `preflight` action is
read-only. Its `run` action requires it to be the main process of the
`astra-leaderboard-wall` systemd unit, with `KillMode=control-group` and the
existing recovery script configured as `ExecStopPost`. Launch with a fresh
output directory and an independently confirmed `--keys-verified` assertion.
It pins the benchmark source, native identities, grading toolchain and geometry,
reserves a single request slot, and disables invasive telemetry. Each arm gets
the same short readiness completion before timed work. Startup and readiness
are outside task timers. Final journal and original-serving audits remain
required after the unit becomes terminal.

```sh
python bench/leaderboard-wall.py qualify --bundle /private/inputs --output /private/qualification.json
python bench/leaderboard-wall.py run --bundle /private/inputs --qualification /private/qualification.json --arm r1 --output /private/r1.json
python bench/leaderboard-wall.py summarize --bundle /private/inputs --reports /private/r1.json /private/r2.json /private/r3.json /private/r4.json --output /private/comparison.json
```

The summary requires the complete task set in both execution orders with an
identical manifest. It reports task success, calls, output tokens, total wall
and model-request time by task and order. Completion rate is the fraction of
attempts that qualify. Successful tasks per hour divides qualified completions
by all attempt wall time, including failed attempts. A zero wall total produces
no rate; a zero baseline completion rate produces no throughput ratio. Report
these alongside raw wall totals, since a quick failure can reduce elapsed time.
Startup and readiness are excluded, so this rate describes the measured task
intervals, not sustained service capacity.

Agent trajectories may differ, so this is task throughput rather than an
isolated runtime speedup. Small task samples do not establish broad quality
equivalence. Compare published rows only using model-request latency: the
historical page excludes tools and grading, and its actual concurrency is not
recorded. The fresh comparison uses one active request. Matching input keys
does not establish matching historical serving conditions. Detailed records
remain private.

Focused wrapper checks pass on Mac and Linux. Both frozen graders reject their
buggy fixtures and accept their references on Linux. The SLO grader has a missing
`sys` import in its failure-reporting branch; that original behavior is retained
for comparability, and the final edits are available for review. Fresh model
runs completed in both execution orders, all generated edits were reviewed, and
automatic recovery returned the original service with a successful completion
and verified GPU ownership. The run contains a model task failure, which remains
in the comparison. Completion-rate reporting was added after the run and applies
to the saved records; it does not change the measured client or serving paths.
This source change does not publish measured performance payloads.
