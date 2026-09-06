# Short leaderboard task wall comparison

`bench/leaderboard-wall.py` runs frozen tasks through the original model-bench
agent loop and hidden graders. The private input bundle contains the original
Python harness, registry, task definitions, fixtures, historical result cells and
reference fixes. Its manifest checks every file before importing the harness,
then recomputes each historical cache key from the prompt, fixture, verifier,
model and budgets. References are outside the fixture copied into the agent's
work directory.

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

```sh
python bench/leaderboard-wall.py qualify --bundle /private/inputs --output /private/qualification.json
python bench/leaderboard-wall.py run --bundle /private/inputs --qualification /private/qualification.json --arm r1 --output /private/r1.json
python bench/leaderboard-wall.py summarize --bundle /private/inputs --reports /private/r1.json /private/r2.json /private/r3.json /private/r4.json --output /private/comparison.json
```

The summary requires the complete task set in both execution orders with an
identical manifest. It reports task success, calls, output tokens, total wall
and model-request time by task and order. Agent trajectories may differ, so this
is task throughput rather than an isolated runtime speedup. Small task samples
do not establish broad quality equivalence. Detailed records remain private.

Focused wrapper checks pass locally. Linux grader qualification and fresh model
runs are pending; no new throughput result is claimed by this source change.
