# Model quality checks

Performance work must not change what the model computes. Two checks catch a
broken build, at different costs:

| Check | Wall time | Catches |
| --- | --- | --- |
| [Output parity](#output-parity-benchparitypy) | minutes | any kernel, cache or restore change that moves the model's outputs, including subtle numeric drift |
| [BFCL subset](#bfcl-subset-benchbfcl) | about an hour | broken agentic tool use: tool-call parsing, multi-turn state, long tool results |

Run output parity on every serving change. Run the BFCL subset before promoting
a build. Neither replaces the workload wall-time comparisons in
[4090-performance.md](4090-performance.md); they check correctness, not speed.

Both talk to a running server over its OpenAI-compatible API and send one
request at a time. Qualifying measurements must run with diagnostics off, as for
wall-time runs.

## Output parity: `bench/parity.py`

`record` sends a fixed set of greedy requests to a trusted build and stores the
exact request bodies with the outputs. `compare` replays those bodies against a
candidate, so the requests are byte-identical whatever checkout the client runs
from.

The cases:

- the three short prompts from `bench/quality.sh`;
- one coding-agent session, with tool definitions, `Read` calls and real repo
  source files as tool results, cut at about 4k, 16k, 40k and 60k tokens. Each
  cut ends on a tool result, so the model's next turn is an agent step. The
  long cuts cross prefill chunk and layer-major group boundaries.

`compare` replays each case twice by default. The second pass is served from the
prefix cache, so it also checks cache and restore paths.

```sh
# on the trusted build
python bench/parity.py record --base-url http://127.0.0.1:8090/v1 --out parity-base.json
# on the candidate
python bench/parity.py compare --base-url http://127.0.0.1:8090/v1 \
  --baseline parity-base.json --report parity-cand.json
```

`compare` exits non-zero when any replay fails. Record the baseline with the same
checkpoint, KV dtype and page size as the candidate. Serving knobs (threads,
budgets, chunk size, HOT settings) may differ; checking that they do not change
outputs is the point. Cut lengths are estimated from characters, so the longest
case lands near, not exactly at, 60k tokens; `record` prints the real prompt
sizes. If a server with a smaller context rejects it, lower `--cuts`.

### Reading the result

With logprobs (the server returns them for non-streaming requests), each replay
is compared token by token:

- `identical`: same tokens, and mean logprob drift within `--max-mean-drift`
  (default 0.05 nats).
- `near-tie flip`: the first differing token is one where the baseline's top two
  candidates were within `--tie-margin` (default 0.1 nats). Reordered
  floating-point sums can flip a near-tie, so this passes, but repeated near-tie
  flips in one area deserve a look.
- `confident flip`: FAIL. The candidate chose a different token where the
  baseline was confident. This is a bug, not noise.
- `logprob drift`: FAIL. The tokens match but the probabilities moved, as
  happens with a subtly wrong scale, dequantization or accumulation.

A baseline recorded from a server without logprobs falls back to exact text
matching, which reports any difference as a failure and cannot separate noise
from bugs.

A case that fails only on the second pass points at prefix-cache or restore
state rather than at compute.

## BFCL subset: `bench/bfcl/`

The [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)
v4 is a public tool-calling benchmark. Its multi-turn categories run the model
against simulated file system, messaging and trading APIs over several user
turns, and grade the final API state. `run.sh` runs a fixed 54-entry slice
(`subset.json`): 29 `multi_turn_base`, 5 `multi_turn_miss_param`, 10
`simple_python` and 10 `parallel_multiple`. Requests go through the server's own
tool-call parser, as agent clients do.

```sh
OUT=results/bfcl-base bash bench/bfcl/run.sh   # trusted build
OUT=results/bfcl-cand bash bench/bfcl/run.sh   # candidate
python3 bench/bfcl/compare.py results/bfcl-base results/bfcl-cand
```

The first run creates a venv at `bench/bfcl/.venv` with a pinned `bfcl-eval`
(it skips `sentence-transformers`, which only the unused `memory_vector`
category needs). Set `BFCL_VENV` to put it elsewhere; the install is about
500 MB, so keep it off a full root disk. Scoring is offline; the multi-turn
backends are plain Python.

Options, as environment variables:

- `BASE_URL`: server URL, default `http://127.0.0.1:8090/v1`.
- `OUT`: a fresh directory per run. BFCL silently reuses existing results, so
  `run.sh` refuses a directory that already has them.
- `THINKING=off`: disable the chat template's thinking. The run is faster but
  no longer measures the thinking path. Use the same setting for both builds.
- `MAX_TOKENS`: per-call output cap, default 4096.
- `SUBSET`: a different test-id file.

`compare.py` prints per-category pass counts and the entries that flipped, with
BFCL's error type for new failures. It exits non-zero when the candidate loses
more than `--max-lost` entries net (default 2). Even at temperature 0, a harmless
numeric change can flip an entry or two. A real regression usually shows up as
a cluster of new failures with one error type, for example a tool-call parse
error.

BFCL's per-call client timeout is 600 seconds. A thinking response slower than
that is recorded as an inference error and fails its entry.

## Choosing between them

A failing parity check is the early warning: it names the case and the token.
BFCL shows whether a change that does move outputs still leaves the model able
to do agentic work. A change expected to move outputs, such as a new
quantization or kernel precision, should record a new parity baseline only
after the BFCL subset shows no loss.
