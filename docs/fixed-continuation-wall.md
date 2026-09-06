# Scripted continuation comparison

Pi can take different solution paths and generate different amounts of text
between arms. This additional mechanism benchmark uses ordinary greedy
generation to copy supplied integer records over three successive chat turns.
The second and third requests include the preceding assistant answers verbatim.
The initial request includes a longer synthetic archive; each followup supplies
32 replacement records. All fixtures are deterministic and prepared before
timing. Conversation prefixes differ within an arm and repeat across arms.

The existing snapshot controller supports this client with:

```sh
python3 bench/pi-decode-prefix-wall-driver.py \
  --fixed-continuation \
  --run-id astra-pi-agentic-fixed-continuation-20260906 \
  --output-dir /private/tmp/astra-pi-agentic-fixed-continuation-20260906
```

Add `--preflight` to inspect the pinned runtime and serving configuration without
stopping serving. The remote server setup, identity and geometry checks,
heartbeat lease, timeout recovery and restoration verification are shared with
the Pi snapshot comparison. Remote start/end helpers run in a dedicated systemd
action unit bound to the lease. Lease shutdown stops that unit before recovery,
and recovery explicitly stops it before the benchmark server. An interrupted
SSH connection therefore cannot leave a startup helper issuing requests after
restoration. Preflight rejects an existing live action unit.
The controller still requires a clean linked
worktree. Its client source hash is recorded and checked before each arm.

The schedule is off/on/on/off, with one warmup and two measured conversations
per fresh server. Each request has a 768-token output limit and a 300-second
transport timeout. There are no repairs, retries, constrained decoding,
logit biases, token forcing or added runtime instrumentation. Both modes use
the same checkpoint, routing, native binary and runtime revision. The snapshot
flag is the only explicit server environment difference. Each complete
conversation's wall clock includes client processing and all three requests.

The client independently checks ordered keys, integer values, duplicate keys,
completion status and token accounting. Complete responses and failed attempts
are retained. Correct JSON with different whitespace may pass copying checks
but does not qualify as matched work.

FreeToken omits `prompt_tokens_details` for zero cache hits. The client interprets
that omission as zero, while the summary separately requires
`--enable-cache-report` in both pinned server command lines. A malformed details
object is still rejected.

After completion and verified restoration:

```sh
python3 bench/pi-agentic-runtime-summary.py \
  --fixed-continuation \
  /private/tmp/astra-pi-agentic-fixed-continuation-20260906
```

The summary requires corresponding request bodies, full assistant messages,
prompt-token counts and completion-token counts to match across all four arms,
including warmups. Cache-read counts may differ. Any mismatch or failed task
suppresses every speedup percentage, retains attempted wall time, and makes
the summary command exit unsuccessfully. Results include both orders and
separate first-request and continuation-request totals. The first request pays
snapshot costs before its generated prefix can benefit a later request.

Matching text and token counts does not prove identical internal token IDs or
expert routes. This synthetic copying fixture does not establish agent task
performance or broad quality equivalence. Interpret it alongside ordinary Pi
wall time and state-resumption correctness checks. Adding this harness makes
no model performance claim; full-model qualification remains required.
