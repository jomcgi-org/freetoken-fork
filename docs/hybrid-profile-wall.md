# Expert-profile retention wall-time gate

The profile-retention fix attaches existing expert-prefetch advice to a hybrid
prefix that survived an unaligned finish. Its state tests establish correct
metadata retention and unchanged KV/state ownership. A model comparison must
establish whether the recovered advice improves wall time.

Use the existing controller with `--fixed-continuation
--hybrid-profile-retention`. It compares parent `58a5355` with fix `5b0ea43` in
parent/fix/fix/parent order. Both already carry prefill snapshots, so that
separate fix does not confound the comparison. All four native extensions must
match, and the only Python runtime difference must be `scheduler/cache.py`.
Commands are identical, with session expert prefetch on and the existing
64-expert protection limit pinned in both arms. Only `PYTHONPATH` differs in the
environment. Decode snapshots, token traces and invasive diagnostics stay off.

Each fresh server runs one warmup and two measured three-turn conversations.
The client performs ordinary greedy chat requests, retaining complete requests
and responses. The summary requires matching request bodies, assistant messages
and token counts across arms. It keeps all failures and both orders, and
suppresses gain percentages if fixed-work qualification fails. First requests
and continuations are reported separately. These checks do not establish broad
model quality equivalence or identical internal token IDs and expert routes.

Run the controller and summary from the same clean linked benchmark worktree:

```sh
python3 bench/pi-decode-prefix-wall-driver.py \
  --fixed-continuation --hybrid-profile-retention \
  --run-id astra-pi-agentic-profile-UNIQUE \
  --output-dir /private/tmp/astra-pi-agentic-profile-UNIQUE
python3 bench/pi-agentic-runtime-summary.py \
  --hybrid-profile-retention /private/tmp/astra-pi-agentic-profile-UNIQUE
```

Replace `UNIQUE` with a fresh lowercase suffix. Add `--preflight` for read-only
runtime checks before taking the serving lease. The mode is propagated to all
remote helpers, including recovery, and retains the existing lease lifetime and
service restoration behavior. Linux validation, native-artifact preparation
and model timing must wait until the current benchmark has restored serving.
Model wall-time qualification remains pending; no measurements are included.
