# Target-only two-token verification cost probe

The existing MTP path loads a draft head and disables ordinary CUDA graphs. That
does not answer whether a headless proposer could amortize expert work on the
4090. `bench/target-verify-cost.py` exercises the existing two-position target
forward with MTP off and the ordinary capacity-one graph available.

This is an invasive component diagnostic. It aborts the dedicated worker after
writing its private report. Never install it in normal serving, a user session,
or a wall-time benchmark. Run only under an exclusive supervisor that restores
the original service on success, failure, timeout, or controller disconnect.

Set `FREETOKEN_TARGET_VERIFY_GRAPH=1` to add a dedicated verification graph. This
adapts the host PLE staging and capture-safe constant indptr approach from the
existing `feat/mtp-graphs` work at `1c42624`. It preserves the selected runtime's
serial width-one GDN updates. PLE gets two staging/hash rows in every arm of this
diagnostic; normal serving imports none of these hooks.

Before starting the isolated server, set `FREETOKEN_TARGET_VERIFY_COST_DIR` to a
new private directory. Put this explicit hook in that run's `sitecustomize.py`,
using the absolute path to this clean worktree:

```python
import runpy
runpy.run_path('/absolute/worktree/bench/target-verify-cost.py')['install_import_hook']()
```

Prepend the hook directory and the same worktree's `python` directory to the
worker's `PYTHONPATH`. The loader waits for Engine's normal import; it does not
import Torch or initialize CUDA in frontend/tokenizer processes. The private
directory and hook must not be carried into the restored serving environment.

Use the selected NVFP4 runtime's existing command with capacity one, CUDA graph
size one, MTP off, and invasive diagnostics off. The explicit probe widens only
the CPU executor's decode workspace to two rows and QSA's pending ring to allow
one speculative position. It applies the same ring capacity to memory budgeting
and allocation. It does not load a draft head or alter routing/precision. Both
probe arms share these allocations; the untouched capacity-one configuration
still requires a later wall-time comparison.

Send a dedicated greedy request with enough output budget to reach at least
sixteen decode steps plus a compression boundary. The trigger selects three
adjacent two-token windows in a fully allocated KV page: the draft closes a
compressed group, the seed closes a group, and the next offset. Every window
gets independently derived ordinary-graph target tokens and state references.
CPU history is updated with the same candidate IDs used on the device because
the ordinary graph stages its PLE lookups from host history.

Each window measures one ordinary graph step, two ordinary graph steps, a
snapshot, acceptance (snapshot plus fused target and greedy acceptance), and
rejection (snapshot plus wrong target candidate, acceptance check, state restore,
and ordinary seed replay). Timing includes stream completion and metadata work.
Reset and correctness copies are outside each component window. Execution order
alternates and warmups are retained but excluded from timing summaries.

Graph mode adds acceptance and rejection through a separately captured two-token
graph. Each diagnostic window owns its graph and addressing tensors. Replay
restages token inputs, QSA addresses/lengths and PLE host lookups before launching
it. Capture failures are fatal to the probe and cannot silently fall back to
eager execution. This does not implement a proposer or serving scheduler.

Untimed numerical checks report finite status, changed-element counts, maximum
absolute error and RMS error for floating state and logits. Integer history is
compared without float conversion. Graph verification is also compared directly
with an independent eager two-token reference, separating capture discrepancies
from differences already present between batched and sequential target forwards.
These measurements do not introduce a tolerance or relax the existing exact
qualification gate. Keep their values private with the rest of the records.

Rejection restores recurrence, convolution, integer PLE history and QSA pending
state. It deliberately leaves speculative KV/index writes beyond the committed
length in place. The check compares all reachable state byte for byte, then
decodes the correct next token and verifies that formerly speculative rows have
been overwritten correctly. Full page restoration is used only between trials.
Acceptance also compares both target tokens and committed state with sequential
ordinary graphs. Any discrepancy, including a warmup failure, suppresses the
component break-even estimate and requires investigation.

Reports contain component times, correctness failures and a theoretical
acceptance threshold. They exclude proposer/scheduler overhead and repeatedly
warm the same expert/file working set. They are not model throughput results or
quality evaluation. Keep all measured artifacts private. A worker exit or an
aborted HTTP response alone is not success: inspect `completed`, every record,
and the external supervisor's serving-restoration audit.

Validation: the focused checks pass on Linux, including the tensor comparisons;
macOS runs the hermetic subset and skips the Torch checks. They exercise boundary
selection, private reporting, lazy installation, break-even qualification, and
committed versus unreachable cache-state comparisons. Full target-model execution
also completed for eager verification. Graph-mode validation and acceptance-state
equivalence remain unresolved, so the change stays draft while those checks run.
All measured records remain private; successful execution does not qualify a
serving throughput improvement.
