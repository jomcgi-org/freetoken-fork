# Compact recurrent rollback component diagnostic

The [wider verification probe](target-multitoken-cost.md) retains a complete
recurrent state after each possible accepted prefix. This optional comparison
keeps one initial recurrent state and owned copies of the GDN inputs needed to
reproduce each prefix. It retains convolution, PLE and QSA prefix state as before.

Enable `FREETOKEN_TARGET_VERIFY_COMPACT_ROLLBACK=1` alongside the wider probe's
graph, serial-linear and seed-checkpoint flags. Width must be three or five and
layer tracing must be off. Normal serving never imports this diagnostic.

Every rejected prefix has a dedicated CUDA rollback graph. It restores the
initial recurrence, invokes the original GDN update kernel with the saved
inputs, and restores the small prefix states. All checkpoint copies and rollback
replay are inside the component timer. Input buffers are allocated before
capture, refreshed by every target replay, and checked for stable bindings.
There is no eager rollback fallback in the measured path.

The comparison includes ordinary decode, full-prefix checkpoints and compact
checkpoints, full acceptance and every rejection position. It checks exact
logits, tokens, committed KV and all mutable request state. Reconstructed
recurrence is compared after every retained prefix. Rejection is followed by an
ordinary decode with its output and state checked against the reference.

`checkpoint_storage` distinguishes checkpoint-owned tensor bytes from CUDA graph
memory. The process allocator snapshots bracket compact target construction and
rollback graph capture with the full-prefix graph still alive. Those snapshots
include graph allocations, workspace and allocator effects; they do not isolate
an individual graph pool or establish a serving memory saving. No allocator
sampling occurs inside the component timer.

Run the focused CPU checks with:

```sh
python -m pytest -q tests/test_target_verify_cost.py tests/test_target_seed_checkpoint.py tests/test_target_multitoken.py tests/test_target_compact_rollback.py
```

An additional real-kernel CUDA check is opt-in and requires exclusive GPU
ownership. It captures target and rollback graphs, refreshes inputs across
generations, overwrites the original activation buffers after target execution,
and compares recurrence against ordinary GDN updates for every prefix:

```sh
FREETOKEN_COMPACT_ROLLBACK_CUDA_TEST=1 python -m pytest -q tests/test_target_compact_rollback.py -k cuda
```

Run that check before model loading under the owned experiment supervisor, with
automatic original-serving recovery. Keep the import hook disabled for the
kernel check. Neither tests nor source changes may overlap measured model runs.

Validation is pending for the compact path on the model. This is a diagnostic,
with no proposer or serving scheduler. Repeated windows warm expert and file
caches. Component timing and local exact checks do not establish end-to-end
throughput or broad quality equivalence. Full records remain private.
