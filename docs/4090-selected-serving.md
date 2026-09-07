# Selected RTX 4090 serving configuration

This handoff selects the ordinary Qwen Flash runtime qualified in #55 and #60.
The runtime source is unchanged from #53 at
`c56790d9298dcbd0dcc3103c8e369e99341f13f1`, which has the same Python and native
sources as the privately measured runtime revision. The service configuration
and these instructions are the only additions. The speculative serving and
paired CPU experiments in #64 are not part of this build.

The selected path combines CPU executor improvements, buffered staged prefill,
published HOT reuse, prefill-marker carry, gated diagnostic work and reclamation
of redundant HOT checkpoint pages. Expert selection, quantization and model
arithmetic retain the qualified behavior. The matched multi-turn comparison
preserved request bodies, complete answers and completion-token counts in both
execution orders. The short coding-task comparison also checked original task
graders and permitted files. These sampled checks do not prove broad quality
equivalence. Full measured records remain private on the serving node.

## Runtime isolation

Use a clean linked worktree at
`/var/lib/longhorn/nvme-02/freetoken/wt-astra-selected-serving`. Pin its revision
in the private deployment manifest before starting the service. Reuse the
qualified Python environment from `wt-plegather/.venv`, but copy the four
qualified native extensions into the new worktree as regular files:
`_cpu_moe`, `_pinned_tensor`, `_ple_uring` and `_uffd_pager`.

Verify every copied binary and its C++ source against the selected runtime's
qualification manifest. The selected CPU extension differs from the original
serving CPU extension. The other three extensions and their sources match the
original. Do not substitute the original CPU binary or retain symlinks to
mutable benchmark libraries. Keep all copies and build products out of git.
Neither the original checkout nor its native libraries need modification.

## Service configuration

`deploy/systemd/4090-selected.conf` is a node-4 drop-in for the existing
`freetoken-serve.service`. It retains the service user, CUDA environment,
restart policy, loopback endpoint and API model alias. `PYTHONPATH` selects the
isolated runtime; the existing `ft` executable supplies the Python environment.
Shutdown uses the service's cgroup instead of the original broad process-name
kill. No benchmark lease or automatic experiment deadline remains attached to
interactive serving.

The configuration reproduces the selected benchmark settings, with the port
changed to the existing interactive endpoint. It supports one running request,
reserves 65536 KV tokens, and disables disk prefix persistence and automatic KV
growth. Ordinary in-memory prefix caching and prefill state carry remain active.
Larger context or concurrency configurations need separate qualification.
Per-step MoE timing, decode-stat collection, continuation traces and reclamation
census probes are disabled. The lightweight API cache report remains enabled.

Before installation, confirm the original server is healthy, no benchmark owns
the GPU, and no client request is active. Save the original unit configuration
and all runtime identities privately. Validate the drop-in with systemd and
compare its command and environment with the qualified selected arm. Install as
`/etc/systemd/system/freetoken-serve.service.d/50-selected-runtime.conf`, reload
systemd and restart `freetoken-serve`. Use a detached supervisor with conditional
rollback until health, a real completion and the worker's loaded-library paths
have all passed. A lost terminal must not strand the service during activation.

After activation, verify the only GPU worker belongs to `freetoken-serve`, all
four mapped native libraries come from the isolated runtime, and their bytes
still match the qualified build. Check actual startup geometry and effective
diagnostic settings. Save the deployment check privately. Interactive clients
continue using their existing endpoint and the `qwen3.6-27b` API alias, which
names this Qwen Flash deployment for compatibility.

## Rollback

After confirming no benchmark or request is active, stop `freetoken-serve` while
the selected drop-in is still loaded. Verify its worker has released the GPU.
Remove only `50-selected-runtime.conf`, reload systemd and start
`freetoken-serve`. Leave the original service file and other drop-ins untouched.
Verify health, an actual completion, the original worker's cgroup and its
original native-library identities. The isolated selected checkout may remain
for inspection or later reuse.

## Remaining experiments

Ordinary batching with the paired CPU kernel, batched GPU dense operations,
lower verification overhead and stronger causal draft matching remain backlog
items. They are not required to use this selected build. Give any future round
a separate budget and a workload-specific wall-time acceptance criterion before
implementation. Keep diagnostic probes separate from qualifying wall runs and
retain all task failures.
