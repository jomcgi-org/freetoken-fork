# First-token checkpoint verification experiment

The target-only diagnostic in `target-verify-cost.md` restores the pre-verification
state and replays the seed when a proposal is rejected. This experiment captures
the state after the seed during the two-token forward, allowing rejection to
restore that state directly. All original token computation, weights, routing
and precision remain unchanged.

Set `FREETOKEN_TARGET_VERIFY_SEED_CHECKPOINT=1` alongside graph mode and the serial
dense-row policy. Activation tracing must be off. The existing isolated worker,
private output and external serving-recovery requirements still apply. Normal
serving imports neither checkpoint hooks nor the diagnostic.

The experiment captures a second verification graph with preallocated checkpoint
buffers. Copies follow the established first-token GDN recurrent/conv update,
each PLE short-convolution update and each QSA attention update. Integer PLE
history is shifted by the seed using the ordinary decode rule. Every mutable
request-state row must be observed; unsupported state or missing/extra updates
fail the capture. The saved buffers belong to this graph and fixed request slot.
This does not implement general request scheduling or a proposer.

Acceptance includes every checkpoint copy even though it retains the two-token
state. Rejection uses the target's first prediction and restores the saved seed
state without a seed replay. Speculative future KV and compressed-index writes
remain in their unreachable rows. The benchmark compares all committed state,
then performs an ordinary next-token decode to check that those rows can safely
be overwritten. Saved seed state is compared against an independent ordinary
decode on both accepted and rejected trials, outside the timed windows.

The existing graph remains a separate comparison without checkpoint copies.
Reports include eager, graph-with-replay and graph-with-checkpoint acceptance
thresholds. All thresholds exclude proposer/scheduler costs and repeatedly warm
the same expert/file working set. These are component costs, never serving wall
time or a broad quality evaluation. All measured records remain private.

Validation: focused Linux checks pass for saved-state selection, other-slot
isolation, incomplete-checkpoint rejection, explicit hook gating and report
qualification. Full target-model execution also completed: the saved seed,
target predictions, logits and committed state matched ordinary decode in the
tested windows. Rejected trials and their subsequent ordinary continuation
passed the same exact comparisons. Both graph variants ran with activation
tracing disabled, and the external supervisor restored original serving.
These local checks do not establish coverage of every input, cache boundary or
sampling setting. A serving integration still requires separate wall-time and
quality validation.
