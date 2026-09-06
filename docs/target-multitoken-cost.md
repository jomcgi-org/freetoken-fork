# Wider target-verification component diagnostic

One proposed token limits the expert work a verification forward can amortize.
This diagnostic extends the existing target-only probe to three or five target
rows, representing two or four proposed tokens plus the seed. It implements no
proposer or serving scheduler.

Set `FREETOKEN_TARGET_VERIFY_WIDTH=3` or `5`, with graph mode, serial dense-row
execution and seed checkpoints enabled. Activation tracing must be off. The
default width remains two. Use the existing private-output import hook in an
isolated worker with external serving recovery. The wider hooks are installed
only for this explicit experiment; normal serving imports none of them.

Every stateful GDN and QSA operation keeps ordinary width-one execution in token
order. PLE convolution uses the same per-token updates. Dense linears retain
ordinary row geometry; expert work sees the full target width. Each target
position has independent QSA addressing and PLE history derived only from its
preceding tokens. CPU workspace, PLE staging and the QSA ring reserve the wider
capacity in every comparison arm.

The graph retains mutable state after every possible accepted prefix. Rejection
restores the prefix ending just before the incorrect proposed input, while
returning the target's corrected prediction. Future KV/index writes remain
unreachable and are overwritten by the next ordinary decode. Every saved prefix,
prediction, logit and committed-state comparison must match ordinary decoding
exactly. Each partial acceptance gets its own forced-rejection trial and subsequent
ordinary continuation check. Unknown state, missing updates or mismatches fail
qualification, including failures during warmup.

Four adjacent windows cover the QSA compression offsets, with the full target
width contained in one allocated KV page. Timings alternate ordinary decode,
ordinary decoding of all rows, full acceptance and each possible rejection
prefix. Reset and correctness copies are outside the timed windows; checkpoint
copies and restoration are inside. The report gives each outcome's cost against
one ordinary decode. A multi-token proposer needs its full accepted-prefix
distribution to estimate cost; a single acceptance percentage is insufficient.

These repeated windows warm the expert/file working set. Component costs exclude
proposer and serving scheduler work, and all measured artifacts remain private.
They cannot qualify serving throughput or broad quality equivalence. Focused
Linux checks and full target-model validation are pending.
