# Paired CPU dots in full-model target verification

This experiment combines the explicit NVFP4 paired CPU kernel with the wider
target and rollback diagnostic. Set `FREETOKEN_TARGET_VERIFY_CPU_PAIR_COMPARE=1`
alongside the existing graph, serial-linear and seed-checkpoint flags. Width
must be three or five. The compact rollback flag additionally tests reconstructed
recurrence state. The native kernel must support the explicit pair setter.

Each token window starts with ordinary single-row decode as the reference.
Paired eager, captured and compact forwards compare logits, greedy tokens and
all committed state to that reference. Every retained rejection prefix is
checked. Timed acceptance and rejection trials alternate paired and ordinary
CPU dispatch for each outcome, reversing order between repetitions. Dispatch
switches happen after stream synchronization and outside the timer.

Trials preserve the existing checks for accepted prefix length, committed state,
retained state and the next ordinary decode after rejection. A missing mode,
incorrect dispatch marker, invalid duration or numerical failure prevents a
qualified cost comparison. The report records native and source identities and
retains each trial privately.

Run under the exclusive supervisor with automatic original-serving recovery.
Builds and focused validation must finish before the model benchmark starts.
The startup configuration requires a single request, ordinary graph size one,
unchanged expert routes and quantization, and invasive telemetry disabled.

This measures target verification components, not serving throughput. Repeated
windows warm expert and file caches, and all trials reserve the wider workspace.
A production proposer and scheduler, broad task validation and separate non-debug
wall measurements remain required. No serving path enables paired execution by
default.
