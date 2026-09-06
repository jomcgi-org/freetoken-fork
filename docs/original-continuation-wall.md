# Original versus selected runtime continuation wall comparison

Run `bench/pi-decode-prefix-wall-driver.py --original-baseline
--fixed-continuation --run-id astra-pi-agentic-<unique-name>
--output-dir /private/tmp/<unique-name>` from a clean linked worktree.
Add `--preflight` to inspect the configuration without stopping serving.

The baseline is the original serving checkout, with CPU DISK prefill and its
original CPU executor binary. The selected runtime is the pinned prefill-marker
carry revision from #45. It includes buffered staged prefill, published HOT
reuse, CPU executor improvements and gates for unnecessary telemetry. The
comparison measures their combined effect and cannot attribute an isolated
speedup to one component. Experimental GPU source staging, host-page reclaim,
profile retention and decode prefix snapshots are excluded.

Both arms request diagnostic flags off. The original still performs legacy
unconditional diagnostic work; removing that work is part of the selected
runtime. The controller uses the same model and actual expert routes, fixed
CPU/GPU placement, HOT capacity, FP8 KV capacity and CUDA graph geometry.
Only the qualified CPU native binary/source and prefill configuration may
differ. The other three native extension hashes and sources must agree.
The original arm must match the identity used by normal serving and recovery.

Run original, selected, selected, original with a fresh server, one warmup and
two measured three-turn conversations per start. Retain the host page cache
and all failures. Use the existing lease recovery on interruption. Snapshot
and token/census diagnostics are off. OS memory and worker I/O snapshots occur
outside client timing and include warmup; they are not peak-memory measures.

Summarize with `bench/pi-agentic-runtime-summary.py <results-dir>
--original-baseline`. The summary requires matching requests, answer bytes
and token counts, including warmups, before reporting a wall reduction. It
retains both execution orders and separate first-request and continuation
costs. If nondeterminism changes the work, report that limitation without
turning unequal work into a runtime speedup claim.

This is a capacity-one conversation benchmark, not a replacement for the
blog's prefill or aggregate decode token-rate workloads. It does not establish
broad quality equivalence. Check journals and original serving restoration
before accepting the result. Keep all measured artifacts private.
