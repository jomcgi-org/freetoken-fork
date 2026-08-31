# Spec: CPU/GPU cross-layer decode pipelining (FreeToken, patch 17)

STATUS: drafted, awaiting dispatch decision (final structural patch of the
program; largest risk).

## Workspace

A worktree on a branch off `feat/moe-disk-tier`. Commit on the branch; do
NOT push. Study first: the CUDA-graph decode step in engine/graph.py and
engine/engine.py, the flag-handshake doorbell in moe/cpu_executor.py +
cpu_moe_ext.cpp (coordinator, ready/done flags), and the DISK-layer
placement (head+tail spill: layers 0-8 and 39-47 typical).

## Problem (measured, node-4: x1 ~44ms/step at 23 tok/s)

Layers execute strictly serially. The DISK (CPU-decoded, cold-tail)
layers bookend the step: the GPU idles while the CPU grinds layers 0-8,
then again for 39-47; the CPU idles during the GPU middle. Both engines
are busy less than half the step.

## Constraint that shapes any design

A layer's input depends on the previous layer's output: true pipelining
across ONE token's layers is impossible. The available overlaps are:
(a) across the top-k routes WITHIN a CPU layer: the cold-tail partial
    (CPU) and hot partial (GPU slot cache) of the SAME layer already run
    on different engines; today the graph waits on the doorbell before
    combining. If the GPU-side hot GEMM runs while the CPU computes the
    cold partial (it may already), the win is bounded by the slower side.
    Measure first and report whether this overlap already exists.
(b) across STEPS at bs>1: while the GPU processes the middle layers of
    the batch's step, the CPU could already run layers 0-8 of the NEXT
    scheduler wave if requests are staggered (continuous-batching skew).
    This changes the scheduler contract; treat as out of scope unless a
    small seam exists.
(c) speculative next-layer weight staging: overlap the CPU tail layers'
    WILLNEED/read traffic with GPU middle compute (cheap, likely already
    covered by the prefetch machinery; verify).

## Task

1. FIRST, instrument: add a per-step phase-timing breakdown behind
   --moe-step-timing (cpu_head_us, gpu_mid_us, cpu_tail_us, overlap_us,
   printed on the decode log line). This alone is a deliverable: it
   turns the inferred step budget into measured numbers.
2. From the measurements, implement the highest-value overlap that does
   NOT change output semantics. The expected candidate: dispatch the
   CPU cold-tail partial for layer L asynchronously and let the GPU
   proceed with layer L's hot partial and the doorbell wait placed as
   late as the dataflow allows (the combine point), rather than at
   submit. If the doorbell is already late-bound, say so and move to the
   next candidate.
3. Anything requiring scheduler-contract changes (b) is OUT OF SCOPE:
   describe the seam in the report instead of implementing it.
4. Correctness bar: bit-identical greedy outputs; the CUDA graph must
   not observe torn CPU outputs (the existing flag protocol's ordering
   guarantees must be preserved).

## Tests

GPU-free: timing plumbing, flag validation. CUDA-gated: greedy parity
with timing on. Honest platform note as usual: tests written on the Mac
cannot run there; never fake a pytest line.

## Deliverable

Commits + report: the measured phase breakdown FIRST, what overlap was
(or was not) implemented and why, deviations. An honest "the doorbell is
already late-bound and the engines already overlap; here are the
numbers" is a fully successful outcome.
