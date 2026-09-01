# Spec: request priority scheduling (FreeToken, patch 20)

## Workspace

A worktree on a branch off `feat/moe-disk-tier` (current tip). Commit on
the branch; do NOT push. Study first: the scheduler admission/queue
(scheduler/scheduler.py, prefill.py PrefillAdder), the one-forward-in-
flight loop, and the OpenAI adapter request plumbing.

## Problem (measured, node-4 production)

FIFO queueing: an interactive chat request queues behind batch/agent
requests. The box serves one human plus a job cluster; the human should
never wait behind bulk work.

## Task

1. Request surface: accept `priority` (integer, higher = sooner) via an
   OpenAI-compatible extension field on the request body, plus an
   `x-request-priority` HTTP header (header wins; both optional,
   default 0).
2. Scheduler: the waiting queue orders by (priority desc, arrival asc).
   Priority affects ADMISSION ORDER only in v1 - no preemption of
   running requests (document the seam for it). A continuously-fed
   stream of high-priority requests must not starve priority-0 forever:
   add a simple aging term (waiting time grants +1 effective priority
   per --priority-aging-seconds, default 30, 0 disables).
3. Chunked prefill interaction: between prefill chunks of a running
   low-priority request, a waiting higher-priority request may win the
   next admission slot (this is what makes priority feel real given
   one-forward-in-flight; if the current structure cannot re-order at
   chunk boundaries, say so precisely - that is the same seam as
   prefill-decode mixing).
4. Stats: per-interval queue depth by priority band, and
   max_wait_seconds.
5. Zero behavior change when no request carries a priority.

## Tests

GPU-free: queue ordering, aging math, header/body parsing precedence,
starvation bound. Platform note: the Mac cannot run the package's tests;
write them, state that plainly, never fake a pytest line.

## Deliverable

Commits + report: files, the chunk-boundary re-ordering answer,
deviations.
