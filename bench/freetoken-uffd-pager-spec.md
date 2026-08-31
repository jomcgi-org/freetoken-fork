# Spec DRAFT: UFFD expert pager for DISK banks (FreeToken, patch 4)

STATUS: draft — finalize with fault-pattern data from the post-PLE bench
(majflt/step, cold-vs-warm gap, per-layer miss profile) before dispatch.

## Motivation

The mmap+MADV path leaves residency to the kernel page LRU: page-granular,
advisory, and adversarial under memory pressure (the cgroup evicts pages we
are about to route to). A userfaultfd pager makes expert residency explicit
and ROW-granular — the same design Firecracker uses for snapshot restore.
Endgame: banks far larger than RAM (DeepSeek-V4: 137 GiB) served from a
fixed byte budget of hot experts, disk -> RAM -> VRAM as a managed pipeline.

## Design sketch

New `--moe-disk-pager {mmap,uffd}` (default mmap, existing behavior).

1. **Region**: per DISK layer, an anonymous MAP_PRIVATE region the size of
   the bank, registered UFFDIO_REGISTER_MODE_MISSING. The FTW file region is
   the backing store, read with io_uring (O_DIRECT, row-aligned).
2. **Handler thread**: services missing-faults by filling the ENTIRE expert
   row containing the faulting page via UFFDIO_COPY (one fault = one row, no
   per-page fault storms). Runs outside the GIL (C, or a dedicated pybind
   module; extending the cpu_moe ext is acceptable).
3. **Prefetch = pre-fill**: the routed-expert hook (exists since 5932caf)
   consults a residency bitmap; missing experts are filled proactively by the
   handler before compute — zero faults on the hot path. WILLNEED disappears
   on this backend.
4. **Eviction**: userspace LRU over expert rows against a configurable byte
   budget (`--moe-pager-budget-gib`); evict via MADV_DONTNEED on the region
   (re-registering not required; next touch refaults). Router stats already
   rank hotness. This REPLACES the external cgroup as the memory governor.
5. **VRAM tier (stretch/phase 2)**: fills can target pinned staging directly
   for H2D into the GPU slot cache, bypassing the CPU-executor read for
   layers that would be better served by GPU compute. Keep out of v1 unless
   it falls out naturally.
6. **Stats**: fills, fills-from-prefetch vs fault-driven, evictions, resident
   bytes, fill latency histogram — the pager must be observable or it is
   undebuggable.

## Preconditions / risks

- `/proc/sys/vm/unprivileged_userfaultfd=1` or CAP_SYS_PTRACE; document and
  probe at startup with a clear error. Fine on hosts we control.
- UFFDIO_COPY into MAP_PRIVATE anon is the well-trodden path (Firecracker);
  keep MAP_SHARED file-backed OUT of scope (uffd shmem semantics differ).
- The CPU executor reads banks via raw pointers from C threads: faults raised
  there are serviced by the handler — verify no deadlock between the GIL, the
  handler thread, and executor workers (handler must never need the GIL).
- Bench guardrail: must not regress the mmap backend numbers; uffd vs mmap is
  a bench2 axis.

## Open questions for the bench to answer first

- Is post-PLE decode fault-bound or CPU-bound? (If CPU-bound, do GPU-fetch
  before the pager.)
- Actual eviction thrash rate at 10-12 layer spill under 64G cap.
- Row fill latency budget: rows are ~2.2-2.6 MiB; io_uring depth needed to
  hide N concurrent layer fills.
