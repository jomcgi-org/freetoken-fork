# Disk prefix cache page pressure on the 4090 disk tier, 2026-09-26

The 8192-token profile prefilled 100k tokens in 89.6 s in every qualification
arm, but 138.6 and 165.9 s on the live server. The arms ran with the disk
prefix cache off; the live server persists every prefix (500 GiB budget). A
matched arm on the same tree and manifest with the prefix cache on and a fresh
directory (`workspacecurve4-*`) reproduced the gap and its cause.

| 100k cold request | TTFT | NVMe read during request | memory stall | IO stall |
| --- | ---: | ---: | ---: | ---: |
| first (no 100k entry written yet) | 95.4 s | 12.7 GiB | 2.5% | 4.8% |
| second (after a 1.4 GiB entry write) | 151.8 s | 69.4 GiB | 15.0% | 22.8% |
| third | 143.4 s | 65.3 GiB | 13.4% | 18.2% |

All 27 responses passed with exact parity against the cache-off arms. Cached
repeats still hit (99,904 tokens, 1.8 to 2.7 s TTFT). Post-prefill decode in
the cold phase fell from 26.6 to 9.8 tok/s while the entry drained to disk.

Mechanism: the file-backed DISK expert banks are streamed through the page
cache once per prefill chunk, so their pages sit on the inactive list. A
prefix entry written or restored through the same page cache takes the active
list and displaces bank pages; the pager then re-reads most of the DISK banks
from NVMe on every chunk. `MemAvailable` never dropped below 28.8 GiB during
the run: the kernel had reclaimable pages, and it chose the bank pages.

Fix: after an entry's write is durable, and after a restore completes, advise
`POSIX_FADV_DONTNEED` over the entry so its pages return to the bank working
set. Entries are read at most once per restore, so no reuse is lost. The
change is in `disk_prefix_cache._drop_file_pages`, called from the writer and
from the restore completion path, and is best effort on platforms without the
advice. Measured as `workspacecurve5-*` with the same protocol.

## Result of the page-drop attempt

`workspacecurve5-*` ran the drop-after-write-and-restore change with the same
protocol. All 27 responses passed with exact parity, but the gap did not close:
100k cold TTFT was 96.3, 141.9 and 138.8 s with 14.5, 58.0 and 58.6 GiB read
from NVMe per request. The cache-off control on the same tree read 10.2, 7.6
and 6.8 GiB for its three 100k requests with no memory stall. The change was
reverted so the branch matches the qualified measurements.

The entry's own pages are therefore not what keeps the bank working set small.
After the first 100k entry is written, roughly 4 to 5 GiB less of the DISK
banks stays resident for every later chunk, and the effect persists across
requests. Candidates for the next session, in order: pinned host staging
buffers retained per written entry (`stage_hybrid_prefix_for_write` allocates
pinned copies and `save_file` serialises a second in-memory copy), the write
path's transient 3 GiB pushing the process into swap (the serve cgroup carried
1.2 GiB of swap after the day's runs), and radix-cache retention of the
previous 100k prefix on the GPU changing what the pager must stream. Each is
testable with the `workspacecurve4` protocol by instrumenting host RSS, pinned
allocations and `memory.swap.current` around the entry write.

Live-server consequence: with the prefix cache on, 100k cold TTFT is 139 to
166 s rather than the 90 s the cache-off arms show, still 1.6 to 1.9x faster
than the 261 s the previous 2048 profile measured under the same cache. Cached
repeats and continuation sessions keep their full benefit. Turning the cache
off would trade that for the cold number, which is the wrong trade for coding
sessions.

## Cgroup trace through a live 100k request

Sampling `freetoken-serve`'s cgroup every 15 s through one 100k request on
the live server (prefix cache on) showed two things. During the prefill the
kernel swapped the engine's anonymous memory almost entirely (anon 2.84 to
0.08 GiB, swap 0 to 2.78 GiB) while bank file pages grew from 29 to 58 GiB.
After the entry write completed, shmem rose from 27.08 to 29.33 GiB and stayed
there. A second test after a fresh restart with `vm.swappiness=10` and two
unused 100k prompts changed nothing on the swap side (anon still swapped,
147 s and 146 s TTFT, 62.6 and 64.3 GiB read) and repeated the shmem step:
27.07 to 29.33 to 31.58 GiB, one 2.25 GiB step per written 100k entry, never
released. Swappiness was restored to 60.

Pinned host memory for each written entry is therefore retained after the
write. The staging path allocates pinned tensors per entry
(`stage_hybrid_prefix_for_write`), and torch's pinned-host caching allocator
keeps such blocks rather than returning them to the OS, so every distinct
entry size grows the resident pinned set. Each step removes that much page
cache from the streamed DISK banks, which is the bank working-set shrink the
arms measured. The next change should stage entries through one reusable
pinned buffer (or pageable memory) and confirm shmem returns to its baseline
after the write, then re-run the `workspacecurve4` protocol.

## Fix: pageable staging released after the write, 2026-09-27

Confirmation on the live server before the change. After a day of traffic the
serve cgroup held 31.58 GiB shmem, two 2.25 GiB steps above its 27.08 GiB
baseline. A third distinct 100k prompt wrote its 1.42 GB entry and shmem did
not move: the write reused one of the two cached pinned blocks. That fits the
caching allocator exactly. Freed pinned blocks are kept, rounded up to a power
of two (a 1.32 GiB entry takes a 2 GiB block), and the writer thread held its
last job while blocked on the queue, so the next entry's staging could not
reuse the previous block and a second one was allocated. With the two blocks
resident, that cold 100k took 196.0 s and read about 126 GiB from NVMe.

The change (`f28f44b`, with a log fix in `8422299`):

- Staging allocates pageable host memory owned by the write job, never
  `pin_memory=True`. The device-to-host copies are synchronous, so no fence
  is needed, and the memory returns to the OS when the job is dropped.
- The writer emits safetensors directly from the staged tensors
  (`write_safetensors_file`) instead of `save_file`, which serialised a second
  full in-memory copy, and fsyncs the same descriptor.
- The writer drops each job before it blocks for the next one. The store
  reports `staged_bytes_live` and `staged_bytes_peak`.
- Tests: two entries of different sizes leave no live staged bytes, peak equals
  the larger entry rather than the sum, the idle writer holds no reference to
  its last job, staging never requests pinned memory, and the streaming writer
  round-trips every dtype through `safetensors`. Entries written by the old
  `save_file` path still restore.

`stagefix1-*` ran the `workspacecurve4` protocol (8192, prefix cache on, fresh
directory) on `f28f44b`. All 27 responses passed with exact parity against
`workspacecurve1-0`. Placement was unchanged: 20 GPU prefill layers, 28 DISK
layers, 82 HOT experts.

| 100k cold request | workspacecurve4 TTFT | NVMe read | stagefix1 TTFT | NVMe read |
| --- | ---: | ---: | ---: | ---: |
| run 0 | 95.4 s | 12.6 GiB | 93.4 s | 13.0 GiB |
| run 1 | 151.8 s | 69.0 GiB | 90.0 s | 10.0 GiB |
| run 2 | 143.4 s | 65.0 GiB | 88.3 s | 7.9 GiB |

The arm's cgroup never exceeded its 27.08 GiB shmem baseline across all 18 entry writes (sampled every 2 s). Means
against the cache-off control (`workspacecurve3-1`) and the unfixed cache-on
arm:

| Arm | 8k cold TTFT | 32k cold TTFT | 100k cold TTFT | 100k cold wall | 100k repeat wall | 100k post-prefill decode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cache off (workspacecurve3-1) | 12.97 s | 32.25 s | 89.64 s | 92.75 s | 6.14 s | 23.39 tok/s |
| cache on, unfixed (workspacecurve4-0) | 14.94 s | 37.92 s | 130.21 s | 138.65 s | 6.31 s | 19.59 tok/s |
| cache on, stagefix1 | 14.31 s | 35.93 s | 90.58 s | 98.06 s | 5.76 s | 22.97 tok/s |

Cold 100k TTFT with the cache on now matches the cache-off arm. The cold wall
is still about 5 s longer than cache-off because the entry write overlaps the
first decode tokens; cached repeats and later decode are unaffected.

Live-server trace after deploying `8422299` (fresh restart, then the finalist
verify requests as warm-up, then two fresh 100k prompts): shmem stayed at
27.08 GiB before, during and after both writes. The first request took 159.7 s
and read 78 GiB while the page cache refilled after the restart. The second
took 108.7 s with 25 GiB read. Staging the 1.32 GiB entry took 1.7 to 2.3 s on
the scheduler thread, inside the drain of the final prefill chunk.

### Tried and reverted: transient pinning

To take that copy off the scheduler thread, `e0e0c13` staged each entry into a
fresh private anonymous mapping, registered it with `cudaHostRegister` for
asynchronous copies, and unregistered it at the writer's fence. Measured as
`stagefix2-*`: parity held and staging fell to 30 to 150 ms for 8k and 32k
entries, but registering the 100k region itself took 0.5 to 1.9 s, 100k cold
TTFT did not improve (93.0 against 90.6 s mean), and cold-phase decode got
slower (8k cold wall 28.2 against 23.0 s, 32k 45.6 against 41.1 s). It was
reverted in `368d9c4`.

### Live server after the fix, and the remaining gap

After the budget sweep, serving came back on `368d9c4` (same tree as
`8422299`). The finalist verify requests passed, then three fresh 100k prompts
ran with 90 s idle between them (`livetrace-final-20260927.jsonl`, driver
`node4-livetrace.py`). Shmem never left 27.08 GiB.

| Live 100k cold request | TTFT | NVMe read (15 s sampling) |
| --- | ---: | ---: |
| first after restart | 179.6 s | 96.1 GiB |
| second | 114.5 s | 33.9 GiB |
| third | 105.9 s | 25.5 GiB |

The prefix-cache leak is gone, but live requests are still 15 to 25 s slower
than the arms. The steady 8192-token chunks take about 7 s on both. The
difference is the first two or three chunks after an idle gap: about 19, 15
and 9 s live against 9 to 10 s in the arms. The HOT adaptation log explains
it. Between live requests the idle ticks execute 193 swaps each toward the
decode aim, and every live prefill then starts at a decayed HOT hit rate of 26
to 27% with a 1,148-swap plan before recovering to 58%. In the arms, the 32k
rows triggered the adaptation's bandwidth back-off (tick interval raised to
12,000 tokens), and each 100k then started near 54% with at most 193 swaps per
tick. The live server had not backed off. The next lever for live cold TTFT is
therefore HOT adaptation across idle gaps, not the prefix cache or host budgets.
