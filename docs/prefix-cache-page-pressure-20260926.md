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
