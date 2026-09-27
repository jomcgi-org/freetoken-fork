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
