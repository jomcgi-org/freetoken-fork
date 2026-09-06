# Reclaiming redundant HOT file pages

`--moe-hot-host-cache reclaim` is an opt-in native NVFP4 experiment. The default
is `retain`. Ordinary GPU layers keep their pinned host sources and existing
transfer path. Expert selection, HOT capacity, quantization and GPU arithmetic
are unchanged.

After complete rows have been staged and their HOT mapping published, request
reclamation of their redundant checkpoint-file pages. GPU installation reads
the separate pinned staging allocation, so reclamation adds no CUDA wait.
Startup reloads and later adaptation use the same policy. Partially cancelled
updates reclaim only the rows actually installed; retired and abandoned
incoming owners are excluded.

The file mapping remains valid. Later CPU reads or demotion transparently fault
the original immutable bytes back from disk. Only full pages contained in the
selected row union are advised away, preserving shared boundary pages and
small scale sidecars. Anonymous, pinned, locked, tmpfs, UFFD and UVM banks are
excluded. An OS advice failure keeps serving and emits at most one warning.

The initial option requires `--moe-backend offload`, `--moe-disk-prefill staged`,
`--moe-disk-decode cpu`, `--moe-prefill-hot-split on`, madvise paging and no tmpfs
mirror. Published HOT rows already bypass file reads during staged prefill and
are excluded from CPU session warming.

The kernel may retain advised pages because of other mappings or readers.
Advised bytes are not a measurement of reclaimed RAM. Validate actual resident
pages separately with a diagnostic census; keep that census out of production
and out of wall-time measurements. Reclamation can add syscall cost and future
page faults, so keep this option disabled until a controlled complete-model
comparison demonstrates a throughput benefit with matching work and outputs.

The explicit diagnostic hook in `bench/hot-host-cache-census.py` records the
first nonempty reclamation per cache. Set `FREETOKEN_HOT_HOST_CACHE_CENSUS_DIR`
to a private output directory and call its `install()` before engine creation.
It queries actual file-page residency and hashes the GPU HOT weights before
and after reclamation. This synchronizes and copies GPU data. Remove the hook
and its environment variable for non-debug wall-time comparisons.

Targeted Linux CPU checks and CUDA staging/publication checks pass, including
actual file-page reclamation and exact cached weight bytes. Use the separate
wall comparison in #54 to qualify a serving configuration before enabling
reclamation. Keep full-model census records and wall measurements private.
