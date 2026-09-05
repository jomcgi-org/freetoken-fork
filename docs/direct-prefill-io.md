# Direct file reads for selected DISK prefill

This experiment tests whether bypassing page-cache insertion during staged
prefill leaves more useful RAM residency for subsequent CPU decode. It adds
`DiskPrefillStaging(..., direct_io=True)` for controlled tests. Serving still
constructs the default buffered reader; there is no automatic policy change.

The reader opens each ordinary file bank with `O_DIRECT`. It aligns a view
inside each existing pinned allocation, reducing usable chunk capacity when
needed while retaining the governor's fixed 64 MiB total allocation. Reads
round outward to 4 KiB blocks, including unaligned payload heads and tails.
Only the exact selected row bytes reach GPU scratch. Published HOT reuse,
router selections, scale bytes, and GPU arithmetic are unchanged.

The implementation reuses the loader's aligned short-read helper. EOF before
all requested payload bytes is an error. Unsupported direct I/O also fails
explicitly, so a benchmark cannot silently measure buffered fallback. Buffer
completion events retain the existing DMA lifetime contract.

Linux documents filesystem-specific [direct-I/O alignment and concurrency
constraints](https://man7.org/linux/man-pages/man2/open.2.html). The target is
node-4's Linux 6.8 ext4 filesystem, using the existing loader's conservative
4 KiB alignment. This reader is created inside the serving worker and must
not run concurrently with a fork. Direct reads can sacrifice useful cache
hits and increase storage traffic, so correctness alone does not qualify
this as a throughput improvement.

Validation must cover exact packed bytes, partial source views, unaligned
allocation addresses and file offsets, repeated asynchronous buffer reuse,
short reads, EOF, and real NVFP4 GEMM output bits. A file-residency check
compares buffered and direct reads using `mincore` on a private test file.
The performance gate uses whole client response time with diagnostics off,
matching memory geometry, full warmups, reversed starts, complete JSON,
and the existing long fidelity questions.
