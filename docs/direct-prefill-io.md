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
all requested payload bytes is an error. Application-level errors propagate
without buffered fallback. Buffer completion events retain the existing DMA
lifetime contract.

Linux documents filesystem-specific [direct-I/O alignment and concurrency
constraints](https://man7.org/linux/man-pages/man2/open.2.html). The target is
node-4's Linux 6.8 ext4 filesystem, using the existing loader's conservative
4 KiB alignment. This reader is created inside the serving worker and must
not run concurrently with a fork. Direct reads can sacrifice useful cache
hits and increase storage traffic, so correctness alone does not qualify
this as a throughput improvement.

The [ext4 read path](https://github.com/torvalds/linux/blob/v6.8/fs/ext4/file.c#L65)
can itself fall back to buffered I/O for some unsupported inode features.
Requesting `O_DIRECT` alone is therefore insufficient proof of transport.
During the direct start, `STATX_DIOALIGN` on all eight model files mapped by
the GPU worker reported 4-byte memory and 512-byte offset alignment, compatible
with this reader's 4 KiB requests. The experiment targets these checked files;
other filesystems and inode configurations are not qualified.
The [metadata record](../bench/results/4090-direct-io-file-alignment-20260905.json)
retains the worker PID, mapped paths, statx masks, and both alignment fields.

Validation must cover exact packed bytes, partial source views, unaligned
allocation addresses and file offsets, repeated asynchronous buffer reuse,
short reads, EOF, and real NVFP4 GEMM output bits. A file-residency check
compares buffered and direct reads using `mincore` on a private test file.
The performance gate uses whole client response time with diagnostics off,
matching memory geometry, full warmups, reversed starts, complete JSON,
and the existing long fidelity questions.

## Initial validation

At `2b6da47`, all 53 focused Linux CUDA checks in
`test_disk_prefill_staging.py` and `test_materialize_hot_slots.py` passed.
The same 53 checks passed under CUDA memcheck with zero errors. Direct and
buffered transports retain identical BF16 GEMM output bits at 16 and 512
tokens. The private-file residency check observed zero resident pages after
direct reads and all sixteen pages resident after buffered reads. This
confirms the intended cache behavior on the target filesystem, without
claiming a model speedup. The [log](../bench/results/4090-direct-io-validation-20260905.txt)
includes exact commands and verification of the restored original service.

## Cache-hit reuse candidates

If direct reads lose too much useful cache reuse, a later experiment could
read resident ranges from RAM and use direct I/O only for the rest. That is
a source-selection policy; it must still execute every selected expert.

`RWF_NOWAIT` is not by itself a cache-only read on the target kernel. Linux
6.8's [generic file read path](https://github.com/torvalds/linux/blob/v6.8/mm/filemap.c#L2558)
explicitly permits readahead with `IOCB_NOWAIT`, and its miss path can start
readahead before returning `EAGAIN`. A successful nonblocking API call is
therefore insufficient evidence that a policy avoids cache insertion.

The [mincore implementation](https://github.com/torvalds/linux/blob/v6.8/mm/mincore.c#L147)
provides a residency snapshot, but it can become stale immediately. It also
reports every page resident when the caller lacks ownership or write access
to the file. Any use as a transport hint must preserve correct reads when
the hint is stale or unavailable, and must measure the cost of inspecting
pages and splitting transfers. These candidates are not implemented or
performance-qualified by the direct-reader tests.
