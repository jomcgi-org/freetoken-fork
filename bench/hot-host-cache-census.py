"""Explicit diagnostic hook for the first nonempty HOT reclamation per cache.

Call install() before engine construction with FREETOKEN_HOT_HOST_CACHE_CENSUS_DIR
set to a private output directory. GPU hashing synchronizes and copies weights;
never install this hook for a wall-time benchmark or ordinary serving.
"""

import ctypes
import hashlib
import json
import mmap
import os
from pathlib import Path
import weakref

import torch

from freetoken.moe.host_banks import _tensor_host_bank, coalesced_row_ranges
from freetoken.moe.offload_cache import OffloadMoeCache


def published_rows(cache, swaps):
    return tuple(s for s in swaps
                 if cache._hot_slot_owners[s.layer_id][s.row] == s.incoming_expert)


def gpu_digest(cache, swaps):
    digest = hashlib.sha256()
    for swap in swaps:
        slot = cache._hot_slot_for_row[swap.layer_id][swap.row]
        for name in cache.bank_schema:
            raw = cache.bank_caches[name][slot].detach().reshape(-1).view(torch.uint8).cpu()
            digest.update(raw.numpy().tobytes())
    return digest.hexdigest()


def resident_hot_bytes(cache, swaps):
    layers = {}
    for swap in swaps:
        layers.setdefault(swap.layer_id, set()).add(swap.incoming_expert)
    libc = ctypes.CDLL(None, use_errno=True)
    page = mmap.PAGESIZE
    total = 0
    for layer, rows in layers.items():
        for name in cache.bank_schema:
            tensor = cache.bank_sources[name][layer]
            owner = _tensor_host_bank(tensor)
            if owner is None or not owner._disk or owner._uffd or owner._tmpfs_backed:
                continue
            count = (owner._mapping_length + page - 1) // page
            vec = (ctypes.c_ubyte * count)()
            if libc.mincore(ctypes.c_void_p(owner._mapping_addr),
                            ctypes.c_size_t(owner._mapping_length), vec):
                raise OSError(ctypes.get_errno(), "mincore failed")
            ranges = coalesced_row_ranges(
                rows, tensor.stride(0) * tensor.element_size(),
                limit=tensor.numel() * tensor.element_size(),
                base_offset=owner._view_offset + tensor.data_ptr() - owner.addr,
            )
            for start, length in ranges:
                total += page * sum(bool(vec[i] & 1) for i in
                                    range((start + page - 1) // page, (start + length) // page))
    return total


def install():
    directory = os.environ.get("FREETOKEN_HOT_HOST_CACHE_CENSUS_DIR")
    if not directory:
        raise RuntimeError("explicit census output directory is required")
    root = Path(directory)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    original = OffloadMoeCache._reclaim_hot_host_rows
    seen = weakref.WeakValueDictionary()

    def census(cache, swaps):
        rows = published_rows(cache, swaps)
        if not rows or seen.get(id(cache)) is cache:
            return original(cache, swaps)
        seen[id(cache)] = cache
        before_digest = gpu_digest(cache, rows)
        before = resident_hot_bytes(cache, rows)
        advised = original(cache, swaps)
        after = resident_hot_bytes(cache, rows)
        after_digest = gpu_digest(cache, rows)
        record = dict(pid=os.getpid(), rows=len(rows), layers=len({s.layer_id for s in rows}),
                      resident_hot_bytes_before=before, resident_hot_bytes_after=after,
                      advised_bytes=advised, gpu_bytes_equal=before_digest == after_digest,
                      gpu_sha256_before=before_digest, gpu_sha256_after=after_digest,
                      diagnostic_only=True, wall_time_qualified=False)
        path = root / (str(os.getpid()) + ".jsonl")
        with os.fdopen(os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600), "w") as f:
            f.write(json.dumps(record) + "\n")
        if not record['gpu_bytes_equal']:
            raise RuntimeError("GPU HOT bytes changed during host-cache reclamation")
        return advised

    OffloadMoeCache._reclaim_hot_host_rows = census
