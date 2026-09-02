"""Crash-safe whole-prefix storage for hybrid KV and recurrent state.

Entries use safetensors so startup can validate metadata without reading tensor data and a
corrupt file cannot execute pickled code. Version 2 adds a page index for demand-loading QSA KV;
version 1 entries remain readable through the eager restore path.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import queue
import threading
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import save_file


FORMAT = "freetoken_disk_prefix"
VERSION = 2
_READABLE_VERSIONS = frozenset((1, VERSION))
_SUFFIX = ".safetensors"
_TMP_MARKER = ".tmp-"
BLOCK_INDEX_TENSOR = "qsa_block_index"


def _stable_json_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            field.name: _stable_json_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(k): _stable_json_value(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_stable_json_value(v) for v in value]
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def model_cache_identity(config) -> tuple[str, str, str]:
    """Return ``(identity, checkpoint_fingerprint, config_hash)`` for disk keys."""
    checkpoint_fingerprint = ""
    index_path = Path(config.model_path) / "freetoken_weight.json"
    try:
        with index_path.open() as f:
            checkpoint_fingerprint = str(json.load(f).get("fingerprint") or "")
    except (OSError, ValueError, TypeError):
        pass
    if not checkpoint_fingerprint:
        # Raw checkpoints have no FTW identity. Bind them to their config bytes and resolved
        # path so two different checkpoint directories cannot share entries accidentally.
        fallback = hashlib.sha256(str(Path(config.model_path).resolve()).encode())
        try:
            fallback.update((Path(config.model_path) / "config.json").read_bytes())
        except OSError:
            pass
        checkpoint_fingerprint = "raw-" + fallback.hexdigest()[:24]

    runtime_geometry = {
        "model_config": _stable_json_value(config.model_config),
        "dtype": str(config.dtype),
        "page_size": int(config.page_size),
        "tp_rank": int(config.tp_info.rank),
        "tp_size": int(config.tp_info.size),
    }
    encoded = json.dumps(runtime_geometry, sort_keys=True, separators=(",", ":")).encode()
    config_hash = hashlib.sha256(encoded).hexdigest()
    identity = hashlib.sha256(
        f"{checkpoint_fingerprint}:{config_hash}".encode()
    ).hexdigest()
    return identity, checkpoint_fingerprint, config_hash


def _token_bytes(token_ids: torch.Tensor) -> bytes:
    ids = token_ids.detach().to(device="cpu", dtype=torch.int64).contiguous()
    return ids.numpy().tobytes()


def token_chain_hash(identity: str, token_ids: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(identity.encode())
    digest.update(_token_bytes(token_ids))
    return digest.hexdigest()


def tensor_nbytes(tensors: Mapping[str, torch.Tensor]) -> int:
    return sum(int(t.numel()) * int(t.element_size()) for t in tensors.values())


@dataclass(frozen=True)
class DiskPrefixEntry:
    tensors: dict[str, torch.Tensor]
    length: int
    path: Path
    file_bytes: int
    restore_ms: float
    expert_profile: Any | None = None
    block_index: torch.Tensor | None = None
    restore_started_at: float = 0.0
    lazy_path_pinned: bool = False

    @property
    def supports_lazy_restore(self) -> bool:
        return (
            self.block_index is not None
            and "qsa_kv" not in self.tensors
            and "qsa_index" in self.tensors
        )


def make_block_index(length: int, page_size: int) -> torch.Tensor:
    """Inclusive token boundaries for page-granular QSA KV reads."""
    if length <= 0 or page_size <= 0 or length % page_size:
        raise ValueError(
            f"QSA block index needs positive page-aligned geometry, got "
            f"length={length}, page_size={page_size}"
        )
    return torch.arange(0, length + 1, page_size, dtype=torch.int64)


def validate_block_index(index: torch.Tensor, length: int) -> torch.Tensor:
    """Validate and normalize a stored block index without trusting advisory metadata."""
    normalized = index.detach().to(device="cpu", dtype=torch.int64).contiguous()
    if normalized.ndim != 1 or normalized.numel() < 2:
        raise ValueError("QSA block index must be a one-dimensional boundary vector")
    if int(normalized[0]) != 0 or int(normalized[-1]) != length:
        raise ValueError("QSA block index does not span the stored prefix")
    widths = normalized[1:] - normalized[:-1]
    if torch.any(widths <= 0) or torch.any(widths != widths[0]):
        raise ValueError("QSA block index must contain fixed-width increasing pages")
    return normalized


def priority_streaming_plan(
    num_blocks: int, hot_blocks: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return eager and background block ids, preserving the sink and newest KV first."""
    if num_blocks < 0 or hot_blocks < 0:
        raise ValueError("block counts must be non-negative")
    if num_blocks == 0:
        return (), ()
    recent_start = max(0, num_blocks - hot_blocks)
    eager = tuple(dict.fromkeys((0, *range(num_blocks - 1, recent_start - 1, -1))))
    eager_set = set(eager)
    streamed = tuple(block for block in range(num_blocks - 1, -1, -1) if block not in eager_set)
    return eager, streamed


class BlockPresence:
    """Thread-safe no-torn publication for one lazy restore's fixed block set."""

    ABSENT = 0
    LOADING = 1
    RESIDENT = 2

    def __init__(self, num_blocks: int) -> None:
        self._states = [self.ABSENT] * num_blocks
        self._condition = threading.Condition()
        self._error: BaseException | None = None

    @property
    def complete(self) -> bool:
        with self._condition:
            return all(state == self.RESIDENT for state in self._states)

    def resident(self, block: int) -> bool:
        with self._condition:
            return self._states[block] == self.RESIDENT

    def install(self, block: int, loader: Callable[[int], None]) -> bool:
        """Install one block once, and return only after its complete bytes are published."""
        with self._condition:
            while self._states[block] == self.LOADING and self._error is None:
                self._condition.wait()
            if self._error is not None:
                raise RuntimeError("lazy QSA KV restore failed") from self._error
            if self._states[block] == self.RESIDENT:
                return False
            self._states[block] = self.LOADING
        try:
            loader(block)
        except BaseException as exc:
            with self._condition:
                self._states[block] = self.ABSENT
                self._error = exc
                self._condition.notify_all()
            raise
        with self._condition:
            self._states[block] = self.RESIDENT
            self._condition.notify_all()
        return True


@dataclass(frozen=True)
class _WriteJob:
    token_ids: torch.Tensor
    tensors: dict[str, torch.Tensor]
    ready: Any | None


class DiskPrefixStore:
    """Byte-budgeted LRU store with one bounded asynchronous writer."""

    def __init__(
        self,
        directory: str | os.PathLike[str],
        budget_bytes: int,
        *,
        identity: str,
        checkpoint_fingerprint: str = "",
        config_hash: str = "",
        queue_size: int = 2,
        lazy_restore: bool = True,
        hot_blocks: int = 32,
    ) -> None:
        if budget_bytes <= 0:
            raise ValueError("disk prefix budget must be positive")
        if queue_size <= 0:
            raise ValueError("disk prefix queue size must be positive")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.budget_bytes = int(budget_bytes)
        self.identity = identity
        self.checkpoint_fingerprint = checkpoint_fingerprint
        self.config_hash = config_hash
        self.lazy_restore = bool(lazy_restore)
        self.hot_blocks = max(0, int(hot_blocks))
        self._entries: dict[tuple[int, str], Path] = {}
        self._lengths: set[int] = set()
        self._stats: dict[str, float | int] = {
            "hits": 0,
            "misses": 0,
            "bytes_restored": 0,
            "restore_ms": 0.0,
            "restore_eager_ms": 0.0,
            "blocks_faulted": 0,
            "blocks_streamed": 0,
            "first_token_after_restore_ms": 0.0,
            "prefill_ms_saved": 0.0,
            "fingerprint_mismatches": 0,
            "corrupt_entries": 0,
            "torn_writes": 0,
            "writes": 0,
            "write_drops": 0,
            "write_errors": 0,
            "lru_evictions": 0,
        }
        self._prefill_tokens_per_s = 0.0
        self._lock = threading.Lock()
        self._budget_lock = threading.Lock()
        self._lazy_restores: set[LazyKVRestore] = set()
        self._lazy_path_pins: dict[Path, int] = {}
        self._queue: queue.Queue[_WriteJob | None] = queue.Queue(maxsize=queue_size)
        self._closed = False
        self._scan()
        self._enforce_budget()
        self._thread = threading.Thread(
            target=self._writer_main, name="ft-disk-prefix", daemon=True
        )
        self._thread.start()

    def _metadata(
        self, token_ids: torch.Tensor, tensors: Mapping[str, torch.Tensor]
    ) -> dict[str, str]:
        with self._lock:
            prefill_rate = self._prefill_tokens_per_s
        metadata = {
            "format": FORMAT,
            "version": str(VERSION),
            "identity": self.identity,
            "checkpoint_fingerprint": self.checkpoint_fingerprint,
            "config_hash": self.config_hash,
            "token_count": str(token_ids.numel()),
            "token_hash": token_chain_hash(self.identity, token_ids),
            "prefill_tokens_per_s": str(prefill_rate),
        }
        if "expert_profile.version" in tensors:
            metadata["expert_profile_version"] = str(
                int(tensors["expert_profile.version"].reshape(-1)[0])
            )
        return metadata

    def _path_for(self, token_ids: torch.Tensor) -> Path:
        key = token_chain_hash(self.identity, token_ids)
        return self.directory / f"{token_ids.numel():012d}-{key}{_SUFFIX}"

    def _scan(self) -> None:
        for path in self.directory.iterdir():
            if path.is_file() and _TMP_MARKER in path.name:
                self._delete(path, "torn_writes")
        for path in self.directory.glob(f"*{_SUFFIX}"):
            try:
                with safe_open(str(path), framework="pt", device="cpu") as handle:
                    meta = handle.metadata() or {}
                if (
                    meta.get("format") != FORMAT
                    or int(meta.get("version", "-1")) not in _READABLE_VERSIONS
                ):
                    raise ValueError("unsupported disk prefix format")
                if meta.get("identity") != self.identity:
                    with self._lock:
                        self._stats["fingerprint_mismatches"] += 1
                    continue
                length = int(meta["token_count"])
                key = str(meta["token_hash"])
                if path != self.directory / f"{length:012d}-{key}{_SUFFIX}":
                    raise ValueError("disk prefix filename does not match metadata")
                self._entries[(length, key)] = path
                self._lengths.add(length)
                rate = float(meta.get("prefill_tokens_per_s", "0"))
                if math.isfinite(rate) and rate > 0:
                    self._prefill_tokens_per_s = rate
            except Exception:
                self._delete(path, "corrupt_entries")

    def _delete(self, path: Path, counter: str) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        finally:
            with self._lock:
                self._stats[counter] += 1

    def can_enqueue(self) -> bool:
        return not self._closed and not self._queue.full()

    def contains(self, token_ids: torch.Tensor) -> bool:
        key = token_chain_hash(self.identity, token_ids)
        with self._lock:
            return (int(token_ids.numel()), key) in self._entries

    def enqueue(
        self,
        token_ids: torch.Tensor,
        tensors: Mapping[str, torch.Tensor],
        *,
        ready: Any | None = None,
    ) -> bool:
        if self._closed:
            return False
        ids = token_ids.detach().to(device="cpu", dtype=torch.int32).contiguous()
        payload = {name: value.detach() for name, value in tensors.items()}
        payload["token_ids"] = ids
        try:
            self._queue.put_nowait(_WriteJob(ids, payload, ready))
            return True
        except queue.Full:
            with self._lock:
                self._stats["write_drops"] += 1
            return False

    def note_write_drop(self) -> None:
        with self._lock:
            self._stats["write_drops"] += 1

    def _writer_main(self) -> None:
        while True:
            job = self._queue.get()
            tmp: Path | None = None
            try:
                if job is None:
                    return
                if job.ready is not None:
                    job.ready.synchronize()
                tensors = {
                    name: value.detach().to(device="cpu").contiguous()
                    for name, value in job.tensors.items()
                }
                path = self._path_for(job.token_ids)
                tmp = path.with_name(path.name + _TMP_MARKER + uuid.uuid4().hex)
                save_file(tensors, str(tmp), metadata=self._metadata(job.token_ids, tensors))
                with tmp.open("rb") as f:
                    os.fsync(f.fileno())
                os.replace(tmp, path)
                try:
                    dir_fd = os.open(self.directory, os.O_RDONLY)
                    try:
                        os.fsync(dir_fd)
                    finally:
                        os.close(dir_fd)
                except OSError:
                    pass
                key = token_chain_hash(self.identity, job.token_ids)
                with self._lock:
                    self._entries[(job.token_ids.numel(), key)] = path
                    self._lengths.add(job.token_ids.numel())
                    self._stats["writes"] += 1
                self._enforce_budget()
            except Exception:
                if tmp is not None:
                    try:
                        tmp.unlink(missing_ok=True)
                    except OSError:
                        pass
                with self._lock:
                    self._stats["write_errors"] += 1
            finally:
                self._queue.task_done()

    def lookup_longest(
        self,
        token_ids: torch.Tensor,
        *,
        longer_than: int = 0,
        record: bool = True,
        pin_lazy_path: bool = False,
    ) -> DiskPrefixEntry | None:
        started = time.perf_counter()
        ids = token_ids.detach().to(device="cpu", dtype=torch.int32).contiguous()
        with self._lock:
            lengths = sorted(
                (n for n in self._lengths if longer_than < n <= ids.numel()), reverse=True
            )
        for length in lengths:
            prefix = ids[:length]
            key = token_chain_hash(self.identity, prefix)
            with self._lock:
                path = self._entries.get((length, key))
            if path is None:
                continue
            path_pinned = False
            try:
                with safe_open(str(path), framework="pt", device="cpu") as handle:
                    meta = handle.metadata() or {}
                    if meta.get("identity") != self.identity:
                        with self._lock:
                            self._stats["fingerprint_mismatches"] += 1
                        continue
                    keys = set(handle.keys())
                    block_index = None
                    if (
                        self.lazy_restore
                        and BLOCK_INDEX_TENSOR in keys
                        and "qsa_kv" in keys
                        and "qsa_index" in keys
                    ):
                        block_index = validate_block_index(
                            handle.get_tensor(BLOCK_INDEX_TENSOR), length
                        )
                        tensors = {
                            name: handle.get_tensor(name)
                            for name in keys
                            if name not in ("qsa_kv", BLOCK_INDEX_TENSOR)
                        }
                    else:
                        tensors = {name: handle.get_tensor(name) for name in keys}
                    path_pinned = pin_lazy_path and block_index is not None
                    if path_pinned:
                        self._pin_path(path)
                stored_ids = tensors.get("token_ids")
                if stored_ids is None or not torch.equal(stored_ids.to(torch.int32), prefix):
                    raise ValueError("stored token ids do not match request prefix")
                if meta.get("token_hash") != key:
                    raise ValueError("stored token hash does not match verified token ids")
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                file_bytes = path.stat().st_size
                os.utime(path, None)
                expert_profile = None
                try:
                    from freetoken.moe.session_profile import SessionExpertProfile

                    expert_profile = SessionExpertProfile.from_tensors(tensors)
                except (KeyError, TypeError, ValueError):
                    # Advisory data must never make otherwise valid KV/GDN state unusable.
                    expert_profile = None
                entry = DiskPrefixEntry(
                    tensors,
                    length,
                    path,
                    file_bytes,
                    elapsed_ms,
                    expert_profile,
                    block_index,
                    started,
                    path_pinned,
                )
                if record:
                    self.record_restore(entry, length - longer_than)
                return entry
            except Exception:
                if path_pinned:
                    self._release_path_pin(path)
                with self._lock:
                    self._entries.pop((length, key), None)
                    if not any(n == length for n, _ in self._entries):
                        self._lengths.discard(length)
                self._delete(path, "corrupt_entries")
        with self._lock:
            self._stats["misses"] += 1
        return None

    def lookup_profile_longest(
        self, token_ids: torch.Tensor, *, longer_than: int = 0
    ) -> tuple[int, Any] | None:
        """Read only the tiny advisory field for admission-time warming.

        Unlike ``lookup_longest``, this does not materialize the KV/GDN tensors and does
        not affect restore hit/miss accounting. Missing or malformed profile fields are
        treated exactly like an older entry that predates the field.
        """
        from freetoken.moe.session_profile import PROFILE_TENSOR_NAMES, SessionExpertProfile

        ids = token_ids.detach().to(device="cpu", dtype=torch.int32).contiguous()
        with self._lock:
            lengths = sorted(
                (n for n in self._lengths if longer_than < n <= ids.numel()), reverse=True
            )
        for length in lengths:
            key = token_chain_hash(self.identity, ids[:length])
            with self._lock:
                path = self._entries.get((length, key))
            if path is None:
                continue
            try:
                with safe_open(str(path), framework="pt", device="cpu") as handle:
                    meta = handle.metadata() or {}
                    if meta.get("identity") != self.identity or meta.get("token_hash") != key:
                        continue
                    keys = set(handle.keys())
                    if "expert_profile.version" not in keys:
                        continue
                    tensors = {
                        name: handle.get_tensor(name)
                        for name in PROFILE_TENSOR_NAMES
                        if name in keys
                    }
                profile = SessionExpertProfile.from_tensors(tensors)
                if profile is not None:
                    return length, profile
            except (OSError, RuntimeError, TypeError, ValueError):
                continue
        return None

    def observe_prefill_rate(self, tokens_per_s: float) -> None:
        if not math.isfinite(tokens_per_s) or tokens_per_s <= 0:
            return
        with self._lock:
            if self._prefill_tokens_per_s <= 0:
                self._prefill_tokens_per_s = float(tokens_per_s)
            else:
                self._prefill_tokens_per_s = (
                    0.8 * self._prefill_tokens_per_s + 0.2 * float(tokens_per_s)
                )

    def note_restore_install(self, elapsed_ms: float) -> None:
        with self._lock:
            self._stats["restore_ms"] += float(elapsed_ms)

    def note_restore_eager(self, elapsed_ms: float) -> None:
        with self._lock:
            self._stats["restore_eager_ms"] += float(elapsed_ms)

    def note_lazy_blocks(self, *, faulted: int = 0, streamed: int = 0) -> None:
        with self._lock:
            self._stats["blocks_faulted"] += int(faulted)
            self._stats["blocks_streamed"] += int(streamed)

    def note_first_token_after_restore(self, elapsed_ms: float) -> None:
        with self._lock:
            self._stats["first_token_after_restore_ms"] = float(elapsed_ms)

    def record_restore(self, entry: DiskPrefixEntry, tokens_restored: int) -> None:
        with self._lock:
            self._stats["hits"] += 1
            self._stats["bytes_restored"] += entry.file_bytes
            self._stats["restore_ms"] += entry.restore_ms
            if self._prefill_tokens_per_s > 0:
                self._stats["prefill_ms_saved"] += (
                    tokens_restored / self._prefill_tokens_per_s * 1000.0
                )

    def invalidate(self, path: Path) -> None:
        with self._lock:
            for entry_key, entry_path in list(self._entries.items()):
                if entry_path == path:
                    self._entries.pop(entry_key, None)
            self._lengths = {n for n, _ in self._entries}
        self._delete(path, "corrupt_entries")

    def stats(self) -> dict[str, float | int]:
        with self._lock:
            result = dict(self._stats)
            result["prefill_tokens_per_s"] = self._prefill_tokens_per_s
            result["queued_writes"] = self._queue.qsize()
        return result

    def _enforce_budget(self) -> None:
        with self._budget_lock:
            self._enforce_budget_locked()

    def _enforce_budget_locked(self) -> None:
        with self._lock:
            pinned_paths = {
                restore.entry.path for restore in self._lazy_restores
            } | set(self._lazy_path_pins)
        records: list[tuple[int, int, Path]] = []
        for path in self.directory.glob(f"*{_SUFFIX}"):
            try:
                stat = path.stat()
            except OSError:
                continue
            records.append((stat.st_mtime_ns, stat.st_size, path))
        total = sum(size for _, size, _ in records)
        for _, size, path in sorted(records):
            if total <= self.budget_bytes:
                break
            if path in pinned_paths:
                continue
            try:
                path.unlink()
                total -= size
                with self._lock:
                    for entry_key, entry_path in list(self._entries.items()):
                        if entry_path == path:
                            self._entries.pop(entry_key, None)
                    self._lengths = {n for n, _ in self._entries}
                    self._stats["lru_evictions"] += 1
            except FileNotFoundError:
                continue
            except OSError:
                continue

    def flush(self) -> None:
        self._queue.join()

    def close(self, *, wait: bool = True) -> None:
        if self._closed:
            return
        if wait:
            self.flush()
        self._closed = True
        self._queue.put(None)
        self._thread.join()
        if wait:
            with self._lock:
                lazy_restores = tuple(self._lazy_restores)
            for restore in lazy_restores:
                restore.join()

    def register_lazy_restore(self, restore: LazyKVRestore) -> None:
        with self._lock:
            self._lazy_restores.add(restore)

    def release_entry_pin(self, entry: DiskPrefixEntry) -> None:
        if entry.lazy_path_pinned:
            self._release_path_pin(entry.path)

    def _pin_path(self, path: Path) -> None:
        # Serialize with the complete LRU pass so it cannot snapshot the old pin set and then
        # unlink this path after lookup has committed to a lazy reader.
        with self._budget_lock:
            if not path.exists():
                raise FileNotFoundError(path)
            with self._lock:
                self._lazy_path_pins[path] = self._lazy_path_pins.get(path, 0) + 1

    def _release_path_pin(self, path: Path) -> None:
        with self._lock:
            count = self._lazy_path_pins.get(path, 0)
            if count <= 1:
                self._lazy_path_pins.pop(path, None)
            else:
                self._lazy_path_pins[path] = count - 1

    def finish_lazy_restore(self, restore: LazyKVRestore) -> None:
        with self._lock:
            self._lazy_restores.discard(restore)
        self._enforce_budget()


def capture_hybrid_prefix_tensors(
    kv_cache,
    linear_pool,
    *,
    kv_indices: torch.Tensor,
    linear_slot: int,
    table_idx: int | None,
) -> dict[str, torch.Tensor]:
    """Capture the same mutable state families used by MTP plus paged QSA data."""
    from freetoken.spec_decode import snapshot_verify_state

    length = int(kv_indices.numel())
    flat = kv_cache._kv_buffer.flatten(2, 3)
    tensors = {"qsa_kv": flat.index_select(2, kv_indices.to(torch.long))}
    page_size = getattr(kv_cache, "_page_size", None)
    if page_size is not None:
        tensors[BLOCK_INDEX_TENSOR] = make_block_index(length, int(page_size))
    cmp_buffer = getattr(kv_cache, "_cmp_k_buffer", None)
    if cmp_buffer is not None:
        ratio = int(kv_cache.index_ratio)
        if length % ratio:
            raise ValueError(f"QSA disk prefix length {length} is not group-aligned to {ratio}")
        rows = kv_indices[::ratio].to(torch.long) // ratio
        tensors["qsa_index"] = cmp_buffer.index_select(1, rows)
    req = type("DiskSnapshotReq", (), {
        "linear_slot_idx": linear_slot,
        "table_idx": 0 if table_idx is None else table_idx,
    })()
    snapshot = snapshot_verify_state(
        linear_pool, kv_cache if table_idx is not None else None, req
    )
    for name in ("conv", "recurrent", "qsa_pending"):
        if name in snapshot:
            tensors[name] = snapshot[name]
    for name, value in snapshot.get("slot_states", {}).items():
        tensors[f"slot_state.{name}"] = value
    return tensors


def restore_hybrid_prefix_tensors(
    kv_cache,
    linear_pool,
    tensors: Mapping[str, torch.Tensor],
    *,
    kv_indices: torch.Tensor,
    linear_slot: int,
    restore_kv: bool = True,
) -> torch.Tensor | None:
    """Install paged data and recurrent state, returning table-local QSA pending state."""
    device = kv_cache.device
    flat = kv_cache._kv_buffer.flatten(2, 3)
    locations = kv_indices.to(torch.long)
    if restore_kv:
        source_kv = tensors["qsa_kv"]
        kv_scratch = torch.empty(source_kv[:, 0].shape, dtype=source_kv.dtype, device=device)
        for layer in range(flat.shape[1]):
            kv_scratch.copy_(source_kv[:, layer])
            flat[:, layer].index_copy_(1, locations, kv_scratch)
    if "qsa_index" in tensors:
        ratio = int(kv_cache.index_ratio)
        rows = locations[::ratio] // ratio
        source_index = tensors["qsa_index"]
        index_scratch = torch.empty(
            source_index[0].shape, dtype=source_index.dtype, device=device
        )
        for layer in range(kv_cache._cmp_k_buffer.shape[0]):
            index_scratch.copy_(source_index[layer])
            kv_cache._cmp_k_buffer[layer].index_copy_(0, rows, index_scratch)
    linear_pool.conv_states[:, linear_slot].copy_(tensors["conv"])
    linear_pool.recurrent_states[:, linear_slot].copy_(
        tensors["recurrent"]
    )
    for name, target in linear_pool.slot_states.items():
        target[:, linear_slot].copy_(tensors[f"slot_state.{name}"])
    return tensors.get("qsa_pending")


class LazyKVRestore:
    """Demand-load one disk prefix's QSA KV pages into their allocated device pages."""

    def __init__(
        self,
        kv_cache,
        entry: DiskPrefixEntry,
        *,
        kv_indices: torch.Tensor,
        already_resident_tokens: int,
        hot_blocks: int,
        on_block: Callable[[str], None] | None = None,
        on_complete: Callable[[], None] | None = None,
    ) -> None:
        if entry.block_index is None:
            raise ValueError("lazy QSA KV restore needs a stored block index")
        self.kv_cache = kv_cache
        self.entry = entry
        self.block_index = validate_block_index(entry.block_index, entry.length)
        self.page_size = int(self.block_index[1] - self.block_index[0])
        cache_page_size = getattr(kv_cache, "_page_size", self.page_size)
        if self.page_size != int(cache_page_size):
            raise ValueError(
                f"stored QSA page size {self.page_size} does not match cache page size "
                f"{cache_page_size}"
            )
        if already_resident_tokens % self.page_size:
            raise ValueError("resident prefix length must be page-aligned")
        if not 0 <= already_resident_tokens <= entry.length:
            raise ValueError("resident prefix length is outside the stored entry")
        locations = kv_indices.detach().to(device="cpu", dtype=torch.int64).contiguous()
        if locations.numel() != entry.length:
            raise ValueError("lazy QSA KV restore locations do not span the entry")
        logical_pages = locations.view(-1, self.page_size)
        offsets = torch.arange(self.page_size, dtype=torch.int64)
        if torch.any(logical_pages != logical_pages[:, :1] + offsets):
            raise ValueError("lazy QSA KV restore needs contiguous physical cache pages")
        if torch.any(logical_pages[:, 0] % self.page_size):
            raise ValueError("lazy QSA KV restore needs page-aligned physical cache pages")
        self.physical_pages = tuple(
            int(locations[offset]) // self.page_size
            for offset in range(0, entry.length, self.page_size)
        )
        if len(set(self.physical_pages)) != len(self.physical_pages):
            raise ValueError("lazy QSA KV restore cannot alias physical cache pages")
        self.first_lazy_block = already_resident_tokens // self.page_size
        self.presence = BlockPresence(len(self.physical_pages))
        for block in range(self.first_lazy_block):
            self.presence.install(block, lambda _: None)
        self._on_block = on_block
        self._on_complete = on_complete
        self._completed_callback = False
        self._callback_lock = threading.Lock()
        self._copy_stream = (
            torch.cuda.Stream(device=self.kv_cache.device)
            if self.kv_cache.device.type == "cuda"
            else None
        )
        eager, streamed = priority_streaming_plan(
            len(self.physical_pages) - self.first_lazy_block, hot_blocks
        )
        self.eager_blocks = tuple(self.first_lazy_block + block for block in eager)
        self.stream_blocks = tuple(self.first_lazy_block + block for block in streamed)
        self._thread: threading.Thread | None = None

    @property
    def complete(self) -> bool:
        return self.presence.complete

    def install_eager(self) -> None:
        if not self.eager_blocks:
            return
        with safe_open(str(self.entry.path), framework="pt", device="cpu") as handle:
            for block in self.eager_blocks:
                self._install(
                    block,
                    "eager",
                    loader=lambda selected, source=handle: self._load_block(
                        selected, source
                    ),
                )

    def start_background(self) -> None:
        if not self.stream_blocks:
            self._finish_once()
            return
        self._thread = threading.Thread(
            target=self._stream_main,
            name="ft-lazy-kv-restore",
            daemon=True,
        )
        self._thread.start()

    def set_on_complete(self, callback: Callable[[], None]) -> None:
        self._on_complete = callback

    def join(self) -> None:
        if self._thread is not None:
            self._thread.join()

    def ensure_blocks(self, blocks: Sequence[int]) -> None:
        for block in dict.fromkeys(int(block) for block in blocks if block >= 0):
            if block < self.first_lazy_block or block >= len(self.physical_pages):
                continue
            self._install(block, "fault")
        if self.complete:
            self._finish_once()

    def _stream_main(self) -> None:
        try:
            with safe_open(
                str(self.entry.path), framework="pt", device="cpu"
            ) as handle:
                for block in self.stream_blocks:
                    self._install(
                        block,
                        "stream",
                        loader=lambda selected, source=handle: self._load_block(
                            selected, source
                        ),
                    )
        except BaseException:
            # BlockPresence retains the failure for the request's next synchronous fault.
            pass
        finally:
            self._finish_once()

    def _install(
        self,
        block: int,
        source: str,
        *,
        loader: Callable[[int], None] | None = None,
    ) -> None:
        installed = self.presence.install(block, loader or self._load_block)
        if installed and source in ("fault", "stream") and self._on_block is not None:
            self._on_block(source)

    def _load_block(self, block: int, handle=None) -> None:
        start = int(self.block_index[block])
        end = int(self.block_index[block + 1])
        physical = self.physical_pages[block]
        locations = torch.arange(
            physical * self.page_size,
            physical * self.page_size + (end - start),
            dtype=torch.int64,
            device=self.kv_cache.device,
        )
        flat = self.kv_cache._kv_buffer.flatten(2, 3)
        stream_context = (
            torch.cuda.stream(self._copy_stream)
            if self._copy_stream is not None
            else nullcontext()
        )
        with stream_context:
            if handle is None:
                with safe_open(str(self.entry.path), framework="pt", device="cpu") as opened:
                    self._copy_block(opened.get_slice("qsa_kv"), flat, locations, start, end)
            else:
                self._copy_block(handle.get_slice("qsa_kv"), flat, locations, start, end)
        if self._copy_stream is not None:
            # Presence is published only after the dedicated copy stream has installed every
            # layer. A demand fault therefore returns before the engine stream launches gather.
            self._copy_stream.synchronize()

    def _copy_block(self, source, flat, locations, start: int, end: int) -> None:
        for layer in range(flat.shape[1]):
            page = source[:, layer, start:end]
            scratch = page.to(device=self.kv_cache.device)
            flat[:, layer].index_copy_(1, locations, scratch)

    def _finish_once(self) -> None:
        if not self.complete:
            return
        with self._callback_lock:
            if self._completed_callback:
                return
            self._completed_callback = True
        if self._on_complete is not None:
            self._on_complete()


def stage_tensors_for_write(
    tensors: Mapping[str, torch.Tensor], device: torch.device
) -> tuple[dict[str, torch.Tensor], Any | None]:
    """Issue nonblocking device-to-host copies and return their completion fence."""
    if device.type != "cuda":
        return {name: value.detach().cpu().clone() for name, value in tensors.items()}, None
    staged: dict[str, torch.Tensor] = {}
    for name, value in tensors.items():
        host = torch.empty_like(value, device="cpu", pin_memory=True)
        host.copy_(value, non_blocking=True)
        staged[name] = host
    ready = torch.cuda.Event()
    ready.record()
    return staged, ready


def stage_hybrid_prefix_for_write(
    kv_cache,
    linear_pool,
    *,
    kv_indices: torch.Tensor,
    linear_slot: int,
    table_idx: int | None,
    extra_tensors: Mapping[str, torch.Tensor] | None = None,
) -> tuple[dict[str, torch.Tensor], Any | None]:
    """Stage a hybrid prefix with one reusable per-layer GPU gather buffer."""
    from freetoken.spec_decode import request_state_views

    device = kv_cache.device
    if device.type != "cuda":
        tensors = capture_hybrid_prefix_tensors(
            kv_cache,
            linear_pool,
            kv_indices=kv_indices,
            linear_slot=linear_slot,
            table_idx=table_idx,
        )
        tensors.update(extra_tensors or {})
        return stage_tensors_for_write(tensors, device)

    locations = kv_indices.to(torch.long)
    flat = kv_cache._kv_buffer.flatten(2, 3)
    kv_shape = (flat.shape[0], flat.shape[1], locations.numel(), *flat.shape[3:])
    host_kv = torch.empty(kv_shape, dtype=flat.dtype, device="cpu", pin_memory=True)
    scratch = torch.empty(
        (flat.shape[0], locations.numel(), *flat.shape[3:]),
        dtype=flat.dtype,
        device=device,
    )
    for layer in range(flat.shape[1]):
        torch.index_select(flat[:, layer], 1, locations, out=scratch)
        for slab in range(flat.shape[0]):
            host_kv[slab, layer].copy_(scratch[slab], non_blocking=True)
    staged: dict[str, torch.Tensor] = {"qsa_kv": host_kv}
    page_size = getattr(kv_cache, "_page_size", None)
    if page_size is not None:
        staged[BLOCK_INDEX_TENSOR] = make_block_index(
            int(locations.numel()), int(page_size)
        )

    cmp_buffer = getattr(kv_cache, "_cmp_k_buffer", None)
    if cmp_buffer is not None:
        ratio = int(kv_cache.index_ratio)
        if locations.numel() % ratio:
            raise ValueError(
                f"QSA disk prefix length {locations.numel()} is not group-aligned to {ratio}"
            )
        rows = locations[::ratio] // ratio
        host_index = torch.empty(
            (cmp_buffer.shape[0], rows.numel(), cmp_buffer.shape[2]),
            dtype=cmp_buffer.dtype,
            device="cpu",
            pin_memory=True,
        )
        index_scratch = torch.empty(
            (rows.numel(), cmp_buffer.shape[2]), dtype=cmp_buffer.dtype, device=device
        )
        for layer in range(cmp_buffer.shape[0]):
            torch.index_select(cmp_buffer[layer], 0, rows, out=index_scratch)
            host_index[layer].copy_(index_scratch, non_blocking=True)
        staged["qsa_index"] = host_index

    req = type("DiskSnapshotReq", (), {
        "linear_slot_idx": linear_slot,
        "table_idx": 0 if table_idx is None else table_idx,
    })()
    views = request_state_views(
        linear_pool, kv_cache if table_idx is not None else None, req
    )
    for name in ("conv", "recurrent", "qsa_pending"):
        if name in views:
            host = torch.empty(
                views[name].shape,
                dtype=views[name].dtype,
                device="cpu",
                pin_memory=True,
            )
            host.copy_(views[name], non_blocking=True)
            staged[name] = host
    for name, value in views.get("slot_states", {}).items():
        host = torch.empty(
            value.shape, dtype=value.dtype, device="cpu", pin_memory=True
        )
        host.copy_(value, non_blocking=True)
        staged[f"slot_state.{name}"] = host

    for name, value in (extra_tensors or {}).items():
        host = torch.empty(
            value.shape, dtype=value.dtype, device="cpu", pin_memory=True
        )
        host.copy_(value, non_blocking=True)
        staged[name] = host

    ready = torch.cuda.Event()
    ready.record()
    return staged, ready


__all__ = [
    "BLOCK_INDEX_TENSOR",
    "BlockPresence",
    "DiskPrefixEntry",
    "DiskPrefixStore",
    "LazyKVRestore",
    "capture_hybrid_prefix_tensors",
    "make_block_index",
    "model_cache_identity",
    "priority_streaming_plan",
    "restore_hybrid_prefix_tensors",
    "stage_hybrid_prefix_for_write",
    "stage_tensors_for_write",
    "tensor_nbytes",
    "token_chain_hash",
    "validate_block_index",
]
