"""GPU-free coverage for PLE io_uring layouts, staging, padding, and failure policy."""

from __future__ import annotations

import ctypes
import errno
import os
import sys
import time
from collections import deque
from types import SimpleNamespace

import pytest
import torch

from freetoken.models.qwen4_exp.ple_uring import (
    UringTable,
    _load_uring_extension,
    _make_store,
    resolve_uring_source,
)
from freetoken.models.qwen4_exp.weight import _ple_table_layout

from .test_weight import (
    NGRAM_ROWS,
    NGRAM_SHARDS,
    QUANT_NGRAM_DIM,
    _quantized_ple_checkpoint,
)


class _FakeRowStore:
    """pread test double for the Linux-only native reader."""

    def __init__(self, **kwargs):
        self.paths = kwargs["paths"]
        self.extent_file = kwargs["extent_file"]
        self.extent_base = kwargs["extent_base"]
        self.rows_per_extent = kwargs["rows_per_extent"]
        self.row_bytes = kwargs["row_bytes"]
        self._queue_depth = kwargs["queue_depth"]

    def read_rows(self, ids_address, count, destination, destination_stride):
        ids = (ctypes.c_int64 * count).from_address(ids_address)
        seen = set()
        for index, row_id in enumerate(ids):
            shard, local = divmod(row_id, self.rows_per_extent)
            path = self.paths[self.extent_file[shard]]
            offset = self.extent_base[shard] + local * self.row_bytes
            with open(path, "rb", buffering=0) as handle:
                payload = os.pread(handle.fileno(), self.row_bytes, offset)
            assert len(payload) == self.row_bytes
            ctypes.memmove(
                destination + index * destination_stride, payload, len(payload)
            )
            seen.add(row_id)
        return len(seen)

    def direct_fallbacks(self):
        return []

    def direct_file_count(self):
        return len(self.paths)

    def file_count(self):
        return len(self.paths)

    def queue_depth(self):
        return self._queue_depth


class _ManualClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class _ScriptedFakeRing:
    """Scripted ring used to cover the native reader's queue state machine."""

    read_patience_seconds = 60
    drain_patience_seconds = 5

    def __init__(
        self, queue_depth, enter_script=(), short_read=None, clock=time.monotonic
    ):
        self.capacity = queue_depth
        self.enter_script = deque(enter_script)
        self.short_read = short_read
        self.clock = clock
        self.to_submit = deque()
        self.completions = deque()
        self.in_flight = 0
        self.submissions = []

    def submit(self, tag, fd, bounce, length, required, offset):
        request = SimpleNamespace(
            tag=tag,
            fd=fd,
            bounce=bounce,
            length=length,
            required=required,
            offset=offset,
        )
        self.to_submit.append(request)
        self.submissions.append(request)
        self.in_flight += 1

    def _enter(self):
        action = (
            self.enter_script.popleft() if self.enter_script else len(self.to_submit)
        )
        if callable(action):
            action = action()
        if isinstance(action, BaseException):
            raise action
        submitted = min(int(action), len(self.to_submit))
        for _ in range(submitted):
            request = self.to_submit.popleft()
            payload = os.pread(request.fd, request.length, request.offset)
            result = len(payload)
            if self.short_read is not None:
                result = min(result, self.short_read)
                payload = payload[:result]
            request.bounce[:result] = payload
            self.completions.append((request.tag, result, request.required))

    def wait_one(self, deadline):
        while True:
            if self.clock() >= deadline:
                raise RuntimeError("io_uring read timed out after 60 seconds")
            if self.completions:
                break
            try:
                self._enter()
            except InterruptedError:
                continue
            except OSError as exc:
                if exc.errno == errno.ETIME:
                    continue
                raise RuntimeError(f"io_uring_enter: {exc.strerror}") from exc
        tag, result, required = self.completions.popleft()
        self.in_flight -= 1
        if result < required:
            raise RuntimeError(
                f"io_uring short read: got {result} bytes, need {required}"
            )
        return tag

    def drain(self):
        deadline = self.clock() + self.drain_patience_seconds
        while self.in_flight:
            if self.completions:
                self.completions.popleft()
                self.in_flight -= 1
                continue
            if self.clock() >= deadline:
                return False
            try:
                self._enter()
            except InterruptedError:
                continue
            except OSError as exc:
                if exc.errno == errno.ETIME:
                    continue
                return False
        return True


class _FakeRingRowStore:
    """Portable specification double for dedup and O_DIRECT bounce handling."""

    alignment = 4096
    bounce_span = 8192

    def __init__(self, path, extent_base, rows, row_bytes, row_stride, ring):
        self.fd = os.open(path, os.O_RDONLY)
        self.extent_base = extent_base
        self.rows = rows
        self.row_bytes = row_bytes
        self.row_stride = row_stride
        self.ring = ring
        self.bounce = [bytearray(self.bounce_span) for _ in range(ring.capacity)]
        self.poisoned = False

    def close(self):
        os.close(self.fd)

    def read_rows(self, ids, destination, destination_stride):
        if self.poisoned:
            raise RuntimeError("PLE uring row store is poisoned")
        pending = []
        pending_index = {}
        for index, row_id in enumerate(ids):
            if row_id in pending_index:
                pending[pending_index[row_id]].destinations.append(index)
                continue
            offset = self.extent_base + row_id * self.row_stride
            read_offset = offset & ~(self.alignment - 1)
            row_offset = offset - read_offset
            read_length = (
                (offset + self.row_bytes + self.alignment - 1)
                & ~(self.alignment - 1)
            ) - read_offset
            pending_index[row_id] = len(pending)
            pending.append(
                SimpleNamespace(
                    read_offset=read_offset,
                    read_length=read_length,
                    row_offset=row_offset,
                    destinations=[index],
                )
            )

        tag_to_pending = [None] * self.ring.capacity
        next_request = 0

        def submit(tag):
            nonlocal next_request
            request = pending[next_request]
            tag_to_pending[tag] = next_request
            next_request += 1
            self.ring.submit(
                tag,
                self.fd,
                self.bounce[tag],
                request.read_length,
                request.row_offset + self.row_bytes,
                request.read_offset,
            )

        try:
            for tag in range(min(self.ring.capacity, len(pending))):
                submit(tag)
            deadline = self.ring.clock() + self.ring.read_patience_seconds
            for _ in pending:
                tag = self.ring.wait_one(deadline)
                deadline = self.ring.clock() + self.ring.read_patience_seconds
                request = pending[tag_to_pending[tag]]
                row = self.bounce[tag][
                    request.row_offset : request.row_offset + self.row_bytes
                ]
                for index in request.destinations:
                    start = index * destination_stride
                    destination[start : start + self.row_bytes] = row
                if next_request < len(pending):
                    submit(tag)
        except Exception:
            if not self.ring.drain():
                self.poisoned = True
            raise
        return len(pending)


_FAKE_EXTENSION = SimpleNamespace(UringRowStore=_FakeRowStore)


def _source(tmp_path, table_format):
    reference = _quantized_ple_checkpoint(tmp_path, table_format)
    args = SimpleNamespace(
        split_ngram_parts=NGRAM_SHARDS, ngram_head_dim=QUANT_NGRAM_DIM
    )
    return resolve_uring_source(str(tmp_path), args), reference, args


@pytest.mark.parametrize(
    "table_format,data_bytes,disk_scale_bytes,packed_bytes",
    [
        ("fp8", 32, 4, 36),
        ("int4g16", 16, 4, 20),
        ("e2m1g16", 16, 2, 20),
    ],
)
def test_quantized_row_offsets_and_widths(
    tmp_path, table_format, data_bytes, disk_scale_bytes, packed_bytes
):
    source, _reference, args = _source(tmp_path, table_format)
    layout = _ple_table_layout(str(tmp_path), args)

    assert source.data.row_nbytes == data_bytes
    assert source.scales is not None
    assert source.scales.row_nbytes == disk_scale_bytes
    assert source.packed_row_nbytes == packed_bytes
    for shard in range(NGRAM_SHARDS):
        for local_row in (0, NGRAM_ROWS - 1):
            row_id = shard * NGRAM_ROWS + local_row
            assert source.data.row_offset(row_id) == (
                layout.parts[shard].path,
                layout.parts[shard].offset + local_row * data_bytes,
            )
            assert source.scales.row_offset(row_id) == (
                layout.scale_parts[shard].path,
                layout.scale_parts[shard].offset + local_row * disk_scale_bytes,
            )


@pytest.mark.parametrize("table_format", ["fp8", "int4g16", "e2m1g16"])
def test_quantized_uring_stage_matches_reference(tmp_path, table_format):
    source, reference, _args = _source(tmp_path, table_format)
    backend = UringTable(
        source,
        staging_mib=1,
        queue_depth=7,
        max_decode_batch_size=2,
        rows_per_token=3,
        required_capacity_rows=6,
        device=torch.device("cpu"),
        prefetch=False,
        extension=_FAKE_EXTENSION,
    )
    ids = torch.tensor([[0, 9, 0], [source.num_rows - 1, 9, 3]])

    backend.prepare_decode(ids)
    got = backend.lookup(torch.zeros_like(ids))
    backend.finish_decode(record_event=False)
    want = reference.index_select(0, ids.reshape(-1)).view(ids.shape[0], -1)

    assert torch.equal(got, want)
    assert backend.local_ids.shape == (2, 3)
    assert backend.uring_stats() == {
        "requested_rows": 6,
        "read_rows": 4,
        "decode_read_rows": 4,
        "decode_gather_ns": backend.uring_stats()["decode_gather_ns"],
        "decode_fills": 1,
        "prefill_gather_ns": 0,
        "prefill_fills": 0,
        "dedup_rate": pytest.approx(1 / 3),
    }
    startup = backend.startup_description()
    assert "backend=uring" in startup
    assert "queue_depth=7" in startup
    assert "O_DIRECT=yes" in startup


@pytest.mark.parametrize("table_format", ["fp8", "int4g16", "e2m1g16"])
def test_quantized_uring_padded_decode_uses_only_live_rows(tmp_path, table_format):
    source, reference, _args = _source(tmp_path, table_format)
    max_decode_batch_size = 3
    backend = UringTable(
        source,
        staging_mib=1,
        queue_depth=4,
        max_decode_batch_size=max_decode_batch_size,
        rows_per_token=3,
        required_capacity_rows=3,
        device=torch.device("cpu"),
        prefetch=False,
        extension=_FAKE_EXTENSION,
    )
    ids = torch.tensor([[0, source.num_rows - 1, 9]])
    unique_count = torch.unique(ids).numel()
    data_tail = backend._stage_bank.tensor[unique_count:].view(torch.uint8).clone()
    scale_tail = (
        None
        if backend._stage_scale_bank is None
        else backend._stage_scale_bank.tensor[unique_count:].view(torch.uint8).clone()
    )
    raw_scale_tail = (
        None
        if backend._raw_scale_bank is None
        else backend._raw_scale_bank.tensor[unique_count:].view(torch.uint8).clone()
    )
    backend.local_ids.fill_(123456)

    backend.prepare_decode(ids)
    with pytest.raises(RuntimeError, match="prepared uring PLE shape"):
        backend.lookup(torch.zeros((max_decode_batch_size, ids.shape[1])))
    got = backend.lookup(torch.zeros_like(ids))
    want = reference.index_select(0, ids.reshape(-1)).view(ids.shape[0], -1)

    assert torch.equal(got, want)
    assert torch.equal(
        backend._stage_bank.tensor[unique_count:].view(torch.uint8), data_tail
    )
    if scale_tail is not None:
        assert torch.equal(
            backend._stage_scale_bank.tensor[unique_count:].view(torch.uint8),
            scale_tail,
        )
    if raw_scale_tail is not None:
        assert torch.equal(
            backend._raw_scale_bank.tensor[unique_count:].view(torch.uint8),
            raw_scale_tail,
        )
    assert torch.equal(
        backend.local_ids[ids.shape[0] :],
        torch.full_like(backend.local_ids[ids.shape[0] :], 123456),
    )
    backend.finish_decode(record_event=False)


def test_uring_table_stats_use_forward_phase_for_eager_decode(tmp_path):
    source, _reference, _args = _source(tmp_path, "fp8")
    backend = UringTable(
        source,
        staging_mib=1,
        queue_depth=4,
        max_decode_batch_size=1,
        rows_per_token=2,
        required_capacity_rows=2,
        device=torch.device("cpu"),
        prefetch=False,
        extension=_FAKE_EXTENSION,
    )
    prefill_ids = torch.tensor([[0, 1]])
    backend.prefetch(prefill_ids)
    backend.lookup(prefill_ids)
    decode_ids = torch.tensor([[2, 3]])
    backend.prefetch(decode_ids, phase="decode")
    backend.lookup(decode_ids.clone())

    stats = backend.uring_stats()
    assert stats["prefill_fills"] == 1
    assert stats["decode_fills"] == 2
    assert stats["read_rows"] == 6
    assert stats["decode_read_rows"] == 4


def test_ple_layer_passes_batch_phase_to_table_prefetch():
    from freetoken.models.qwen4_exp.ple import PLELayer

    calls = []

    class Table:
        def prefetch(self, row_ids, *, phase):
            calls.append((row_ids, phase))

    row_ids = torch.tensor([[1, 2]])
    layer = PLELayer.__new__(PLELayer)
    layer.ple_embedding = SimpleNamespace(
        row_ids=lambda _meta: row_ids,
        table=Table(),
    )
    batch = SimpleNamespace(phase="decode")
    meta = object()

    layer.start_prefetch(batch, meta)

    assert calls == [(row_ids, "decode")]


def test_fake_ring_deduplicates_and_aligns_direct_reads_after_partial_eintr(
    tmp_path,
):
    path = tmp_path / "rows.bin"
    payload = bytes(index % 251 for index in range(3 * 4096))
    path.write_bytes(payload)
    ring = _ScriptedFakeRing(
        2,
        enter_script=[
            InterruptedError(errno.EINTR, "Interrupted system call"),
            1,
            1,
        ],
    )
    store = _FakeRingRowStore(
        str(path),
        extent_base=4090,
        rows=3,
        row_bytes=32,
        row_stride=37,
        ring=ring,
    )
    destination = bytearray(4 * 40)
    try:
        assert store.read_rows([0, 0, 1, 2], destination, 40) == 3
    finally:
        store.close()

    for index, row_id in enumerate([0, 0, 1, 2]):
        offset = 4090 + row_id * 37
        assert destination[index * 40 : index * 40 + 32] == payload[
            offset : offset + 32
        ]
    assert [(item.offset, item.length, item.required) for item in ring.submissions] == [
        (0, 8192, 4122),
        (4096, 4096, 63),
        (4096, 4096, 100),
    ]


def test_fake_ring_retries_etime_without_poisoning(tmp_path):
    path = tmp_path / "etime.bin"
    payload = bytes(index % 251 for index in range(8192))
    path.write_bytes(payload)
    ring = _ScriptedFakeRing(
        1,
        enter_script=[OSError(errno.ETIME, "Timer expired"), 1],
    )
    store = _FakeRingRowStore(
        str(path),
        extent_base=4090,
        rows=1,
        row_bytes=32,
        row_stride=32,
        ring=ring,
    )
    destination = bytearray(32)
    try:
        assert store.read_rows([0], destination, 32) == 1
        assert destination == payload[4090:4122]
        assert not store.poisoned
    finally:
        store.close()


def test_fake_ring_resets_deadline_after_each_slow_completion(tmp_path):
    path = tmp_path / "slow-fill.bin"
    payload = bytes(index % 251 for index in range(4096))
    path.write_bytes(payload)
    clock = _ManualClock()

    def complete_slowly():
        clock.advance(59)
        return 1

    ring = _ScriptedFakeRing(
        1,
        enter_script=[complete_slowly] * 8,
        clock=clock,
    )
    store = _FakeRingRowStore(
        str(path),
        extent_base=0,
        rows=8,
        row_bytes=32,
        row_stride=32,
        ring=ring,
    )
    destination = bytearray(8 * 32)
    try:
        assert store.read_rows(list(range(8)), destination, 32) == 8
        assert destination == payload[: 8 * 32]
        assert clock() == 8 * 59
        assert not store.poisoned
    finally:
        store.close()


def test_fake_ring_drain_retries_etime_without_poisoning(tmp_path):
    path = tmp_path / "drain-etime.bin"
    path.write_bytes(bytes(range(256)) * 32)
    ring = _ScriptedFakeRing(
        1,
        enter_script=[
            OSError(errno.EIO, "Input/output error"),
            OSError(errno.ETIME, "Timer expired"),
            1,
        ],
    )
    store = _FakeRingRowStore(
        str(path),
        extent_base=0,
        rows=1,
        row_bytes=32,
        row_stride=32,
        ring=ring,
    )
    try:
        with pytest.raises(RuntimeError, match="io_uring_enter: Input/output error"):
            store.read_rows([0], bytearray(32), 32)
        assert not store.poisoned
        assert ring.in_flight == 0
    finally:
        store.close()


def test_fake_ring_drain_stops_after_etime_patience_window(tmp_path):
    path = tmp_path / "drain-timeout.bin"
    path.write_bytes(bytes(range(256)) * 32)
    clock = _ManualClock()

    def wait_one_second():
        clock.advance(1)
        return OSError(errno.ETIME, "Timer expired")

    ring = _ScriptedFakeRing(
        1,
        enter_script=[
            OSError(errno.EIO, "Input/output error"),
            *([wait_one_second] * 6),
            1,
        ],
        clock=clock,
    )
    store = _FakeRingRowStore(
        str(path),
        extent_base=0,
        rows=1,
        row_bytes=32,
        row_stride=32,
        ring=ring,
    )
    try:
        with pytest.raises(RuntimeError, match="io_uring_enter: Input/output error"):
            store.read_rows([0], bytearray(32), 32)
        assert store.poisoned
        assert ring.in_flight == 1
        assert clock() == ring.drain_patience_seconds
    finally:
        store.close()


def test_fake_ring_rejects_short_read(tmp_path):
    path = tmp_path / "short.bin"
    path.write_bytes(bytes(range(256)) * 32)
    ring = _ScriptedFakeRing(2, short_read=131)
    store = _FakeRingRowStore(
        str(path),
        extent_base=100,
        rows=1,
        row_bytes=32,
        row_stride=32,
        ring=ring,
    )
    try:
        with pytest.raises(RuntimeError, match="short read: got 131 bytes, need 132"):
            store.read_rows([0], bytearray(32), 32)
    finally:
        store.close()


def test_fake_ring_non_eintr_failure_poison_is_sticky(tmp_path):
    path = tmp_path / "poison.bin"
    path.write_bytes(bytes(range(256)) * 32)
    failure = OSError(errno.EIO, "Input/output error")
    ring = _ScriptedFakeRing(2, enter_script=[failure, failure])
    store = _FakeRingRowStore(
        str(path),
        extent_base=0,
        rows=1,
        row_bytes=32,
        row_stride=32,
        ring=ring,
    )
    try:
        with pytest.raises(RuntimeError, match="io_uring_enter: Input/output error"):
            store.read_rows([0], bytearray(32), 32)
        with pytest.raises(RuntimeError, match="row store is poisoned"):
            store.read_rows([0], bytearray(32), 32)
    finally:
        store.close()


def test_uring_table_failed_fill_poison_is_sticky(tmp_path):
    source, _reference, _args = _source(tmp_path, "fp8")

    class EnterFailsAfterProbe(_FakeRowStore):
        calls = 0

        def read_rows(self, *args):
            type(self).calls += 1
            if type(self).calls == 3:
                raise RuntimeError("io_uring_enter: Input/output error")
            return super().read_rows(*args)

    backend = UringTable(
        source,
        staging_mib=2,
        queue_depth=4,
        max_decode_batch_size=1,
        rows_per_token=1,
        required_capacity_rows=1,
        device=torch.device("cpu"),
        extension=SimpleNamespace(UringRowStore=EnterFailsAfterProbe),
    )
    ids = torch.tensor([[1]])
    with pytest.raises(RuntimeError, match="io_uring_enter: Input/output error"):
        backend.prepare_decode(ids)
    calls_after_failure = EnterFailsAfterProbe.calls
    with pytest.raises(RuntimeError, match="table is poisoned"):
        backend.prepare_decode(ids)
    assert EnterFailsAfterProbe.calls == calls_after_failure


def test_decode_preparation_uses_padded_reqs():
    from freetoken.models.qwen4_exp.model import Qwen4ExpForCausalLM

    captured = []

    class Embedding:
        def host_decode_row_ids(self, contexts, current_ids):
            captured.append((contexts.clone(), current_ids.clone()))
            return torch.zeros((contexts.shape[0], 4), dtype=torch.int64)

    class Backend:
        def prepare_decode(self, ids):
            self.ids = ids.clone()

    requests = [
        SimpleNamespace(
            uid=index,
            cached_len=2,
            input_ids=torch.tensor([10 + index, 20 + index, 30 + index]),
            pending_token_cpu=None,
            sample_copy_done=None,
        )
        for index in range(3)
    ]
    backend = Backend()
    model = Qwen4ExpForCausalLM.__new__(Qwen4ExpForCausalLM)
    model._config = SimpleNamespace(
        qwen4_args=SimpleNamespace(ngram_size=3, ngram_boundary_token_id=2)
    )
    model.model = SimpleNamespace(
        ple_layers=[SimpleNamespace(ple_embedding=Embedding())]
    )
    model._ple_disk_backends = [backend]
    model._ple_disk_decode = tuple(zip(model.model.ple_layers, [backend]))
    model._ple_decode_contexts = torch.empty((3, 2), dtype=torch.int64)
    model._ple_decode_input_ids = torch.empty(3, dtype=torch.int64)
    model._ple_waited_events = [None] * 3
    model._ple_staging_ns = 0

    batch = SimpleNamespace(padded_reqs=requests, reqs=requests[:1])
    model.prepare_cuda_graph_replay(batch)

    contexts, current = captured[0]
    assert contexts.shape == (3, 2)
    assert current.tolist() == [30, 31, 32]
    assert backend.ids.shape == (3, 4)


def test_non_linux_degrade_names_safe_alternatives(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(RuntimeError) as error:
        _load_uring_extension()
    message = str(error.value)
    assert "requires Linux io_uring" in message
    assert "--ple-backend pinned, cached, or disk" in message
    assert "hmm" not in message


def test_server_flags_select_uring_and_tuning_values():
    from freetoken.server.args import parse_args

    args, _run_shell = parse_args(
        [
            "--model",
            "/tmp/model",
            "--dtype",
            "bfloat16",
            "--ple-backend",
            "uring",
            "--ple-uring-staging-mib",
            "96",
            "--ple-uring-queue-depth",
            "128",
        ]
    )
    assert args.ple_backend == "uring"
    assert args.ple_uring_staging_mib == 96
    assert args.ple_uring_queue_depth == 128


def test_model_logs_one_per_layer_budget_summary(monkeypatch):
    from freetoken.models.qwen4_exp import model as qwen_model
    import freetoken.models.qwen4_exp.ple_uring as ple_uring

    layers = [
        SimpleNamespace(
            ple_embedding=SimpleNamespace(
                attach_table=lambda _table: None,
                snapshot_host_hash_constants=lambda _size: None,
            )
        )
        for _ in range(2)
    ]
    model = qwen_model.Qwen4ExpForCausalLM.__new__(
        qwen_model.Qwen4ExpForCausalLM
    )
    model.model = SimpleNamespace(ple_layers=layers)
    model._config = SimpleNamespace(
        qwen4_args=SimpleNamespace(num_ngram_heads=2, ngram_size=3)
    )

    class FakeUring:
        staging_nbytes = 123

        def __init__(self, *_args, **_kwargs):
            pass

        def startup_description(self):
            return "backend=uring, test"

    logs = []
    monkeypatch.setattr(ple_uring, "resolve_uring_source", lambda *_args: object())
    monkeypatch.setattr(ple_uring, "UringTable", FakeUring)
    monkeypatch.setattr(qwen_model.logger, "info_rank0", logs.append)
    engine_config = SimpleNamespace(
        model_path="/tmp/model",
        ple_backend="uring",
        ple_uring_staging_mib=64,
        ple_uring_queue_depth=4,
        use_dummy_weight=False,
        max_running_req=1,
        max_forward_len=1,
        cuda_graph_bs=[],
        cuda_graph_max_bs=0,
    )

    assert model.load_host_tables(engine_config) == 246
    assert logs == [
        "PLE startup: layers=2, per_layer_budget_mib=64, "
        "total_resident_bytes=246, backend=uring, test"
    ]
    assert not hasattr(model, "_ple_table")


def test_seccomp_style_setup_failure_is_strict(tmp_path):
    source, _reference, _args = _source(tmp_path, "fp8")

    class Rejected:
        def __init__(self, **_kwargs):
            raise RuntimeError("io_uring_setup failed: Operation not permitted")

    with pytest.raises(RuntimeError) as error:
        _make_store(SimpleNamespace(UringRowStore=Rejected), source.data, 64)
    message = str(error.value)
    assert "Operation not permitted" in message
    assert "--ple-backend pinned, cached, or disk" in message
    assert "hmm" not in message


def test_startup_read_probe_failure_is_strict(tmp_path):
    source, _reference, _args = _source(tmp_path, "fp8")

    class RejectedRead(_FakeRowStore):
        def read_rows(self, *_args):
            raise RuntimeError("io_uring_enter: Operation not permitted")

    with pytest.raises(RuntimeError) as error:
        UringTable(
            source,
            staging_mib=2,
            queue_depth=64,
            max_decode_batch_size=1,
            rows_per_token=1,
            required_capacity_rows=1,
            device=torch.device("cpu"),
            extension=SimpleNamespace(UringRowStore=RejectedRead),
        )
    message = str(error.value)
    assert "startup row-read probe" in message
    assert "--ple-backend pinned, cached, or disk" in message
    assert "hmm" not in message


def test_required_capacity_rows_must_be_positive():
    with pytest.raises(ValueError, match="required capacity must be positive"):
        UringTable(
            object(),
            staging_mib=1,
            queue_depth=4,
            max_decode_batch_size=1,
            rows_per_token=1,
            required_capacity_rows=0,
            device=torch.device("cpu"),
            extension=_FAKE_EXTENSION,
        )


def test_row_count_limit_does_not_recommend_more_staging_mib(tmp_path):
    source, _reference, _args = _source(tmp_path, "fp8")
    with pytest.raises(ValueError) as error:
        UringTable(
            source,
            staging_mib=1,
            queue_depth=4,
            max_decode_batch_size=1,
            rows_per_token=1,
            required_capacity_rows=source.num_rows + 1,
            device=torch.device("cpu"),
            extension=_FAKE_EXTENSION,
        )
    message = str(error.value)
    assert f"source row count {source.num_rows} bounds staging capacity" in message
    assert "increasing --ple-uring-staging-mib will not raise" in message
    assert "use at least" not in message


def test_staging_nbytes_charges_raw_scales_and_bounce_per_layer(tmp_path):
    source, _reference, _args = _source(tmp_path, "e2m1g16")

    class RoundedQueueDepth(_FakeRowStore):
        def queue_depth(self):
            return 8

    backend = UringTable(
        source,
        staging_mib=1,
        queue_depth=7,
        max_decode_batch_size=2,
        rows_per_token=3,
        required_capacity_rows=6,
        device=torch.device("cpu"),
        extension=SimpleNamespace(UringRowStore=RoundedQueueDepth),
    )
    capacity = backend._capacity
    expected = (
        capacity * source.data.row_nbytes
        + capacity * 4
        + capacity * 2
        + 2 * 3 * 4
        + 2 * 8 * 8192
        + 4 * len(source.shard_global_scales)
    )
    assert backend.staging_nbytes == expected
    assert backend.staging_nbytes <= 1 << 20


def test_bounce_buffers_are_part_of_capacity_budget(tmp_path):
    source, _reference, _args = _source(tmp_path, "fp8")
    with pytest.raises(ValueError, match="staging capacity.*after .* fixed bytes"):
        UringTable(
            source,
            staging_mib=1,
            queue_depth=64,
            max_decode_batch_size=1,
            rows_per_token=1,
            required_capacity_rows=1,
            device=torch.device("cpu"),
            extension=_FAKE_EXTENSION,
        )


def test_uring_stats_separate_decode_steps_and_prefill_chunks():
    from freetoken.models.qwen4_exp.model import Qwen4ExpForCausalLM

    first = SimpleNamespace(
        prefetch_pages=0,
        uring_stats=lambda: {
            "requested_rows": 20,
            "read_rows": 12,
            "decode_read_rows": 8,
            "decode_gather_ns": 2_000_000,
            "decode_fills": 2,
            "prefill_gather_ns": 7_000_000,
            "prefill_fills": 1,
        },
        reset_stats=lambda: None,
    )
    second = SimpleNamespace(
        prefetch_pages=0,
        uring_stats=lambda: {
            "requested_rows": 20,
            "read_rows": 10,
            "decode_read_rows": 6,
            "decode_gather_ns": 3_000_000,
            "decode_fills": 2,
            "prefill_gather_ns": 5_000_000,
            "prefill_fills": 1,
        },
        reset_stats=lambda: None,
    )
    model = Qwen4ExpForCausalLM.__new__(Qwen4ExpForCausalLM)
    model._ple_disk_backends = [first, second]
    model._ple_major_fault_base = None
    model._ple_staging_ns = 0

    stats = model.ple_disk_stats()
    assert stats["ple_rows_per_step"] == 7
    assert stats["ple_gather_ms_per_decode_step"] == 2.5
    assert stats["ple_gather_ms_per_prefill_chunk"] == 12
    assert stats["ple_dedup_rate"] == pytest.approx(0.45)
