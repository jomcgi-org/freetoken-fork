from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.attention.linear import build_fla_metadata
from freetoken.attention.qsa_sparse import QSASparseAttnBackend, QSASparseMetadata
from freetoken.distributed import DistributedInfo
from freetoken.engine.config import EngineConfig
from freetoken.engine.graph import GraphCaptureBuffer, GraphRunner
from freetoken.spec_decode import (
    MTPVerifyHostStaging,
    configure_mtp_decode_step,
    configure_mtp_fused_step,
    reserve_mtp_window,
    restore_verify_state,
    snapshot_verify_state,
)


class _Batch(SimpleNamespace):
    @property
    def is_decode(self) -> bool:
        return self.phase == "decode"

    @property
    def is_prefill(self) -> bool:
        return self.phase == "prefill"

    @property
    def size(self) -> int:
        return len(self.reqs)

    @property
    def padded_size(self) -> int:
        return len(self.padded_reqs)


def test_engine_config_rejects_more_than_one_mtp_draft():
    with pytest.raises(ValueError, match=r"--mtp-draft-tokens.*fixed at 1"):
        EngineConfig(
            model_path="unused",
            tp_info=DistributedInfo(rank=0, size=1),
            dtype=torch.bfloat16,
            mtp_draft_tokens=2,
        )


def test_engine_config_validates_mtp_verify_graph():
    base = dict(
        model_path="unused",
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.bfloat16,
    )
    assert EngineConfig(**base).mtp_verify_graph == "on"
    assert EngineConfig(**base, mtp_verify_graph="off").mtp_verify_graph == "off"
    with pytest.raises(ValueError, match=r"--mtp-verify-graph.*off.*on"):
        EngineConfig(**base, mtp_verify_graph="auto")


@pytest.mark.parametrize(
    ("bs", "width", "cu_seqlens"),
    [
        (1, 1, [0, 1]),
        (4, 1, [0, 1, 2, 3, 4]),
        (1, 2, [0, 2]),
        (3, 2, [0, 2, 4, 6]),
        (2, 4, [0, 4, 8]),
    ],
)
def test_graph_capture_buffer_sizes_rows_and_strides_cu_seqlens(
    bs, width, cu_seqlens
):
    vocab_size = 11
    buffer = GraphCaptureBuffer.init(
        bs, vocab_size, torch.device("cpu"), width=width
    )

    rows = bs * width
    assert buffer.input_ids.shape == (rows,)
    assert buffer.positions.shape == (rows,)
    assert buffer.out_loc.shape == (rows,)
    assert buffer.table_idx.shape == (rows,)
    assert buffer.request_table_idx.shape == (rows,)
    assert buffer.logits.shape == (rows, vocab_size)
    assert buffer.fla_cu_seqlens.tolist() == cu_seqlens


@pytest.mark.parametrize(
    ("bs", "width", "cu_seqlens"),
    [
        (1, 1, [0, 1]),
        (3, 1, [0, 1, 2, 3]),
        (1, 2, [0, 2]),
        (3, 2, [0, 2, 4, 6]),
    ],
)
def test_eager_fla_metadata_matches_fixed_query_width(bs, width, cu_seqlens):
    reqs = [SimpleNamespace() for _ in range(bs)]
    batch = _Batch(
        reqs=reqs,
        padded_reqs=reqs,
        phase="decode",
        mtp_fused=width > 1,
        input_ids=torch.zeros(bs * width, dtype=torch.int32),
        linear_table_idx=torch.arange(bs, dtype=torch.int32),
    )

    metadata = build_fla_metadata(batch, torch.device("cpu"))

    assert metadata.cu_seqlens.tolist() == cu_seqlens
    assert metadata.cache_indices is batch.linear_table_idx


def test_verify_buffer_binds_token_and_request_slices_separately():
    req = SimpleNamespace()
    batch = _Batch(reqs=[req], padded_reqs=[req], phase="decode")
    buffer = GraphCaptureBuffer.init(1, 5, torch.device("cpu"), width=2)

    buffer.set_batch(batch)

    assert batch.input_ids.shape == (2,)
    assert batch.positions.shape == (2,)
    assert batch.out_loc.shape == (2,)
    assert batch.linear_table_idx.shape == (1,)
    assert batch.active_table_idx.shape == (1,)
    assert batch.fla_metadata.cu_seqlens.tolist() == [0, 2]


def test_qsa_verify_steps_keep_distinct_persistent_lengths():
    backend = QSASparseAttnBackend.__new__(QSASparseAttnBackend)
    backend.device = torch.device("cpu")
    backend.page_size = 1
    backend._graph = {
        "block_table": torch.zeros(1, 3, dtype=torch.int32),
        "kvlen": torch.zeros(1, dtype=torch.int32),
        "table_idx": torch.zeros(1, dtype=torch.int32),
        "token_to_req": torch.zeros(1, dtype=torch.int32),
        "cu_seqlens": torch.tensor([0, 1], dtype=torch.int32),
    }
    backend._block_base_view = lambda: torch.tensor(
        [[0, 0, 0], [3, 4, 5]], dtype=torch.int32
    )
    metadata = [
        QSASparseMetadata(
            is_decode=True,
            last_indices=torch.zeros(1, dtype=torch.int32),
            qo_indptr_cpu=torch.tensor([0, 1], dtype=torch.int32),
            kv_len_cpu=torch.tensor([length], dtype=torch.int32),
        )
        for length in (8, 9)
    ]

    for step, md in enumerate(metadata):
        backend._stage_mtp_decode(
            md, step, 1, torch.tensor([1], dtype=torch.int64)
        )

    assert metadata[0].seq_lens.tolist() == [8]
    assert metadata[1].seq_lens.tolist() == [9]
    assert metadata[0].seq_lens.data_ptr() != metadata[1].seq_lens.data_ptr()
    assert metadata[0].block_table.tolist() == [[3, 4, 5]]


def _mtp_graph_batch(width=2):
    req = SimpleNamespace(lazy_kv_restore=None)
    batch = _Batch(
        reqs=[req],
        padded_reqs=[req],
        phase="decode",
        mtp_fused=True,
        input_ids=torch.arange(width, dtype=torch.int32),
        positions=torch.arange(width, dtype=torch.int32),
        out_loc=torch.arange(width, dtype=torch.int32),
        linear_table_idx=torch.tensor([3], dtype=torch.int32),
        active_table_idx=torch.tensor([4], dtype=torch.int32),
        lazy_restore_pending=False,
    )
    return batch


def test_mtp_verify_graph_eligibility_is_width_keyed_and_requires_mtp():
    runner = GraphRunner.__new__(GraphRunner)
    runner.mtp_enabled = True
    runner.mtp_graph_map = {2: object()}
    batch = _mtp_graph_batch()

    assert runner.can_use_mtp_verify_graph(batch, 2)
    assert not runner.can_use_mtp_verify_graph(batch, 3)

    runner.mtp_enabled = False
    assert not runner.can_use_mtp_verify_graph(batch, 2)


@pytest.mark.parametrize("backend", ["cached", "disk", "uring"])
def test_staged_ple_backends_remain_eligible_for_mtp_verify_capture(
    monkeypatch, backend
):
    monkeypatch.setattr(GraphRunner, "_capture_graphs", lambda *args: None)
    model = SimpleNamespace(
        _ple_disk_backends=[SimpleNamespace(name=backend)]
    )

    runner = GraphRunner(
        stream=None,
        device=torch.device("cpu"),
        model=model,
        attn_backend=SimpleNamespace(),
        cuda_graph_bs=[1],
        cuda_graph_max_bs=None,
        free_memory=0,
        max_seq_len=1,
        vocab_size=5,
        dummy_req=SimpleNamespace(),
        mtp_enabled=True,
        mtp_verify_widths=[2],
    )

    assert runner.mtp_verify_widths == [2]


def _ple_prefetch_spy_model(monkeypatch):
    import importlib.util
    import sys
    from pathlib import Path
    from types import ModuleType

    package_name = "freetoken.models.qwen4_exp"
    source_dir = Path(__file__).resolve().parents[2] / "python/freetoken/models/qwen4_exp"
    package = ModuleType(package_name)
    package.__path__ = [str(source_dir)]
    monkeypatch.setitem(sys.modules, package_name, package)

    dependencies = {
        "attention": "Qwen4ExpAttention",
        "hc": "GatedResidual",
        "moe": "Qwen4ExpMoE",
        "ple": "PLELayer",
    }
    modules = {}
    for name, symbol in dependencies.items():
        module_name = f"{package_name}.{name}"
        module = ModuleType(module_name)
        setattr(module, symbol, object)
        monkeypatch.setitem(sys.modules, module_name, module)
        modules[name] = module

    model_name = f"{package_name}.model"
    spec = importlib.util.spec_from_file_location(model_name, source_dir / "model.py")
    assert spec is not None and spec.loader is not None
    model_module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, model_name, model_module)
    spec.loader.exec_module(model_module)

    calls = []

    class _Embedding:
        def host_decode_row_ids(self, contexts, input_ids):
            return input_ids.view(-1, 1).expand(-1, 4).clone()

    class _PLE:
        args = SimpleNamespace()
        ple_embedding = _Embedding()

        def start_prefetch(self, batch, meta):
            calls.append((batch.phase, meta))

    ple = _PLE()
    meta = object()
    modules["ple"].build_ple_metadata = lambda *args: meta
    modules["ple"].commit_ngram_context = lambda *args: None

    model = model_module.Qwen4ExpModel.__new__(model_module.Qwen4ExpModel)
    model.hc_count = 1
    model.embed_tokens = SimpleNamespace(
        forward=lambda input_ids: input_ids.float().unsqueeze(1)
    )
    model.layers = SimpleNamespace(op_list=[])
    model.hyper_connection_mixer = SimpleNamespace(
        mix=lambda hidden: (hidden, None)
    )
    model._ple = (ple,)
    model._mtp_verify_ple_staging_required = True
    model._mtp_verify_ple_staging_prepared = False
    return model_module, model, ple, calls


def _ple_forward_batch(*, phase, mtp_fused, graph_capture=False):
    req = SimpleNamespace(
        uid=1,
        cached_len=2,
        input_ids=torch.tensor([10, 11, 12], dtype=torch.int32),
        pending_token_cpu=None,
        sample_copy_done=None,
    )
    return _Batch(
        reqs=[req],
        padded_reqs=[req],
        phase=phase,
        mtp_fused=mtp_fused,
        mtp_verify_graph_capture=graph_capture,
        input_ids=torch.tensor([12, 13], dtype=torch.int32),
        linear_table_idx=torch.tensor([0], dtype=torch.int32),
        fla_metadata=None,
    )


def test_mtp_verify_capture_forward_skips_prepared_ple_prefetch(monkeypatch):
    model_module, model, ple, calls = _ple_prefetch_spy_model(monkeypatch)
    batch = _ple_forward_batch(
        phase="decode", mtp_fused=True, graph_capture=True
    )

    class _Backend:
        _decode_shape = None

        def prepare_decode(self, row_ids):
            self._decode_shape = row_ids.shape

        def finish_decode(self, *, record_event):
            self._decode_shape = None

    backend = _Backend()
    causal_model = model_module.Qwen4ExpForCausalLM.__new__(
        model_module.Qwen4ExpForCausalLM
    )
    causal_model.model = model
    causal_model._config = SimpleNamespace(
        qwen4_args=SimpleNamespace(
            ngram_boundary_token_id=2, num_ngram_heads=4
        )
    )
    causal_model._ple_disk_backends = [backend]
    causal_model._ple_disk_decode = ((ple, backend),)
    causal_model._ple_mtp_host_staging = MTPVerifyHostStaging.init(2, 2)
    causal_model._ple_staging_ns = 0

    assert causal_model.prepare_mtp_verify_cuda_graph_replay(batch)
    model.forward(batch.input_ids, batch)

    assert calls == []
    assert not model._mtp_verify_ple_staging_prepared


@pytest.mark.parametrize("phase", ["decode", "prefill"])
def test_non_mtp_forward_still_prefetches_ple(monkeypatch, phase):
    _model_module, model, _ple, calls = _ple_prefetch_spy_model(monkeypatch)
    batch = _ple_forward_batch(phase=phase, mtp_fused=False)

    model.forward(batch.input_ids, batch)

    assert [call[0] for call in calls] == [phase]


def test_mtp_verify_capture_forward_requires_prepared_ple_staging(monkeypatch):
    _model_module, model, _ple, calls = _ple_prefetch_spy_model(monkeypatch)
    batch = _ple_forward_batch(
        phase="decode", mtp_fused=True, graph_capture=True
    )

    with pytest.raises(AssertionError, match="requires prepared PLE staging"):
        model.forward(batch.input_ids, batch)

    assert calls == []


def test_mtp_verify_eager_fallback_still_prefetches_ple(monkeypatch):
    _model_module, model, _ple, calls = _ple_prefetch_spy_model(monkeypatch)
    batch = _ple_forward_batch(phase="decode", mtp_fused=True)

    model.forward(batch.input_ids, batch)

    assert [call[0] for call in calls] == ["decode"]


def test_mtp_verify_replay_restores_the_captured_hidden_storage():
    runner = GraphRunner.__new__(GraphRunner)
    runner.mtp_enabled = True
    captured_hidden = torch.zeros(2, 3)
    model_core = SimpleNamespace(_last_hc_hidden=torch.full((2, 3), -1.0))
    runner.model = SimpleNamespace(model=model_core)
    runner.attn_backend = SimpleNamespace(prepare_for_replay=lambda batch: None)
    runner.moe_offload_cache = None
    runner.mtp_graph_map = {}
    runner.mtp_hidden_map = {2: captured_hidden}
    buffer = GraphCaptureBuffer.init(1, 5, torch.device("cpu"), width=2)
    runner.mtp_buffer_map = {2: buffer}

    class _Graph:
        def replay(self):
            captured_hidden.fill_(7)

    runner.mtp_graph_map[2] = _Graph()
    batch = _mtp_graph_batch()

    logits = runner.replay_mtp_verify(batch, 2)

    assert logits.shape == (2, 5)
    assert buffer.input_ids.tolist() == [0, 1]
    assert buffer.table_idx[0].item() == 3
    assert buffer.request_table_idx[0].item() == 4
    assert model_core._last_hc_hidden is captured_hidden
    assert torch.equal(model_core._last_hc_hidden, torch.full((2, 3), 7.0))


def _staged_mtp_replay(*, fail_staging=False):
    order = []
    captured = []

    class _Embedding:
        def host_decode_row_ids(self, contexts, input_ids):
            order.append("metadata")
            captured.append((contexts.clone(), input_ids.clone()))
            return input_ids.view(-1, 1).expand(-1, 4).clone()

    class _Backend:
        def __init__(self):
            self._decode_shape = None
            self.staging = torch.empty(8, dtype=torch.bfloat16)

        def prepare_decode(self, row_ids):
            order.append("prefetch")
            if fail_staging:
                raise RuntimeError("staging unavailable")
            self.staging[: row_ids.numel()].fill_(1)
            order.append("await")
            self._decode_shape = row_ids.shape

        def finish_decode(self, *, record_event):
            order.append("finish")
            self._decode_shape = None

    backend = _Backend()
    embedding = _Embedding()

    class _Model:
        def __init__(self):
            self.model = SimpleNamespace(_last_hc_hidden=torch.zeros(2, 3))
            self.host_staging = MTPVerifyHostStaging.init(2, 2)
            self.staging_ready = False

        def prepare_mtp_verify_cuda_graph_replay(self, batch):
            self.staging_ready = False
            try:
                contexts, input_ids = self.host_staging.prepare(batch, 2)
                row_ids = embedding.host_decode_row_ids(contexts, input_ids)
                backend.prepare_decode(row_ids)
            except RuntimeError:
                backend.finish_decode(record_event=False)
                return False
            self.staging_ready = True
            return True

        def mtp_verify_cuda_graph_ready(self, batch):
            return self.staging_ready and backend._decode_shape == (2, 4)

        def finish_cuda_graph_replay(self, *, record_event):
            backend.finish_decode(record_event=record_event)
            self.staging_ready = False

    model = _Model()

    req = SimpleNamespace(
        uid=1,
        cached_len=2,
        input_ids=torch.tensor([10, 11, 12], dtype=torch.int32),
        pending_token_cpu=None,
        sample_copy_done=None,
        lazy_kv_restore=None,
    )
    batch = _Batch(
        reqs=[req],
        padded_reqs=[req],
        phase="decode",
        mtp_fused=True,
        input_ids=torch.tensor([12, 13], dtype=torch.int32),
        positions=torch.tensor([2, 3], dtype=torch.int32),
        out_loc=torch.tensor([4, 5], dtype=torch.int32),
        linear_table_idx=torch.tensor([0], dtype=torch.int32),
        active_table_idx=torch.tensor([0], dtype=torch.int32),
        lazy_restore_pending=False,
    )
    runner = GraphRunner.__new__(GraphRunner)
    runner.mtp_enabled = True
    runner.model = model
    runner.attn_backend = SimpleNamespace(
        prepare_for_replay=lambda batch: order.append("attention")
    )
    runner.moe_offload_cache = None
    runner.mtp_hidden_map = {2: model.model._last_hc_hidden}
    runner.mtp_buffer_map = {
        2: GraphCaptureBuffer.init(1, 5, torch.device("cpu"), width=2)
    }

    class _Graph:
        def replay(self):
            assert model.mtp_verify_cuda_graph_ready(batch)
            order.append("replay")

    runner.mtp_graph_map = {2: _Graph()}
    return runner, model, backend, batch, order, captured


def test_staged_mtp_verify_reuses_buffers_and_completes_before_replay():
    runner, model, backend, batch, order, captured = _staged_mtp_replay()
    pointers = (
        model.host_staging.contexts.data_ptr(),
        model.host_staging.input_ids.data_ptr(),
        model.host_staging.draft_token.data_ptr(),
        backend.staging.data_ptr(),
    )

    assert runner.replay_mtp_verify(batch, 2) is not None
    assert order == ["metadata", "prefetch", "await", "attention", "replay", "finish"]
    assert captured[0][0].tolist() == [[10, 11], [11, 12]]
    assert captured[0][1].tolist() == [12, 13]

    order.clear()
    batch.input_ids = torch.tensor([12, 14], dtype=torch.int32)
    assert runner.replay_mtp_verify(batch, 2) is not None
    assert order == ["metadata", "prefetch", "await", "attention", "replay", "finish"]
    assert captured[1][1].tolist() == [12, 14]
    assert pointers == (
        model.host_staging.contexts.data_ptr(),
        model.host_staging.input_ids.data_ptr(),
        model.host_staging.draft_token.data_ptr(),
        backend.staging.data_ptr(),
    )


def test_staged_mtp_verify_falls_back_when_staging_cannot_complete():
    runner, _model, _backend, batch, order, _captured = _staged_mtp_replay(
        fail_staging=True
    )

    assert runner.replay_mtp_verify(batch, 2) is None
    assert order == ["metadata", "prefetch", "finish"]


def test_engine_config_disk_prefix_cache_defaults_off_and_requires_directory():
    base = dict(
        model_path="unused",
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.bfloat16,
    )
    assert EngineConfig(**base).kv_disk_cache_gib == 0
    assert EngineConfig(**base).lazy_restore == "on"
    with pytest.raises(ValueError, match=r"--kv-disk-cache-dir.*required"):
        EngineConfig(**base, kv_disk_cache_gib=1.0)
    configured = EngineConfig(
        **base, kv_disk_cache_gib=8.0, kv_disk_cache_dir="/nvme/prefixes"
    )
    assert configured.kv_disk_cache_dir == "/nvme/prefixes"
    assert EngineConfig(**base, lazy_restore="off").lazy_restore == "off"
    with pytest.raises(ValueError, match=r"--lazy-restore.*on.*off"):
        EngineConfig(**base, lazy_restore="maybe")
    with pytest.raises(ValueError, match=r"--kv-harness-prefixes.*kind=prefix"):
        EngineConfig(**base, kv_harness_prefixes=("missing-separator",))


def test_verify_state_snapshot_restore_round_trip_cpu():
    pool = SimpleNamespace(
        conv_states=torch.arange(2 * 3 * 4, dtype=torch.float32).view(2, 3, 4),
        recurrent_states=torch.arange(2 * 3 * 2 * 2, dtype=torch.float32).view(2, 3, 2, 2),
        slot_states={
            "ple_conv": torch.arange(2 * 3 * 5, dtype=torch.float32).view(2, 3, 5),
            "ple_ngram": torch.arange(3 * 2, dtype=torch.int32).view(1, 3, 2),
        },
    )
    kv_cache = SimpleNamespace(
        _pending_ring=torch.arange(3 * 2 * 4, dtype=torch.float32).view(3, 2, 4)
    )
    req = SimpleNamespace(table_idx=1, linear_slot_idx=2)

    expected = {
        "conv": pool.conv_states[:, 2].clone(),
        "recurrent": pool.recurrent_states[:, 2].clone(),
        "ple_conv": pool.slot_states["ple_conv"][:, 2].clone(),
        "ple_ngram": pool.slot_states["ple_ngram"][:, 2].clone(),
        "qsa_pending": kv_cache._pending_ring[1].clone(),
    }
    snapshot = snapshot_verify_state(pool, kv_cache, req)

    pool.conv_states[:, 2].fill_(-1)
    pool.recurrent_states[:, 2].fill_(-2)
    pool.slot_states["ple_conv"][:, 2].fill_(-3)
    pool.slot_states["ple_ngram"][:, 2].fill_(-4)
    kv_cache._pending_ring[1].fill_(-5)
    restore_verify_state(pool, kv_cache, req, snapshot)

    assert torch.equal(pool.conv_states[:, 2], expected["conv"])
    assert torch.equal(pool.recurrent_states[:, 2], expected["recurrent"])
    assert torch.equal(pool.slot_states["ple_conv"][:, 2], expected["ple_conv"])
    assert torch.equal(pool.slot_states["ple_ngram"][:, 2], expected["ple_ngram"])
    assert torch.equal(kv_cache._pending_ring[1], expected["qsa_pending"])


def test_mtp_window_reservation_stays_decode():
    req = SimpleNamespace(cached_len=7, device_len=8)
    batch = _Batch(reqs=[req], phase="decode")

    reserve_mtp_window(batch, width=2)

    assert batch.is_decode
    assert not batch.is_prefill
    assert req.device_len == 9
    assert req.device_len - req.cached_len == 2
    assert batch.mtp_original_device_len == 8
    assert batch.mtp_original_cached_len == 7
    assert batch.mtp_allocated_end == 9


def test_fused_verify_exposes_seed_and_one_draft_as_one_decode_routed_step():
    req = SimpleNamespace(
        cached_len=7,
        device_len=9,
    )
    batch = _Batch(
        reqs=[req],
        phase="decode",
        mtp_original_cached_len=7,
        mtp_original_device_len=8,
    )
    verify_ids = torch.tensor([10, 11], dtype=torch.int32)
    positions = torch.arange(7, 9, dtype=torch.int32)
    out_loc = torch.arange(20, 22, dtype=torch.int32)

    configure_mtp_fused_step(batch, verify_ids, positions, out_loc)

    assert batch.phase == "decode"
    assert batch.mtp_fused
    assert req.cached_len == 7
    assert req.device_len == 9
    assert req.device_len - req.cached_len == 2
    assert torch.equal(batch.input_ids, verify_ids)
    assert torch.equal(batch.positions, positions)
    assert torch.equal(batch.out_loc, out_loc)


def test_reject_replay_reverts_to_one_decode_position():
    req = SimpleNamespace(cached_len=7, device_len=9)
    batch = _Batch(
        reqs=[req], phase="decode", mtp_original_cached_len=7,
        mtp_original_device_len=8, mtp_fused=True,
    )
    verify_ids = torch.tensor([10, 11], dtype=torch.int32)
    positions = torch.arange(7, 9, dtype=torch.int32)
    out_loc = torch.arange(20, 22, dtype=torch.int32)

    configure_mtp_decode_step(batch, verify_ids, positions, out_loc, 0)

    assert not batch.mtp_fused
    assert req.cached_len == 7
    assert req.device_len == 8
    assert req.device_len - req.cached_len == 1
    assert batch.input_ids.tolist() == [10]
    assert batch.positions.tolist() == [7]
    assert batch.out_loc.tolist() == [20]
