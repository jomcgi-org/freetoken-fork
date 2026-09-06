"""Page relocation preserves live prefix bytes and isolates both slot spaces."""

import importlib.util
from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / "bench" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


relocation = load("boundary_relocation", "target_relocation.py")
base = load("boundary_base", "target-verify-cost.py")
multi = load("boundary_multi", "target_multitoken.py")


@pytest.fixture
def fixture():
    torch = pytest.importorskip("torch")
    pool = SimpleNamespace(num_slots=3,
                            conv_states=torch.arange(18, dtype=torch.float32).reshape(2, 3, 3),
                            recurrent_states=torch.arange(24, dtype=torch.float32).reshape(2, 3, 4),
                            slot_states={"ple_conv": torch.arange(18, dtype=torch.float32).reshape(2, 3, 3),
                                         "ple_ngram_ctx": torch.arange(6).reshape(1, 3, 2)})
    kv = SimpleNamespace(index_ratio=4, cmp_scratch_base=10,
                         _kv_buffer=torch.arange(320, dtype=torch.float32).reshape(2, 2, 5, 8, 1, 2),
                         _cmp_k_buffer=torch.arange(72, dtype=torch.float32).reshape(2, 12, 3),
                         _pending_ring=torch.arange(96, dtype=torch.float32).reshape(2, 2, 8, 3))
    engine = SimpleNamespace(config=SimpleNamespace(page_size=8, max_running_req=1), num_pages=4,
                             page_table=torch.tensor([list(range(16, 24)) + [0] * 16, [32] * 24], dtype=torch.int32),
                             kv_cache=kv, linear_state_pool=pool,
                             dummy_req=SimpleNamespace(table_idx=1), stream=SimpleNamespace(synchronize=lambda: None))
    req = SimpleNamespace(table_idx=0, linear_slot_idx=0, cached_len=6, device_len=7, remain_len=20)
    source = SimpleNamespace(reqs=[req], padded_reqs=[req], is_decode=True, lazy_restore_pending=False,
                             linear_table_idx=torch.tensor([0], dtype=torch.int32),
                             active_table_idx=torch.tensor([0], dtype=torch.int32))

    def views(engine, req):
        slot = relocation.linear_slot(req)
        return {"conv": engine.linear_state_pool.conv_states[:, slot],
                "recurrent": engine.linear_state_pool.recurrent_states[:, slot],
                **{"slot/" + k: value[:, slot] for k, value in engine.linear_state_pool.slot_states.items()},
                "qsa_pending": engine.kv_cache._pending_ring[req.table_idx]}

    return SimpleNamespace(torch=torch, engine=engine, source=source, views=views)


def test_boundary_window_snapshots_noncontiguous_pages_and_only_compares_visible_rows(fixture, monkeypatch):
    f = fixture
    monkeypatch.setattr(base, "state_views", f.views)
    f.engine.page_table[0, 8:16] = f.torch.arange(8)
    window = base.StateWindow(f.engine, f.source.reqs[0], 6, width=5)
    assert window.physical_pages == (2, 0)
    assert window.locations.tolist() == [22, 23, 0, 1, 2]
    saved = window.capture()
    window.views["kv_page"][:, :, 7].fill_(999)
    window.views["cmp_page"][:, 1].fill_(999)
    window.views["kv_page/1"][:, :, 0].fill_(999)
    window.views["cmp_page/1"][:, 0].fill_(999)
    assert window.compare(saved, committed_end=7) == []
    assert set(window.compare(saved, committed_end=8)) == {"kv_page", "cmp_page"}
    assert set(window.compare(saved, committed_end=9)) == {"kv_page", "cmp_page", "kv_page/1"}
    assert set(window.compare(saved, committed_end=12)) == {"kv_page", "cmp_page", "kv_page/1", "cmp_page/1"}
    window.reset(saved)
    assert window.compare(saved, committed_end=13) == []


@pytest.mark.parametrize("failure", ["missing", "negative", "alias", "reserved", "missing_compressed"])
def test_window_rejects_bad_mapping_before_any_state_write(fixture, monkeypatch, failure):
    f = fixture
    monkeypatch.setattr(base, "state_views", f.views)
    f.engine.page_table[0, 8:16] = f.torch.arange(8)
    if failure == "missing":
        f.engine.page_table[0, 8:16].zero_()
    elif failure == "negative":
        f.engine.page_table[0, :8] = f.torch.arange(-8, 0)
    elif failure == "alias":
        f.engine.page_table[0, 8:16] = f.torch.arange(16, 24)
    elif failure == "reserved":
        f.engine.page_table[0, 8:16] = f.torch.arange(32, 40)
    else:
        f.engine.kv_cache.cmp_scratch_base = 4
    before = f.engine.kv_cache._kv_buffer.clone()
    with pytest.raises(RuntimeError):
        base.StateWindow(f.engine, f.source.reqs[0], 6, width=5)
    assert f.torch.equal(f.engine.kv_cache._kv_buffer, before)


@pytest.mark.parametrize("source_linear", [0, 2])
def test_lease_rotates_prefix_pages_moves_both_slot_spaces_and_restores_borrowed_state(fixture, monkeypatch, source_linear):
    f = fixture
    monkeypatch.setattr(base, "state_views", f.views)
    f.source.reqs[0].linear_slot_idx = source_linear
    f.source.linear_table_idx.fill_(source_linear)
    lease = relocation.RelocationLease(f.engine, f.source, 6, 5, f.views)
    borrowed = {name: value.clone() for name, value in lease.borrowed_views.items()}
    original_kv = f.engine.kv_cache._kv_buffer.clone()
    original_cmp = f.engine.kv_cache._cmp_k_buffer.clone()
    original_table = f.engine.page_table.clone()
    seen = []
    with lease:
        assert f.engine.page_table[0, :16].tolist() == list(range(32, 40)) + list(range(16, 24))
        assert f.torch.equal(f.engine.page_table[0], f.engine.page_table[1])
        assert f.torch.equal(f.engine.kv_cache._kv_buffer[:, :, 4], original_kv[:, :, 2])
        assert f.torch.equal(f.engine.kv_cache._cmp_k_buffer[:, 8:10], original_cmp[:, 4:6])
        assert f.torch.count_nonzero(f.engine.kv_cache._kv_buffer[:, :, 2]) == 0
        assert f.torch.count_nonzero(f.engine.kv_cache._cmp_k_buffer[:, 4:6]) == 0
        window = base.StateWindow(f.engine, f.source.reqs[0], 6, width=5, allow_reserved_page=True)
        assert window.physical_pages == (4, 2)
        assert window.locations.tolist() == [38, 39, 16, 17, 18]
        current = {name: value.clone() for name, value in f.views(f.engine, f.source.reqs[0]).items()}
        for case in range(4):
            batch = lease.select(case)
            req = batch.reqs[0]
            seen.append((req.table_idx, req.linear_slot_idx))
            active = f.views(f.engine, req)
            assert all(f.torch.equal(value, current[name]) for name, value in active.items())
            neighbours = lease.neighbours()
            untouched = {name: value.clone() for name, value in neighbours.items()}
            for value in active.values():
                value.add_(case + 10)
            assert all(f.torch.equal(value, untouched[name]) for name, value in neighbours.items())
            assert batch.active_table_idx.tolist() == [req.table_idx]
            assert batch.linear_table_idx.tolist() == [req.linear_slot_idx]
            current = {name: value.clone() for name, value in active.items()}
    assert seen == list(lease.plan)
    assert f.torch.equal(f.engine.page_table, original_table)
    assert f.torch.equal(f.engine.kv_cache._kv_buffer, original_kv)
    assert f.torch.equal(f.engine.kv_cache._cmp_k_buffer[:, :10], original_cmp[:, :10])
    assert all(f.torch.equal(value, borrowed[name]) for name, value in lease.borrowed_views.items())
    lease.close()
    with pytest.raises(RuntimeError, match="once"):
        lease.__enter__()


def test_lease_restores_reserved_storage_after_failure_and_rejects_out_of_order_cases(fixture):
    f = fixture
    lease = relocation.RelocationLease(f.engine, f.source, 6, 5, f.views)
    before = [value.clone() for value in lease.restore_views]
    with pytest.raises(ValueError, match="test failure"):
        with lease:
            with pytest.raises(RuntimeError, match="in order"):
                lease.select(1)
            lease.select(0)
            with pytest.raises(RuntimeError, match="in order"):
                lease.select(0)
            for value in lease.restore_views:
                value.fill_(999)
            raise ValueError("test failure")
    assert all(f.torch.equal(value, saved) for value, saved in zip(lease.restore_views, before))


@pytest.mark.parametrize("failure", ["position", "remaining", "capacity", "linear", "table", "storage", "lazy"])
def test_lease_rejects_invalid_ownership_or_layout_without_mutation(fixture, failure):
    f = fixture
    position = 6
    if failure == "position":
        position = 5
    elif failure == "remaining":
        f.source.reqs[0].remain_len = 7
    elif failure == "capacity":
        f.engine.config.max_running_req = 2
    elif failure == "linear":
        f.source.reqs[0].linear_slot_idx = 3
    elif failure == "table":
        f.source.reqs[0].table_idx = 1
    elif failure == "storage":
        f.engine.kv_cache.cmp_scratch_base = 9
    else:
        f.source.reqs[0].lazy_kv_restore = SimpleNamespace(complete=False)
    before = f.engine.page_table.clone()
    with pytest.raises(ValueError):
        relocation.RelocationLease(f.engine, f.source, position, 5, f.views)
    assert f.torch.equal(before, f.engine.page_table)


@pytest.mark.parametrize("failure", [None, "forward", "health", "graph"])
def test_original_mapping_control_restores_state_and_kv_even_after_failure(fixture, monkeypatch, failure):
    f = fixture
    # Load the real CPU metadata builder without the package's model-download imports.
    path = Path(__file__).parents[1] / "python/freetoken/attention/linear.py"
    spec = importlib.util.spec_from_file_location("freetoken.attention.linear", path)
    linear = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, linear)
    spec.loader.exec_module(linear)
    monkeypatch.setattr(base, "state_views", f.views)
    f.engine.device = f.torch.device("cpu")
    touched = list(f.views(f.engine, f.source.reqs[0]).values())
    touched.append(f.engine.kv_cache._kv_buffer[:, :, 2])
    before = [value.clone() for value in touched]

    def replay(batch):
        for value in touched:
            value.add_(11)
        if failure == "forward":
            raise RuntimeError("control forward failure")
        return f.torch.tensor([[1., 2., 3.]])

    def healthy():
        if failure == "health":
            raise RuntimeError("control health failure")

    f.engine.graph_runner = SimpleNamespace(can_use_cuda_graph=lambda batch: failure != "graph", replay=replay)
    f.engine.ctx = SimpleNamespace(forward_batch=lambda batch: nullcontext())
    f.engine.attn_backend = SimpleNamespace(prepare_metadata=lambda batch: None)
    f.engine.cpu_moe_executor = SimpleNamespace(begin_decode_step=lambda: None, raise_if_unhealthy=healthy)
    if failure:
        with pytest.raises(RuntimeError):
            multi.original_seed_logits(f.engine, f.source, vars(base))
    else:
        result = multi.original_seed_logits(f.engine, f.source, vars(base))
        assert result.tolist() == [[1., 2., 3.]]
    assert all(f.torch.equal(value, old) for value, old in zip(touched, before))


@pytest.mark.parametrize("failure", [None, "missing_case", "wrong_slot", "same_slot", "wrong_page",
                                    "contiguous", "neighbour", "missing_layout"])
def test_qualification_requires_independent_slot_moves_and_the_page_transition(failure):
    layout = dict(boundary=64, page_size=64, source_table=0, source_linear=2,
                  spare_table=1, spare_linear=0, old_page=3, reserved_page=20)
    rows = []
    for i, (table, linear) in enumerate(relocation.request_plan(0, 2, 1, 0)):
        rows.append(dict(case=str(62 + i), request_table=table, linear_slot=linear,
                         physical_pages=[20, 3] if i < 2 else [3], neighbours_unchanged=True))
    if failure == "missing_case":
        rows.pop()
    elif failure == "wrong_slot":
        rows[-1]["linear_slot"] = 2
    elif failure == "same_slot":
        layout["spare_linear"] = 2
    elif failure == "wrong_page":
        rows[0]["physical_pages"] = [20, 4]
    elif failure == "contiguous":
        layout["old_page"] = 21
    elif failure == "neighbour":
        rows[0]["neighbours_unchanged"] = False
    elif failure == "missing_layout":
        layout = None
    assert relocation.qualify(rows, layout, 5) is (failure is None)
