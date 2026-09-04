"""GPU-free coverage for persisted HOT expert plans."""

from __future__ import annotations

import json
import os

import pytest

from freetoken.moe.hot_adapt import (
    HOT_PLAN_TIER_COMMIT_ENV,
    HOT_PLAN_FILENAME,
    atomic_write_hot_plan,
    checkpoint_identity,
    hot_plan_directory_writable,
    load_hot_plan,
    make_hot_plan_document,
    resolve_tier_commit,
)


IDENTITY = {
    "kind": "ftw",
    "path": "/models/example",
    "index": "freetoken_weight.json",
    "index_sha256": "abc123",
    "shards": [
        {"file": "freetoken-00000.ftw", "size": 4096, "mtime_ns": 1234}
    ],
}


def test_checkpoint_identity_hashes_index_and_stats_shards(tmp_path):
    shard = tmp_path / "freetoken-00000.ftw"
    shard.write_bytes(b"weights")
    index = {
        "format": "freetoken_weight",
        "shards": [{"file": shard.name, "global_off": 0, "nbytes": 7}],
    }
    (tmp_path / "freetoken_weight.json").write_text(json.dumps(index))

    identity = checkpoint_identity(str(tmp_path))

    assert identity["kind"] == "ftw"
    assert identity["path"] == str(tmp_path.resolve())
    assert identity["shards"] == [
        {
            "file": shard.name,
            "size": 7,
            "mtime_ns": shard.stat().st_mtime_ns,
        }
    ]
    assert len(identity["index_sha256"]) == 64


def _document(
    *, layers=(0, 1), budget=400, written_at=1000.0, prefill_counters=None
):
    counters = {
        0: (1.0, 9.0, 8.0, 2.0),
        1: (7.0, 1.0, 6.0, 3.0),
    }
    return make_hot_plan_document(
        identity=IDENTITY,
        disk_layer_ids=(0, 1),
        num_layers=2,
        num_experts=4,
        hot_budget_bytes=budget,
        tier_commit="tier-old",
        protected_slots={layer: ((0, 3) if layer == 0 else (2, 1)) for layer in layers},
        decayed_counters={layer: counters[layer] for layer in layers},
        decayed_prefill_counters=(
            {layer: prefill_counters[layer] for layer in layers}
            if prefill_counters is not None
            else None
        ),
        written_at=written_at,
    )


def _load(path, *, budget=400, capacity=None, identity=IDENTITY):
    return load_hot_plan(
        str(path),
        identity=identity,
        disk_layer_ids=frozenset({0, 1}),
        num_layers=2,
        num_experts=4,
        current_capacity=capacity or {0: 2, 1: 2},
        current_hot_budget_bytes=budget,
        static_expert_ids={0: (1, 2), 1: (0, 3)},
        tier_commit="tier-new",
        now=1060.0,
    )


def test_hot_plan_round_trip_preserves_slot_order_counters_and_metadata(tmp_path):
    path = tmp_path / HOT_PLAN_FILENAME
    document = _document()
    assert document is not None
    atomic_write_hot_plan(str(path), document)

    seed = _load(path)

    assert seed.expert_ids == {0: (3, 0), 1: (2, 1)}
    assert seed.counters[0] == pytest.approx((1.0, 9.0, 8.0, 2.0))
    assert seed.seeded_layers == frozenset({0, 1})
    assert seed.age_seconds == 60.0
    assert seed.saved_hot_budget_bytes == 400
    assert seed.tier_commit == "tier-old"
    assert seed.tier_mismatch
    assert seed.prefill_counters == {}


def test_hot_plan_round_trip_preserves_optional_prefill_counters(tmp_path):
    path = tmp_path / HOT_PLAN_FILENAME
    prefill = {
        0: (10.0, 20.0, 30.0, 40.0),
        1: (4.0, 3.0, 2.0, 1.0),
    }
    document = _document(prefill_counters=prefill)
    assert document is not None
    assert "decayed_prefill_counters" in document
    atomic_write_hot_plan(str(path), document)

    seed = _load(path)

    assert seed.prefill_counters[0] == pytest.approx(prefill[0])
    assert seed.prefill_counters[1] == pytest.approx(prefill[1])


def test_older_hot_plan_without_prefill_section_seeds_prefill_as_zero(tmp_path):
    path = tmp_path / HOT_PLAN_FILENAME
    document = _document()
    assert document is not None
    assert "decayed_prefill_counters" not in document
    atomic_write_hot_plan(str(path), document)

    seed = _load(path)
    prefill_seed = {
        layer_id: seed.prefill_counters.get(layer_id, (0.0,) * 4)
        for layer_id in seed.seeded_layers
    }

    assert prefill_seed == {0: (0.0,) * 4, 1: (0.0,) * 4}


def test_hot_plan_identity_mismatch_is_ignored_by_loader(tmp_path):
    path = tmp_path / HOT_PLAN_FILENAME
    document = _document()
    assert document is not None
    atomic_write_hot_plan(str(path), document)
    changed = dict(IDENTITY, index_sha256="different")

    with pytest.raises(ValueError, match="FTW identity mismatch"):
        _load(path, identity=changed)


def test_smaller_budget_uses_counter_ranked_hottest_prefix(tmp_path):
    path = tmp_path / HOT_PLAN_FILENAME
    document = _document(budget=400)
    assert document is not None
    atomic_write_hot_plan(str(path), document)

    seed = _load(path, budget=200, capacity={0: 1, 1: 1})

    assert seed.expert_ids == {0: (3,), 1: (2,)}


def test_larger_budget_extends_saved_residents_from_counter_ranking(tmp_path):
    path = tmp_path / HOT_PLAN_FILENAME
    document = _document(budget=400)
    assert document is not None
    atomic_write_hot_plan(str(path), document)

    seed = _load(path, budget=600, capacity={0: 3, 1: 3})

    assert seed.expert_ids[0] == (3, 0, 1)
    assert seed.expert_ids[1] == (2, 1, 0)


def test_missing_plan_layer_keeps_static_seed_and_is_marked_unseeded(tmp_path):
    path = tmp_path / HOT_PLAN_FILENAME
    document = _document(layers=(0,))
    assert document is not None
    atomic_write_hot_plan(str(path), document)

    seed = _load(path)

    assert seed.expert_ids[0] == (3, 0)
    assert seed.expert_ids[1] == (0, 3)
    assert seed.seeded_layers == frozenset({0})
    assert 1 not in seed.counters


def test_atomic_write_replaces_only_after_complete_temp_file(tmp_path, monkeypatch):
    path = tmp_path / HOT_PLAN_FILENAME
    document = _document()
    assert document is not None
    real_replace = os.replace
    observed = []

    def checked_replace(source, target):
        with open(source, encoding="utf-8") as handle:
            staged = json.load(handle)
        observed.append((source, target, staged))
        assert not path.exists()
        real_replace(source, target)

    monkeypatch.setattr(os, "replace", checked_replace)
    atomic_write_hot_plan(str(path), document)

    assert len(observed) == 1
    assert observed[0][0] != str(path)
    assert observed[0][1] == str(path)
    assert json.loads(path.read_text()) == document
    assert not list(tmp_path.glob(f".{HOT_PLAN_FILENAME}.*.tmp"))


def test_atomic_write_cancel_fences_publish_after_staging(tmp_path):
    path = tmp_path / HOT_PLAN_FILENAME
    document = _document()
    assert document is not None

    assert not atomic_write_hot_plan(
        str(path), document, publish=lambda _source, _target: False
    )
    assert not path.exists()
    assert not list(tmp_path.glob(f".{HOT_PLAN_FILENAME}.*.tmp"))


def test_plan_directory_writable_probes_actual_create_access(monkeypatch, tmp_path):
    path = str(tmp_path / HOT_PLAN_FILENAME)
    assert hot_plan_directory_writable(path)

    def deny_create(*_args, **_kwargs):
        raise OSError("read only")

    monkeypatch.setattr(
        "freetoken.moe.hot_adapt.tempfile.NamedTemporaryFile", deny_create
    )
    assert not hot_plan_directory_writable(path)


def test_tier_commit_environment_override_precedes_git(monkeypatch):
    logs = []
    monkeypatch.setenv(HOT_PLAN_TIER_COMMIT_ENV, "deployed-tier")
    monkeypatch.setattr(
        "freetoken.moe.hot_adapt.subprocess.run",
        lambda *_args, **_kwargs: pytest.fail("git must not run for an environment override"),
    )
    monkeypatch.setattr("freetoken.moe.hot_adapt.logger.info_rank0", logs.append)

    assert resolve_tier_commit() == "deployed-tier"
    assert logs == [
        "HOT plan tier commit='deployed-tier' source=FREETOKEN_TIER_COMMIT"
    ]


def test_tier_commit_skips_git_for_installed_package(monkeypatch):
    from freetoken import version

    warnings = []
    monkeypatch.delenv(HOT_PLAN_TIER_COMMIT_ENV, raising=False)
    monkeypatch.setattr(
        version, "__file__", "/venv/lib/python3.10/site-packages/freetoken/version.py"
    )
    monkeypatch.setattr(
        "freetoken.moe.hot_adapt.subprocess.run",
        lambda *_args, **_kwargs: pytest.fail("git must not run for an installed package"),
    )
    monkeypatch.setattr("freetoken.moe.hot_adapt.logger.warning_rank0", warnings.append)

    assert resolve_tier_commit().startswith("package-")
    assert len(warnings) == 1
    assert "source=package-version" in warnings[0]


def test_all_zero_counters_are_not_made_persistable():
    assert make_hot_plan_document(
        identity=IDENTITY,
        disk_layer_ids=(0,),
        num_layers=1,
        num_experts=2,
        hot_budget_bytes=100,
        tier_commit="tier",
        protected_slots={0: (0,)},
        decayed_counters={0: (0.0, 0.0)},
    ) is None


def test_compact_float32_format_stays_near_ninety_kibibytes():
    layers = 28
    experts = 512
    document = make_hot_plan_document(
        identity=IDENTITY,
        disk_layer_ids=range(layers),
        num_layers=layers,
        num_experts=experts,
        hot_budget_bytes=1 << 30,
        tier_commit="tier",
        protected_slots={layer: range(80) for layer in range(layers)},
        decayed_counters={
            layer: [float(expert + 1) for expert in range(experts)]
            for layer in range(layers)
        },
    )
    assert document is not None

    encoded = json.dumps(document, separators=(",", ":")).encode()
    assert 80_000 < len(encoded) < 100_000
