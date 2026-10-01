"""/v1/stats host_memory block: tracker intake and document schema."""

from __future__ import annotations

from types import SimpleNamespace

from freetoken.server.stats import StatsTracker, build_stats

_BLOCK = {
    "estimate": {"reserve_gib": 9.29, "disk_tier_cache_gib": 3.64,
                 "prefix_cache_headroom_gib": 3.0},
    "live": {"majflt_per_step": 12.5, "cached_gib": 20.0, "mem_available_gib": 41.0,
             "pressure": False, "pressure_reasons": []},
    "pressure": False, "pressure_reasons": [], "max_majflt_per_step": 1000.0,
}


def _state(tracker):
    config = SimpleNamespace(
        served_model_name="m", max_seq_len=1024, page_size=16,
        model_config=SimpleNamespace(),
    )
    return SimpleNamespace(stats=tracker, config=config, ready_at=None, instance_id="i")


def test_host_memory_is_null_until_a_reply_carries_it():
    doc = build_stats(_state(StatsTracker()), 0, 0)
    assert doc["host_memory"] is None


def test_host_memory_block_is_exposed_with_age():
    tr = StatsTracker()
    tr.observe(SimpleNamespace(host_memory=_BLOCK), now=1.0)
    # a later reply without the block keeps the last one
    tr.observe(SimpleNamespace(completion_tokens_delta=1, host_memory=None), now=2.0)
    doc = build_stats(_state(tr), 0, 0)
    hm = doc["host_memory"]
    assert hm["estimate"]["disk_tier_cache_gib"] == 3.64
    assert hm["live"]["mem_available_gib"] == 41.0
    assert hm["pressure"] is False and hm["max_majflt_per_step"] == 1000.0
    assert isinstance(hm["age_s"], float)
    # the other fields are untouched
    assert {"model", "kv", "throughput", "requests"} <= set(doc)
