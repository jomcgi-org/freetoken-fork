"""Integer routing, adaptive histories, and LRU state must survive parallelization."""

import pytest
import torch

from freetoken.moe import offload_kernels as kernels
from freetoken.moe.offload_cache import OffloadMoeCache


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

STATE = (
    "slot_for_id", "id_of_slot", "usage", "step", "active_mask",
    "num_indices", "num_missing_full", "stat_hot_pairs", "stat_hot_total_pairs",
    "decayed_decode_freq", "decayed_prefill_freq",
)


def make_cache(*, experts=512, missing=False, device="cuda", collect=True, adapt=True):
    cache = OffloadMoeCache(
        num_layers=3, num_experts=experts, cache_size=3 * experts + 7,
        device=torch.device(device),
    )
    rng = torch.Generator().manual_seed(4090)
    owners = torch.randperm(3 * experts, generator=rng)
    cache.id_of_slot[:owners.numel()].copy_(owners)
    cache.slot_for_id.view(-1)[owners.to(device)] = torch.arange(owners.numel(), device=device)
    cache.usage.copy_(torch.randint(0, 25, (cache.cache_size,), generator=rng))
    cache.step.fill_(25)
    hot = sorted(set(range(0, experts, 6)) | {experts - 1})
    cache.hot_row_for_expert[1, hot] = torch.arange(len(hot), device=device, dtype=torch.int32)
    if missing:
        for expert in hot[:3]:
            slot = int(cache.slot_for_id[1, expert])
            cache.id_of_slot[slot] = -1
            cache.slot_for_id[1, expert] = -1
    cache.collect_stats = collect
    cache.hot_adapt_enabled = adapt
    cache._hot_decay_factor = 0.9996534864594093
    cache.decayed_decode_freq.copy_(torch.rand(3, experts, generator=rng))
    cache.decayed_prefill_freq = torch.rand(3, experts, generator=rng).to(device)
    return cache


def assert_state_equal(left, right):
    for name in STATE:
        assert torch.equal(getattr(left, name).cpu(), getattr(right, name).cpu()), name
    missing = int(left.num_indices)
    for name in ("src_indices", "evict_slots"):
        assert torch.equal(getattr(left, name)[:missing].cpu(), getattr(right, name)[:missing].cpu()), name


@pytest.mark.parametrize("count", [1, 63, 64, 65, 1023, 1024, 1025, 20480])
@pytest.mark.parametrize("pattern", ["mixed", "single", "all_cold"])
def test_parallel_matches_serial_for_all_metadata(monkeypatch, count, pattern):
    rng = torch.Generator().manual_seed(count)
    raw = torch.randint(0, 512, (count,), generator=rng, dtype=torch.int32).cuda()
    if pattern == "single":
        raw.fill_(511)
    elif pattern == "all_cold":
        raw.fill_(1)
    a = make_cache()
    b = make_cache()
    for history, weight, record in (("prefill", 0.1, True), ("decode", 1.0, True), ("prefill", 0.1, False)):
        ids_a, ids_b = raw.clone(), raw.clone()
        monkeypatch.setattr(kernels, "_PARALLEL_HOT_ROUTING", False)
        kernels.ensure_experts_hot(a, 1, ids_a, route_weight=weight, history=history, record_stats=record)
        monkeypatch.setattr(kernels, "_PARALLEL_HOT_ROUTING", True)
        kernels.ensure_experts_hot(b, 1, ids_b, route_weight=weight, history=history, record_stats=record)
        assert torch.equal(ids_a, ids_b)
        assert_state_equal(a, b)


@pytest.mark.parametrize("collect,adapt", [(False, False), (False, True), (True, False), (True, True)])
def test_non_power_of_two_experts_and_missing_rows_match_cpu(monkeypatch, collect, adapt):
    a = make_cache(experts=19, missing=True, device="cpu", collect=collect, adapt=adapt)
    b = make_cache(experts=19, missing=True, collect=collect, adapt=adapt)
    raw = (torch.arange(65, dtype=torch.int32) % 19)
    gpu = raw.cuda()
    monkeypatch.setattr(kernels, "_PARALLEL_HOT_ROUTING", True)
    kernels.ensure_experts_hot(a, 1, raw, history="prefill", route_weight=0.1)
    kernels.ensure_experts_hot(b, 1, gpu, history="prefill", route_weight=0.1)
    assert torch.equal(raw, gpu.cpu())
    # PyTorch and Triton may differ in FMA contraction. Compare adaptation with
    # the serial GPU kernel bitwise above; use close for the CPU arithmetic oracle.
    for name in ("decayed_decode_freq", "decayed_prefill_freq"):
        torch.testing.assert_close(getattr(a, name), getattr(b, name).cpu())
        getattr(a, name).copy_(getattr(b, name).cpu())
    assert_state_equal(a, b)


def test_graph_replay_keeps_changing_routes_and_histories(monkeypatch):
    a, b = make_cache(), make_cache()
    host_routes = torch.arange(1025, device="cuda", dtype=torch.int32) % 512
    ids_a, ids_b = torch.empty_like(host_routes), torch.empty_like(host_routes)
    # Compile both variants outside capture, on separate scratch caches.
    for parallel in (False, True):
        monkeypatch.setattr(kernels, "_PARALLEL_HOT_ROUTING", parallel)
        kernels.ensure_experts_hot(make_cache(), 1, host_routes.clone())
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ids_b.copy_(host_routes)
        kernels.ensure_experts_hot(b, 1, ids_b, route_weight=0.1, history="prefill")
    # Capture does not run a live invocation. Each replay must observe new data.
    for offset in (7, 12, 31):
        host_routes.add_(offset).remainder_(512)
        ids_a.copy_(host_routes)
        monkeypatch.setattr(kernels, "_PARALLEL_HOT_ROUTING", False)
        kernels.ensure_experts_hot(a, 1, ids_a, route_weight=0.1, history="prefill")
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(ids_a, ids_b)
        assert_state_equal(a, b)
