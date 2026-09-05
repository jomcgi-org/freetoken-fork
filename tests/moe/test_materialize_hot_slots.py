"""Whole-layer GPU prefill must preserve protected expert ownership for decode."""

import pytest
import torch

from freetoken.moe.offload_cache import OffloadMoeCache


def nvfp4_banks(layers, experts, *, hidden=64, intermediate=32):
    shapes = {
        "gate_up_packed": (2 * intermediate, hidden // 2),
        "gate_up_scale": (2 * intermediate, hidden // 16),
        "gate_up_global": (2 * intermediate,),
        "down_packed": (hidden, intermediate // 2),
        "down_scale": (hidden, intermediate // 16),
        "down_global": (hidden,),
    }
    result = {}
    for name, shape in shapes.items():
        dtype = (
            torch.uint8 if name.endswith("packed") else
            torch.float8_e4m3fn if name.endswith("scale") else torch.float16
        )
        result[name] = []
        for layer in range(layers):
            bank = torch.empty(experts, *shape, dtype=dtype)
            for expert in range(experts):
                # Distinguish source rows in every bank, including their scales.
                value = (17 * layer + expert) % 113 + 1
                bank[expert].fill_(value if dtype == torch.uint8 else value / 16)
            result[name].append(bank)
    return result


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("experts,slots", [(8, 32), (512, 4045)])
@pytest.mark.parametrize("route_repeats", [5, 40])
def test_materialized_hot_layers_survive_reuse_and_decode(experts, slots, route_repeats):
    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(0, 1)
    sources = nvfp4_banks(3, experts)
    seeds = {0: (1, 3), 2: (2, experts - 2)}
    cache = OffloadMoeCache(
        num_layers=3, num_experts=experts, cache_size=slots,
        device=torch.device("cuda"), quant_format="nvfp4",
        decode_target="cpu", prefill_overlap=True, moe_disk_prefill="copy",
    )
    cache.cpu_layer_ids = frozenset(range(3))
    cache.set_bank_sources(
        sources, layer_residency=["disk"] * 3,
        hot_expert_ids=seeds, hot_expert_capacity={0: 2, 2: 2},
    )
    expert_bytes = sum(bank[0][0].numel() * bank[0].element_size() for bank in sources.values())
    cache.configure_hot_adaptation(
        half_life_steps=2000, interval_steps=0,
        max_swap_bytes=expert_bytes, expert_bytes=expert_bytes,
    )
    try:
        protected = {(layer, expert): cache._hot_slot_for_row[layer][row]
                     for layer, owners in seeds.items() for row, expert in enumerate(owners)}
        for layer in (0, 1, 2, 0):
            cache.materialize_layer(layer)
            cache.copy_missing()
            assert int(cache.num_indices) == experts
            assert cache.src_indices[:experts].tolist() == list(range(experts))
            assert cache.evict_slots[:experts].tolist() == list(range(experts))
            for name, bank in sources.items():
                assert torch.equal(
                    cache.bank_caches[name][:experts].view(torch.uint8).cpu(),
                    bank[layer].view(torch.uint8),
                )
            # A later pinned prefill can reuse the same scratch rows.
            cache._invalidate_prefill_buffer(0)

        for layer, owners in seeds.items():
            # Ten routes use scalar decode; eighty routes exercise the tiled remap.
            original = torch.tensor(
                list(owners) * route_repeats, device="cuda", dtype=torch.int32,
            )
            remapped = original.clone()
            cache.ensure_experts_hot(layer, remapped)
            cache.copy_missing()
            for name, bank in sources.items():
                actual = cache.bank_caches[name].view(torch.uint8)[remapped.long()].cpu()
                expected = bank[layer].view(torch.uint8)[original.cpu().long()]
                assert torch.equal(actual, expected), f"decode read the wrong {name} rows"
            for expert in owners:
                slot = protected[layer, expert]
                assert int(cache.slot_for_id[layer, expert]) == slot
                assert int(cache.id_of_slot[slot]) == layer * experts + expert
                assert int(cache.usage[slot]) == torch.iinfo(torch.int64).max
    finally:
        cache.shutdown_hot_adaptation()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("protected", [False, True])
@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("reuse_backend", ["fused", "disabled", "ablation"])
def test_selected_staging_leaves_uncopied_rows_unowned(
    tmp_path, monkeypatch, protected, overlap, reuse_backend,
):
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.moe.host_banks import HostBank

    if try_get_tp_info() is None:
        set_tp_info(0, 1)
    sources = nvfp4_banks(2, 8)
    for name, layers in sources.items():
        for layer, tensor in enumerate(layers):
            path = tmp_path / f"{name}-{layer}.ftw"
            path.write_bytes(tensor.view(torch.uint8).numpy().tobytes())
            bank = HostBank(tensor.shape, tensor.dtype, backing="file", file_path=str(path))
            layers[layer] = bank.tensor
    cache = OffloadMoeCache(
        num_layers=2, num_experts=8, cache_size=24, device=torch.device("cuda"),
        quant_format="nvfp4", decode_target="cpu", prefill_overlap=overlap,
        moe_disk_prefill="staged",
    )
    cache.cpu_layer_ids = frozenset({0, 1})
    cache.set_bank_sources(
        sources, layer_residency=["disk"] * 2,
        hot_expert_ids={0: (1,)} if protected else {},
        hot_expert_capacity={0: 1} if protected else {},
    )
    cache.init_disk_prefill_staging()
    assert cache._disk_prefill_staging.pinned_bytes == 64 << 20
    expert_bytes = sum(bank[0][0].numel() * bank[0].element_size() for bank in sources.values())
    cache.configure_hot_adaptation(
        half_life_steps=2000, interval_steps=0,
        max_swap_bytes=expert_bytes, expert_bytes=expert_bytes,
    )
    if reuse_backend == "disabled":
        cache._copy_fused_ok = False
    elif reuse_backend == "ablation":
        monkeypatch.setenv("FREETOKEN_SKIP_FAST_INDEX_COPY", "1")
    reused = int(protected and reuse_backend == "fused")
    try:
        # Prime ordinary cache ownership, then overwrite only two expert rows.
        cache.materialize_layer(1)
        cache.copy_missing()
        old = [destination[:8].view(torch.uint8).cpu().clone() for _, destination in cache.banks]
        ids = torch.tensor([[3, 1], [3, 1]], dtype=torch.int32, device="cuda")
        for stats in (False, True):
            cache.collect_stats = stats
            cache.stage_disk_prefill_layer(0, ids)
            expected_bytes = 0
            for (per_layer, destination), previous in zip(cache.banks, old):
                expected = previous.clone()
                expected[[1, 3]] = per_layer[0].view(torch.uint8)[[1, 3]]
                assert torch.equal(destination[:8].view(torch.uint8).cpu(), expected)
                expected_bytes += (2 - reused) * per_layer[0][0].numel() * per_layer[0].element_size()
            assert cache.disk_prefill_staged_h2d_bytes == (expected_bytes if stats else 0)
            assert cache.disk_prefill_staged_d2d_bytes == (reused * expert_bytes if stats else 0)
            assert cache.prefill_h2d_bytes == 0
            assert cache.id_of_slot[:8].tolist() == [-1] * 8
            assert cache.slot_for_id[0, 6].item() == -1
            assert cache.slot_for_id[1].tolist() == [-1] * 8
            if protected:
                slot = cache._hot_slot_for_row[0][0]
                assert cache.slot_for_id[0, 1].item() == slot
                assert cache.id_of_slot[slot].item() == 1
                assert cache.usage[slot].item() == torch.iinfo(torch.int64).max
        cache.reset_stats()
        assert cache.disk_prefill_staged_h2d_bytes == 0
        assert cache.disk_prefill_staged_d2d_bytes == 0
    finally:
        cache.synchronize_disk_prefill_staging()
        cache.shutdown_hot_adaptation()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("state", ["published", "retired", "republished"])
@pytest.mark.parametrize("tokens", [16, 512])
def test_staged_hot_reuse_matches_nvfp4_gemm_across_publication(tmp_path, state, tokens):
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.moe.fused_nvfp4 import fused_experts_nvfp4
    from freetoken.moe.host_banks import HostBank
    from freetoken.moe.hot_adapt import HotSwap

    if try_get_tp_info() is None:
        set_tp_info(0, 1)
    experts, hidden, intermediate = 8, 256, 128
    sources = nvfp4_banks(1, experts, hidden=hidden, intermediate=intermediate)
    for name, layers in sources.items():
        tensor = layers[0]
        path = tmp_path / f"{name}.ftw"
        path.write_bytes(tensor.view(torch.uint8).numpy().tobytes())
        layers[0] = HostBank(
            tensor.shape, tensor.dtype, backing="file", file_path=str(path),
        ).tensor
    cache = OffloadMoeCache(
        num_layers=1, num_experts=experts, cache_size=24,
        device=torch.device("cuda"), quant_format="nvfp4",
        decode_target="cpu", prefill_overlap=False, moe_disk_prefill="staged",
    )
    cache.cpu_layer_ids = frozenset({0})
    cache.set_bank_sources(
        sources, layer_residency=["disk"],
        hot_expert_ids={0: (1, 6)}, hot_expert_capacity={0: 2},
    )
    cache.init_disk_prefill_staging()
    expert_bytes = sum(bank[0][0].numel() * bank[0].element_size() for bank in sources.values())
    cache.configure_hot_adaptation(
        half_life_steps=2000, interval_steps=1, idle_ms=0,
        max_swap_bytes=expert_bytes, expert_bytes=expert_bytes,
    )
    try:
        if state != "published":
            cache._retire_hot_adaptation_swaps((HotSwap(0, 0, 4, 1),), tick_count=2)
            # The real worker has replaced the bytes, but the request thread
            # has not published the new owner. slot_for_id still names expert 1.
            cache._hot_adapt_future.result(timeout=30)
            slot = cache._hot_slot_for_row[0][0]
            assert cache._hot_slot_owners[0][0] is None
            assert cache.slot_for_id[0, 1].item() == slot
            for per_layer, destination in cache.banks:
                assert torch.equal(
                    destination[slot].view(torch.uint8).cpu(),
                    per_layer[0][4].view(torch.uint8),
                )
            if state == "republished":
                cache._poll_hot_adaptation()
                assert cache._hot_slot_owners[0][0] == 4
                assert cache.slot_for_id[0, 1].item() == -1

        reference = []
        for per_layer, destination in cache.banks:
            full = torch.empty_like(destination[:experts])
            cache._disk_prefill_staging.copy_bank(per_layer[0], full)
            reference.append(full)
            destination[:experts].fill_(255 if destination.dtype == torch.uint8 else float("nan"))
        routes = torch.tensor([[1, 3], [4, 6]], dtype=torch.int32, device="cuda").repeat(tokens // 2, 1)
        original_routes = routes.clone()
        owners = cache.hot_slot_owners()
        cache.collect_stats = True
        cache.stage_disk_prefill_layer(0, routes)
        hot_rows = 1 if state == "retired" else 2
        assert cache.disk_prefill_staged_h2d_bytes == (4 - hot_rows) * expert_bytes
        assert cache.disk_prefill_staged_d2d_bytes == hot_rows * expert_bytes
        assert cache.hot_slot_owners() == owners
        assert torch.equal(routes, original_routes)
        assert cache.id_of_slot[:experts].tolist() == [-1] * experts
        actual_banks = [destination[:experts] for _, destination in cache.banks]
        for a, b in zip(actual_banks, reference):
            assert torch.equal(a.view(torch.uint8)[[1, 3, 4, 6]], b.view(torch.uint8)[[1, 3, 4, 6]])
        generator = torch.Generator().manual_seed(4090)
        x = (torch.randn(tokens, hidden, generator=generator) / 4).to("cuda", torch.bfloat16)
        router = torch.rand(tokens, 2, generator=generator).to("cuda")
        expected = fused_experts_nvfp4(x, *reference, router, routes, experts, "silu", False)
        actual = fused_experts_nvfp4(x, *actual_banks, router, routes, experts, "silu", False)
        assert torch.isfinite(actual).all()
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    finally:
        cache.synchronize_disk_prefill_staging()
        cache.shutdown_hot_adaptation()
