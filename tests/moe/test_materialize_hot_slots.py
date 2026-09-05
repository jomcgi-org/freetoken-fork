"""Whole-layer GPU prefill must preserve protected expert ownership for decode."""

import pytest
import torch

from freetoken.moe.offload_cache import OffloadMoeCache


def nvfp4_banks(layers, experts):
    hidden, intermediate = 64, 32
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
