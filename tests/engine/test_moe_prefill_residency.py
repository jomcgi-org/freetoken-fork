from __future__ import annotations

import pytest

from freetoken.engine.engine import _plan_gpu_prefill_layers as plan_prefill
from freetoken.engine.engine import _gpu_prefill_plan_log as plan_log


def test_auto_reserves_full_candidate_set_when_it_fits():
    plan = plan_prefill(
        (100, 100, 100, 100),
        budget_bytes=450,
        reserved_bytes=50,
    )

    assert set(plan.chosen_layer_ids) == {0, 1, 2, 3}
    assert plan.chosen_bytes == 400
    assert plan.reason == (
        "auto fit 4/4 candidates using the current middle-first ordering"
    )


def test_partial_auto_fit_picks_highest_traffic_layers_first():
    plan = plan_prefill(
        (100, 100, 100, 100),
        budget_bytes=200,
        traffic_scores={0: 1.0, 1: 9.0, 2: 7.0, 3: 0.0},
    )

    assert plan.candidate_layer_ids == (1, 2, 0, 3)
    assert plan.chosen_layer_ids == (1, 2)
    assert plan.chosen_bytes == 200


def test_auto_keeps_exactly_one_layer_when_only_one_fits():
    plan = plan_prefill((7, 5, 9), budget_bytes=5)

    assert plan.chosen_layer_ids == (1,)
    assert plan.chosen_bytes == 5


def test_auto_refuses_when_no_layer_fits():
    with pytest.raises(ValueError, match="cannot fit one MoE layer"):
        plan_prefill((7, 5, 9), budget_bytes=4)


def test_off_assigns_the_whole_current_candidate_set_to_the_remainder():
    plan = plan_prefill((100, 100, 100, 100), budget_bytes=400, setting="off")

    assert plan.chosen_layer_ids == ()
    assert set(plan.candidate_layer_ids) == {0, 1, 2, 3}


def test_forced_count_reserves_exactly_n_layers():
    plan = plan_prefill((100, 100, 100, 100), budget_bytes=250, setting="2")

    assert plan.chosen_layer_ids == (2, 1)
    assert plan.chosen_bytes == 200
    assert plan.reason == (
        "forced exactly 2 GPU prefill layers using the current middle-first ordering"
    )


def test_forced_count_refuses_when_exact_set_does_not_fit():
    with pytest.raises(ValueError, match=r"--moe-gpu-prefill-layers 3 requires 300 bytes"):
        plan_prefill((100, 100, 100, 100), budget_bytes=250, setting="3")


def test_auto_without_profile_preserves_middle_first_ordering():
    plan = plan_prefill((100,) * 6, budget_bytes=300)

    assert plan.candidate_layer_ids == (3, 2, 4, 1, 5, 0)
    assert plan.chosen_layer_ids == (3, 2, 4)


def test_plan_log_contains_sets_bytes_and_budget_arithmetic():
    plan = plan_prefill((100, 100), budget_bytes=250, reserved_bytes=50)

    message = plan_log(plan)
    assert "GPU candidates=[1, 0]" in message
    assert "chosen=[1, 0], bytes=200 B" in message
    assert "budget=250 B - reserved=50 B = available=200 B" in message


@pytest.mark.parametrize("value", ["auto", "3", "off"])
def test_server_cli_accepts_gpu_prefill_layer_modes(value):
    from freetoken.server.args import parse_args

    args, _ = parse_args([
        "--model", "/tmp/nonexistent-model",
        "--dtype", "bfloat16",
        "--moe-gpu-prefill-layers", value,
    ])

    assert args.moe_gpu_prefill_layers == value


def test_server_cli_rejects_zero_for_gpu_prefill_layers():
    from freetoken.server.args import parse_args

    with pytest.raises(SystemExit):
        parse_args([
            "--model", "/tmp/nonexistent-model",
            "--dtype", "bfloat16",
            "--moe-gpu-prefill-layers", "0",
        ])
