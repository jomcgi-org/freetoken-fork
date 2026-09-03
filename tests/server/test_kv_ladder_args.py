import pytest

from freetoken.server.args import parse_args


ANON_PATH = "/models/anon"


def test_kv_ladder_defaults_on_without_changing_the_fixed_reserve_default():
    args, _ = parse_args(["--model", ANON_PATH, "--dtype", "bfloat16"])
    assert args.kv_ladder == "on"
    assert args.kv_ladder_explicit is False
    assert args.kv_reserve_tokens == 8192


def test_kv_ladder_off_keeps_the_explicit_fixed_reservation_exactly():
    args, _ = parse_args(
        [
            "--model",
            ANON_PATH,
            "--dtype",
            "bfloat16",
            "--kv-ladder",
            "off",
            "--kv-reserve-tokens",
            "163840",
        ]
    )
    assert args.kv_ladder == "off"
    assert args.kv_ladder_explicit is True
    assert args.kv_reserve_tokens == 163_840


def test_kv_ladder_equals_syntax_is_recorded_as_explicit():
    args, _ = parse_args(
        ["--model", ANON_PATH, "--dtype", "bfloat16", "--kv-ladder=on"]
    )
    assert args.kv_ladder == "on"
    assert args.kv_ladder_explicit is True


def test_kv_ladder_abbreviation_is_recorded_as_explicit_and_rejected_early():
    with pytest.raises(ValueError, match="requires --max-running-requests 1"):
        parse_args(
            [
                "--model",
                ANON_PATH,
                "--dtype",
                "bfloat16",
                "--moe-backend",
                "offload",
                "--kv-ladd",
                "on",
            ]
        )
