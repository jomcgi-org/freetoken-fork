from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from freetoken.moe.cpu_executor import (
    CpuMoeExecutor,
    _spin_core_cpus,
    decide_cpu_executor_mode,
    parse_cpu_list,
    read_cpu_topology,
)


def _cli_modules(monkeypatch):
    """Import config and CLI without loading Triton-backed flashlib kernels."""
    kernels = ModuleType("flashlib.kernels")
    kernels.__path__ = []
    slot_cache = ModuleType("flashlib.kernels.slot_cache")
    slot_cache.N_STATS = 3
    slot_cache.Stat = SimpleNamespace(ACTIVE=0, MISS=1, CALLS=2)
    slot_cache.lru_ensure = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "flashlib.kernels", kernels)
    monkeypatch.setitem(sys.modules, "flashlib.kernels.slot_cache", slot_cache)
    from freetoken.engine.config import EngineConfig
    from freetoken.server.args import parse_args

    return EngineConfig, parse_args


def _write_topology(
    root: Path, *, sockets: int, cores_per_socket: int, threads_per_core: int
) -> set[int]:
    allowed = set()
    logical_per_socket = cores_per_socket * threads_per_core
    for package_id in range(sockets):
        package_base = package_id * logical_per_socket
        for core_id in range(cores_per_socket):
            siblings = [
                package_base + core_id + thread * cores_per_socket
                for thread in range(threads_per_core)
            ]
            sibling_text = ",".join(str(cpu) for cpu in siblings)
            for cpu in siblings:
                allowed.add(cpu)
                topology = root / f"cpu{cpu}" / "topology"
                topology.mkdir(parents=True)
                (topology / "core_id").write_text(str(core_id))
                (topology / "physical_package_id").write_text(str(package_id))
                (topology / "thread_siblings_list").write_text(sibling_text)
    return allowed


def _topology(
    tmp_path: Path,
    *,
    sockets: int = 1,
    cores_per_socket: int,
    threads_per_core: int,
    machine: str = "x86_64",
):
    sys_root = tmp_path / "sys"
    allowed = _write_topology(
        sys_root,
        sockets=sockets,
        cores_per_socket=cores_per_socket,
        threads_per_core=threads_per_core,
    )
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("flags : fpu sse2 ht\n")
    return read_cpu_topology(
        sys_root,
        cpuinfo,
        machine=machine,
        allowed_cpus=allowed,
    )


@pytest.mark.parametrize("mode", ["sleep", "spin", "auto"])
def test_executor_mode_argument_accepts_all_modes(monkeypatch, mode):
    _, parse_args = _cli_modules(monkeypatch)
    args, _ = parse_args(
        [
            "--model",
            "/tmp/nonexistent-model",
            "--dtype",
            "bfloat16",
            "--moe-cpu-executor-mode",
            mode,
        ]
    )
    assert args.moe_cpu_executor_mode == mode


def test_executor_mode_argument_rejects_invalid_value(monkeypatch):
    _, parse_args = _cli_modules(monkeypatch)
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--model",
                "/tmp/nonexistent-model",
                "--dtype",
                "bfloat16",
                "--moe-cpu-executor-mode",
                "park",
            ]
        )


def test_executor_mode_defaults_to_sleep(monkeypatch):
    EngineConfig, parse_args = _cli_modules(monkeypatch)
    args, _ = parse_args(
        ["--model", "/tmp/nonexistent-model", "--dtype", "bfloat16"]
    )
    assert EngineConfig.__dataclass_fields__["moe_cpu_executor_mode"].default == "sleep"
    assert args.moe_cpu_executor_mode == "sleep"


def test_executor_constructor_rejects_invalid_mode_before_native_import():
    with pytest.raises(ValueError, match="executor_mode"):
        CpuMoeExecutor(
            None,
            top_k=1,
            activation="silu",
            apply_router_weight_on_input=False,
            num_threads=1,
            max_tokens=1,
            device=None,
            executor_mode="park",
        )


@pytest.mark.parametrize(
    ("cores", "threads_per_core", "executor_threads"),
    [(16, 8, 16), (8, 2, 8), (32, 1, 30)],
)
def test_auto_selects_spin_on_supported_single_socket_x86(
    tmp_path, cores, threads_per_core, executor_threads
):
    topology = _topology(
        tmp_path,
        cores_per_socket=cores,
        threads_per_core=threads_per_core,
    )
    decision = decide_cpu_executor_mode(topology, executor_threads)
    assert decision.mode == "spin"
    assert decision.single_socket_x86
    assert decision.cpu_count_le_32
    assert decision.cpus_free_ge_2


def test_auto_selects_sleep_on_multi_socket_x86(tmp_path):
    topology = _topology(
        tmp_path, sockets=2, cores_per_socket=8, threads_per_core=2
    )
    decision = decide_cpu_executor_mode(topology, 8)
    assert decision.mode == "sleep"
    assert decision.reason == "multi-socket system detected"


def test_auto_selects_sleep_on_single_socket_arm(tmp_path):
    topology = _topology(
        tmp_path,
        cores_per_socket=8,
        threads_per_core=2,
        machine="aarch64",
    )
    decision = decide_cpu_executor_mode(topology, 8)
    assert decision.mode == "sleep"
    assert decision.reason == "non-x86 architecture detected"


def test_auto_selects_sleep_with_only_one_cpu_free(tmp_path):
    topology = _topology(tmp_path, cores_per_socket=8, threads_per_core=2)
    decision = decide_cpu_executor_mode(topology, 15)
    assert decision.mode == "sleep"
    assert decision.cpus_free == 1
    assert decision.reason == "fewer than 2 CPUs free"


def test_cpu_list_and_physical_core_parser(tmp_path):
    assert parse_cpu_list("0-3,8-11") == [0, 1, 2, 3, 8, 9, 10, 11]
    topology = _topology(tmp_path, cores_per_socket=4, threads_per_core=2)
    assert len(topology.logical_cpus) == 8
    assert len(topology.physical_cores) == 4
    assert topology.threads_per_core == 2
    assert _spin_core_cpus(topology) == [0, 1, 2, 3]
