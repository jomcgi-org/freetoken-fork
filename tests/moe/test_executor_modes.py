from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from freetoken.moe.cpu_executor import (
    CPU_MOE_SPIN_IDLE_US_MAX,
    CPU_MOE_SPIN_IDLE_US_MIN,
    CpuMoeExecutor,
    _spin_core_cpus,
    decide_cpu_executor_mode,
    parse_cpu_list,
    plan_spin_worker_placement,
    read_cpu_topology,
    validate_spin_idle_us,
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
    assert EngineConfig.__dataclass_fields__["moe_cpu_spin_idle_us"].default == 500
    assert args.moe_cpu_spin_idle_us == 500


@pytest.mark.parametrize(
    "value", [CPU_MOE_SPIN_IDLE_US_MIN, 500, CPU_MOE_SPIN_IDLE_US_MAX]
)
def test_spin_idle_argument_accepts_boundaries(monkeypatch, value):
    _, parse_args = _cli_modules(monkeypatch)
    args, _ = parse_args(
        [
            "--model",
            "/tmp/nonexistent-model",
            "--dtype",
            "bfloat16",
            "--moe-cpu-spin-idle-us",
            str(value),
        ]
    )
    assert args.moe_cpu_spin_idle_us == value
    assert validate_spin_idle_us(value) == value


@pytest.mark.parametrize("value", ["-1", "1000001", "1.5", "five"])
def test_spin_idle_argument_rejects_invalid_values(monkeypatch, value):
    _, parse_args = _cli_modules(monkeypatch)
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--model",
                "/tmp/nonexistent-model",
                "--dtype",
                "bfloat16",
                "--moe-cpu-spin-idle-us",
                value,
            ]
        )


@pytest.mark.parametrize("value", [True, 1.0, "500", None])
def test_spin_idle_validation_rejects_non_integer_types(value):
    with pytest.raises(TypeError, match="must be an integer"):
        validate_spin_idle_us(value)


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


@pytest.mark.parametrize("value", [True, 1.0, "500", None])
def test_executor_constructor_rejects_invalid_spin_idle_type(value):
    with pytest.raises(TypeError, match="spin_idle_us must be an integer"):
        CpuMoeExecutor(
            None,
            top_k=1,
            activation="silu",
            apply_router_weight_on_input=False,
            num_threads=1,
            max_tokens=1,
            device=None,
            spin_idle_us=value,
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


def test_auto_guard_counts_only_workers_that_actually_spin(tmp_path):
    topology = _topology(tmp_path, cores_per_socket=8, threads_per_core=2)
    decision = decide_cpu_executor_mode(topology, 15)
    assert decision.mode == "spin"
    assert decision.cpus_free == 10
    assert decision.reason == "auto-detected suitable CPU topology"


def test_cpu_list_and_physical_core_parser(tmp_path):
    assert parse_cpu_list("0-3,8-11") == [0, 1, 2, 3, 8, 9, 10, 11]
    topology = _topology(tmp_path, cores_per_socket=4, threads_per_core=2)
    assert len(topology.logical_cpus) == 8
    assert len(topology.physical_cores) == 4
    assert topology.threads_per_core == 2
    assert _spin_core_cpus(topology) == [0, 1, 2, 3]


def test_spin_placement_excludes_main_cpu_and_smt_sibling(tmp_path):
    topology = _topology(tmp_path, cores_per_socket=8, threads_per_core=2)

    placement = plan_spin_worker_placement(topology, 14, main_cpu=3)

    assert placement.can_spin
    assert placement.excluded_cpus == (3, 11)
    assert placement.spin_threads == 6
    assert placement.spin_cpus == (0, 1, 2, 4, 5, 6)
    assert len(placement.worker_cpus) == 14
    assert 3 not in placement.worker_cpus
    assert 11 not in placement.worker_cpus


def test_spin_guard_falls_back_on_two_core_topology(tmp_path):
    topology = _topology(tmp_path, cores_per_socket=2, threads_per_core=2)

    decision = decide_cpu_executor_mode(topology, executor_threads=1)
    placement = plan_spin_worker_placement(topology, 1, main_cpu=0)

    assert decision.mode == "sleep"
    assert decision.reason == "fewer than 3 physical cores available for spin placement"
    assert not placement.can_spin
    assert placement.reason == decision.reason


def test_explicit_spin_logs_guard_fallback_on_two_core_topology(
    monkeypatch, tmp_path
):
    import freetoken.kernel as kernel
    import freetoken.moe.cpu_executor as cpu_executor_module

    topology = _topology(tmp_path, cores_per_socket=2, threads_per_core=2)
    native_kwargs = {}
    logs = []

    class NativeExecutor:
        def __init__(self, **kwargs):
            native_kwargs.update(kwargs)

        def isa_name(self):
            return "test"

    pointer_args = {
        "gate_up_ptr": 0,
        "down_ptr": 0,
        "gate_up_scale_ptr": 0,
        "gate_up_global_ptr": 0,
        "down_scale_ptr": 0,
        "down_global_ptr": 0,
        "gate_up_bias_ptr": 0,
        "down_bias_ptr": 0,
    }
    monkeypatch.setattr(
        kernel,
        "_cpu_moe",
        SimpleNamespace(CpuMoeExecutor=NativeExecutor),
        raising=False,
    )
    monkeypatch.setattr(cpu_executor_module, "read_cpu_topology", lambda: topology)
    monkeypatch.setattr(cpu_executor_module, "_current_logical_cpu", lambda: 0)
    monkeypatch.setattr(cpu_executor_module, "physical_core_cpus", lambda: [0, 1])
    monkeypatch.setattr(cpu_executor_module.logger, "info_rank0", logs.append)
    monkeypatch.setattr(cpu_executor_module.torch, "get_num_threads", lambda: 1)
    monkeypatch.setattr(cpu_executor_module.torch, "set_num_threads", lambda _n: None)
    monkeypatch.setattr(
        CpuMoeExecutor,
        "_resolve_banks",
        lambda _self, _banks, _fmt: (pointer_args, (8, 16)),
    )

    executor = CpuMoeExecutor(
        SimpleNamespace(
            quant_format="bf16",
            num_layers=1,
            num_experts=1,
            bank_sources={},
        ),
        top_k=1,
        activation="silu",
        apply_router_weight_on_input=False,
        num_threads=1,
        max_tokens=1,
        device=SimpleNamespace(type="cpu"),
        executor_mode="spin",
        prefill_batch="off",
    )

    assert executor._executor_mode == "sleep"
    assert native_kwargs["spin_mode"] is False
    assert native_kwargs["spin_thread_count"] == 0
    assert any(
        "explicit spin fell back to sleep "
        "(fewer than 3 physical cores available for spin placement)" in message
        for message in logs
    )


def test_sleep_worker_submit_and_sync_paths_have_no_spin_checks_or_clocks():
    source = (
        Path(__file__).parents[2]
        / "python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp"
    ).read_text()
    worker = source.partition("  void worker_loop(int tid) {")[2].partition(
        "  void timed_worker_loop(int tid) {"
    )[0]
    submit = source.partition("  void submit(MoeTask* t,")[2].partition(
        "  void sync(MoeTask* timing_task"
    )[0]
    sync = source.partition("  void sync(MoeTask* timing_task")[2].partition(
        "  void spin_sync(MoeTask* timing_task"
    )[0]

    assert "spin_" not in worker
    assert "steady_clock" not in worker
    assert "spin_" not in submit
    assert "steady_clock" not in submit
    assert submit.count("task_cv.notify_all();") == 2
    assert "spin_" not in sync
    assert "steady_clock" not in sync
