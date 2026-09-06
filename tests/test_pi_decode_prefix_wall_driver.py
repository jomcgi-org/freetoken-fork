"""Focused checks for service recovery, lease EOF and geometry qualification."""

import importlib.util
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location('agentic_gate', Path(__file__).parents[1] / 'bench/pi-decode-prefix-wall-driver.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def test_stopping_an_absent_or_inactive_service_is_idempotent(monkeypatch):
    monkeypatch.setattr(gate, 'state', lambda _: {'ActiveState': 'inactive'})
    monkeypatch.setattr(gate, 'run', lambda *_args, **_kwargs: pytest.fail('inactive unit must not be stopped'))
    gate.stop(gate.SERVER)


def test_existing_original_service_survives_recovery(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(gate, 'R', tmp_path)
    monkeypatch.setattr(gate, 'state', lambda _: {'ActiveState': 'active'})
    monkeypatch.setattr(gate, 'stop', lambda unit: calls.append(unit))
    monkeypatch.setattr(gate, 'wait_gpu_release', lambda: pytest.fail('original GPU process is expected'))
    monkeypatch.setattr(gate, 'run', lambda *_args, **_kwargs: pytest.fail('must not restart a healthy original'))
    monkeypatch.setattr(gate, 'ready', lambda port: {'port': port})
    monkeypatch.setattr(gate, 'completion', lambda port: {'verified_port': port})
    result = gate.remote_main(SimpleNamespace(run_id='test', action='restore'))
    assert calls == [gate.ACTION, gate.SERVER]
    assert result['verified'] and result['completion']['verified_port'] == 8090
    assert json.loads((tmp_path / 'results/test/restoration.json').read_text())['verified']


def test_recovery_waits_for_gpu_before_starting_original(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(gate, 'R', tmp_path)
    monkeypatch.setattr(gate, 'state', lambda _: {'ActiveState': 'inactive'})
    monkeypatch.setattr(gate, 'stop', lambda unit: calls.append(('stop', unit)))
    monkeypatch.setattr(gate, 'wait_gpu_release', lambda: calls.append(('gpu',)))
    monkeypatch.setattr(gate, 'run', lambda *args, **kwargs: calls.append(args))
    monkeypatch.setattr(gate, 'ready', lambda port: calls.append(('ready', port)))
    monkeypatch.setattr(gate, 'completion', lambda port: calls.append(('completion', port)))
    result = gate.remote_main(SimpleNamespace(run_id='test', action='restore'))
    assert result['verified']
    assert calls[:4] == [('stop', gate.ACTION), ('stop', gate.SERVER), ('gpu',),
                         ('sudo', '-n', 'systemctl', 'start', 'freetoken-serve')]
    assert calls[-2:] == [('ready', 8090), ('completion', 8090)]


def test_failed_restoration_is_retained(monkeypatch, tmp_path):
    monkeypatch.setattr(gate, 'R', tmp_path)
    monkeypatch.setattr(gate, 'stop', lambda _: None)
    monkeypatch.setattr(gate, 'state', lambda _: {'ActiveState': 'active'})

    def fail(_):
        raise TimeoutError('original never became ready')

    monkeypatch.setattr(gate, 'ready', fail)
    with pytest.raises(TimeoutError):
        gate.remote_main(SimpleNamespace(run_id='test', action='restore'))
    result = json.loads((tmp_path / 'results/test/restoration.json').read_text())
    assert not result['verified'] and 'never became ready' in result['error']


@pytest.mark.parametrize('payload', [b'', b'ping\ndone\n', b'ping\nping\n'])
def test_lease_eof_and_batched_heartbeats_close_without_waiting(payload):
    read_fd, write_fd = os.pipe()
    with os.fdopen(read_fd, 'rb') as stream:
        os.write(write_fd, payload)
        os.close(write_fd)
        assert gate.hold_connection(stream, idle_s=0.05) == {'lease': 'closed'}


def test_missing_heartbeats_expire_lease():
    read_fd, write_fd = os.pipe()
    try:
        with os.fdopen(read_fd, 'rb') as stream:
            with pytest.raises(TimeoutError, match='renewing'):
                gate.hold_connection(stream, idle_s=0.01)
    finally:
        os.close(write_fd)


def test_geometry_ignores_log_timestamps_but_rejects_missing_fields():
    text = '\n'.join([
        'INFO MoE bank split residency: 26.44 GiB pinned + 37.02 GiB file-backed',
        'INFO MoE HOT expert residency: 2296 protected GPU rows',
        'INFO MoE activation dtype: bf16',
        'INFO --moe-cache-auto resolved moe_cache_size=3753 num_pages=1024',
        'INFO Allocating 65536 tokens for KV cache, K + V = 0.80 GiB',
        'INFO CPU MoE executor ready: threads=14 isa=avx512vnni(nvfp4-w4a8) max_tokens=1',
        'INFO Start capturing CUDA graphs with sizes: [1]',
        'INFO KV cache dtype: kv_dtype=fp8_e4m3',
    ])
    assert gate.geometry(text) == gate.geometry(text.replace('INFO', '[different timestamp] INFO'))
    with pytest.raises(AssertionError):
        gate.geometry(text.replace('max_tokens=1', 'max_tokens=4'))
    with pytest.raises(AssertionError):
        gate.geometry(text.replace('KV cache dtype:', 'missing dtype:'))
    with pytest.raises(AssertionError):
        gate.geometry(text.replace('moe_cache_size=3753', 'moe_cache_size=4045'))


def test_only_snapshot_flag_changes_between_serving_arms(monkeypatch, tmp_path):
    service = tmp_path / 'original.service'
    service.write_text(
        '[Service]\nExecStart=/original/bin/ft serve model '
        '--port 8090 --moe-disk-prefill cpu --max-running-requests 1 '
        '--kv-disk-cache-gib 16 --moe-collect-stats\n'
    )
    monkeypatch.setattr(gate, 'SERVICE', service)
    commands = [gate.server_command(mode) for mode in gate.MODES]
    assert commands[0] == commands[1]
    command = commands[0]
    for flag, value in [('--moe-disk-prefill', 'staged'),
                        ('--moe-disk-prefill-io', 'buffered'),
                        ('--moe-hot-staging-io', 'mmap'),
                        ('--cache-type', 'radix'), ('--kv-disk-cache-gib', '0'),
                        ('--kv-reserve-tokens', '65536'), ('--cuda-graph-max-bs', '1')]:
        assert command[command.index(flag) + 1] == value
    assert '--moe-collect-stats' not in command and '--moe-step-timing' not in command
    off, on = (gate.server_env(mode) for mode in gate.MODES)
    assert off.pop('FREETOKEN_DECODE_PREFIX_SNAPSHOT') == '0'
    assert on.pop('FREETOKEN_DECODE_PREFIX_SNAPSHOT') == '1'
    assert off == on and off['FREETOKEN_CONTINUATION_TRACE_DIR'] == ''
    assert off['PYTHONPATH'] == str(gate.OPT / 'python')
    assert gate.REVISIONS['off'] == gate.REVISIONS['on']
    assert gate.BINARIES['off'] == gate.BINARIES['on']


@pytest.mark.parametrize('mismatch', ['command', 'environment'])
def test_preflight_rejects_confounded_comparison_before_touching_services(monkeypatch, mismatch):
    monkeypatch.setattr(gate, 'server_command', lambda mode: [mode] if mismatch == 'command' else ['same'])
    def environment(mode):
        return {'FREETOKEN_DECODE_PREFIX_SNAPSHOT': str(int(mode == 'on')),
                'UNRELATED': mode if mismatch == 'environment' else 'same'}
    monkeypatch.setattr(gate, 'server_env', environment)
    monkeypatch.setattr(gate, 'state', lambda _: pytest.fail('configuration must be rejected before service access'))
    with pytest.raises(AssertionError):
        gate.preflight()


def complete_schedule():
    return [dict(arm=arm, mode=mode, client_returncode=0,
                 warmup={'passed': True}, measured=[{'passed': True}, {'passed': True}])
            for arm, mode in gate.ARMS]


def test_both_orders_and_all_tasks_are_required():
    arms = complete_schedule()
    assert gate.all_tasks_passed(arms)
    assert not gate.all_tasks_passed([])
    assert not gate.all_tasks_passed(arms[:-1])
    arms[1], arms[2] = arms[2], arms[1]
    assert not gate.all_tasks_passed(arms)


@pytest.mark.parametrize('failure', ['warmup', 'measured', 'missing_task', 'client'])
def test_failed_or_incomplete_task_cannot_qualify(failure):
    arms = copy.deepcopy(complete_schedule())
    if failure == 'warmup':
        arms[0]['warmup']['passed'] = False
    elif failure == 'measured':
        arms[2]['measured'][1]['passed'] = False
    elif failure == 'missing_task':
        arms[1]['measured'].pop()
    else:
        arms[3]['client_returncode'] = 1
    assert not gate.all_tasks_passed(arms)


def test_scripted_client_can_use_controller_without_pi(monkeypatch, tmp_path):
    monkeypatch.setattr(gate.sys, 'argv', ['driver', '--fixed-continuation', '--preflight',
                        '--run-id', 'astra-pi-agentic-fixed-test', '--output-dir', str(tmp_path)])
    seen = []
    monkeypatch.setattr(gate, 'local_main', lambda args: seen.append(args) or 0)
    assert gate.main() == 0
    assert seen[0].fixed_continuation and seen[0].preflight and seen[0].pi is None


def test_pi_client_still_requires_its_executable(monkeypatch, tmp_path):
    monkeypatch.setattr(gate.sys, 'argv', ['driver', '--run-id', 'astra-pi-agentic-fixed-test',
                        '--output-dir', str(tmp_path)])
    monkeypatch.setattr(gate, 'local_main', lambda _: pytest.fail('invalid client must not run'))
    with pytest.raises(SystemExit) as failure:
        gate.main()
    assert failure.value.code == 2


@pytest.mark.parametrize('action', ['start', 'end'])
def test_remote_mutating_actions_share_the_lease_lifetime(action):
    command = gate.remote_command('/tmp/driver.py', action, 'astra-pi-agentic-test', 'r1')
    assert '--property=BindsTo=' + gate.LEASE + '.service' in command
    assert '--property=After=' + gate.LEASE + '.service' in command
    assert '--property=KillMode=control-group' in command
    assert '--property=RuntimeMaxSec=660' in command
    assert '--unit=' + gate.ACTION in command
    assert command[-8:] == ['/usr/bin/python3', '/tmp/driver.py', '--remote', action,
                            '--run-id', 'astra-pi-agentic-test', '--arm', 'r1']


def test_restore_can_run_after_lease_termination():
    command = gate.remote_command('/tmp/driver.py', 'restore', 'astra-pi-agentic-test')
    assert command == ['/usr/bin/python3', '/tmp/driver.py', '--remote', 'restore',
                       '--run-id', 'astra-pi-agentic-test']


def test_orphan_action_prevents_starting_a_new_comparison(monkeypatch):
    monkeypatch.setattr(gate, 'server_command', lambda mode: ['same'])
    monkeypatch.setattr(gate, 'server_env', lambda mode: {
        'FREETOKEN_DECODE_PREFIX_SNAPSHOT': '1' if mode == 'on' else '0'})
    monkeypatch.setattr(gate, 'live', lambda unit: unit == gate.ACTION)
    with pytest.raises(AssertionError, match='another benchmark is live'):
        gate.preflight()


def test_prefill_carry_changes_only_runtime_path_with_decode_snapshots_off(monkeypatch):
    monkeypatch.setattr(gate, 'PREFILL_CARRY', True)
    off, on = (gate.server_env(mode) for mode in gate.MODES)
    assert off.pop('PYTHONPATH') == str(gate.OPT / 'python')
    assert on.pop('PYTHONPATH') == str(gate.CARRY / 'python')
    assert off == on and off['FREETOKEN_DECODE_PREFIX_SNAPSHOT'] == '0'
    assert gate.runtime_revision('off') == gate.REVISIONS['off']
    assert gate.runtime_revision('on') != gate.runtime_revision('off')
    assert gate.runtime_revision('original') == gate.REVISIONS['original']


@pytest.mark.parametrize('action', ['preflight', 'hold', 'start', 'end', 'restore', 'restoration'])
def test_prefill_carry_selection_reaches_every_remote_action(monkeypatch, action):
    monkeypatch.setattr(gate, 'PREFILL_CARRY', True)
    command = gate.remote_command('/tmp/driver.py', action, 'astra-pi-agentic-carry-test')
    assert command.count('--prefill-snapshot-carry') == 1
    if action in ('start', 'end'):
        assert '--property=BindsTo=' + gate.LEASE + '.service' in command
    else:
        assert command[0] == '/usr/bin/python3'


def test_prefill_carry_rejects_decode_snapshot_confounds_before_service_access(monkeypatch):
    monkeypatch.setattr(gate, 'PREFILL_CARRY', True)
    monkeypatch.setattr(gate, 'server_command', lambda _: ['same'])
    monkeypatch.setattr(gate, 'server_env', lambda mode: {
        'PYTHONPATH': mode, 'FREETOKEN_DECODE_PREFIX_SNAPSHOT': '1'})
    monkeypatch.setattr(gate, 'state', lambda _: pytest.fail('must reject before accessing services'))
    with pytest.raises(AssertionError):
        gate.preflight()
