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


def source_log(mode):
    gpu = list(range(20))
    disk = list(range(20, 48)) if mode == 'off' else list(range(48))
    bank = (f'MoE bank split residency: 0.00 GiB pinned (cudaHostRegister, '
            f'{20 if mode == "off" else 0} GPU layers) + '
            f'0.00 GiB OS-locked (0 CPU layers: []) + 0.00 GiB file-backed '
            f'({len(disk)} DISK layers: {disk})')
    paths = f'MoE prefill paths: overlap=0; GPU candidates={gpu}, chosen={gpu}, bytes=0 B; budget=100 B - reserved=0 B = available=100'
    staging = f'MoE GPU source staging: layers={gpu}; synthetic fixture' if mode == 'on' else ''
    return '\n'.join([bank, paths, staging])


def test_gpu_source_changes_only_explicit_source_flag(monkeypatch, tmp_path):
    monkeypatch.setattr(gate, 'GPU_SOURCE_STAGING', True)
    service = tmp_path / 'original.service'
    service.write_text('ExecStart=/bin/ft --port 8090 --moe-disk-prefill cpu '
                       '--max-running-requests 1 --kv-disk-cache-gib 1 --moe-collect-stats')
    monkeypatch.setattr(gate, 'SERVICE', service)
    commands = {mode: gate.server_command(mode) for mode in gate.MODES}
    gate.qualify_source_commands(commands)
    assert gate.server_env('off') == gate.server_env('on')
    assert gate.server_env('on')['FREETOKEN_DECODE_PREFIX_SNAPSHOT'] == '0'
    assert gate.runtime_tree('off') == gate.runtime_tree('on') == gate.GPU_SOURCE
    assert gate.runtime_revision('off') == gate.runtime_revision('on') == gate.GPU_SOURCE_REVISION
    assert gate.runtime_revision('original') == gate.REVISIONS['original']


@pytest.mark.parametrize('change', ['duplicate', 'unrelated', 'wrong', 'missing_value'])
def test_gpu_source_commands_reject_confounds(change):
    commands = {'off': ['ft', '--moe-gpu-source', 'pinned'],
                'on': ['ft', '--moe-gpu-source', 'staged']}
    if change == 'duplicate':
        commands['on'] += ['--moe-gpu-source', 'staged']
    elif change == 'unrelated':
        commands['on'] += ['--extra']
    elif change == 'wrong':
        commands['on'][-1] = 'pinned'
    else:
        commands['on'].pop()
    with pytest.raises(AssertionError):
        gate.qualify_source_commands(commands)


def test_source_layout_allows_backing_changes_but_preserves_gpu_compute():
    off, on = (gate.source_layout(source_log(mode), mode) for mode in gate.MODES)
    assert off['gpu_layer_ids'] == on['gpu_layer_ids'] == list(range(20))
    assert off['file_layer_ids'] == list(range(20, 48))
    assert on['file_layer_ids'] == list(range(48))
    with pytest.raises(AssertionError, match='wrong file-backed'):
        gate.source_layout(source_log('off'), 'on')
    with pytest.raises(AssertionError, match='wrong pinned'):
        gate.source_layout(source_log('on').replace('0 GPU layers)', '20 GPU layers)'), 'on')
    with pytest.raises(AssertionError, match='wrong GPU staging'):
        gate.source_layout(source_log('on').replace('staging: layers=[0,', 'staging: layers=[99,'), 'on')


def test_gpu_placement_ignores_budget_estimate_but_rejects_changed_selection():
    original = source_log('off')
    changed_budget = original.replace('budget=100', 'budget=110').replace('available=100', 'available=110')
    assert gate.gpu_compute_placement(original) == gate.gpu_compute_placement(changed_budget)
    for before, after in [('chosen=[0,', 'chosen=[99,'), ('candidates=[0,', 'candidates=[99,'),
                          ('bytes=0', 'bytes=1'), ('reserved=0', 'reserved=1')]:
        assert gate.gpu_compute_placement(original) != gate.gpu_compute_placement(original.replace(before, after))


@pytest.mark.parametrize('action', ['preflight', 'hold', 'start', 'end', 'restore', 'restoration'])
def test_gpu_source_mode_reaches_remote_recovery_and_serving(monkeypatch, action):
    monkeypatch.setattr(gate, 'GPU_SOURCE_STAGING', True)
    command = gate.remote_command('/tmp/driver.py', action, 'astra-pi-agentic-source-test')
    assert command.count('--gpu-source-staging') == 1


@pytest.mark.parametrize('unit', ['astra-gpu-source-staging-cost', 'astra-gpu-source-staging-validation',
                                 'astra-hot-host-cache-reclaim-validation', 'astra-hot-host-cache-reclaim-census'])
def test_staging_component_job_prevents_model_benchmark(monkeypatch, unit):
    monkeypatch.setattr(gate, 'server_command', lambda _: ['same'])
    monkeypatch.setattr(gate, 'server_env', lambda mode: {
        'FREETOKEN_DECODE_PREFIX_SNAPSHOT': '1' if mode == 'on' else '0'})
    monkeypatch.setattr(gate, 'live', lambda name: name == unit)
    with pytest.raises(AssertionError, match='another benchmark is live'):
        gate.preflight()


def test_os_snapshot_uses_bytes_and_preserves_shared_memory(monkeypatch):
    def read(path):
        fields = gate.SYSTEM_MEMORY_FIELDS if str(path) == '/proc/meminfo' else gate.WORKER_MEMORY_FIELDS
        return '\n'.join(f'{key}: {i + 1} kB' for i, key in enumerate(fields))
    monkeypatch.setattr(gate.Path, 'read_text', read)
    snapshot = gate.memory_snapshot(123)
    assert snapshot['worker_bytes']['RssShmem'] == 5 * 1024
    assert snapshot['system_bytes']['MemAvailable'] == 1024


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


@pytest.mark.parametrize('missing', [None, *gate.EXTENSION_SOURCES])
def test_native_identity_requires_every_serving_extension(tmp_path, missing):
    kernel = tmp_path / 'python/freetoken/kernel'
    for name, source in gate.EXTENSION_SOURCES.items():
        code = kernel / 'csrc' / source
        code.parent.mkdir(parents=True, exist_ok=True)
        code.write_bytes(b'synthetic source')
        if name != missing:
            (kernel / (name + '.test.so')).write_bytes(b'synthetic extension')
    if missing:
        with pytest.raises(AssertionError, match='missing or ambiguous native extension'):
            gate.native_extensions(tmp_path)
    else:
        result = gate.native_extensions(tmp_path)
        assert set(result) == set(gate.EXTENSION_SOURCES)
        for name, row in result.items():
            assert row['sha256'] == gate.sha(kernel / (name + '.test.so'))
            assert row['source_sha256'] == gate.sha(kernel / 'csrc' / gate.EXTENSION_SOURCES[name])


def test_profile_retention_compares_with_carried_prefill_parent(monkeypatch):
    monkeypatch.setattr(gate, 'PROFILE_RETENTION', True)
    off, on = (gate.server_env(mode) for mode in gate.MODES)
    assert off.pop('PYTHONPATH') == str(gate.CARRY / 'python')
    assert on.pop('PYTHONPATH') == str(gate.PROFILE / 'python')
    assert off == on and off['FREETOKEN_DECODE_PREFIX_SNAPSHOT'] == '0'
    assert gate.runtime_revision('off') == gate.PREFILL_CARRY_REVISIONS['on']
    assert gate.runtime_revision('on') == '5b0ea43e03971759f7355f5a9c80e8065636f31b'
    assert gate.runtime_revision('original') == gate.REVISIONS['original']


def test_profile_retention_pins_advice_policy_in_both_commands(monkeypatch):
    monkeypatch.setattr(gate, 'PROFILE_RETENTION', True)
    service = ('ExecStart=/bin/ft --port 8090 --moe-disk-prefill staged '
               '--max-running-requests 4 --kv-disk-cache-gib 2 --moe-collect-stats '
               '--session-expert-prefetch off --session-protect-experts 32')
    monkeypatch.setattr(gate, 'SERVICE', SimpleNamespace(read_text=lambda: service))
    off, on = (gate.server_command(mode) for mode in gate.MODES)
    assert off == on
    for flag, value in [('--session-expert-prefetch', 'on'), ('--session-protect-experts', '64'),
                        ('--max-running-requests', '1')]:
        assert off.count(flag) == 1 and off[off.index(flag) + 1] == value
    assert '--moe-collect-stats' not in off and '--moe-step-timing' not in off


@pytest.mark.parametrize('action', ['preflight', 'hold', 'start', 'end', 'restore', 'restoration'])
def test_profile_retention_selection_reaches_every_remote_action(monkeypatch, action):
    monkeypatch.setattr(gate, 'PROFILE_RETENTION', True)
    command = gate.remote_command('/tmp/driver.py', action, 'astra-pi-agentic-profile-test')
    assert command.count('--hybrid-profile-retention') == 1
    assert '--prefill-snapshot-carry' not in command
    if action in ('start', 'end'):
        assert '--property=BindsTo=' + gate.LEASE + '.service' in command


@pytest.mark.parametrize('files', ['cache.py', 'prefill.py', 'cache.py\nprefill.py'])
def test_profile_retention_preflight_allows_only_the_cache_finish_change(monkeypatch, files):
    monkeypatch.setattr(gate, 'PROFILE_RETENTION', True)
    command = ['ft', '--session-expert-prefetch', 'on', '--session-protect-experts', '64']
    monkeypatch.setattr(gate, 'server_command', lambda _: command)
    monkeypatch.setattr(gate, 'live', lambda _: False)
    monkeypatch.setattr(gate, 'state', lambda _: {'ActiveState': 'active', 'ControlGroup': 'group'})
    monkeypatch.setattr(gate, 'gpu_pids', lambda: [123])
    monkeypatch.setattr(Path, 'read_text', lambda _: 'group')
    monkeypatch.setattr(gate, 'sha', lambda _: 'sha')
    identity = dict(native='same', native_sha256='same', cpp_sha256='same', native_extensions='same')
    monkeypatch.setattr(gate, 'identities', lambda: {m: identity for m in gate.MODES})

    def diff(*args, **kwargs):
        assert args == ('git', '-C', str(gate.PROFILE), 'diff', '--name-only',
                        gate.PROFILE_RETENTION_REVISIONS['off'], gate.PROFILE_RETENTION_REVISIONS['on'],
                        '--', 'python/')
        return SimpleNamespace(stdout='\n'.join('python/freetoken/scheduler/' + f
                                                for f in files.splitlines()))

    monkeypatch.setattr(gate, 'run', diff)
    if files == 'cache.py':
        assert gate.preflight()['experiment'] == 'hybrid-profile-retention'
    else:
        with pytest.raises(AssertionError):
            gate.preflight()


def test_conflicting_source_pairs_are_rejected_before_services(monkeypatch):
    monkeypatch.setattr(gate, 'PREFILL_CARRY', True)
    monkeypatch.setattr(gate, 'PROFILE_RETENTION', True)
    monkeypatch.setattr(gate, 'server_command', lambda _: pytest.fail('must reject first'))
    with pytest.raises(AssertionError, match='conflicting experiments'):
        gate.preflight()


@pytest.mark.parametrize('action', ['preflight', 'hold', 'start', 'end', 'restore', 'restoration'])
def test_hot_host_mode_preserves_runtime_and_reaches_recovery(monkeypatch, tmp_path, action):
    monkeypatch.setattr(gate, 'HOT_HOST_RECLAIM', True)
    service = tmp_path / 'original.service'
    service.write_text('ExecStart=/bin/ft --port 8090 --moe-disk-prefill cpu '
                       '--max-running-requests 1 --kv-disk-cache-gib 1 --moe-collect-stats')
    monkeypatch.setattr(gate, 'SERVICE', service)
    commands = {mode: gate.server_command(mode) for mode in gate.MODES}
    gate.qualify_hot_host_commands(commands)
    assert gate.server_env('off') == gate.server_env('on')
    env = gate.server_env('on')
    assert env['FREETOKEN_DECODE_PREFIX_SNAPSHOT'] == '0'
    assert env['FREETOKEN_HOT_HOST_CACHE_CENSUS_DIR'] == ''
    assert env['PYTHONPATH'] == str(gate.HOT_HOST / 'python')
    assert gate.runtime_tree('off') == gate.runtime_tree('on') == gate.HOT_HOST
    assert gate.runtime_revision('off') == gate.runtime_revision('on') == gate.HOT_HOST_REVISION
    assert gate.runtime_revision('original') == gate.REVISIONS['original']
    command = gate.remote_command('/tmp/driver.py', action, 'astra-pi-agentic-hot-host-test')
    assert command.count('--hot-host-cache-reclaim') == 1


@pytest.mark.parametrize('change', ['duplicate', 'unrelated', 'wrong', 'missing_value'])
def test_hot_host_commands_reject_confounds(change):
    commands = {'off': ['ft', '--moe-hot-host-cache', 'retain'],
                'on': ['ft', '--moe-hot-host-cache', 'reclaim']}
    if change == 'duplicate':
        commands['on'] += ['--moe-hot-host-cache', 'reclaim']
    elif change == 'unrelated':
        commands['on'] += ['--extra']
    elif change == 'wrong':
        commands['on'][-1] = 'retain'
    else:
        commands['on'].pop()
    with pytest.raises(AssertionError):
        gate.qualify_hot_host_commands(commands)


@pytest.mark.parametrize('action', ['preflight', 'hold', 'start', 'end', 'restore', 'restoration'])
def test_original_baseline_selects_qualified_runtime_and_recovery(monkeypatch, tmp_path, action):
    monkeypatch.setattr(gate, 'ORIGINAL_BASELINE', True)
    service = tmp_path / 'original.service'
    service.write_text('ExecStart=/bin/ft --port 8090 --moe-disk-prefill cpu '
                       '--max-running-requests 1 --kv-disk-cache-gib 1 --moe-collect-stats')
    monkeypatch.setattr(gate, 'SERVICE', service)
    commands = {mode: gate.server_command(mode) for mode in gate.MODES}
    gate.qualify_original_commands(commands)
    env = {mode: gate.server_env(mode) for mode in gate.MODES}
    assert {k for k in env['off'] if env['off'][k] != env['on'][k]} == {'PYTHONPATH'}
    assert all(e['FREETOKEN_DECODE_PREFIX_SNAPSHOT'] == '0' for e in env.values())
    assert all(e['FREETOKEN_HOT_HOST_CACHE_CENSUS_DIR'] == '' for e in env.values())
    assert gate.runtime_tree('off') == gate.SRC
    assert gate.runtime_tree('on') == gate.HOT_HOST
    assert gate.runtime_revision('off') == gate.REVISIONS['original']
    assert gate.runtime_revision('on') == gate.HOT_HOST_REVISION
    command = gate.remote_command('/tmp/driver.py', action, 'astra-pi-agentic-original-test')
    assert command.count('--original-baseline') == 1


@pytest.mark.parametrize('change', ['baseline_reader', 'duplicate', 'unrelated', 'wrong', 'missing_value'])
def test_original_baseline_commands_reject_confounds(change):
    commands = {'off': ['ft', '--moe-disk-prefill', 'cpu'],
                'on': ['ft', '--moe-disk-prefill', 'staged', '--moe-disk-prefill-io', 'buffered',
                       '--moe-hot-staging-io', 'mmap', '--moe-hot-host-cache', 'reclaim']}
    if change == 'baseline_reader':
        commands['off'] += ['--moe-disk-prefill-io', 'buffered']
    elif change == 'duplicate':
        commands['on'] += ['--moe-disk-prefill', 'staged']
    elif change == 'unrelated':
        commands['on'] += ['--extra']
    elif change == 'wrong':
        commands['on'][-1] = 'buffered'
    else:
        commands['on'].pop()
    with pytest.raises(AssertionError):
        gate.qualify_original_commands(commands)
