"""Controlled decode-snapshot off/on sessions on node-4, with remote recovery.

The same file runs as a Mac controller and a Linux service helper. All model
work is remote. Select Pi tasks or scripted continuation requests. No runtime edits.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import select
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request


R = Path('/var/lib/longhorn/nvme-02/freetoken')
SERVICE = Path('/etc/systemd/system/freetoken-serve.service')
SRC = R / 'wt-plegather'
OPT = R / 'wt-astra-decode-prefix-snapshot'
CARRY = R / 'wt-astra-prefill-snapshot-carry'
PREFILL_CARRY = False
PROFILE = R / 'wt-astra-hybrid-profile-retention'
PROFILE_RETENTION = False
GPU_SOURCE = R / 'wt-astra-gpu-source-staging'
GPU_SOURCE_STAGING = False
HOT_HOST = R / 'wt-astra-hot-host-cache-reclaim'
HOT_HOST_RECLAIM = False
HOT_HOST_REVISION = 'f15cb030ced4fbb6ab7971485528cc972d288aea'
GPU_SOURCE_REVISION = '3f1e4dfc860232d4eafb7638423ee883ff879599'
EXTENSION_SOURCES = {
    '_cpu_moe': 'cpu_moe/cpu_moe_ext.cpp',
    '_pinned_tensor': 'pinned_tensor.cpp',
    '_ple_uring': 'ple_uring/ple_uring_ext.cpp',
    '_uffd_pager': 'uffd_pager.cpp',
}
SERVER = 'astra-pi-agentic-server'
LEASE = 'astra-pi-agentic-lease'
ACTION = 'astra-pi-agentic-action'
SSH = ['ssh', '-oIdentityAgent=none', '-oIdentitiesOnly=yes', '-oBatchMode=yes',
       '-oServerAliveInterval=15', '-oServerAliveCountMax=3']
REVISIONS = {'original': '3a67403a293be20836604bc25729329997848e09',
             'off': '9c2eb77e9abe3f843b5e50ed55e90871d57ed8d9',
             'on': '9c2eb77e9abe3f843b5e50ed55e90871d57ed8d9'}
PREFILL_CARRY_REVISIONS = {'off': REVISIONS['off'],
                         'on': '58a5355fc671a5dc5892bbb76f006950e4b5ab54'}
PROFILE_RETENTION_REVISIONS = {'off': PREFILL_CARRY_REVISIONS['on'],
                              'on': '5b0ea43e03971759f7355f5a9c80e8065636f31b'}
BINARIES = {'original': 'aad1f3169b9b39829a356877de7e5add9b8c8a0173837f54140871b24e0c0048',
            'off': 'c88ed9f877a5a6c4cb3eb4c172b0a7a953794e3ff1104a12b8dcb0f22fb4810f',
            'on': 'c88ed9f877a5a6c4cb3eb4c172b0a7a953794e3ff1104a12b8dcb0f22fb4810f'}
CLIENT_SHA = '986875b712ff05862c03417e56e45c18559d82e05af913e134055a84268715d5'
MODES = ('off', 'on')
ARMS = [('r1', 'off'), ('r2', 'on'), ('r3', 'on'), ('r4', 'off')]



def run(*args, **kwargs):
    return subprocess.run(args, text=True, check=True, capture_output=True, **kwargs)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def state(unit):
    result = run('systemctl', 'show', unit, '-p', 'ActiveState', '-p', 'MainPID',
                 '-p', 'InvocationID', '-p', 'ControlGroup', '-p', 'Result')
    return dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)


def live(unit):
    return state(unit)['ActiveState'] in ('active', 'activating', 'deactivating')


def stop(unit):
    if state(unit)['ActiveState'] != 'inactive':
        run('sudo', '-n', 'systemctl', 'stop', unit, timeout=180)


def gpu_pids():
    result = run('nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits', timeout=10)
    return [int(line) for line in result.stdout.splitlines() if line.strip()]


def wait_gpu_release():
    deadline = time.monotonic() + 45
    while gpu_pids():
        if time.monotonic() >= deadline:
            raise RuntimeError('GPU still has a process after service shutdown')
        time.sleep(1)


def ready(port):
    deadline = time.monotonic() + 420
    while True:
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=3) as response:
                result = json.load(response)
            if result.get('status') == 'error':
                raise RuntimeError(result)
            if result.get('status') == 'ok':
                return result
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        if time.monotonic() > deadline:
            raise TimeoutError(f'server on {port} did not become ready')
        time.sleep(2)


def completion(port):
    payload = dict(model='qwen3.6-27b', messages=[dict(role='user', content='Reply with only OK.')],
                   max_tokens=8, temperature=0, chat_template_kwargs=dict(enable_thinking=False))
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',
                                 data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=120) as response:
        result = json.load(response)
    assert result['choices'][0]['message']['content'].strip() == 'OK', result
    return result


def runtime_tree(mode):
    if HOT_HOST_RECLAIM:
        return HOT_HOST
    if GPU_SOURCE_STAGING:
        return GPU_SOURCE
    if PROFILE_RETENTION:
        return PROFILE if mode == 'on' else CARRY
    return CARRY if PREFILL_CARRY and mode == 'on' else OPT


def runtime_revision(mode):
    if HOT_HOST_RECLAIM and mode != 'original':
        return HOT_HOST_REVISION
    if GPU_SOURCE_STAGING and mode != 'original':
        return GPU_SOURCE_REVISION
    if PROFILE_RETENTION and mode != 'original':
        return PROFILE_RETENTION_REVISIONS[mode]
    return PREFILL_CARRY_REVISIONS[mode] if PREFILL_CARRY and mode != 'original' else REVISIONS[mode]


def runtime_pair():
    return PREFILL_CARRY or PROFILE_RETENTION


def same_runtime_policy():
    return GPU_SOURCE_STAGING or HOT_HOST_RECLAIM


def experiment_name():
    if HOT_HOST_RECLAIM:
        return 'hot-host-cache-reclaim'
    if GPU_SOURCE_STAGING:
        return 'gpu-source-staging'
    if PROFILE_RETENTION:
        return 'hybrid-profile-retention'
    return 'prefill-snapshot-carry' if PREFILL_CARRY else 'decode-prefix-snapshot'


def native_extensions(tree):
    kernel = tree / 'python/freetoken/kernel'
    result = {}
    for name, source in EXTENSION_SOURCES.items():
        binaries = list(kernel.glob(name + '.*.so'))
        assert len(binaries) == 1, ('missing or ambiguous native extension', name)
        native = binaries[0].resolve(strict=True)
        result[name] = dict(path=str(native), sha256=sha(native),
                            source_sha256=sha(kernel / 'csrc' / source))
    return result


def identities():
    result = {}
    for mode, tree in [('original', SRC), *[(m, runtime_tree(m)) for m in MODES]]:
        revision = run('git', '-C', str(tree), 'rev-parse', 'HEAD').stdout.strip()
        assert revision == runtime_revision(mode), (mode, revision)
        assert not run('git', '-C', str(tree), 'status', '--porcelain').stdout.strip(), tree
        extensions = native_extensions(tree)
        native = Path(extensions['_cpu_moe']['path'])
        assert sha(native) == BINARIES[mode], native
        result[mode] = dict(revision=revision, tree=str(tree), native=str(native), native_sha256=sha(native),
                            cpp_sha256=sha(tree / 'python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp'),
                            native_extensions=extensions)
    return result


def server_command(mode):
    assert mode in MODES, mode
    service = SERVICE.read_text().replace('\\\n', ' ')
    args = shlex.split(next(line[10:] for line in service.splitlines() if line.startswith('ExecStart=')))
    args[0] = str(SRC / '.venv/bin/ft')
    for flag, value in [('--port', '18090'), ('--moe-disk-prefill', 'staged'),
                        ('--max-running-requests', '1'), ('--kv-disk-cache-gib', '0')]:
        args[args.index(flag) + 1] = value
    args.remove('--moe-collect-stats')
    args.extend(['--moe-hot-plan-persist', 'off', '--cache-type', 'radix', '--kv-ladder', 'off',
                 '--kv-reserve-tokens', '65536', '--cuda-graph-max-bs', '1'])
    args.extend(['--moe-disk-prefill-io', 'buffered', '--moe-hot-staging-io', 'mmap'])
    if PROFILE_RETENTION:
        for flag, value in [('--session-expert-prefetch', 'on'), ('--session-protect-experts', '64')]:
            if flag in args:
                args[args.index(flag) + 1] = value
            else:
                args.extend([flag, value])
    if GPU_SOURCE_STAGING:
        assert '--moe-gpu-source' not in args, 'original service already overrides GPU sources'
        args.extend(['--moe-gpu-source', 'staged' if mode == 'on' else 'pinned'])
    if HOT_HOST_RECLAIM:
        assert '--moe-hot-host-cache' not in args, 'original service already overrides HOT host cache'
        args.extend(['--moe-hot-host-cache', 'reclaim' if mode == 'on' else 'retain'])
    assert '--moe-collect-stats' not in args and '--moe-step-timing' not in args
    return args


def server_env(mode):
    assert mode in MODES, mode
    return dict(CUDA_HOME='/usr/local/cuda-13.0',
                PATH=f'{SRC}/.venv/bin:/usr/local/cuda-13.0/bin:/usr/bin:/bin', TMPDIR=str(R / 'tmp'),
                PYTHONPATH=str(runtime_tree(mode) / 'python'),
                FREETOKEN_DECODE_PREFIX_SNAPSHOT='1' if mode == 'on' and not runtime_pair() and not same_runtime_policy() else '0',
                FREETOKEN_CONTINUATION_TRACE_DIR='', FREETOKEN_HOT_HOST_CACHE_CENSUS_DIR='',
                FREETOKEN_PREFILL_SELECTIVE_MAX_TOKENS='128', FREETOKEN_PREFILL_HOT_OVERLAP='0')


def qualify_source_commands(commands):
    normalized = []
    for mode, value in [('off', 'pinned'), ('on', 'staged')]:
        command = commands[mode].copy()
        assert command.count('--moe-gpu-source') == 1, 'missing or duplicate source flag'
        index = command.index('--moe-gpu-source')
        assert command[index + 1:index + 2] == [value], 'wrong GPU source flag'
        del command[index:index + 2]
        normalized.append(command)
    assert normalized[0] == normalized[1], 'unrelated GPU source command difference'


def qualify_hot_host_commands(commands):
    normalized = []
    for mode, value in [('off', 'retain'), ('on', 'reclaim')]:
        command = commands[mode].copy()
        assert command.count('--moe-hot-host-cache') == 1, 'missing or duplicate HOT host cache flag'
        index = command.index('--moe-hot-host-cache')
        assert command[index + 1:index + 2] == [value], 'wrong HOT host cache flag'
        del command[index:index + 2]
        normalized.append(command)
    assert normalized[0] == normalized[1], 'unrelated HOT host cache command difference'


def preflight(allow_lease=False):
    assert sum((PREFILL_CARRY, PROFILE_RETENTION, GPU_SOURCE_STAGING, HOT_HOST_RECLAIM)) <= 1, 'conflicting experiments'
    commands = {mode: server_command(mode) for mode in MODES}
    env = {mode: server_env(mode) for mode in MODES}
    if GPU_SOURCE_STAGING:
        qualify_source_commands(commands)
    elif HOT_HOST_RECLAIM:
        qualify_hot_host_commands(commands)
    else:
        assert commands['off'] == commands['on']
    differences = {key for key in env['off'].keys() | env['on'].keys()
                   if env['off'].get(key) != env['on'].get(key)}
    expected = set() if same_runtime_policy() else ({'PYTHONPATH'} if runtime_pair() else {'FREETOKEN_DECODE_PREFIX_SNAPSHOT'})
    assert differences == expected, differences
    if runtime_pair() or same_runtime_policy():
        assert all(e['FREETOKEN_DECODE_PREFIX_SNAPSHOT'] == '0' for e in env.values())
    if HOT_HOST_RECLAIM:
        assert all(e['FREETOKEN_HOT_HOST_CACHE_CENSUS_DIR'] == '' for e in env.values()), 'HOT census enabled'
    if PROFILE_RETENTION:
        for flag, value in [('--session-expert-prefetch', 'on'), ('--session-protect-experts', '64')]:
            assert commands['off'].count(flag) == 1
            assert commands['off'][commands['off'].index(flag) + 1] == value
    for unit in ['astra-decode-weight-reuse-wall-driver', 'astra-concurrent-wall-driver',
                 'astra-decode-weight-reuse-validation-v2', 'astra-sustained-hot-staging-wall-driver',
                 'astra-gpu-source-staging-cost', 'astra-gpu-source-staging-validation',
                 'astra-hot-host-cache-reclaim-validation', 'astra-hot-host-cache-reclaim-census',
                 'astra-sustained-reader-wall-driver', SERVER, ACTION, *([] if allow_lease else [LEASE])]:
        assert not live(unit), ('another benchmark is live', unit)
    original = state('freetoken-serve')
    assert original['ActiveState'] == 'active', original
    workers = gpu_pids()
    assert len(workers) == 1 and original['ControlGroup'] in Path(f'/proc/{workers[0]}/cgroup').read_text()
    runtime_ids = identities()
    if same_runtime_policy():
        assert runtime_ids['off'] == runtime_ids['on'], 'policy runtime identities differ'
    if runtime_pair():
        for key in ('native', 'native_sha256', 'cpp_sha256', 'native_extensions'):
            assert runtime_ids['off'][key] == runtime_ids['on'][key], key
        changed = run('git', '-C', str(runtime_tree('on')), 'diff', '--name-only',
                      runtime_revision('off'), runtime_revision('on'),
                      '--', 'python/').stdout.splitlines()
        expected_file = 'cache.py' if PROFILE_RETENTION else 'prefill.py'
        assert changed == ['python/freetoken/scheduler/' + expected_file], changed
    return dict(identities=runtime_ids, original_unit=original,
                experiment=experiment_name(),
                service_sha256=sha(SERVICE),
                model_config_sha256=sha(R / 'models/flash-e2m1.ftw/config.json'),
                layer_profile_sha256=sha(R / 'layer-profile-v3.json'),
                driver_sha256=sha(Path(__file__)),
                commands=commands, env=env)


def journal(invocation):
    return run('journalctl', '_SYSTEMD_INVOCATION_ID=' + invocation, '--no-pager', '-o', 'cat').stdout


def geometry(text, *, source_staging=False):
    lines = re.sub(r'\x1b\[[0-9;]*m', '', text).splitlines()
    markers = ['MoE bank split residency:', 'MoE HOT expert residency:', 'MoE activation dtype:',
               '--moe-cache-auto resolved', 'Allocating 65536 tokens for KV cache',
               'CPU MoE executor ready:', 'Start capturing CUDA graphs with sizes:', 'KV cache dtype:']
    result = {}
    for marker in markers:
        found = [line[line.index(marker):] for line in lines if marker in line]
        assert len(found) == 1, (marker, found)
        result[marker] = found[0]
    assert 'threads=14' in result['CPU MoE executor ready:']
    assert 'max_tokens=1' in result['CPU MoE executor ready:']
    assert 'avx512vnni(nvfp4-w4a8)' in result['CPU MoE executor ready:']
    assert result['Start capturing CUDA graphs with sizes:'].endswith('[1]')
    assert 'fp8_e4m3' in result['KV cache dtype:']
    assert 'moe_cache_size=3753' in result['--moe-cache-auto resolved']
    if source_staging:
        # Only the host backing and overlap counts may change. Preserve the
        # ordinary GPU selection, HOT residency and all cache geometry checks.
        del result['MoE bank split residency:']
        found = [line[line.index('GPU candidates='):] for line in lines
                 if 'MoE prefill paths:' in line and 'GPU candidates=' in line]
        assert len(found) == 1, 'missing or ambiguous GPU compute placement'
        result['GPU compute placement'] = gpu_compute_placement(found[0])
    return result


def gpu_compute_placement(text):
    """Compare placement, excluding the volatile available-host-memory estimate."""
    result = {}
    for name in ('candidates', 'chosen'):
        match = re.search(name + r'=(\[[0-9, ]*\])', text)
        assert match, ('missing GPU placement field', name)
        result[name] = json.loads(match[1])
    for name in ('bytes', 'reserved'):
        match = re.search(name + r'=(\d+)', text)
        assert match, ('missing GPU placement field', name)
        result[name] = int(match[1])
    return result


def source_layout(text, mode):
    lines = re.sub(r'\x1b\[[0-9;]*m', '', text).splitlines()
    def one(marker):
        found = [line[line.index(marker):] for line in lines if marker in line]
        assert len(found) == 1, ('missing or ambiguous source layout', marker)
        return found[0]
    bank = one('MoE bank split residency:')
    paths = one('MoE prefill paths:')
    chosen = re.search(r'chosen=(\[[0-9, ]*\])', paths)
    assert chosen, 'missing GPU layer selection'
    gpu = json.loads(chosen[1])
    assert len(gpu) == len(set(gpu)) == 20 and set(gpu) <= set(range(48)), 'wrong GPU layer selection'
    disk = re.search(r'\((\d+) DISK layers: (\[[0-9, ]*\])\)', bank)
    pinned = re.search(r', (\d+) GPU layers\)', bank)
    assert disk and pinned and '0.00 GiB OS-locked (0 CPU layers: [])' in bank, 'wrong host backing'
    expected_disk = sorted(set(range(48)) - set(gpu)) if mode == 'off' else list(range(48))
    assert int(disk[1]) == len(expected_disk) and json.loads(disk[2]) == expected_disk, 'wrong file-backed layers'
    assert int(pinned[1]) == (20 if mode == 'off' else 0), 'wrong pinned layer count'
    markers = [line[line.index('MoE GPU source staging:'):] for line in lines
               if 'MoE GPU source staging:' in line]
    if mode == 'on':
        assert len(markers) == 1 and markers[0].startswith(
            'MoE GPU source staging: layers=' + str(sorted(gpu)) + ';'), 'wrong GPU staging selection'
    else:
        assert not markers, 'pinned arm enabled GPU staging'
    return dict(bank=bank, prefill_paths=paths, staging=markers,
                gpu_layer_ids=gpu, file_layer_ids=expected_disk)


WORKER_MEMORY_FIELDS = ('VmRSS', 'VmHWM', 'RssAnon', 'RssFile', 'RssShmem', 'VmLck', 'VmSwap')
SYSTEM_MEMORY_FIELDS = ('MemAvailable', 'Cached', 'Shmem', 'AnonPages', 'Unevictable', 'Mlocked')


def memory_snapshot(pid):
    """Read OS counters only before/after the client, outside timed requests."""
    def kib(path, fields):
        values = dict(line.split(':', 1) for line in path.read_text().splitlines() if ':' in line)
        result = {}
        for key in fields:
            count, unit = values[key].split()
            assert unit == 'kB' and int(count) >= 0, ('invalid memory counter', key)
            result[key] = int(count) * 1024
        return result
    return dict(worker_bytes=kib(Path(f'/proc/{pid}/status'), WORKER_MEMORY_FIELDS),
                system_bytes=kib(Path('/proc/meminfo'), SYSTEM_MEMORY_FIELDS))


def io_snapshot(pid):
    return dict(line.split(': ', 1) for line in Path(f'/proc/{pid}/io').read_text().splitlines())


def hold_connection(stream, idle_s=90):
    buffer = b''
    while select.select([stream], [], [], idle_s)[0]:
        data = os.read(stream.fileno(), 4096)
        if not data:
            return dict(lease='closed')
        buffer += data
        while b'\n' in buffer:
            line, buffer = buffer.split(b'\n', 1)
            if line == b'done':
                return dict(lease='closed')
            assert line == b'ping', 'invalid lease heartbeat'
    raise TimeoutError('client stopped renewing the benchmark lease')


def remote_main(args):
    out = R / 'results' / args.run_id
    if args.action == 'preflight':
        assert not out.exists(), out
        return preflight()
    if args.action == 'restore':
        result = dict(started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        try:
            # SSH termination alone can leave a remote start helper waiting for
            # readiness. Stop its whole unit before restoring the original port.
            stop(ACTION)
            stop(SERVER)
            if state('freetoken-serve')['ActiveState'] not in ('active', 'activating'):
                wait_gpu_release()
                run('sudo', '-n', 'systemctl', 'start', 'freetoken-serve', timeout=180)
            result['health'] = ready(8090)
            result['completion'] = completion(8090)
            result['unit'] = state('freetoken-serve')
            result['verified'] = True
        except Exception as exc:
            result.update(verified=False, error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            out.mkdir(parents=True, exist_ok=True)
            save(out / 'restoration.json', result)
        return result
    if args.action == 'hold':
        result = preflight(allow_lease=True)
        out.mkdir(parents=True, exist_ok=False)
        save(out / 'preflight.json', result)
        print('LEASE_READY', flush=True)
        # EOF or a broken connection ends the lease. ExecStopPost restores serving.
        return hold_connection(sys.stdin)
    if args.action == 'restoration':
        return json.loads((out / 'restoration.json').read_text())
    plan = json.loads((out / 'preflight.json').read_text())
    assert live(LEASE), 'benchmark lease is not live'
    assert sha(Path(__file__)) == plan['driver_sha256']
    start_path = out / (args.arm + '-start.json')
    if args.action == 'start':
        assert not start_path.exists(), 'refuse to rerun an existing arm'
        mode = dict(ARMS)[args.arm]
        assert identities() == plan['identities']
        assert sha(SERVICE) == plan['service_sha256']
        assert sha(R / 'models/flash-e2m1.ftw/config.json') == plan['model_config_sha256']
        assert sha(R / 'layer-profile-v3.json') == plan['layer_profile_sha256']
        stop(SERVER)
        stop('freetoken-serve')
        wait_gpu_release()
        command = ['sudo', '-n', 'systemd-run', '--unit=' + SERVER, '--collect', '--uid=jomcgi',
                   '--working-directory=' + plan['identities'][mode]['tree'],
                   '--property=KillMode=control-group', '--property=TimeoutStopSec=90',
                   '--property=RuntimeMaxSec=3400']
        command += ['--setenv=' + k + '=' + v for k, v in server_env(mode).items()]
        command += ['--setenv=HOME=/home/jomcgi']
        run(*(command + server_command(mode)))
        health = ready(18090)
        check = completion(18090)
        unit = state(SERVER)
        workers = gpu_pids()
        assert len(workers) == 1
        worker = workers[0]
        assert unit['ControlGroup'] in Path(f'/proc/{worker}/cgroup').read_text()
        maps = [line for line in Path(f'/proc/{worker}/maps').read_text().splitlines() if '_cpu_moe.' in line]
        assert maps and all(line.split()[-1] == plan['identities'][mode]['native'] for line in maps)
        all_maps = Path(f'/proc/{worker}/maps').read_text().splitlines()
        for name, identity in plan['identities'][mode]['native_extensions'].items():
            loaded = [line.split()[-1] for line in all_maps if '/' + name + '.' in line]
            if name in ('_cpu_moe', '_ple_uring'):
                assert loaded, ('required extension not mapped', name)
            assert all(path == identity['path'] for path in loaded), ('native mapping changed', name)
        actual_env = dict(item.split(b'=', 1) for item in Path(f'/proc/{worker}/environ').read_bytes().split(b'\0') if b'=' in item)
        assert all(actual_env[k.encode()].decode() == v for k, v in server_env(mode).items())
        text = journal(unit['InvocationID'])
        assert 'moe_collect_stats=False' in text and 'moe_step_timing=False' in text
        assert "cache_type='radix'" in text and "kv_ladder='off'" in text
        assert 'kv_disk_cache_gib=0.0' in text and "moe_hot_plan_persist='off'" in text
        assert any('DISK staged prefill:' in line and 'file_io=buffered' in line for line in text.splitlines())
        assert any('MoE HOT staging:' in line and 'file_io=mmap' in line for line in text.splitlines())
        assert "speculative_mtp='off'" in text and 'special_token_ckpt=False' in text
        if PROFILE_RETENTION:
            assert "session_expert_prefetch='on'" in text and 'session_protect_experts=64' in text
        snapshot_enabled = mode == 'on' and not runtime_pair() and not same_runtime_policy()
        assert ('Aligned decode prefix snapshots enabled' in text) == snapshot_enabled
        shape = geometry(text, source_staging=GPU_SOURCE_STAGING)
        layout = source_layout(text, mode) if GPU_SOURCE_STAGING else None
        if args.arm != 'r1':
            first = json.loads((out / 'r1-start.json').read_text())
            assert shape == first['geometry'], dict(first=first['geometry'], current=shape)
        result = dict(arm=args.arm, mode=mode, revision=runtime_revision(mode), unit=unit, worker=worker,
                      health=health, startup_completion=check, geometry=shape, native_maps=maps,
                      command=server_command(mode), env=server_env(mode), io_before=io_snapshot(worker),
                      worker_stat=Path(f'/proc/{worker}/stat').read_text(),
                      diagnostics_disabled=True, snapshot_enabled=snapshot_enabled, original_unit=state('freetoken-serve'),
                      identity=plan['identities'][mode], driver_sha256=plan['driver_sha256'],
                      cache_policy='RAM radix reuse on; disk prefix cache and HOT persistence off')
        if GPU_SOURCE_STAGING:
            assert ("moe_gpu_source='staged'" if mode == 'on' else "moe_gpu_source='pinned'") in text
            result['source_layout'] = layout
        if HOT_HOST_RECLAIM:
            policy = 'reclaim' if mode == 'on' else 'retain'
            assert f"moe_hot_host_cache='{policy}'" in text
            assert f'MoE HOT host cache: {policy}' in text
            result['hot_host_cache'] = policy
        if same_runtime_policy():
            result['memory_before'] = memory_snapshot(worker)
        save(start_path, result)
        (out / (args.arm + '-startup.log')).write_text(text)
        return result
    if args.action == 'end':
        before = json.loads(start_path.read_text())
        current = state(SERVER)
        assert current['InvocationID'] == before['unit']['InvocationID'] and current['ActiveState'] == 'active'
        result = dict(unit=current, io_after=io_snapshot(before['worker']),
                      gpu_pids=gpu_pids(), original_unit=state('freetoken-serve'))
        if same_runtime_policy():
            result['memory_after'] = memory_snapshot(before['worker'])
        assert result['gpu_pids'] == [before['worker']]
        text = journal(current['InvocationID'])
        (out / (args.arm + '-journal.log')).write_text(text)
        result['journal_sha256'] = sha(out / (args.arm + '-journal.log'))
        save(out / (args.arm + '-end.json'), result)
        stop(SERVER)
        return result
    raise ValueError(args.action)


def all_tasks_passed(arms):
    """A complete schedule includes passing warmups and measured tasks in both orders."""
    return (
        [(arm['arm'], arm['mode']) for arm in arms] == ARMS
        and all(
            arm['client_returncode'] == 0 and arm['warmup']['passed']
            and len(arm['measured']) == 2
            and all(row['passed'] for row in arm['measured'])
            for arm in arms
        )
    )


def remote_command(script, action, run_id, arm=None):
    command = ['/usr/bin/python3', str(script), '--remote', action, '--run-id', run_id]
    if PREFILL_CARRY:
        command += ['--prefill-snapshot-carry']
    if PROFILE_RETENTION:
        command += ['--hybrid-profile-retention']
    if GPU_SOURCE_STAGING:
        command += ['--gpu-source-staging']
    if HOT_HOST_RECLAIM:
        command += ['--hot-host-cache-reclaim']
    if arm:
        command += ['--arm', arm]
    if action in ('start', 'end'):
        # After + BindsTo makes lease shutdown stop the helper before the lease's
        # restoration hook. This also covers a killed local SSH process.
        command = ['sudo', '-n', 'systemd-run', '--unit=' + ACTION, '--pipe', '--wait',
                   '--collect', '--uid=jomcgi', '--property=KillMode=control-group',
                   '--property=TimeoutStopSec=15', '--property=RuntimeMaxSec=660',
                   '--property=BindsTo=' + LEASE + '.service',
                   '--property=After=' + LEASE + '.service'] + command
    return command


def local_main(args):
    here = Path(__file__).resolve().parent
    pi_client = here / 'pi-agentic-wall.py'
    assert sha(pi_client) == CLIENT_SHA, 'Pi client changed after Linux qualification'
    client = here / ('fixed-continuation-wall.py' if args.fixed_continuation else 'pi-agentic-wall.py')
    assert (here.parent / '.git').is_file(), 'run from a linked worktree'
    assert not run('git', '-C', str(here.parent), 'status', '--porcelain').stdout.strip()
    root = args.output_dir.resolve()
    if root.exists():
        assert sorted(p.name for p in root.iterdir()) == ['preflight.json'], root
    else:
        root.mkdir(parents=True)
    remote_script = R / 'tmp' / (args.run_id + '-driver.py')
    run('scp', '-oIdentityAgent=none', '-oIdentitiesOnly=yes', '-oBatchMode=yes',
        str(Path(__file__).resolve()), 'node-4:' + str(remote_script))

    def remote(action, arm=None):
        command = remote_command(remote_script, action, args.run_id, arm)
        return json.loads(run(*(SSH + ['node-4', shlex.join(command)]), timeout=600).stdout)

    plan = remote('preflight')
    plan.update(local_driver_sha256=sha(Path(__file__)), client_sha256=sha(client),
                client_kind='fixed-continuation' if args.fixed_continuation else 'pi',
                local_revision=run('git', '-C', str(here.parent), 'rev-parse', 'HEAD').stdout.strip(),
                design='Snapshot off/on/on/off: one warmup and two measured three-turn conversations per start. Same runtime and native binary; only FREETOKEN_DECODE_PREFIX_SNAPSHOT differs. All failures retained. Capacity one, graph one, 65536 FP8 KV tokens, 3753 expert slots, radix prefixes enabled, token trace and invasive diagnostics off. Host page cache retained. No model routing or quantization change.')
    if runtime_pair():
        label = 'Hybrid expert profile retention' if PROFILE_RETENTION else 'Prefill marker carry'
        plan['design'] = (label + ' parent/fix/fix/parent: one warmup and two measured '
                          'three-turn conversations per start. Pinned runtime revisions; identical '
                          'native binary, command and environment except PYTHONPATH. Decode snapshots '
                          'and invasive diagnostics off in both arms. Same capacity-one geometry, '
                          'quantization and routing. Retain all failures and both orders.')
    if GPU_SOURCE_STAGING:
        plan['design'] = ('Pinned/staged/staged/pinned GPU sources: one warmup and two measured '
                          'three-turn conversations per fresh server. Identical runtime, native '
                          'extensions and environment; only --moe-gpu-source differs. Decode '
                          'snapshots and invasive diagnostics off. Preserve compute placement, '
                          'HOT residency and cache geometry. Read OS memory counters only before '
                          'and after the client; these include warmup and are not peak measurements. '
                          'Host page cache retained. Retain failures and both orders.')
    if HOT_HOST_RECLAIM:
        plan['design'] = ('Retain/reclaim/reclaim/retain HOT host file-cache pages: one warmup and '
                          'two measured three-turn conversations per fresh server. Same runtime, '
                          'native extensions and environment; only --moe-hot-host-cache differs. '
                          'Decode snapshots, token traces and HOT census hook off. Identical '
                          'compute placement, HOT and cache geometry. OS memory counters outside '
                          'the client include warmup; they do not measure peak usage. Host page '
                          'cache retained. Keep failures and both execution orders.')
    if not args.fixed_continuation:
        plan.update(pi_version=run(str(args.pi), '--version').stdout.strip(),
                    pi_executable_sha256=sha(args.pi.resolve()))
        assert plan['pi_version'] == '0.85.1'
    save(root / 'preflight.json', plan)
    if args.preflight:
        print(json.dumps(plan, indent=2))
        return 0
    # The tunnel belongs to this controller, separate from the user's 8090 tunnel.
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 18092))
    report = dict(preflight=plan, arms=[], completed=False, restored=False, all_tasks_passed=False)
    save(root / 'driver.json', report)
    done = threading.Event()
    heartbeat_error = []
    lease = tunnel = active_client = None

    def interrupted(signum, _frame):
        raise RuntimeError(f'controller interrupted by signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location('pi_wall_client', pi_client)
    client_module = module_from_spec(spec)
    spec.loader.exec_module(client_module)

    try:
        restore = shlex.join(remote_command(remote_script, 'restore', args.run_id))
        command = ['sudo', '-n', 'systemd-run', '--unit=' + LEASE, '--pipe', '--wait', '--collect', '--uid=jomcgi',
                   '--property=RuntimeMaxSec=14400', '--property=TimeoutStopSec=600',
                   '--property=ExecStopPost=' + restore] + remote_command(remote_script, 'hold', args.run_id)
        with (root / 'lease-stderr.log').open('wb') as err:
            lease = subprocess.Popen(SSH + ['node-4', shlex.join(command)], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=err, start_new_session=True)
        deadline = time.monotonic() + 60
        lease_output = b''
        while b'LEASE_READY\n' not in lease_output:
            if time.monotonic() > deadline or lease.poll() is not None:
                raise RuntimeError('remote lease did not become ready')
            if select.select([lease.stdout], [], [], 1)[0]:
                lease_output += os.read(lease.stdout.fileno(), 65536)
        (root / 'lease-start.log').write_bytes(lease_output)

        def pulse():
            try:
                while not done.is_set():
                    lease.stdin.write(b'ping\n')
                    lease.stdin.flush()
                    done.wait(20)
            except Exception as exc:
                heartbeat_error.append(str(exc))

        heartbeat = threading.Thread(target=pulse, daemon=True)
        heartbeat.start()
        with (root / 'tunnel.log').open('wb') as err:
            tunnel = subprocess.Popen(SSH + ['-oExitOnForwardFailure=yes', '-N',
                                      '-L', '127.0.0.1:18092:127.0.0.1:18090', 'node-4'],
                                      stdout=err, stderr=err, start_new_session=True)
        time.sleep(1)
        assert tunnel.poll() is None, 'SSH tunnel failed'
        for arm, mode in ARMS:
            assert not heartbeat_error and lease.poll() is None
            start = remote('start', arm)
            save(root / (arm + '-server-start.json'), start)
            current = root / 'client-current'
            assert not current.exists()
            assert sha(client) == plan['client_sha256'], 'client changed during comparison'
            command = [sys.executable, str(client),
                       '--base-url', 'http://127.0.0.1:18092/v1', '--output-dir', str(current),
                       '--label', mode + '-' + arm, '--server-metadata', str(root / (arm + '-server-start.json'))]
            if not args.fixed_continuation:
                command += ['--pi', str(args.pi.resolve()), '--sessions', '3', '--timeout', '900',
                            '--max-model-calls', '30', '--max-tokens', '8192',
                            '--context-tokens', '32768', '--repairs', '1']
            print('STARTED ' + arm + ' ' + mode, flush=True)
            with (root / (arm + '-client.log')).open('wb') as log:
                active_client = subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True)
                deadline = time.monotonic() + 2850
                while active_client.poll() is None:
                    if heartbeat_error or lease.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError('client exceeded controller budget or lease failed')
                    time.sleep(2)
            result = dict(arm=arm, mode=mode, client_returncode=active_client.returncode)
            active_client = None
            # Archive, then reuse exactly the same path for the next arm's prompt.
            current.rename(root / arm)
            rows = [json.loads((root / arm / f'session-{n}/result.json').read_text()) for n in (1, 2, 3)]
            assert all(not row['trace'] for row in rows)
            result['warmup'] = {k: rows[0].get(k) for k in ['passed', 'task_wall_s', 'model_calls', 'error']}
            result['measured'] = [{k: row.get(k) for k in ['passed', 'task_wall_s', 'model_calls', 'repair_prompts', 'error']} for row in rows[1:]]
            result['end'] = remote('end', arm)
            save(root / (arm + '-server-end.json'), result['end'])
            report['arms'].append(result)
            save(root / 'driver.json', report)
            print('COMPLETED ' + json.dumps(result), flush=True)
        report['completed'] = True
        report['all_tasks_passed'] = all_tasks_passed(report['arms'])
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        print('FAILED ' + report['error'], flush=True)
    finally:
        if active_client is not None:
            # Let the client preserve cancellation evidence and stop its Pi group.
            active_client.send_signal(signal.SIGTERM)
            try:
                active_client.wait(timeout=30)
            except subprocess.TimeoutExpired:
                client_module.kill_group(active_client)
        done.set()
        if lease is not None:
            if 'heartbeat' in locals():
                heartbeat.join(timeout=5)
            try:
                lease.stdin.write(b'done\n')
                lease.stdin.flush()
                lease.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            try:
                lease.wait(timeout=600)
                report['lease_returncode'] = lease.returncode
                (root / 'lease-end.log').write_bytes(lease.stdout.read())
                report['restoration'] = remote('restoration')
                report['restored'] = report['restoration']['verified']
            except Exception as exc:
                report['restoration_error'] = f'{type(exc).__name__}: {exc}'
        if tunnel is not None:
            client_module.kill_group(tunnel)
        save(root / 'driver.json', report)
        print('FINAL ' + json.dumps({k: report.get(k) for k in ['completed', 'all_tasks_passed', 'restored', 'error', 'restoration_error']}), flush=True)
    return 0 if report['completed'] and report['all_tasks_passed'] and report['restored'] else 1


def main():
    global PREFILL_CARRY, PROFILE_RETENTION, GPU_SOURCE_STAGING, HOT_HOST_RECLAIM
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote', dest='action', choices=['preflight', 'hold', 'start', 'end', 'restore', 'restoration'])
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--arm', choices=[arm for arm, _ in ARMS])
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--pi', type=Path)
    parser.add_argument('--fixed-continuation', action='store_true',
                        help='use ordinary greedy scripted copying instead of Pi tasks')
    experiments = parser.add_mutually_exclusive_group()
    experiments.add_argument('--prefill-snapshot-carry', action='store_true',
                             help='compare the pinned prefill-marker fix with its parent; decode snapshots stay off')
    experiments.add_argument('--hybrid-profile-retention', action='store_true',
                             help='compare the pinned expert-profile fix with its parent; decode snapshots stay off')
    experiments.add_argument('--gpu-source-staging', action='store_true',
                             help='compare pinned and staged GPU sources on the same runtime; decode snapshots stay off')
    experiments.add_argument('--hot-host-cache-reclaim', action='store_true',
                             help='compare retaining and reclaiming HOT source pages on one runtime')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    PREFILL_CARRY = args.prefill_snapshot_carry
    PROFILE_RETENTION = args.hybrid_profile_retention
    GPU_SOURCE_STAGING = args.gpu_source_staging
    HOT_HOST_RECLAIM = args.hot_host_cache_reclaim
    if not re.fullmatch(r'astra-pi-agentic-[a-z0-9-]+', args.run_id):
        parser.error('run-id must start astra-pi-agentic- and contain only lowercase letters, digits and hyphens')
    if args.action:
        print(json.dumps(remote_main(args), indent=2), flush=True)
        return 0
    if (runtime_pair() or same_runtime_policy()) and not args.fixed_continuation:
        parser.error('source experiments require --fixed-continuation')
    if not args.output_dir or (not args.pi and not args.fixed_continuation):
        parser.error('local controller requires --output-dir and either --pi or --fixed-continuation')
    return local_main(args)


if __name__ == '__main__':
    raise SystemExit(main())
