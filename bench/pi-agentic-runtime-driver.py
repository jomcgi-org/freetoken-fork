"""Controlled original/optimized Pi sessions on node-4, with remote recovery.

The same file runs as a Mac controller and a Linux service helper. All model
work is remote; Pi and its fixture tools run on the Mac. No runtime edits.
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
SRC = R / 'wt-plegather'
OPT = R / 'wt-astra-concurrent-wall-client'
SERVER = 'astra-pi-agentic-server'
LEASE = 'astra-pi-agentic-lease'
SSH = ['ssh', '-oIdentityAgent=none', '-oIdentitiesOnly=yes', '-oBatchMode=yes',
       '-oServerAliveInterval=15', '-oServerAliveCountMax=3']
REVISIONS = {'baseline': '3a67403a293be20836604bc25729329997848e09',
             'optimized': 'c0775ea48b108dcf31d80091a8d36c81d5239edf'}
BINARIES = {'baseline': 'aad1f3169b9b39829a356877de7e5add9b8c8a0173837f54140871b24e0c0048',
            'optimized': 'c88ed9f877a5a6c4cb3eb4c172b0a7a953794e3ff1104a12b8dcb0f22fb4810f'}
CLIENT_SHA = '986875b712ff05862c03417e56e45c18559d82e05af913e134055a84268715d5'
ARMS = [('r1', 'baseline'), ('r2', 'optimized'), ('r3', 'optimized'), ('r4', 'baseline')]


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


def identities():
    result = {}
    for mode, tree in [('baseline', SRC), ('optimized', OPT)]:
        revision = run('git', '-C', str(tree), 'rev-parse', 'HEAD').stdout.strip()
        assert revision == REVISIONS[mode], (mode, revision)
        assert not run('git', '-C', str(tree), 'status', '--porcelain').stdout.strip(), tree
        binaries = list((tree / 'python/freetoken/kernel').glob('_cpu_moe.*.so'))
        assert len(binaries) == 1
        native = binaries[0].resolve()
        assert sha(native) == BINARIES[mode], native
        result[mode] = dict(revision=revision, tree=str(tree), native=str(native), native_sha256=sha(native),
                            cpp_sha256=sha(tree / 'python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp'))
    return result


def server_command(mode):
    service = Path('/etc/systemd/system/freetoken-serve.service').read_text().replace('\\\n', ' ')
    args = shlex.split(next(line[10:] for line in service.splitlines() if line.startswith('ExecStart=')))
    args[0] = str(SRC / '.venv/bin/ft')
    for flag, value in [('--port', '18090'), ('--moe-disk-prefill', 'cpu' if mode == 'baseline' else 'staged'),
                        ('--max-running-requests', '1'), ('--kv-disk-cache-gib', '0')]:
        args[args.index(flag) + 1] = value
    args.remove('--moe-collect-stats')
    args.extend(['--moe-hot-plan-persist', 'off', '--cache-type', 'radix', '--kv-ladder', 'off',
                 '--kv-reserve-tokens', '65536', '--cuda-graph-max-bs', '1'])
    if mode == 'optimized':
        args.extend(['--moe-disk-prefill-io', 'buffered', '--moe-hot-staging-io', 'mmap'])
    assert '--moe-collect-stats' not in args and '--moe-step-timing' not in args
    return args


def server_env(mode):
    return dict(CUDA_HOME='/usr/local/cuda-13.0',
                PATH=f'{SRC}/.venv/bin:/usr/local/cuda-13.0/bin:/usr/bin:/bin', TMPDIR=str(R / 'tmp'),
                PYTHONPATH=str((SRC if mode == 'baseline' else OPT) / 'python'),
                FREETOKEN_PREFILL_SELECTIVE_MAX_TOKENS='128', FREETOKEN_PREFILL_HOT_OVERLAP='0')


def preflight(allow_lease=False):
    for unit in ['astra-decode-weight-reuse-wall-driver', 'astra-concurrent-wall-driver',
                 'astra-decode-weight-reuse-validation-v2', 'astra-sustained-hot-staging-wall-driver',
                 'astra-sustained-reader-wall-driver', SERVER, *([] if allow_lease else [LEASE])]:
        assert not live(unit), ('another benchmark is live', unit)
    original = state('freetoken-serve')
    assert original['ActiveState'] == 'active', original
    workers = gpu_pids()
    assert len(workers) == 1 and original['ControlGroup'] in Path(f'/proc/{workers[0]}/cgroup').read_text()
    return dict(identities=identities(), original_unit=original,
                service_sha256=sha(Path('/etc/systemd/system/freetoken-serve.service')),
                model_config_sha256=sha(R / 'models/flash-e2m1.ftw/config.json'),
                layer_profile_sha256=sha(R / 'layer-profile-v3.json'),
                driver_sha256=sha(Path(__file__)),
                commands={mode: server_command(mode) for mode in REVISIONS},
                env={mode: server_env(mode) for mode in REVISIONS})


def journal(invocation):
    return run('journalctl', '_SYSTEMD_INVOCATION_ID=' + invocation, '--no-pager', '-o', 'cat').stdout


def geometry(text):
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
    return result


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
        assert sha(Path('/etc/systemd/system/freetoken-serve.service')) == plan['service_sha256']
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
        actual_env = dict(item.split(b'=', 1) for item in Path(f'/proc/{worker}/environ').read_bytes().split(b'\0') if b'=' in item)
        assert all(actual_env[k.encode()].decode() == v for k, v in server_env(mode).items())
        text = journal(unit['InvocationID'])
        assert 'moe_collect_stats=False' in text and 'moe_step_timing=False' in text
        assert "cache_type='radix'" in text and "kv_ladder='off'" in text
        assert 'kv_disk_cache_gib=0.0' in text and "moe_hot_plan_persist='off'" in text
        if mode == 'optimized':
            assert any('DISK staged prefill:' in line and 'file_io=buffered' in line for line in text.splitlines())
            assert any('MoE HOT staging:' in line and 'file_io=mmap' in line for line in text.splitlines())
        shape = geometry(text)
        if args.arm != 'r1':
            first = json.loads((out / 'r1-start.json').read_text())
            assert shape == first['geometry'], dict(first=first['geometry'], current=shape)
        result = dict(arm=args.arm, mode=mode, revision=REVISIONS[mode], unit=unit, worker=worker,
                      health=health, startup_completion=check, geometry=shape, native_maps=maps,
                      command=server_command(mode), env=server_env(mode), io_before=io_snapshot(worker),
                      worker_stat=Path(f'/proc/{worker}/stat').read_text(),
                      diagnostics_disabled=True, original_unit=state('freetoken-serve'),
                      identity=plan['identities'][mode], driver_sha256=plan['driver_sha256'],
                      cache_policy='RAM radix reuse on; disk prefix cache and HOT persistence off')
        save(start_path, result)
        (out / (args.arm + '-startup.log')).write_text(text)
        return result
    if args.action == 'end':
        before = json.loads(start_path.read_text())
        current = state(SERVER)
        assert current['InvocationID'] == before['unit']['InvocationID'] and current['ActiveState'] == 'active'
        result = dict(unit=current, io_after=io_snapshot(before['worker']),
                      gpu_pids=gpu_pids(), original_unit=state('freetoken-serve'))
        assert result['gpu_pids'] == [before['worker']]
        text = journal(current['InvocationID'])
        (out / (args.arm + '-journal.log')).write_text(text)
        result['journal_sha256'] = sha(out / (args.arm + '-journal.log'))
        save(out / (args.arm + '-end.json'), result)
        stop(SERVER)
        return result
    raise ValueError(args.action)


def local_main(args):
    here = Path(__file__).resolve().parent
    client = here / 'pi-agentic-wall.py'
    assert sha(client) == CLIENT_SHA, 'client changed after Linux qualification'
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
        command = ['python3', str(remote_script), '--remote', action, '--run-id', args.run_id]
        if arm:
            command += ['--arm', arm]
        return json.loads(run(*(SSH + ['node-4', shlex.join(command)]), timeout=600).stdout)

    plan = remote('preflight')
    plan.update(local_driver_sha256=sha(Path(__file__)), client_sha256=sha(client),
                local_revision=run('git', '-C', str(here.parent), 'rev-parse', 'HEAD').stdout.strip(),
                pi_version=run(str(args.pi), '--version').stdout.strip(),
                pi_executable_sha256=sha(args.pi.resolve()),
                design='A/B/B/A: one warmup and two measured Pi sessions per start. All three stages and all failures retained. Same Pi/workspace path, source-verified runtimes, capacity one, graph one, 65536 reserved KV tokens, radix prefixes enabled, diagnostics off. Host page cache retained. No model routing or quantization change.')
    assert plan['pi_version'] == '0.85.1'
    save(root / 'preflight.json', plan)
    if args.preflight:
        print(json.dumps(plan, indent=2))
        return 0
    # The tunnel belongs to this controller, separate from the user's 8090 tunnel.
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 18092))
    report = dict(preflight=plan, arms=[], completed=False, restored=False)
    save(root / 'driver.json', report)
    done = threading.Event()
    heartbeat_error = []
    lease = tunnel = active_client = None

    def interrupted(signum, _frame):
        raise RuntimeError(f'controller interrupted by signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location('pi_wall_client', client)
    client_module = module_from_spec(spec)
    spec.loader.exec_module(client_module)

    try:
        restore = shlex.join(['/usr/bin/python3', str(remote_script), '--remote', 'restore', '--run-id', args.run_id])
        command = ['sudo', '-n', 'systemd-run', '--unit=' + LEASE, '--pipe', '--wait', '--collect', '--uid=jomcgi',
                   '--property=RuntimeMaxSec=14400', '--property=TimeoutStopSec=600',
                   '--property=ExecStopPost=' + restore,
                   '/usr/bin/python3', str(remote_script), '--remote', 'hold', '--run-id', args.run_id]
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
            command = [sys.executable, str(client), '--pi', str(args.pi.resolve()),
                       '--base-url', 'http://127.0.0.1:18092/v1', '--output-dir', str(current),
                       '--label', mode + '-' + arm, '--server-metadata', str(root / (arm + '-server-start.json')),
                       '--sessions', '3', '--timeout', '900', '--max-model-calls', '30',
                       '--max-tokens', '8192', '--context-tokens', '32768', '--repairs', '1']
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
        print('FINAL ' + json.dumps({k: report.get(k) for k in ['completed', 'restored', 'error', 'restoration_error']}), flush=True)
    return 0 if report['completed'] and report['restored'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote', dest='action', choices=['preflight', 'hold', 'start', 'end', 'restore', 'restoration'])
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--arm', choices=[arm for arm, _ in ARMS])
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--pi', type=Path)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'astra-pi-agentic-[a-z0-9-]+', args.run_id):
        parser.error('run-id must start astra-pi-agentic- and contain only lowercase letters, digits and hyphens')
    if args.action:
        print(json.dumps(remote_main(args), indent=2), flush=True)
        return 0
    if not args.output_dir or not args.pi:
        parser.error('local controller requires --output-dir and --pi')
    return local_main(args)


if __name__ == '__main__':
    raise SystemExit(main())
