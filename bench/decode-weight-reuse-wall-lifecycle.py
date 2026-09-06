"""Exclusive serving ownership and recovery for the native weight-reuse gate."""

import argparse
import json
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request


SERVER = 'astra-decode-weight-reuse-wall-server'
ORIGINAL = 'freetoken-serve'
OTHER_JOBS = (
    'astra-concurrent-wall-driver',
    'astra-decode-weight-reuse-validation-v2',
    'astra-sustained-hot-staging-wall-driver',
    'astra-sustained-reader-wall-driver',
    'astra-hot-staging-validation',
    'astra-sustained-cached-wall-driver',
    'astra-pi-agentic-server',
    'astra-pi-agentic-lease',
    'astra-pi-agentic-action',
)
BUSY = {'active', 'activating', 'deactivating'}


def run(*args, **kwargs):
    return subprocess.run(args, check=True, text=True, capture_output=True, **kwargs)


def state(unit):
    raw = run('systemctl', 'show', unit, '-p', 'ActiveState', '-p', 'MainPID',
              '-p', 'ControlGroup', '-p', 'InvocationID', timeout=15).stdout
    return dict(line.split('=', 1) for line in raw.splitlines() if '=' in line)


def other_jobs():
    return [unit for unit in OTHER_JOBS if state(unit)['ActiveState'] in BUSY]


def gpu_pids():
    raw = run('nvidia-smi', '--query-compute-apps=pid',
              '--format=csv,noheader,nounits', timeout=15).stdout
    return [int(line) for line in raw.splitlines() if line.strip()]


def original_owns_gpu(original):
    workers = gpu_pids()
    group = original['ControlGroup']
    return (len(workers) == 1 and bool(group)
            and group in Path(f'/proc/{workers[0]}/cgroup').read_text())


def preflight():
    busy = other_jobs()
    if busy:
        raise RuntimeError('another benchmark is live: ' + ', '.join(busy))
    if state(SERVER)['ActiveState'] in BUSY:
        raise RuntimeError('weight-reuse benchmark server is already live')
    original = state(ORIGINAL)
    if original['ActiveState'] != 'active' or not original_owns_gpu(original):
        raise RuntimeError('original serving does not own the sole GPU process')
    return dict(original_unit=original, exclusive=True)


def wait_gpu_release():
    deadline = time.monotonic() + 45
    while gpu_pids():
        if time.monotonic() >= deadline:
            raise TimeoutError('GPU remains occupied; original serving was not started')
        time.sleep(1)


def completion():
    deadline = time.monotonic() + 420
    while True:
        try:
            with urllib.request.urlopen('http://127.0.0.1:8090/health', timeout=3) as response:
                health = json.load(response)
            if health.get('status') == 'error':
                raise RuntimeError('restored server reports a backend error')
            if health.get('status') == 'ok':
                break
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError('restored server did not become ready')
        time.sleep(2)
    payload = dict(model='qwen3.6-27b', messages=[dict(role='user', content='Reply with only OK.')],
                   max_tokens=8, temperature=0, chat_template_kwargs=dict(enable_thinking=False))
    request = urllib.request.Request('http://127.0.0.1:8090/v1/chat/completions',
                                    data=json.dumps(payload).encode(),
                                    headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.load(response)
    if result['choices'][0]['message']['content'].strip() != 'OK':
        raise RuntimeError('restored server failed the completion check')
    return result


def restore():
    # ExecStopPost also runs when preflight refused this job. Another benchmark
    # then owns the serving pause; do not interfere with its units or requests.
    busy = other_jobs()
    if busy:
        return dict(verified=False, skipped_other_jobs=busy)
    if state(SERVER)['ActiveState'] != 'inactive':
        run('sudo', '-n', 'systemctl', 'stop', SERVER, timeout=180)
    original = state(ORIGINAL)
    if original['ActiveState'] == 'active' and not original_owns_gpu(original):
        raise RuntimeError('active original serving does not own the sole GPU process')
    if original['ActiveState'] not in ('active', 'activating'):
        wait_gpu_release()
        # Recheck after waiting, before any attempt to take the GPU.
        busy = other_jobs()
        if busy:
            return dict(verified=False, skipped_other_jobs=busy)
        run('sudo', '-n', 'systemctl', 'start', ORIGINAL, timeout=180)
    result = completion()
    original = state(ORIGINAL)
    if original['ActiveState'] != 'active' or not original_owns_gpu(original):
        raise RuntimeError('restored serving does not own the sole GPU process')
    return dict(verified=True, original_unit=original, completion=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('preflight', 'restore'))
    args = parser.parse_args()
    result = preflight() if args.action == 'preflight' else restore()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
