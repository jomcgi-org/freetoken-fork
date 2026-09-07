"""Sustained mmap versus buffered HOT staging on one frozen runtime."""
import hashlib, importlib.util, json, os, pathlib, shlex, signal, subprocess, sys, time, urllib.error, urllib.request
R = pathlib.Path('/var/lib/longhorn/nvme-02/freetoken')
WT = R / 'wt-astra-hot-staging-io'
SRC = R / 'wt-plegather'
OUT = R / 'results'
UNIT = 'astra-sustained-hot-staging-wall-server'

def run(*args, **kwargs):
    return subprocess.run(args, check=True, text=True, **kwargs)

def stop():
    subprocess.run(['sudo', '-n', 'systemctl', 'stop', UNIT], check=False)

def interrupted(*_):
    raise RuntimeError('benchmark interrupted, restoring service')

signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)
signal.signal(signal.SIGHUP, interrupted)
service = pathlib.Path('/etc/systemd/system/freetoken-serve.service').read_text().replace('\\\n', ' ')
args = shlex.split(next(line[10:] for line in service.splitlines() if line.startswith('ExecStart=')))
args[0] = str(SRC / '.venv/bin/ft')
for flag, value in [('--moe-disk-prefill', 'staged'), ('--port', '18090'), ('--moe-hot-adapt-interval-steps', 'auto'), ('--kv-disk-cache-gib', '0')]:
    args[args.index(flag) + 1] = value
args.remove('--moe-collect-stats')
args.extend(['--moe-hot-plan-persist', 'off', '--cache-type', 'naive', '--moe-disk-prefill-io', 'buffered'])
assert '--moe-collect-stats' not in args
assert '--moe-hot-staging-io' not in args
env = dict(CUDA_HOME='/usr/local/cuda-13.0', PATH=f'{SRC}/.venv/bin:/usr/local/cuda-13.0/bin:/usr/bin:/bin', TMPDIR=str(R/'tmp'), PYTHONPATH=f'{WT}/python', FREETOKEN_PREFILL_SELECTIVE_MAX_TOKENS='128', FREETOKEN_PREFILL_HOT_OVERLAP='0')
revisions = {mode: run('git', '-C', str(tree), 'rev-parse', 'HEAD', capture_output=True).stdout.strip() for mode, tree in [('baseline', WT), ('optimized', WT)]}
assert revisions['baseline'].startswith('878d723'), revisions
assert revisions['baseline'] == revisions['optimized'], revisions
assert revisions['optimized'].startswith('878d723'), revisions
assert not run('git','-C',str(SRC),'status','--porcelain',capture_output=True).stdout.strip()
assert not run('git','-C',str(WT),'status','--porcelain',capture_output=True).stdout.strip()
base_report = dict(
    revisions=revisions,
    driver_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
    adaptation='automatic interval, split histories, unchanged phase aim and swap bounds, persistence off',
    design='Four isolated starts: mmap, buffered, buffered, mmap HOT staging. All use the same frozen optimized runtime and buffered staged GPU prefill. Only --moe-hot-staging-io changes. Each source tree stays fixed throughout startup, four warmup responses, twelve measured complete-response tasks, and eight fidelity cases. HOT assignments and routing histories persist within each process. Host page cache is retained between starts. Six measured JSON/prose blocks expose early-to-late behavior; all prompts are prepared before warmup. Diagnostics, GPU timing, KV reuse, and HOT persistence are off.',
    policies={
        'baseline': 'Default mmap HOT staging with buffered GPU DISK prefill on runtime 878d723.',
        'optimized': 'Buffered HOT staging with buffered GPU DISK prefill on the identical runtime 878d723.',
    },
)

native_extensions = {}
for mode, tree in [('baseline', WT), ('optimized', WT)]:
    candidates = list((tree/'python/freetoken/kernel').glob('_cpu_moe.*.so'))
    assert len(candidates) == 1, (mode, candidates)
    binary = candidates[0].resolve()
    data = binary.read_bytes()
    has_input_reuse = b'set_prefill_input_reuse' in data
    assert has_input_reuse, (mode, binary, has_input_reuse)
    source = tree/'python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp'
    native_extensions[mode] = dict(path=str(binary), sha256=hashlib.sha256(data).hexdigest(), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), contains_input_reuse_symbol=has_input_reuse)
assert native_extensions['baseline'] == native_extensions['optimized']
base_report['native_extensions'] = native_extensions

report = dict(base_report)
report_path = OUT / 'astra-sustained-hot-staging-wall-driver-metadata.json'

def save():
    report_path.write_text(json.dumps(report, indent=2) + '\n')

def ready(port):
    deadline = time.monotonic() + 420
    while True:
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2) as response:
                health = json.load(response)
            if health.get('status') == 'error':
                raise RuntimeError(health)
            if health.get('status') == 'ok':
                return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        if time.monotonic() > deadline:
            raise RuntimeError(f'port {port} readiness timeout')
        time.sleep(3)

def completion(port, prompt='Reply with only OK.', max_tokens=8):
    payload = dict(model='qwen3.6-27b', messages=[dict(role='user', content=prompt)], max_tokens=max_tokens, temperature=0, chat_template_kwargs=dict(enable_thinking=False))
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=120) as response:
        result = json.load(response)
    if prompt == 'Reply with only OK.':
        assert result['choices'][0]['message']['content'].strip() == 'OK', result
    return result

baseenv = dict(os.environ, CUDA_HOME=env['CUDA_HOME'], PATH=env['PATH'], TMPDIR=env['TMPDIR'], PYTHONPATH=str(WT/'python'))
# Refuse to overwrite a previous attempt or replace a live benchmark server.
assert not list(OUT.glob('astra-sustained-hot-staging-wall-r[1-4]*'))
existing = subprocess.run(['systemctl', 'show', UNIT, '-p', 'ActiveState', '--value'], text=True, capture_output=True)
assert existing.stdout.strip() not in ('active', 'activating', 'deactivating'), existing.stdout
for other in ('astra-sustained-reader-wall-driver', 'astra-hot-staging-validation', 'astra-sustained-cached-wall-driver'):
    state = run('systemctl', 'show', other, '-p', 'ActiveState', '--value', capture_output=True).stdout.strip()
    assert state not in ('active', 'activating', 'deactivating'), (other, state)
assert sys.argv[1:] in ([], ['--preflight']), sys.argv
if sys.argv[1:] == ['--preflight']:
    print(json.dumps(dict(base_report, args=args, env=env, preflight=True), indent=2), flush=True)
    raise SystemExit(0)
try:
    run('sudo', '-n', 'systemctl', 'stop', 'freetoken-serve')
    for label, mode in [('r1', 0), ('r2', 1), ('r3', 1), ('r4', 0)]:
        policy = ('baseline', 'optimized')[mode]
        tree = WT
        file_io = ('mmap', 'buffered')[mode]
        env['PYTHONPATH'] = str(tree/'python')
        revision = run('git', '-C', str(tree), 'rev-parse', 'HEAD', capture_output=True).stdout.strip()
        assert revision == revisions[policy], (policy, revision, revisions)
        start_args = list(args)
        start_args.extend(['--moe-hot-staging-io', file_io])
        report = dict(base_report, args=start_args, env=dict(env), revision=revision, source_tree=str(tree), label=label, mode=policy, file_io=file_io, records=[], phase='startup')
        report_path = OUT / f'astra-sustained-hot-staging-wall-{label}-metadata.json'
        stop()
        since = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        cmd = ['sudo', '-n', 'systemd-run', '--unit='+UNIT, '--collect', '--uid=jomcgi', '--working-directory='+str(tree), '--property=KillMode=control-group', '--property=TimeoutStopSec=30', '--property=RuntimeMaxSec=2700']
        for key, value in env.items():
            cmd.append('--setenv='+key+'='+value)
        cmd.append('--setenv=HOME=/home/jomcgi')
        save()
        run(*(cmd + start_args))
        try:
            ready(18090)
            report['startup_completion'] = completion(18090)
            save()
            gpu_pids = [int(x) for x in run('nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits', capture_output=True).stdout.splitlines() if x.strip()]
            assert len(gpu_pids) == 1, gpu_pids
            worker = gpu_pids[0]
            group = run('systemctl', 'show', UNIT, '-p', 'ControlGroup', '--value', capture_output=True).stdout.strip()
            assert group and group in pathlib.Path(f'/proc/{worker}/cgroup').read_text(), (group, worker)
            report['io_worker_pid'] = worker
            native_maps = [line for line in pathlib.Path(f'/proc/{worker}/maps').read_text().splitlines() if '_cpu_moe.' in line]
            assert native_maps, ('native CPU extension absent from worker maps', worker)
            assert all(line.split()[-1] == native_extensions[policy]['path'] for line in native_maps), native_maps
            report['native_extension_maps'] = native_maps
            journal = run('sudo','-n','journalctl','-u',UNIT,'--since',since,'--no-pager','-o','cat',capture_output=True).stdout
            report['transport_startup'] = [line for line in journal.splitlines() if 'DISK staged prefill:' in line]
            assert any('ring=64 MiB' in line and 'minimum_chunk=1024 tokens' in line and 'file_io=buffered' in line for line in report['transport_startup']), report['transport_startup']
            report['hot_transport_startup'] = [line for line in journal.splitlines() if 'MoE HOT staging:' in line]
            assert len(report['hot_transport_startup']) == 1, report['hot_transport_startup']
            assert 'file_io='+file_io in report['hot_transport_startup'][0], report['hot_transport_startup']
            report['startup_geometry'] = [line for line in journal.splitlines() if 'MoE bank split residency:' in line or 'MoE HOT expert residency:' in line or 'MoE activation dtype:' in line or 'MoE HOT adaptation intervals:' in line]
            assert len(report['startup_geometry']) == 4, report['startup_geometry']
            save()
            report['phase'] = 'timing'
            save()
            print('READY '+label+' '+file_io, flush=True)
            run(str(SRC/'.venv/bin/python'), str(WT/'bench/sustained-prefill-wall.py'), '--tokenizer', str(R/'models/flash-e2m1.ftw'), '--output', str(OUT/f'astra-sustained-hot-staging-wall-{label}.jsonl'), '--mode', policy, '--warmup-pairs', '2', '--blocks', '6', '--io-pid', str(worker), env=baseenv, timeout=2400)
            rows = [json.loads(line) for line in (OUT/f'astra-sustained-hot-staging-wall-{label}.jsonl').read_text().splitlines()]
            assert len(rows) == 16, len(rows)
            assert all(row['diagnostic_phase_io'] is False for row in rows), 'diagnostic rows cannot qualify wall time'
            report['client_manifest'] = json.loads((OUT/f'astra-sustained-hot-staging-wall-{label}.prompts.json').read_text())
            report['timing_cumulative'] = rows[-1]['cumulative']
            report['phase'] = 'fidelity'
            save()
            essay = next(row for row in rows if row['kind'] == 'essay')
            background = essay['prompt'].split('<background>\n', 1)[1].split('\n</background>', 1)[0]
            spec = importlib.util.spec_from_file_location('moe_fidelity', WT/'bench/moe-fidelity.py')
            fidelity = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(fidelity)
            for name, question, expected in fidelity.cases():
                prompt = question if name == 'long_retrieval' else (
                    'The following source excerpt is background only. Answer the question after it.\n'
                    f'<background>\n{background}\n</background>\nQuestion: {question}'
                )
                response = completion(18090, prompt, 64)
                text = response['choices'][0]['message']['content']
                assert response['usage']['prompt_tokens'] >= 1024, response
                row = dict(kind='fidelity_'+name, mode=report['mode'], prompt=prompt, text=text, expected=expected, passed=text.strip()==expected, usage=response['usage'], finish_reason=response['choices'][0]['finish_reason'])
                report['records'].append(row)
                save()
                print(json.dumps({key:value for key,value in row.items() if key!='prompt'}), flush=True)
            report['fidelity_score'] = sum(row['passed'] for row in report['records'])
            report['completed'] = True
            report['phase'] = 'completed'
            save()
            print('COMPLETED '+label, flush=True)
        finally:
            stop()
            with (OUT/f'astra-sustained-hot-staging-wall-{label}-journal.log').open('w') as log:
                subprocess.run(['sudo', '-n', 'journalctl', '-u', UNIT, '--since', since, '--no-pager', '-o', 'cat'], stdout=log, check=False)
finally:
    stop()
    run('sudo', '-n', 'systemctl', 'start', 'freetoken-serve')
    print('PRODUCTION STARTED', flush=True)
    ready(8090)
    report['original_serving_restoration'] = completion(8090)
    save()
    print('PRODUCTION VERIFIED '+json.dumps(report['original_serving_restoration']), flush=True)
