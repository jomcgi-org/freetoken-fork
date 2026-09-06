"""Node-4 off/on/on/off comparison using the frozen Pi recovery helper.

Requires the helper and runtime installed by the preceding recorded validations.
Preflight is read-only. Before launching, exercise restore_command() through the
intended systemd ExecStopPost and retain its result as recovery-probe.json.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time


R = Path('/var/lib/longhorn/nvme-02/freetoken')
WT = R / 'wt-astra-populate-resident'
REVISION = 'e865d198bff1fec5dde6c6acd7639d3bba65fe57'
NATIVE_SHA = 'c88ed9f877a5a6c4cb3eb4c172b0a7a953794e3ff1104a12b8dcb0f22fb4810f'
HELPER = R / 'tmp/astra-pi-agentic-runtime-20260906-driver.py'
HELPER_SHA = '6829b8f07b6e3d223338018fea730c88d0cf50383d44c082d81d39dcbf543ddb'
RUN_ID = 'astra-pi-agentic-resident-populate-wall-20260906-v2'
OUT = R / 'results' / RUN_ID
DRIVER_UNIT = 'astra-resident-populate-wall-driver-v2'
FLAG = 'FREETOKEN_PREFILL_POPULATE_SKIP_RESIDENT'
SUPPORT_SHA = {
    '_ple_uring': 'dd58afde7b49fd8b6ef3f819181a906d8f2689dbe055985f3c27e6682d04c9cf',
    '_pinned_tensor': '9e1ee95c87e8590ef87c824f6f2626427e32814888cc1043e42c9037c4224df1',
    '_uffd_pager': '071e22fdc54f31bd7f137856ed52f0522570a5cd116daf1ecbaa5f24866e51b4',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def helper():
    assert sha(HELPER) == HELPER_SHA, 'recovery helper changed'
    return module('resident_recovery', HELPER)


def restore_command():
    return ['/usr/bin/python3', str(HELPER), '--remote', 'restore', '--run-id', RUN_ID]


def identity(h):
    assert (WT / '.git').is_file()
    assert h.run('git', '-C', str(WT), 'rev-parse', 'HEAD').stdout.strip() == REVISION
    assert not h.run('git', '-C', str(WT), 'status', '--porcelain').stdout.strip()
    binaries = list((WT / 'python/freetoken/kernel').glob('_cpu_moe.*.so'))
    assert len(binaries) == 1 and sha(binaries[0]) == NATIVE_SHA
    support = {}
    for name, digest in SUPPORT_SHA.items():
        paths = list((WT / 'python/freetoken/kernel').glob(name + '.*.so'))
        assert len(paths) == 1 and sha(paths[0]) == digest, ('missing or changed native support', name)
        support[name] = dict(path=str(paths[0].resolve()), sha256=digest)
    sources = ['python/freetoken/moe/host_banks.py', 'python/freetoken/moe/resident_range.py',
               'python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp',
               'python/freetoken/kernel/csrc/ple_uring/ple_uring_ext.cpp',
               'python/freetoken/kernel/csrc/pinned_tensor.cpp', 'python/freetoken/kernel/csrc/uffd_pager.cpp',
               'bench/sustained-prefill-wall.py', 'bench/selective-prefill.py',
               'bench/staged-prefill-long-output.py', 'bench/moe-fidelity.py']
    return dict(revision=REVISION, native=str(binaries[0].resolve()), native_sha256=NATIVE_SHA,
                support_extensions=support, sources={name: sha(WT / name) for name in sources})


def preflight(h, *, allow_driver=False):
    for unit in ['astra-populate-component-20260906', 'astra-populate-component-recovery-20260906',
                 'astra-resident-populate-wall-driver',
                 *([] if allow_driver else [DRIVER_UNIT])]:
        assert not h.live(unit), ('another experiment is live', unit)
    plan = h.preflight()
    current = identity(h)
    import_code = (
        "import importlib,json,pathlib,torch; "
        "names=['_cpu_moe','_ple_uring','_pinned_tensor','_uffd_pager']; "
        "modules={n:str(pathlib.Path(importlib.import_module('freetoken.kernel.'+n).__file__).resolve()) for n in names}; "
        "print(json.dumps(dict(modules=modules,cuda_initialized=torch.cuda.is_initialized())))"
    )
    imported = h.run(str(h.SRC / '.venv/bin/python'), '-c', import_code,
                     env=dict(os.environ, PYTHONPATH=str(WT / 'python'), CUDA_VISIBLE_DEVICES='',
                              OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'), timeout=120)
    import_probe = json.loads(imported.stdout)
    assert import_probe['cuda_initialized'] is False
    assert import_probe['modules']['_cpu_moe'] == current['native']
    assert all(import_probe['modules'][n] == data['path'] for n, data in current['support_extensions'].items())
    if OUT.exists():
        assert {p.name for p in OUT.iterdir()} == {'recovery-probe.json'}, OUT
    args = h.server_command('optimized')
    args[args.index('--cache-type') + 1] = 'naive'
    args.extend(['--moe-prefill-coalesce', 'populate'])
    env = h.server_env('optimized')
    env['PYTHONPATH'] = str(WT / 'python')
    manifest = json.loads((R / 'results/astra-populate-mixed-prompts-20260906.json').read_text())
    assert manifest['client_sha256'] == current['sources']['bench/sustained-prefill-wall.py']
    assert len(manifest['cases']) == 16 and all(c['predicted_prefill_band_valid'] for c in manifest['cases'])
    return dict(original=plan, identity=current, args=args, env=env, manifest=manifest,
                native_import_probe=import_probe, driver_sha256=sha(Path(__file__)), restore_command=restore_command())


def run_gate(h, plan):
    probe = json.loads((OUT / 'recovery-probe.json').read_text())
    assert probe['verified'] and probe['unit']['InvocationID'] == plan['original']['original_unit']['InvocationID']
    assert probe['restore_command'] == plan['restore_command'] and probe['helper_sha256'] == HELPER_SHA
    report = dict(preflight=plan, recovery_probe=probe, completed=False, arms=[],
                  driver_source=Path(__file__).read_text(), helper_source=HELPER.read_text(),
                  started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    h.save(OUT / 'driver.json', report)
    geometry = None
    expected_cases = [{k: v for k, v in c.items() if not k.startswith('predicted_')}
                      for c in plan['manifest']['cases']]
    try:
        h.stop('freetoken-serve')
        h.wait_gpu_release()
        for arm, flag in [('r1', '0'), ('r2', '1'), ('r3', '1'), ('r4', '0')]:
            assert identity(h) == plan['identity']
            assert sha(Path(__file__)) == plan['driver_sha256'] and sha(HELPER) == HELPER_SHA
            for path, key in [(Path('/etc/systemd/system/freetoken-serve.service'), 'service_sha256'),
                              (R / 'models/flash-e2m1.ftw/config.json', 'model_config_sha256'),
                              (R / 'layer-profile-v3.json', 'layer_profile_sha256')]:
                assert sha(path) == plan['original'][key]
            h.stop(h.SERVER)
            h.wait_gpu_release()
            env = dict(plan['env'], **{FLAG: flag})
            command = ['sudo', '-n', 'systemd-run', '--unit=' + h.SERVER, '--collect', '--uid=jomcgi',
                       '--working-directory=' + str(WT), '--property=KillMode=control-group',
                       '--property=TimeoutStopSec=90', '--property=RuntimeMaxSec=3400']
            command += ['--setenv=' + k + '=' + v for k, v in env.items()]
            record = dict(arm=arm, flag=flag, env=env, command=plan['args'], phase='startup')
            report['arms'].append(record)
            h.save(OUT / 'driver.json', report)
            h.run(*(command + plan['args']))
            record['unit'] = h.state(h.SERVER)
            h.save(OUT / 'driver.json', report)
            record['health'] = h.ready(18090)
            record['readiness'] = h.completion(18090)
            unit = h.state(h.SERVER)
            pids = h.gpu_pids()
            assert len(pids) == 1
            worker = pids[0]
            assert unit['ControlGroup'] in Path(f'/proc/{worker}/cgroup').read_text()
            all_maps = Path(f'/proc/{worker}/maps').read_text().splitlines()
            mapped = [line for line in all_maps if '_cpu_moe.' in line]
            assert mapped and all(line.split()[-1] == plan['identity']['native'] for line in mapped)
            support_maps = {name: [line for line in all_maps if name + '.' in line] for name in SUPPORT_SHA}
            assert support_maps['_ple_uring'], 'native PLE reader absent from worker mappings'
            assert all(line.split()[-1] == plan['identity']['support_extensions'][name]['path']
                       for name, lines in support_maps.items() for line in lines)
            assert (FLAG + '=' + flag).encode() in Path(f'/proc/{worker}/environ').read_bytes().split(b'\0')
            assert h.state('freetoken-serve')['ActiveState'] == 'inactive'
            startup = h.journal(unit['InvocationID'])
            resolved = h.geometry(startup)
            if geometry is None:
                geometry = resolved
            assert resolved == geometry, 'memory or execution geometry changed'
            assert 'CPU MoE prefill coalesce: populate' in startup
            assert 'minimum_chunk=1024 tokens, file_io=buffered' in startup
            assert 'MoE HOT staging: file_io=mmap' in startup
            record.update(unit=unit, worker=worker, native_maps=mapped, native_support_maps=support_maps, geometry=resolved,
                          io_before=h.io_snapshot(worker), phase='timing')
            h.save(OUT / 'driver.json', report)
            print('ARM_READY ' + arm + ' flag=' + flag, flush=True)
            client_command = [str(h.SRC / '.venv/bin/python'), str(WT / 'bench/sustained-prefill-wall.py'),
                              '--tokenizer', str(R / 'models/flash-e2m1.ftw'), '--output', str(OUT / (arm + '.jsonl')),
                              '--mode', 'optimized' if flag == '1' else 'baseline', '--warmup-pairs', '2',
                              '--blocks', '6', '--io-pid', str(worker), '--mixed-prefill']
            with (OUT / (arm + '-client.log')).open('w') as stream:
                client = subprocess.run(client_command, env=dict(os.environ, **plan['env']),
                                        stdout=stream, stderr=subprocess.STDOUT, timeout=2400)
            record['client_returncode'] = client.returncode
            record['client_command'] = client_command
            rows = [json.loads(line) for line in (OUT / (arm + '.jsonl')).read_text().splitlines()]
            assert len(rows) == 16 and [r['ordinal'] for r in rows] == list(range(16))
            assert client.returncode == 0 and all(r['diagnostic_phase_io'] is False for r in rows)
            cases = json.loads((OUT / (arm + '.prompts.json')).read_text())
            assert cases == expected_cases, 'prompt manifest changed'
            assert all(r['prefill_band_valid'] for r in rows), 'unexpected prefill execution band'
            assert all(r['usage']['prompt_tokens'] == c['predicted_prompt_tokens']
                       for r, c in zip(rows, plan['manifest']['cases'])), 'tokenizer prediction mismatch'
            record['response_checks_passed'] = all(r['completed'] and (r['passed'] if r['kind'] == 'json'
                                                       else r['prose_format_passed']) for r in rows)
            record['phase'] = 'fidelity'
            record['fidelity'] = []
            h.save(OUT / 'driver.json', report)
            fidelity = module('resident_fidelity', WT / 'bench/moe-fidelity.py')
            essay = next(c for c in cases if c['kind'] == 'essay' and c['prefill_band'] == 'long')
            background = essay['prompt'].split('<background>\n', 1)[1].split('\n</background>', 1)[0]
            for name, question, expected in fidelity.cases():
                prompt = question if name == 'long_retrieval' else (
                    'The following source excerpt is background only. Answer the question after it.\n'
                    f'<background>\n{background}\n</background>\nQuestion: {question}')
                payload = dict(model='qwen3.6-27b', messages=[dict(role='user', content=prompt)],
                               max_tokens=64, temperature=0, chat_template_kwargs=dict(enable_thinking=False))
                request = h.urllib.request.Request('http://127.0.0.1:18090/v1/chat/completions',
                    data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
                with h.urllib.request.urlopen(request, timeout=300) as response:
                    result = json.load(response)
                choice = result['choices'][0]
                record['fidelity'].append(dict(name=name, prompt=prompt, expected=expected, response=result,
                    passed=choice['message']['content'].strip() == expected and choice['finish_reason'] == 'stop'))
                h.save(OUT / 'driver.json', report)
            assert identity(h) == plan['identity'] and h.state(h.SERVER)['InvocationID'] == unit['InvocationID']
            assert h.gpu_pids() == [worker] and h.state('freetoken-serve')['ActiveState'] == 'inactive'
            record.update(io_after=h.io_snapshot(worker), phase='completed')
            h.stop(h.SERVER)
            h.wait_gpu_release()
            journal = h.journal(unit['InvocationID'])
            (OUT / (arm + '-journal.log')).write_text(journal)
            record['journal_sha256'] = hashlib.sha256(journal.encode()).hexdigest()
            record['http_completions'] = journal.count('POST /v1/chat/completions HTTP/1.1')
            assert record['http_completions'] == 25, 'unaccounted or missing inference requests'
            h.save(OUT / 'driver.json', report)
            print('ARM_COMPLETE ' + arm, flush=True)
        report['completed'] = True
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        report['ended_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        h.save(OUT / 'driver.json', report)
        h.stop(h.SERVER)
        if report['arms'] and report['arms'][-1]['phase'] != 'completed':
            record = report['arms'][-1]
            invocation = record.get('unit', {}).get('InvocationID')
            if invocation:
                try:
                    journal = h.journal(invocation)
                    (OUT / (record['arm'] + '-journal.log')).write_text(journal)
                    record['journal_sha256'] = hashlib.sha256(journal.encode()).hexdigest()
                except Exception as error:
                    record['journal_error'] = f'{type(error).__name__}: {error}'
                h.save(OUT / 'driver.json', report)
        # The enclosing systemd unit executes the pretested recovery command.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    h = helper()
    plan = preflight(h, allow_driver=not args.preflight)
    if args.preflight:
        print(json.dumps(plan, indent=2))
        return

    def interrupted(*_):
        raise RuntimeError('benchmark interrupted')

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)
    run_gate(h, plan)


if __name__ == '__main__':
    main()
