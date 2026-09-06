"""Summarize a completed A/B/B/A Pi runtime comparison without dropping failures.

Reads archived client results and controller evidence only. This never contacts
the server, executes model-written files, or qualifies broad model quality.
"""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path


ARMS = [('r1', 'baseline'), ('r2', 'optimized'),
        ('r3', 'optimized'), ('r4', 'baseline')]
SNAPSHOT_ARMS = [('r1', 'off'), ('r2', 'on'), ('r3', 'on'), ('r4', 'off')]
CLIENT_FIELDS = ('pi_version', 'pi_executable_sha256', 'command', 'client_host',
                 'python', 'task', 'model_config', 'settings', 'requested_sessions',
                 'budgets', 'sources')
FIXED_CLIENT_FIELDS = ('command', 'client_host', 'python', 'task', 'model_config',
                       'requested_sessions', 'budgets', 'sources', 'fixture_sha256')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def session(row, arm, mode, ordinal):
    require(row['ordinal'] == ordinal, f'{arm}: session ordinal mismatch')
    require(type(row['passed']) is bool, f'{arm}: invalid task success value')
    require(row.get('trace') is False, f'{arm}: client trace is enabled or missing')
    wall = row['task_wall_s']
    require(isinstance(wall, (int, float)) and not isinstance(wall, bool)
            and math.isfinite(wall) and wall > 0, f'{arm}: invalid task wall time')
    if row['passed']:
        stages = row['stages']
        require([s['stage'] for s in stages] == [1, 2, 3],
                f'{arm}: successful task lacks all three stages')
        require(all(s['passed'] is True and s['attempts']
                    and s['attempts'][-1]['verification']['passed'] is True
                    and s['attempts'][-1]['verification']['exit_code'] == 0
                    for s in stages), f'{arm}: successful task has a failed check')
        require(row.get('error') is None and row.get('artifact_error') is None,
                f'{arm}: successful task has an error or missing artifacts')
        require(row['verified_task_wall_s'] == wall,
                f'{arm}: successful task wall clocks disagree')
    metrics = row['event_metrics']
    usage = metrics.get('model_usage', [])
    return dict(arm=arm, mode=mode, ordinal=ordinal, warmup=ordinal == 1,
                passed=row['passed'], task_wall_s=wall, error=row.get('error'),
                startup_s=row.get('startup_s'), model_calls=row.get('model_calls'),
                tool_calls=metrics.get('tool_calls'), tool_errors=metrics.get('tool_errors'),
                tool_wall_s=metrics.get('tool_wall_s'), repair_prompts=row.get('repair_prompts'),
                tokens={key: sum(u.get(key, 0) for u in usage)
                        for key in ('input', 'output', 'cacheRead', 'cacheWrite')},
                stages=row['stages'])


def totals(rows):
    wall = sum(row['task_wall_s'] for row in rows)
    passed = sum(row['passed'] is True for row in rows)
    counts = {key: sum(row[key] for row in rows)
              if all(row.get(key) is not None for row in rows) else None
              for key in ('model_calls', 'tool_errors', 'repair_prompts')}
    return dict(sessions=len(rows), passed=passed, failed=len(rows) - passed,
                attempted_task_wall_s=wall,
                checked_tasks_per_hour=3600 * passed / wall
                if rows and passed == len(rows) else None,
                **counts,
                tokens={key: sum(row['tokens'][key] for row in rows)
                        for key in ('input', 'output', 'cacheRead', 'cacheWrite')})


def compare(rows, mode_names=('baseline', 'optimized')):
    modes = {mode: totals([r for r in rows if r['mode'] == mode])
             for mode in mode_names}
    complete = all(m['sessions'] and not m['failed'] for m in modes.values())
    baseline, optimized = (modes[m]['attempted_task_wall_s']
                           for m in mode_names)
    return dict(modes=modes,
                wall_reduction_percent=100 * (1 - optimized / baseline) if complete else None,
                throughput_ratio=baseline / optimized if complete else None)


def summarize(root, *, decode_prefix_snapshot=False, fixed_continuation=False,
              prefill_snapshot_carry=False, hybrid_profile_retention=False, gpu_source_staging=False,
              hot_host_cache_reclaim=False, original_baseline=False):
    require(sum((prefill_snapshot_carry, hybrid_profile_retention, decode_prefix_snapshot, gpu_source_staging, hot_host_cache_reclaim, original_baseline)) <= 1,
            'conflicting experiments')
    runtime_pair = prefill_snapshot_carry or hybrid_profile_retention
    different_runtimes = runtime_pair or original_baseline
    same_runtime_policy = gpu_source_staging or hot_host_cache_reclaim
    fixed_continuation = fixed_continuation or different_runtimes or same_runtime_policy
    decode_prefix_snapshot = decode_prefix_snapshot or fixed_continuation
    hashes = {}
    expected_arms = SNAPSHOT_ARMS if decode_prefix_snapshot else ARMS
    mode_names = ('off', 'on') if decode_prefix_snapshot else ('baseline', 'optimized')

    def read(relative):
        data = (root / relative).read_bytes()
        hashes[str(relative)] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    driver = read(Path('driver.json'))
    require(driver.get('completed') is True, 'controller has not completed all four arms')
    require(driver.get('restored') is True
            and driver.get('restoration', {}).get('verified') is True,
            'original service restoration is not verified')
    require([(a['arm'], a['mode']) for a in driver['arms']] == expected_arms,
            'controller does not contain exactly A/B/B/A')
    plan = driver['preflight']
    if fixed_continuation:
        require(plan.get('client_kind') == 'fixed-continuation', 'wrong continuation client')
        require(all('--enable-cache-report' in command for command in plan['commands'].values()),
                'fixed continuation requires enabled cache reporting')
    else:
        require(plan.get('client_kind', 'pi') == 'pi', 'use --fixed-continuation for scripted requests')
    if decode_prefix_snapshot:
        experiment = 'prefill-snapshot-carry' if prefill_snapshot_carry else 'decode-prefix-snapshot'
        if hybrid_profile_retention:
            experiment = 'hybrid-profile-retention'
        if gpu_source_staging:
            experiment = 'gpu-source-staging'
        if hot_host_cache_reclaim:
            experiment = 'hot-host-cache-reclaim'
        if original_baseline:
            experiment = 'original-continuation'
        require(plan.get('experiment', 'decode-prefix-snapshot') == experiment, 'wrong experiment')
        if different_runtimes or same_runtime_policy:
            spec = importlib.util.spec_from_file_location(
                'carry_gate', Path(__file__).with_name('pi-decode-prefix-wall-driver.py'))
            gate = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(gate)
            ids = plan['identities']
        if runtime_pair:
            revisions = (gate.PROFILE_RETENTION_REVISIONS if hybrid_profile_retention
                         else gate.PREFILL_CARRY_REVISIONS)
            require({m: ids[m]['revision'] for m in ('off', 'on')} == revisions,
                    'wrong runtime pair revisions')
            require(ids['off']['tree'] != ids['on']['tree'], 'same runtime pair tree')
            require(all(ids['off'].get(k) and ids['off'][k] == ids['on'].get(k)
                        for k in ('native', 'native_sha256', 'cpp_sha256')),
                    'runtime pair native binary or source changed')
            extensions = ids['off'].get('native_extensions', {})
            require(set(extensions) == set(gate.EXTENSION_SOURCES)
                    and extensions == ids['on'].get('native_extensions')
                    and all(all(row.get(k) for k in ('path', 'sha256', 'source_sha256'))
                            for row in extensions.values()),
                    'runtime pair native extension identity missing or changed')
        elif original_baseline:
            try:
                gate.qualify_original_identities(ids)
            except AssertionError as exc:
                raise ValueError('combined-runtime identity mismatch: ' + str(exc)) from exc
        else:
            require(plan['identities']['off'] == plan['identities']['on'],
                    'snapshot arms used different runtime identities')
        if same_runtime_policy:
            revision = gate.HOT_HOST_REVISION if hot_host_cache_reclaim else gate.GPU_SOURCE_REVISION
            tree = gate.HOT_HOST if hot_host_cache_reclaim else gate.GPU_SOURCE
            require(ids['off']['revision'] == revision
                    and ids['off']['tree'] == str(tree), 'wrong policy runtime')
            extensions = ids['off'].get('native_extensions', {})
            require(set(extensions) == set(gate.EXTENSION_SOURCES)
                    and all(all(row.get(k) for k in ('path', 'sha256', 'source_sha256'))
                            for row in extensions.values()), 'policy native identity missing')
            try:
                qualify = gate.qualify_hot_host_commands if hot_host_cache_reclaim else gate.qualify_source_commands
                qualify(plan['commands'])
            except AssertionError as exc:
                raise ValueError('policy command mismatch: ' + str(exc)) from exc
        elif original_baseline:
            try:
                gate.qualify_original_commands(plan['commands'])
            except AssertionError as exc:
                raise ValueError('combined-runtime command mismatch: ' + str(exc)) from exc
        else:
            require(plan['commands']['off'] == plan['commands']['on'],
                    'snapshot arms used different command lines')
        off, on = plan['env']['off'], plan['env']['on']
        require(off['FREETOKEN_DECODE_PREFIX_SNAPSHOT'] == '0'
                and on['FREETOKEN_DECODE_PREFIX_SNAPSHOT'] == ('0' if different_runtimes or same_runtime_policy else '1'),
                'wrong snapshot flags')
        require({key for key in off.keys() | on.keys() if off.get(key) != on.get(key)}
                == (set() if same_runtime_policy else ({'PYTHONPATH'} if different_runtimes else {'FREETOKEN_DECODE_PREFIX_SNAPSHOT'})),
                'unrelated environment difference')
        if different_runtimes or same_runtime_policy:
            require(all(plan['env'][m]['PYTHONPATH'] == str(Path(ids[m]['tree']) / 'python')
                        for m in ('off', 'on')), 'runtime pair environment points at wrong tree')
        if hybrid_profile_retention:
            require({m: ids[m]['tree'] for m in ('off', 'on')}
                    == {'off': str(gate.CARRY), 'on': str(gate.PROFILE)},
                    'wrong profile retention runtime tree')
            for flag, value in [('--session-expert-prefetch', 'on'), ('--session-protect-experts', '64')]:
                command = plan['commands']['off']
                require(command.count(flag) == 1 and command.index(flag) + 1 < len(command)
                        and command[command.index(flag) + 1] == value,
                        'wrong session prefetch policy')
        require(off['FREETOKEN_CONTINUATION_TRACE_DIR'] == '', 'engine token trace enabled')
        if hot_host_cache_reclaim or original_baseline:
            require(all(e.get('FREETOKEN_HOT_HOST_CACHE_CENSUS_DIR') == '' for e in (off, on)),
                    'HOT census enabled or unset')
        require(driver.get('lease_returncode') == 0 and not driver.get('error')
                and not driver.get('restoration_error'), 'snapshot controller or lease failed')
    rows, arms = [], []
    first_geometry = first_client = None
    for control, (arm, mode) in zip(driver['arms'], expected_arms):
        start = read(Path(arm + '-server-start.json'))
        end = read(Path(arm + '-server-end.json'))
        metadata = read(Path(arm) / 'metadata.json')
        client_summary = read(Path(arm) / 'summary.json')
        require(start['arm'] == arm and start['mode'] == mode, f'{arm}: wrong server arm')
        if decode_prefix_snapshot:
            require(start.get('snapshot_enabled') is (mode == 'on' and not different_runtimes and not same_runtime_policy),
                    f'{arm}: wrong snapshot marker')
        require(start['identity'] == plan['identities'][mode]
                and start['revision'] == plan['identities'][mode]['revision']
                and start['driver_sha256'] == plan['driver_sha256'],
                f'{arm}: server source identity mismatch')
        require(start['command'] == plan['commands'][mode]
                and start['env'] == plan['env'][mode], f'{arm}: server configuration mismatch')
        require(start['diagnostics_disabled'] is True
                and '--moe-collect-stats' not in start['command']
                and '--moe-step-timing' not in start['command'], f'{arm}: server diagnostics enabled')
        require(metadata['server_metadata'] == start, f'{arm}: client/server evidence mismatch')
        require(metadata['trace'] is False and metadata['requested_sessions'] == 3,
                f'{arm}: wrong client trace or session count')
        client_file = 'fixed-continuation-wall.py' if fixed_continuation else 'pi-agentic-wall.py'
        require(metadata['sources'][client_file] == plan['client_sha256'],
                f'{arm}: client source mismatch')
        require(client_summary['completed_schedule'] is True
                and client_summary['cancelled'] is False and client_summary['sessions'] == 3,
                f'{arm}: client schedule incomplete')
        require(end == control['end'] and end['unit']['ActiveState'] == 'active'
                and end['unit']['InvocationID'] == start['unit']['InvocationID']
                and end['gpu_pids'] == [start['worker']], f'{arm}: server changed during session')
        require(all(record['original_unit']['ActiveState'] == 'inactive' for record in (start, end)),
                f'{arm}: original service was active during comparison')
        client = {key: metadata[key] for key in (FIXED_CLIENT_FIELDS if fixed_continuation else CLIENT_FIELDS)}
        if first_geometry is None:
            first_geometry, first_client = start['geometry'], client
        require(start['geometry'] == first_geometry, f'{arm}: server geometry mismatch')
        require(client == first_client, f'{arm}: client configuration mismatch')
        if gpu_source_staging:
            layout = start['source_layout']
            try:
                layout_text = '\n'.join([layout['bank'], layout['prefill_paths'], *layout['staging']])
                require(gate.source_layout(layout_text, mode) == layout,
                        f'{arm}: source layout metadata mismatch')
                placement = gate.gpu_compute_placement(layout['prefill_paths'])
                require(start['geometry']['GPU compute placement'] == placement,
                        f'{arm}: source compute placement mismatch')
            except AssertionError as exc:
                raise ValueError(f'{arm}: source layout mismatch: {exc}') from exc
        if hot_host_cache_reclaim or (original_baseline and mode == 'on'):
            require(start.get('hot_host_cache') == ('retain' if mode == 'off' else 'reclaim'),
                    f'{arm}: wrong HOT host cache policy marker')
        if same_runtime_policy or original_baseline:
            for snapshot in (start['memory_before'], end['memory_after']):
                for section, keys in [('worker_bytes', gate.WORKER_MEMORY_FIELDS),
                                      ('system_bytes', gate.SYSTEM_MEMORY_FIELDS)]:
                    require(set(snapshot[section]) == set(keys) and all(
                        type(v) is int and v >= 0 for v in snapshot[section].values()),
                        f'{arm}: invalid OS memory counters')
        arm_rows = [session(read(Path(arm) / f'session-{n}/result.json'), arm, mode, n)
                    for n in (1, 2, 3)]
        expected_code = 0 if all(row['passed'] for row in arm_rows) else 1
        require(control['client_returncode'] == expected_code, f'{arm}: unexpected client exit code')
        delta = int(end['io_after']['read_bytes']) - int(start['io_before']['read_bytes'])
        require(delta >= 0, f'{arm}: worker I/O counter decreased')
        arms.append(dict(arm=arm, mode=mode, warmup=arm_rows[0],
                         measured=totals(arm_rows[1:]),
                         worker_read_bytes_including_warmup=delta))
        if same_runtime_policy or original_baseline:
            arms[-1].update(memory_before=start['memory_before'], memory_after=end['memory_after'])
        if gpu_source_staging:
            arms[-1]['source_layout'] = layout
        if hot_host_cache_reclaim:
            arms[-1]['hot_host_cache'] = start['hot_host_cache']
        rows.extend(arm_rows)
    measured = [row for row in rows if not row['warmup']]
    result = dict(completed_protocol=True, broad_quality_equivalence=False,
                review_required='Review journals, complete model outputs and final workspaces. '
                                'Recorded checks do not establish broad quality equivalence.',
                comparison=compare(measured, mode_names),
                orders=[dict(order=order, **compare([r for r in measured if r['arm'] in pair], mode_names))
                        for order, pair in [('A/B', ('r1', 'r2')), ('B/A', ('r3', 'r4'))]],
                arms=arms, sessions=rows, geometry=first_geometry, input_sha256=hashes)
    if decode_prefix_snapshot:
        result['experiment'] = experiment
        all_passed = all(row['passed'] for row in rows)
        require(driver.get('all_tasks_passed') is all_passed,
                'controller success summary disagrees with task records')
        result['all_tasks_passed_including_warmups'] = all_passed
        if not all_passed:
            for comparison in [result['comparison'], *result['orders']]:
                comparison['wall_reduction_percent'] = None
                comparison['throughput_ratio'] = None
    if fixed_continuation:
        spec = importlib.util.spec_from_file_location(
            'fixed_continuation_client', Path(__file__).with_name('fixed-continuation-wall.py'))
        client_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(client_module)
        mismatches = client_module.fixed_work_mismatches(rows)
        result['fixed_work_qualified'] = not mismatches and all(row['passed'] for row in rows)
        result['fixed_work_mismatches'] = mismatches
        result['request_phases'] = {}
        for mode in ('off', 'on'):
            phases = {}
            for name, stage_ids in [('first_request', (1,)), ('continuations', (2, 3))]:
                responses = [stage['attempts'][0]['response']
                             for row in measured if row['mode'] == mode
                             for stage in row['stages'] if stage['stage'] in stage_ids]
                completed = [r['completion'] for r in responses if 'completion' in r]
                usages = [r.get('usage', {}) for r in completed]
                available = [u for u in usages if all(type(u.get(k)) is int for k in
                             ('prompt_tokens', 'completion_tokens'))]
                phases[name] = dict(attempted_requests=len(responses),
                                    completed_requests=len(completed),
                                    usage_records=len(available),
                                    attempted_request_wall_s=sum(r['wall_s'] for r in responses),
                                    prompt_tokens=sum(u['prompt_tokens'] for u in available),
                                    completion_tokens=sum(u['completion_tokens'] for u in available),
                                    cached_tokens=sum(u.get('prompt_tokens_details', {}).get(
                                        'cached_tokens', 0) for u in available))
            result['request_phases'][mode] = phases
        result['review_required'] = (
            'Review service journals and full responses. Matched request bodies, answer bytes and '
            'token counts control work on this synthetic fixture; they do not prove identical '
            'internal token IDs, expert routes or broad model quality.')
        if not result['fixed_work_qualified']:
            for comparison in [result['comparison'], *result['orders']]:
                comparison['wall_reduction_percent'] = None
                comparison['throughput_ratio'] = None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--decode-prefix-snapshot', action='store_true',
                        help='verify the same-runtime snapshot off/on/on/off protocol')
    parser.add_argument('--fixed-continuation', action='store_true',
                        help='also require matched scripted requests, answers and token counts')
    parser.add_argument('--prefill-snapshot-carry', action='store_true',
                        help='verify the pinned parent/fix scripted comparison with decode snapshots off')
    parser.add_argument('--hybrid-profile-retention', action='store_true',
                        help='verify the pinned expert-profile parent/fix scripted comparison')
    parser.add_argument('--gpu-source-staging', action='store_true',
                        help='verify the same-runtime pinned/staged GPU source comparison')
    parser.add_argument('--hot-host-cache-reclaim', action='store_true',
                        help='verify the same-runtime HOT source page retain/reclaim comparison')
    parser.add_argument('--original-baseline', action='store_true',
                        help='verify original versus selected runtime on identical continuations')
    args = parser.parse_args()
    try:
        result = summarize(args.results, decode_prefix_snapshot=args.decode_prefix_snapshot,
                           fixed_continuation=args.fixed_continuation,
                           prefill_snapshot_carry=args.prefill_snapshot_carry,
                           hybrid_profile_retention=args.hybrid_profile_retention,
                           gpu_source_staging=args.gpu_source_staging,
                           hot_host_cache_reclaim=args.hot_host_cache_reclaim,
                           original_baseline=args.original_baseline)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f'Cannot summarize comparison: {exc}\n')
    print(json.dumps(result, indent=2, allow_nan=False))
    fixed = args.fixed_continuation or args.prefill_snapshot_carry or args.hybrid_profile_retention or args.gpu_source_staging or args.hot_host_cache_reclaim or args.original_baseline
    if fixed and not result['fixed_work_qualified']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
