"""Summarize a completed A/B/B/A Pi runtime comparison without dropping failures.

Reads archived client results and controller evidence only. This never contacts
the server, executes model-written files, or qualifies broad model quality.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path


ARMS = [('r1', 'baseline'), ('r2', 'optimized'),
        ('r3', 'optimized'), ('r4', 'baseline')]
CLIENT_FIELDS = ('pi_version', 'pi_executable_sha256', 'command', 'client_host',
                 'python', 'task', 'model_config', 'settings', 'requested_sessions',
                 'budgets', 'sources')


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


def compare(rows):
    modes = {mode: totals([r for r in rows if r['mode'] == mode])
             for mode in ('baseline', 'optimized')}
    complete = all(m['sessions'] and not m['failed'] for m in modes.values())
    baseline, optimized = (modes[m]['attempted_task_wall_s']
                           for m in ('baseline', 'optimized'))
    return dict(modes=modes,
                wall_reduction_percent=100 * (1 - optimized / baseline) if complete else None,
                throughput_ratio=baseline / optimized if complete else None)


def summarize(root):
    hashes = {}

    def read(relative):
        data = (root / relative).read_bytes()
        hashes[str(relative)] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    driver = read(Path('driver.json'))
    require(driver.get('completed') is True, 'controller has not completed all four arms')
    require(driver.get('restored') is True
            and driver.get('restoration', {}).get('verified') is True,
            'original service restoration is not verified')
    require([(a['arm'], a['mode']) for a in driver['arms']] == ARMS,
            'controller does not contain exactly A/B/B/A')
    plan = driver['preflight']
    rows, arms = [], []
    first_geometry = first_client = None
    for control, (arm, mode) in zip(driver['arms'], ARMS):
        start = read(Path(arm + '-server-start.json'))
        end = read(Path(arm + '-server-end.json'))
        metadata = read(Path(arm) / 'metadata.json')
        client_summary = read(Path(arm) / 'summary.json')
        require(start['arm'] == arm and start['mode'] == mode, f'{arm}: wrong server arm')
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
        require(metadata['sources']['pi-agentic-wall.py'] == plan['client_sha256'],
                f'{arm}: client source mismatch')
        require(client_summary['completed_schedule'] is True
                and client_summary['cancelled'] is False and client_summary['sessions'] == 3,
                f'{arm}: client schedule incomplete')
        require(end == control['end'] and end['unit']['ActiveState'] == 'active'
                and end['unit']['InvocationID'] == start['unit']['InvocationID']
                and end['gpu_pids'] == [start['worker']], f'{arm}: server changed during session')
        require(all(record['original_unit']['ActiveState'] == 'inactive' for record in (start, end)),
                f'{arm}: original service was active during comparison')
        client = {key: metadata[key] for key in CLIENT_FIELDS}
        if first_geometry is None:
            first_geometry, first_client = start['geometry'], client
        require(start['geometry'] == first_geometry, f'{arm}: server geometry mismatch')
        require(client == first_client, f'{arm}: client configuration mismatch')
        arm_rows = [session(read(Path(arm) / f'session-{n}/result.json'), arm, mode, n)
                    for n in (1, 2, 3)]
        expected_code = 0 if all(row['passed'] for row in arm_rows) else 1
        require(control['client_returncode'] == expected_code, f'{arm}: unexpected client exit code')
        delta = int(end['io_after']['read_bytes']) - int(start['io_before']['read_bytes'])
        require(delta >= 0, f'{arm}: worker I/O counter decreased')
        arms.append(dict(arm=arm, mode=mode, warmup=arm_rows[0],
                         measured=totals(arm_rows[1:]),
                         worker_read_bytes_including_warmup=delta))
        rows.extend(arm_rows)
    measured = [row for row in rows if not row['warmup']]
    return dict(completed_protocol=True, broad_quality_equivalence=False,
                review_required='Review journals, complete model outputs and final workspaces. '
                                'Recorded checks do not establish broad quality equivalence.',
                comparison=compare(measured),
                orders=[dict(order=order, **compare([r for r in measured if r['arm'] in pair]))
                        for order, pair in [('A/B', ('r1', 'r2')), ('B/A', ('r3', 'r4'))]],
                arms=arms, sessions=rows, geometry=first_geometry, input_sha256=hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    try:
        result = summarize(args.results)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f'Cannot summarize comparison: {exc}\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
