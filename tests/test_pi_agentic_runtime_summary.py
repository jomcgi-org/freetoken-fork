"""Keep warmup costs, failures and mismatched runs out of speedup claims."""

import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    'pi_summary', Path(__file__).parents[1] / 'bench/pi-agentic-runtime-summary.py')
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def write(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def change(root, name, mutate):
    path = root / name
    value = json.loads(path.read_text())
    mutate(value)
    path.write_text(json.dumps(value))


@pytest.fixture
def records(tmp_path):
    plan = dict(identities={m: dict(revision=m) for m in ('baseline', 'optimized')},
                commands={m: ['ft', m] for m in ('baseline', 'optimized')},
                env={m: {} for m in ('baseline', 'optimized')},
                driver_sha256='driver', client_sha256='client')
    controls = []
    for arm, mode in summary.ARMS:
        start = dict(arm=arm, mode=mode, identity=plan['identities'][mode], revision=mode,
                     driver_sha256='driver', command=plan['commands'][mode], env={},
                     diagnostics_disabled=True, geometry={'expert_slots': 3753},
                     unit={'InvocationID': arm}, worker=123,
                     original_unit={'ActiveState': 'inactive'}, io_before={'read_bytes': '100'})
        end = dict(unit={'InvocationID': arm, 'ActiveState': 'active'}, gpu_pids=[123],
                   original_unit={'ActiveState': 'inactive'}, io_after={'read_bytes': '3100'})
        metadata = dict.fromkeys(summary.CLIENT_FIELDS, 'same')
        metadata.update(requested_sessions=3, sources={'pi-agentic-wall.py': 'client'},
                        server_metadata=start, trace=False)
        write(tmp_path, arm + '-server-start.json', start)
        write(tmp_path, arm + '-server-end.json', end)
        write(tmp_path, arm + '/metadata.json', metadata)
        write(tmp_path, arm + '/summary.json', dict(completed_schedule=True, cancelled=False, sessions=3))
        controls.append(dict(arm=arm, mode=mode, client_returncode=0, end=end))
        for ordinal in (1, 2, 3):
            # Large, unequal warmups make accidental inclusion visibly wrong.
            wall = (1000 if mode == 'baseline' else 10) if ordinal == 1 else (
                100 if mode == 'baseline' else 80)
            stages = [dict(stage=n, passed=True, attempts=[dict(verification=dict(
                passed=True, exit_code=0))]) for n in (1, 2, 3)]
            write(tmp_path, f'{arm}/session-{ordinal}/result.json', dict(
                ordinal=ordinal, passed=True, trace=False, task_wall_s=wall,
                verified_task_wall_s=wall, stages=stages, error=None, model_calls=4,
                event_metrics=dict(tool_errors=0, tool_calls=3,
                                   model_usage=[dict(input=100, output=200, cacheRead=300)])))
    write(tmp_path, 'driver.json', dict(completed=True, restored=True,
                                       restoration={'verified': True}, arms=controls, preflight=plan))
    return tmp_path


def test_warmups_are_retained_but_never_counted_as_measured_time(records):
    result = summary.summarize(records)
    comparison = result['comparison']
    assert comparison['modes']['baseline']['attempted_task_wall_s'] == 400
    assert comparison['modes']['optimized']['attempted_task_wall_s'] == 320
    assert comparison['wall_reduction_percent'] == pytest.approx(20)
    assert comparison['throughput_ratio'] == pytest.approx(1.25)
    assert comparison['modes']['optimized']['checked_tasks_per_hour'] == 45
    assert [order['order'] for order in result['orders']] == ['A/B', 'B/A']
    assert all(order['wall_reduction_percent'] == pytest.approx(20) for order in result['orders'])
    assert len(result['sessions']) == 12 and sum(s['warmup'] for s in result['sessions']) == 4
    assert result['arms'][0]['worker_read_bytes_including_warmup'] == 3000
    assert not result['broad_quality_equivalence']


def test_fast_failure_retains_time_without_claiming_a_speedup(records):
    change(records, 'r2/session-2/result.json', lambda d: d.update(
        passed=False, stages=[], task_wall_s=1, verified_task_wall_s=None, error='timeout'))
    change(records, 'driver.json', lambda d: d['arms'][1].update(client_returncode=1))
    result = summary.summarize(records)
    comparison = result['comparison']
    assert comparison['modes']['optimized']['attempted_task_wall_s'] == 241
    assert comparison['modes']['optimized']['passed'] == 3
    assert comparison['modes']['optimized']['failed'] == 1
    assert comparison['modes']['optimized']['checked_tasks_per_hour'] is None
    assert comparison['wall_reduction_percent'] is None
    assert comparison['throughput_ratio'] is None
    assert result['orders'][0]['wall_reduction_percent'] is None
    assert result['orders'][1]['wall_reduction_percent'] == pytest.approx(20)


@pytest.mark.parametrize(('name', 'mutate', 'message'), [
    ('driver.json', lambda d: d.update(completed=False), 'not completed'),
    ('driver.json', lambda d: d.update(restored=False), 'restoration'),
    ('driver.json', lambda d: d['arms'].reverse(), 'A/B/B/A'),
    ('r2/metadata.json', lambda d: d.update(budgets={'timeout': 100}), 'client configuration'),
    ('r2/metadata.json', lambda d: d.update(trace=True), 'client trace'),
    ('r2-server-start.json', lambda d: d.update(diagnostics_disabled=False), 'server diagnostics'),
    ('r2-server-end.json', lambda d: d['unit'].update(InvocationID='other'), 'server changed'),
    ('r2/summary.json', lambda d: d.update(cancelled=True), 'schedule incomplete'),
    ('r2/session-2/result.json', lambda d: d['stages'].pop(), 'all three stages'),
    ('r2/session-2/result.json', lambda d: d.update(ordinal=1), 'ordinal mismatch'),
    ('r2/session-2/result.json', lambda d: d.update(passed='true'), 'success value'),
    ('r2/session-2/result.json', lambda d: d.update(task_wall_s=float('nan')), 'invalid task wall'),
])
def test_incomplete_or_confounded_records_are_rejected(records, name, mutate, message):
    change(records, name, mutate)
    with pytest.raises(ValueError, match=message):
        summary.summarize(records)


def test_changed_geometry_is_rejected_even_when_client_agrees(records):
    change(records, 'r2-server-start.json', lambda d: d['geometry'].update(expert_slots=3900))
    start = json.loads((records / 'r2-server-start.json').read_text())
    change(records, 'r2/metadata.json', lambda d: d.update(server_metadata=start))
    with pytest.raises(ValueError, match='geometry mismatch'):
        summary.summarize(records)
