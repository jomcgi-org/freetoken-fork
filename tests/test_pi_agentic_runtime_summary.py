"""Keep warmup costs, failures and mismatched runs out of speedup claims."""

import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    'pi_summary', Path(__file__).parents[1] / 'bench/pi-agentic-runtime-summary.py')
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)

fixed_spec = importlib.util.spec_from_file_location(
    'fixed_client', Path(__file__).parents[1] / 'bench/fixed-continuation-wall.py')
fixed_client = importlib.util.module_from_spec(fixed_spec)
fixed_spec.loader.exec_module(fixed_client)


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


@pytest.fixture
def snapshot_records(records):
    driver = json.loads((records / 'driver.json').read_text())
    plan = driver['preflight']
    plan['identities'] = {mode: {'revision': 'same', 'native_sha256': 'same'}
                          for mode in ('off', 'on')}
    plan['commands'] = {mode: ['ft', 'same'] for mode in ('off', 'on')}
    plan['env'] = {mode: {'FREETOKEN_DECODE_PREFIX_SNAPSHOT': '1' if mode == 'on' else '0',
                          'FREETOKEN_CONTINUATION_TRACE_DIR': ''}
                   for mode in ('off', 'on')}
    driver.update(all_tasks_passed=True, lease_returncode=0)
    for control, (arm, mode) in zip(driver['arms'], summary.SNAPSHOT_ARMS):
        control['mode'] = mode
        start = json.loads((records / (arm + '-server-start.json')).read_text())
        start.update(mode=mode, revision='same', identity=plan['identities'][mode],
                     command=plan['commands'][mode], env=plan['env'][mode],
                     snapshot_enabled=(mode == 'on'))
        write(records, arm + '-server-start.json', start)
        change(records, arm + '/metadata.json', lambda d: d.update(server_metadata=start))
    write(records, 'driver.json', driver)
    return records


def test_snapshot_comparison_preserves_both_orders_and_warmup_accounting(snapshot_records):
    result = summary.summarize(snapshot_records, decode_prefix_snapshot=True)
    comparison = result['comparison']
    assert comparison['modes']['off']['attempted_task_wall_s'] == 400
    assert comparison['modes']['on']['attempted_task_wall_s'] == 320
    assert comparison['wall_reduction_percent'] == pytest.approx(20)
    assert all(order['wall_reduction_percent'] == pytest.approx(20) for order in result['orders'])
    assert result['all_tasks_passed_including_warmups']
    assert len(result['sessions']) == 12
    assert not result['broad_quality_equivalence']


@pytest.mark.parametrize(('field', 'message'), [
    ('identities', 'different runtime identities'),
    ('commands', 'different command lines'),
    ('env', 'unrelated environment difference'),
])
def test_snapshot_comparison_rejects_additional_changes(snapshot_records, field, message):
    def mutate(driver):
        value = driver['preflight'][field]['on']
        if field == 'identities':
            value['native_sha256'] = 'different'
        elif field == 'commands':
            value.append('--different')
        else:
            value['EXTRA'] = '1'
    change(snapshot_records, 'driver.json', mutate)
    with pytest.raises(ValueError, match=message):
        summary.summarize(snapshot_records, decode_prefix_snapshot=True)


def test_snapshot_engine_trace_is_rejected_even_if_both_arms_use_it(snapshot_records):
    def mutate(driver):
        for env in driver['preflight']['env'].values():
            env['FREETOKEN_CONTINUATION_TRACE_DIR'] = '/trace'
    change(snapshot_records, 'driver.json', mutate)
    with pytest.raises(ValueError, match='engine token trace'):
        summary.summarize(snapshot_records, decode_prefix_snapshot=True)


@pytest.mark.parametrize('ordinal', [1, 2])
def test_snapshot_failed_warmup_or_measurement_prevents_speedup_claim(snapshot_records, ordinal):
    change(snapshot_records, f'r2/session-{ordinal}/result.json', lambda d: d.update(
        passed=False, stages=[], task_wall_s=1, verified_task_wall_s=None, error='timeout'))
    change(snapshot_records, 'driver.json', lambda d: (
        d['arms'][1].update(client_returncode=1), d.update(all_tasks_passed=False)))
    result = summary.summarize(snapshot_records, decode_prefix_snapshot=True)
    assert not result['all_tasks_passed_including_warmups']
    assert result['comparison']['wall_reduction_percent'] is None
    assert all(order['wall_reduction_percent'] is None for order in result['orders'])
    assert result['comparison']['modes']['on']['attempted_task_wall_s'] == (320 if ordinal == 1 else 241)


@pytest.mark.parametrize(('fields', 'message'), [
    ({'all_tasks_passed': False}, 'success summary disagrees'),
    ({'lease_returncode': 1}, 'controller or lease failed'),
    ({'error': 'transport closed'}, 'controller or lease failed'),
])
def test_snapshot_controller_claims_must_match_evidence(snapshot_records, fields, message):
    change(snapshot_records, 'driver.json', lambda d: d.update(fields))
    with pytest.raises(ValueError, match=message):
        summary.summarize(snapshot_records, decode_prefix_snapshot=True)


def test_snapshot_start_marker_must_match_the_arm(snapshot_records):
    change(snapshot_records, 'r2-server-start.json', lambda d: d.update(snapshot_enabled=False))
    with pytest.raises(ValueError, match='wrong snapshot marker'):
        summary.summarize(snapshot_records, decode_prefix_snapshot=True)


@pytest.fixture
def fixed_records(snapshot_records):
    root = snapshot_records
    change(root, 'driver.json', lambda d: d['preflight'].update(client_kind='fixed-continuation'))
    change(root, 'driver.json', lambda d: [command.append('--enable-cache-report')
           for command in d['preflight']['commands'].values()])
    for arm, _mode in summary.SNAPSHOT_ARMS:
        change(root, arm + '-server-start.json', lambda d: d['command'].append('--enable-cache-report'))
        start = json.loads((root / (arm + '-server-start.json')).read_text())
        change(root, arm + '/metadata.json', lambda d: d.update(
            sources={'fixed-continuation-wall.py': 'client'}, fixture_sha256='fixture', server_metadata=start))
        for ordinal in (1, 2, 3):
            def mutate(row):
                row['event_metrics']['model_usage'] = [dict(input=1936, output=400, cacheRead=64,
                                                           cacheWrite=0) for _ in (1, 2, 3)]
                for stage, case in zip(row['stages'], fixed_client.fixture(ordinal)):
                    stage['attempts'][0]['response'] = dict(
                        wall_s=5,
                        request={'messages': [{'role': 'user', 'content': case['prompt']}]},
                        completion=dict(choices=[dict(finish_reason='stop', message=dict(
                            role='assistant', content=json.dumps(case['expected'], indent=2)))],
                            usage=dict(prompt_tokens=2000, completion_tokens=400,
                                       prompt_tokens_details={'cached_tokens': 64})))
            change(root, f'{arm}/session-{ordinal}/result.json', mutate)
    return root


def test_fixed_work_keeps_warmups_and_both_orders(fixed_records):
    result = summary.summarize(fixed_records, fixed_continuation=True)
    assert result['fixed_work_qualified'] and result['fixed_work_mismatches'] == []
    assert result['comparison']['wall_reduction_percent'] == pytest.approx(20)
    assert len(result['sessions']) == 12 and len(result['orders']) == 2
    assert result['request_phases']['off']['first_request']['attempted_request_wall_s'] == 20
    assert result['request_phases']['off']['continuations']['attempted_request_wall_s'] == 40
    assert result['request_phases']['on']['continuations']['completed_requests'] == 8


@pytest.mark.parametrize('ordinal', [1, 2])
@pytest.mark.parametrize('field', ['text', 'length'])
def test_fixed_work_rejects_drift_in_warmups_or_measurements(fixed_records, ordinal, field):
    def mutate(row):
        completion = row['stages'][0]['attempts'][0]['response']['completion']
        if field == 'text':
            completion['choices'][0]['message']['content'] += '\n'
        else:
            completion['usage']['completion_tokens'] += 1
            row['event_metrics']['model_usage'][0]['output'] += 1
    change(fixed_records, f'r2/session-{ordinal}/result.json', mutate)
    result = summary.summarize(fixed_records, fixed_continuation=True)
    assert result['all_tasks_passed_including_warmups'] and not result['fixed_work_qualified']
    assert result['comparison']['wall_reduction_percent'] is None
    assert all(order['wall_reduction_percent'] is None for order in result['orders'])
    assert result['comparison']['modes']['on']['attempted_task_wall_s'] == 320


def test_scripted_work_cannot_be_reported_as_a_pi_comparison(fixed_records):
    with pytest.raises(ValueError, match='use --fixed-continuation'):
        summary.summarize(fixed_records, decode_prefix_snapshot=True)


def test_fixed_work_rechecks_answers_instead_of_trusting_pass_flag(fixed_records):
    def mutate(row):
        row['stages'][0]['attempts'][0]['response']['completion']['choices'][0]['message']['content'] = '{}'
    change(fixed_records, 'r2/session-2/result.json', mutate)
    with pytest.raises(ValueError, match='ordered integer records'):
        summary.summarize(fixed_records, fixed_continuation=True)


def test_fixed_work_rechecks_usage_instead_of_trusting_totals(fixed_records):
    change(fixed_records, 'r2/session-2/result.json',
           lambda d: d['event_metrics']['model_usage'][0].update(output=1))
    with pytest.raises(ValueError, match='reported usage differs'):
        summary.summarize(fixed_records, fixed_continuation=True)


def test_fixed_work_keeps_incomplete_response_cost(fixed_records):
    def mutate(row):
        row.update(passed=False, verified_task_wall_s=None, error='HTTP response has no usage')
        row['stages'][0]['attempts'][0]['response']['completion'] = {'error': 'failure'}
        row['stages'] = row['stages'][:1]
        row['stages'][0]['passed'] = False
    change(fixed_records, 'r2/session-2/result.json', mutate)
    change(fixed_records, 'driver.json', lambda d: (
        d['arms'][1].update(client_returncode=1), d.update(all_tasks_passed=False)))
    result = summary.summarize(fixed_records, fixed_continuation=True)
    assert not result['fixed_work_qualified']
    assert result['comparison']['wall_reduction_percent'] is None
    assert result['comparison']['modes']['on']['attempted_task_wall_s'] == 320
    phase = result['request_phases']['on']['first_request']
    assert phase['attempted_request_wall_s'] == 20 and phase['usage_records'] == 3


def test_omitted_zero_hits_cannot_hide_disabled_cache_reporting(fixed_records):
    change(fixed_records, 'driver.json', lambda d:
           d['preflight']['commands']['on'].remove('--enable-cache-report'))
    with pytest.raises(ValueError, match='enabled cache reporting'):
        summary.summarize(fixed_records, fixed_continuation=True)


@pytest.fixture
def carry_records(fixed_records):
    root = fixed_records
    spec = importlib.util.spec_from_file_location(
        'carry_gate_fixture', Path(__file__).parents[1] / 'bench/pi-decode-prefix-wall-driver.py')
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    driver = json.loads((root / 'driver.json').read_text())
    plan = driver['preflight']
    plan['experiment'] = 'prefill-snapshot-carry'
    for mode in ('off', 'on'):
        plan['identities'][mode] = dict(revision=gate.PREFILL_CARRY_REVISIONS[mode],
                                       tree='/runtime/' + mode, native='/same/native.so',
                                       native_sha256='native', cpp_sha256='cpp',
                                       native_extensions={name: dict(path='/same/' + name,
                                           sha256='binary', source_sha256='source')
                                           for name in gate.EXTENSION_SOURCES})
        plan['env'][mode].update(PYTHONPATH='/runtime/' + mode + '/python',
                                 FREETOKEN_DECODE_PREFIX_SNAPSHOT='0')
    for arm, mode in summary.SNAPSHOT_ARMS:
        start = json.loads((root / (arm + '-server-start.json')).read_text())
        start.update(identity=plan['identities'][mode], revision=gate.PREFILL_CARRY_REVISIONS[mode],
                     env=plan['env'][mode], snapshot_enabled=False)
        write(root, arm + '-server-start.json', start)
        change(root, arm + '/metadata.json', lambda d: d.update(server_metadata=start))
    write(root, 'driver.json', driver)
    return root


def test_prefill_carry_qualifies_fixed_work_with_both_orders(carry_records):
    result = summary.summarize(carry_records, prefill_snapshot_carry=True)
    assert result['experiment'] == 'prefill-snapshot-carry'
    assert result['fixed_work_qualified'] and not result['broad_quality_equivalence']
    assert result['comparison']['wall_reduction_percent'] == pytest.approx(20)
    assert len(result['sessions']) == 12 and len(result['orders']) == 2


@pytest.mark.parametrize(('section', 'key', 'value', 'message'), [
    ('identities', 'revision', 'other', 'runtime pair revisions'),
    ('identities', 'native_sha256', 'other', 'native binary'),
    ('identities', 'cpp_sha256', 'other', 'native binary'),
    ('env', 'FREETOKEN_DECODE_PREFIX_SNAPSHOT', '1', 'snapshot flags'),
    ('env', 'PYTHONPATH', '/wrong/python', 'wrong tree'),
    ('env', 'EXTRA', '1', 'environment difference'),
])
def test_prefill_carry_rejects_unrelated_changes(carry_records, section, key, value, message):
    change(carry_records, 'driver.json', lambda d:
           d['preflight'][section]['on'].update({key: value}))
    with pytest.raises(ValueError, match=message):
        summary.summarize(carry_records, prefill_snapshot_carry=True)


def test_prefill_carry_cannot_be_reported_as_decode_snapshot_experiment(carry_records):
    with pytest.raises(ValueError, match='wrong experiment'):
        summary.summarize(carry_records, fixed_continuation=True)


def test_prefill_carry_still_rejects_answer_drift(carry_records):
    change(carry_records, 'r2/session-2/result.json', lambda d:
           d['stages'][0]['attempts'][0]['response']['completion']['choices'][0]['message'].update(
               content=d['stages'][0]['attempts'][0]['response']['completion']['choices'][0]['message']['content'] + '\n'))
    result = summary.summarize(carry_records, prefill_snapshot_carry=True)
    assert not result['fixed_work_qualified']
    assert result['comparison']['wall_reduction_percent'] is None
    assert all(order['wall_reduction_percent'] is None for order in result['orders'])


@pytest.mark.parametrize('change_kind', ['missing', 'changed'])
def test_prefill_carry_requires_matching_ple_native_identity(carry_records, change_kind):
    def mutate(driver):
        ids = driver['preflight']['identities']
        if change_kind == 'missing':
            for row in ids.values():
                row['native_extensions'].pop('_ple_uring')
        else:
            ids['on']['native_extensions']['_ple_uring']['sha256'] = 'other'
    change(carry_records, 'driver.json', mutate)
    with pytest.raises(ValueError, match='native extension identity missing or changed'):
        summary.summarize(carry_records, prefill_snapshot_carry=True)


@pytest.fixture
def profile_records(carry_records):
    root = carry_records
    spec = importlib.util.spec_from_file_location(
        'profile_gate_fixture', Path(__file__).parents[1] / 'bench/pi-decode-prefix-wall-driver.py')
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    driver = json.loads((root / 'driver.json').read_text())
    plan = driver['preflight']
    plan['experiment'] = 'hybrid-profile-retention'
    for mode, tree in [('off', gate.CARRY), ('on', gate.PROFILE)]:
        plan['identities'][mode].update(revision=gate.PROFILE_RETENTION_REVISIONS[mode], tree=str(tree))
        plan['env'][mode]['PYTHONPATH'] = str(tree / 'python')
        plan['commands'][mode] += ['--session-expert-prefetch', 'on', '--session-protect-experts', '64']
    for arm, mode in summary.SNAPSHOT_ARMS:
        start = json.loads((root / (arm + '-server-start.json')).read_text())
        start.update(identity=plan['identities'][mode], revision=plan['identities'][mode]['revision'],
                     env=plan['env'][mode], command=plan['commands'][mode])
        write(root, arm + '-server-start.json', start)
        change(root, arm + '/metadata.json', lambda d: d.update(server_metadata=start))
    write(root, 'driver.json', driver)
    return root


def test_profile_retention_qualifies_same_work_and_reports_both_orders(profile_records):
    result = summary.summarize(profile_records, hybrid_profile_retention=True)
    assert result['experiment'] == 'hybrid-profile-retention'
    assert result['fixed_work_qualified'] and not result['broad_quality_equivalence']
    assert result['comparison']['wall_reduction_percent'] == pytest.approx(20)
    assert len(result['sessions']) == 12 and len(result['orders']) == 2


@pytest.mark.parametrize(('section', 'key', 'value', 'message'), [
    ('identities', 'revision', 'other', 'runtime pair revisions'),
    ('identities', 'native_extensions', {}, 'native extension identity'),
    ('env', 'FREETOKEN_DECODE_PREFIX_SNAPSHOT', '1', 'snapshot flags'),
    ('env', 'PYTHONPATH', '/wrong/python', 'wrong tree'),
    ('env', 'EXTRA', '1', 'environment difference'),
])
def test_profile_retention_rejects_unrelated_changes(profile_records, section, key, value, message):
    change(profile_records, 'driver.json', lambda d:
           d['preflight'][section]['on'].update({key: value}))
    with pytest.raises(ValueError, match=message):
        summary.summarize(profile_records, hybrid_profile_retention=True)


@pytest.mark.parametrize(('flag', 'value'), [('--session-expert-prefetch', 'off'),
                                           ('--session-protect-experts', '0')])
def test_profile_retention_requires_the_same_enabled_advice_policy(profile_records, flag, value):
    def mutate(driver):
        for command in driver['preflight']['commands'].values():
            command[command.index(flag) + 1] = value
    change(profile_records, 'driver.json', mutate)
    with pytest.raises(ValueError, match='wrong session prefetch policy'):
        summary.summarize(profile_records, hybrid_profile_retention=True)


def test_profile_retention_cannot_be_reported_as_prefill_carry(profile_records):
    with pytest.raises(ValueError, match='wrong experiment'):
        summary.summarize(profile_records, prefill_snapshot_carry=True)


def test_profile_retention_still_rejects_answer_drift(profile_records):
    change(profile_records, 'r2/session-2/result.json', lambda d:
           d['stages'][0]['attempts'][0]['response']['completion']['choices'][0]['message'].update(
               content=d['stages'][0]['attempts'][0]['response']['completion']['choices'][0]['message']['content'] + '\n'))
    result = summary.summarize(profile_records, hybrid_profile_retention=True)
    assert not result['fixed_work_qualified']
    assert result['comparison']['wall_reduction_percent'] is None
    assert all(order['wall_reduction_percent'] is None for order in result['orders'])


def test_summary_rejects_conflicting_source_pairs_before_reading_files(tmp_path):
    with pytest.raises(ValueError, match='conflicting experiments'):
        summary.summarize(tmp_path, prefill_snapshot_carry=True, hybrid_profile_retention=True)
