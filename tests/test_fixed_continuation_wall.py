"""Reject changed or failed work before interpreting continuation wall time."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    'fixed_wall', Path(__file__).parents[1] / 'bench/fixed-continuation-wall.py')
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)


def reply(expected):
    return dict(choices=[dict(finish_reason='stop', message=dict(
        role='assistant', content=json.dumps(expected, indent=2)))],
        usage=dict(prompt_tokens=2000, completion_tokens=400,
                   prompt_tokens_details=dict(cached_tokens=64)))


def run_conversation(ordinal=1):
    cases = iter(client.fixture(ordinal))
    return client.conversation(ordinal, base_url='unused', model='qwen3.6-27b',
                               send=lambda _url, _body: reply(next(cases)['expected']))


def test_fixture_is_stable_distinct_and_preserves_prior_answers_in_requests():
    assert client.fixture(1) == client.fixture(1)
    assert len({client.fixture(i)[0]['prompt'] for i in (1, 2, 3)}) == 3
    row = run_conversation()
    assert row['passed'] and row['model_calls'] == 3 and row['repair_prompts'] == 0
    assert row['verified_task_wall_s'] == row['task_wall_s'] > 0
    requests = [s['attempts'][0]['response']['request'] for s in row['stages']]
    assert [len(r['messages']) for r in requests] == [1, 3, 5]
    assert requests[2]['messages'][:3] == requests[1]['messages']
    assert all(r['temperature'] == 0 and r['chat_template_kwargs']['enable_thinking'] is False
               and 'response_format' not in r and 'logit_bias' not in r for r in requests)


@pytest.mark.parametrize('fault', ['duplicate', 'boolean', 'wrong_value', 'extra_key',
                                   'reordered', 'truncated', 'reasoning', 'missing_usage',
                                   'missing_cache_usage', 'invalid_cache_usage'])
def test_invalid_answers_and_accounting_are_rejected(fault):
    expected = {'a': 1, 'b': 2}
    result = reply(expected)
    message = result['choices'][0]['message']
    if fault == 'duplicate': message['content'] = '{"a":1,"b":2,"a":1}'
    if fault == 'boolean': message['content'] = '{"a":true,"b":2}'
    if fault == 'wrong_value': message['content'] = '{"a":0,"b":2}'
    if fault == 'extra_key': message['content'] = '{"a":1,"b":2,"c":3}'
    if fault == 'reordered': message['content'] = '{"b":2,"a":1}'
    if fault == 'truncated': result['choices'][0]['finish_reason'] = 'length'
    if fault == 'reasoning': message['reasoning_content'] = 'extra work'
    if fault == 'missing_usage': result['usage'].pop('completion_tokens')
    if fault == 'missing_cache_usage': result['usage']['prompt_tokens_details'].pop('cached_tokens')
    if fault == 'invalid_cache_usage': result['usage']['prompt_tokens_details']['cached_tokens'] = 9999
    with pytest.raises(ValueError):
        client.checked_answer(result, expected)


def test_omitted_cache_details_mean_zero_hits_as_in_the_server_wire_format():
    expected = {'a': 1, 'b': 2}
    result = reply(expected)
    result['usage'].pop('prompt_tokens_details')
    text, usage = client.checked_answer(result, expected)
    assert json.loads(text) == expected
    assert usage == dict(input=2000, output=400, cacheRead=0, cacheWrite=0)


def test_failure_keeps_attempt_response_and_wall_time_without_fabricating_later_turns():
    calls = []
    def send(_url, _body):
        calls.append(1)
        if len(calls) == 2:
            raise TimeoutError('deliberate transport timeout')
        return reply(client.fixture(1)[0]['expected'])
    row = client.conversation(1, base_url='unused', model='test', send=send)
    assert not row['passed'] and row['verified_task_wall_s'] is None
    assert row['task_wall_s'] > 0 and row['model_calls'] == len(row['stages']) == 2
    assert row['stages'][0]['passed'] and not row['stages'][1]['passed']
    assert row['stages'][1]['attempts'][0]['response']['wall_s'] > 0
    assert 'deliberate transport timeout' in row['error']


def test_incorrect_full_response_is_retained():
    invalid = reply({'wrong': 1})
    row = client.conversation(1, base_url='unused', model='test', send=lambda *_: invalid)
    assert not row['passed'] and row['model_calls'] == 1
    assert row['stages'][0]['attempts'][0]['response']['completion'] == invalid


@pytest.mark.parametrize('field', ['request', 'message', 'prompt_tokens', 'completion_tokens'])
def test_same_values_with_different_bytes_or_lengths_are_not_fixed_work(field):
    first = dict(run_conversation(), arm='r1')
    second = copy.deepcopy(first)
    second['arm'] = 'r2'
    response = second['stages'][1]['attempts'][0]['response']
    if field == 'request': response['request']['messages'][0]['content'] += ' '
    if field == 'message': response['completion']['choices'][0]['message']['content'] += '\n'
    if field in ('prompt_tokens', 'completion_tokens'): response['completion']['usage'][field] += 1
    assert client.fixed_work_mismatches([first, second]) == [
        dict(arm='r2', ordinal=1, stage=2, field=field)]


def test_cache_usage_can_change_while_work_stays_identical():
    first = dict(run_conversation(), arm='r1')
    second = copy.deepcopy(first)
    second['arm'] = 'r2'
    for stage in second['stages']:
        stage['attempts'][0]['response']['completion']['usage']['prompt_tokens_details']['cached_tokens'] += 640
    assert client.fixed_work_mismatches([first, second]) == []
