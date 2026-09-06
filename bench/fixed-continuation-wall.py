"""Ordinary greedy copy-and-continue requests with independently checked answers.

This is a mechanism benchmark, not an agent or a quality evaluation. The server
controller runs the same three conversations in each arm. Cross-arm analysis
must additionally verify identical request bodies, answer bytes and token counts.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import signal
import sys
import time
import urllib.request


MAX_TOKENS = 768
TIMEOUT = 300
SESSIONS = 3


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def fixture(ordinal):
    """Distinct prefixes between conversations, identical inputs across arms."""
    if ordinal not in (1, 2, 3):
        raise ValueError('the protocol requires one warmup and two measured conversations')
    nonce = hashlib.sha256(f'fixed-continuation/{ordinal}'.encode()).hexdigest()[:24]
    background = '\n'.join(
        f'Background row {i:03}: archived value {(i * 7919 + ordinal * 101) % 99999:05}.'
        for i in range(96)
    )
    cases = []
    for stage in (1, 2, 3):
        expected = {f'r{i:02}': (24531 + 7919 * i + 104729 * ordinal + 65537 * stage) % 99999
                    for i in range(32)}
        records = '\n'.join(f'{key} = {value}' for key, value in expected.items())
        instruction = (
            'Copy only the 32 current records below into a complete JSON object. '
            'Preserve their key order and integer values. Include every key exactly once. '
            'Use two spaces of indentation, one record per line, no markdown, no explanation, '
            'and no trailing newline. Earlier records and background rows are irrelevant.\n'
            'Current records:\n' + records
        )
        if stage == 1:
            instruction = f'Conversation {nonce}.\nBackground archive:\n{background}\n\n' + instruction
        cases.append(dict(stage=stage, prompt=instruction, expected=expected))
    return cases


def checked_answer(completion, expected):
    """Reject duplicate keys, booleans posing as integers, truncation and extra text."""
    choices = completion['choices']
    if len(choices) != 1 or choices[0]['finish_reason'] != 'stop':
        raise ValueError('response did not complete normally')
    message = choices[0]['message']
    if message.get('tool_calls') or message.get('reasoning_content'):
        raise ValueError('unexpected tool call or reasoning output')
    text = message['content']
    pairs = json.loads(text, object_pairs_hook=list)
    if (not isinstance(pairs, list) or len(pairs) != len(expected)
            or any(not isinstance(pair, tuple) or len(pair) != 2 for pair in pairs)
            or [key for key, _ in pairs] != list(expected)
            or any(type(value) is not int or value != expected[key] for key, value in pairs)):
        raise ValueError('answer does not exactly copy the ordered integer records')
    usage = completion['usage']
    for field in ('prompt_tokens', 'completion_tokens'):
        if type(usage.get(field)) is not int or usage[field] <= 0:
            raise ValueError('missing or invalid token usage')
    # FreeToken's _usage omits the entire details object for a zero-token hit.
    # The controller/summary must separately verify --enable-cache-report.
    details = usage.get('prompt_tokens_details', {'cached_tokens': 0})
    cached = details.get('cached_tokens') if isinstance(details, dict) else None
    if type(cached) is not int or not 0 <= cached <= usage['prompt_tokens']:
        raise ValueError('missing or invalid cached-token accounting')
    return text, dict(input=usage['prompt_tokens'] - cached,
                      output=usage['completion_tokens'], cacheRead=cached, cacheWrite=0)


def request(base_url, payload):
    req = urllib.request.Request(base_url.rstrip('/') + '/chat/completions',
                                 data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as response:
        return json.load(response)


def conversation(ordinal, *, base_url, model, send=request):
    cases = fixture(ordinal)
    messages, stages, usage_rows = [], [], []
    model_calls = 0
    error = None
    started = time.perf_counter()
    try:
        for case in cases:
            messages.append(dict(role='user', content=case['prompt']))
            payload = dict(model=model, messages=[dict(m) for m in messages],
                           max_tokens=MAX_TOKENS, temperature=0, stream=False,
                           chat_template_kwargs=dict(enable_thinking=False))
            response = dict(request=payload)
            attempt = dict(response=response, verification=dict(passed=False, exit_code=1))
            stage = dict(stage=case['stage'], passed=False, attempts=[attempt])
            stages.append(stage)
            model_calls += 1
            call_start = time.perf_counter()
            try:
                completion = send(base_url, payload)
                response['completion'] = completion
            finally:
                response['wall_s'] = time.perf_counter() - call_start
            text, usage = checked_answer(completion, case['expected'])
            if case['stage'] == 1 and completion['usage']['prompt_tokens'] < 1024:
                raise ValueError('initial request did not exercise long prefill')
            if usage['output'] > MAX_TOKENS:
                raise ValueError('completion exceeded its fixed token budget')
            usage_rows.append(usage)
            attempt['verification'] = dict(passed=True, exit_code=0)
            stage['passed'] = True
            messages.append(dict(role='assistant', content=text))
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    wall = time.perf_counter() - started
    passed = error is None and len(stages) == 3 and all(s['passed'] for s in stages)
    return dict(ordinal=ordinal, passed=passed, stages=stages, error=error,
                task_wall_s=wall, verified_task_wall_s=wall if passed else None,
                startup_s=0.0, model_calls=model_calls, repair_prompts=0, trace=False,
                event_metrics=dict(tool_calls=0, tool_errors=0, tool_wall_s=0.0,
                                   model_usage=usage_rows),
                fixture_sha256=digest(cases), broad_quality_equivalence=False)


def fixed_work_mismatches(rows):
    """Compare raw requests and complete results, never canonicalize model answers."""
    reference = {}
    mismatches = []
    for row in rows:
        ordinal, arm = row['ordinal'], row['arm']
        if not row['passed'] or len(row['stages']) != 3:
            mismatches.append(dict(arm=arm, ordinal=ordinal, field='incomplete conversation'))
            continue
        verified_usage = []
        for stage in row['stages']:
            response = stage['attempts'][0]['response']
            request_body, completion = response['request'], response['completion']
            case = fixture(ordinal)[stage['stage'] - 1]
            _text, usage = checked_answer(completion, case['expected'])
            verified_usage.append(usage)
            signature = dict(request=request_body,
                             message=completion['choices'][0]['message'],
                             finish_reason=completion['choices'][0]['finish_reason'],
                             prompt_tokens=completion['usage']['prompt_tokens'],
                             completion_tokens=completion['usage']['completion_tokens'])
            key = ordinal, stage['stage']
            if key not in reference:
                reference[key] = signature
            else:
                for field in signature:
                    if signature[field] != reference[key][field]:
                        mismatches.append(dict(arm=arm, ordinal=ordinal,
                                               stage=stage['stage'], field=field))
        totals = {key: sum(u[key] for u in verified_usage)
                  for key in ('input', 'output', 'cacheRead', 'cacheWrite')}
        reported = row.get('tokens')
        if reported is not None and reported != totals:
            raise ValueError(f'{arm}: reported usage differs from full responses')
    return mismatches


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--model', default='qwen3.6-27b')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--server-metadata', required=True, type=Path)
    parser.add_argument('--label', required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    def interrupted(signum, _frame):
        raise RuntimeError(f'client interrupted by signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    source = Path(__file__)
    metadata = dict(command='fixed-continuation-wall', client_host=platform.node(),
                    python=sys.version, task='three-turn ordered JSON copying',
                    model_config=dict(model=args.model, temperature=0, enable_thinking=False),
                    requested_sessions=SESSIONS,
                    budgets=dict(max_tokens=MAX_TOKENS, request_timeout_s=TIMEOUT),
                    sources={source.name: hashlib.sha256(source.read_bytes()).hexdigest()},
                    fixture_sha256=digest([fixture(i) for i in (1, 2, 3)]),
                    trace=False, label=args.label,
                    server_metadata=json.loads(args.server_metadata.read_text()))
    save(args.output_dir / 'metadata.json', metadata)
    rows = []
    cancelled = False
    try:
        for ordinal in (1, 2, 3):
            row = conversation(ordinal, base_url=args.base_url, model=args.model)
            directory = args.output_dir / f'session-{ordinal}'
            directory.mkdir()
            save(directory / 'result.json', row)
            rows.append(row)
            print(json.dumps({k: row[k] for k in ('ordinal', 'passed', 'task_wall_s', 'error')}), flush=True)
            if row['error'] and 'interrupted by signal' in row['error']:
                cancelled = True
                break
    except Exception:
        cancelled = True
        raise
    finally:
        save(args.output_dir / 'summary.json', dict(
            completed_schedule=len(rows) == SESSIONS, cancelled=cancelled, sessions=len(rows),
            all_tasks_passed=len(rows) == SESSIONS and all(r['passed'] for r in rows),
            broad_quality_equivalence=False))
    return 0 if len(rows) == SESSIONS and all(r['passed'] for r in rows) else 1


if __name__ == '__main__':
    raise SystemExit(main())
