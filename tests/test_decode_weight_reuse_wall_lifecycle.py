"""Hermetic ownership and recovery checks; no GPU, service or network calls."""

import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    'reuse_lifecycle', Path(__file__).parents[1] / 'bench/decode-weight-reuse-wall-lifecycle.py')
lifecycle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lifecycle)


@pytest.fixture
def host(monkeypatch):
    states = {name: dict(ActiveState='inactive', ControlGroup='', MainPID='0', InvocationID='')
              for name in (*lifecycle.OTHER_JOBS, lifecycle.SERVER, lifecycle.ORIGINAL)}
    states[lifecycle.ORIGINAL].update(ActiveState='active', ControlGroup='/original',
                                      MainPID='123', InvocationID='original-instance')
    calls = []
    monkeypatch.setattr(lifecycle, 'state', lambda name: states[name].copy())
    monkeypatch.setattr(lifecycle, 'original_owns_gpu', lambda _: True)
    monkeypatch.setattr(lifecycle, 'wait_gpu_release', lambda: calls.append('wait_gpu'))
    monkeypatch.setattr(lifecycle, 'completion', lambda: calls.append('completion') or {'ok': True})

    def run(*args, **kwargs):
        calls.append(args)
        if args[:4] == ('sudo', '-n', 'systemctl', 'stop'):
            states[args[4]]['ActiveState'] = 'inactive'
        elif args[:4] == ('sudo', '-n', 'systemctl', 'start'):
            states[args[4]]['ActiveState'] = 'active'
        else:
            pytest.fail('unexpected external command')
    monkeypatch.setattr(lifecycle, 'run', run)
    return states, calls


@pytest.mark.parametrize('unit', lifecycle.OTHER_JOBS)
@pytest.mark.parametrize('active_state', ['active', 'activating', 'deactivating'])
def test_preflight_and_recovery_preserve_another_jobs_pause(host, unit, active_state):
    states, calls = host
    states[unit]['ActiveState'] = active_state
    states[lifecycle.ORIGINAL]['ActiveState'] = 'inactive'
    with pytest.raises(RuntimeError, match='another benchmark is live'):
        lifecycle.preflight()
    result = lifecycle.restore()
    assert not result['verified'] and result['skipped_other_jobs'] == [unit]
    assert calls == []
    assert states[lifecycle.ORIGINAL]['ActiveState'] == 'inactive'


def test_preflight_accepts_only_an_idle_benchmark_with_sole_original_gpu(host):
    states, calls = host
    result = lifecycle.preflight()
    assert result['exclusive'] and result['original_unit']['MainPID'] == '123'
    assert calls == []
    states[lifecycle.SERVER]['ActiveState'] = 'active'
    with pytest.raises(RuntimeError, match='server is already live'):
        lifecycle.preflight()


def test_foreign_gpu_prevents_preflight_and_probe_on_active_original(host, monkeypatch):
    _states, calls = host
    monkeypatch.setattr(lifecycle, 'original_owns_gpu', lambda _: False)
    with pytest.raises(RuntimeError, match='sole GPU process'):
        lifecycle.preflight()
    with pytest.raises(RuntimeError, match='sole GPU process'):
        lifecycle.restore()
    assert calls == []


def test_recovery_preserves_a_healthy_original_process(host):
    states, calls = host
    before = states[lifecycle.ORIGINAL].copy()
    result = lifecycle.restore()
    assert result['verified'] and result['original_unit'] == before
    assert result['completion'] == {'ok': True}
    assert calls == ['completion']


def test_recovery_stops_owned_server_and_waits_before_starting_original(host):
    states, calls = host
    states[lifecycle.SERVER]['ActiveState'] = 'active'
    states[lifecycle.ORIGINAL]['ActiveState'] = 'inactive'
    result = lifecycle.restore()
    assert result['verified']
    assert calls == [('sudo', '-n', 'systemctl', 'stop', lifecycle.SERVER),
                     'wait_gpu', ('sudo', '-n', 'systemctl', 'start', lifecycle.ORIGINAL),
                     'completion']


def test_gpu_release_failure_never_starts_original(host, monkeypatch):
    states, calls = host
    states[lifecycle.ORIGINAL]['ActiveState'] = 'inactive'
    def occupied():
        raise TimeoutError('GPU remains occupied')
    monkeypatch.setattr(lifecycle, 'wait_gpu_release', occupied)
    with pytest.raises(TimeoutError, match='GPU remains occupied'):
        lifecycle.restore()
    assert calls == []


def test_another_job_appearing_during_release_prevents_original_start(host, monkeypatch):
    states, calls = host
    states[lifecycle.ORIGINAL]['ActiveState'] = 'inactive'
    def race():
        states['astra-pi-agentic-lease']['ActiveState'] = 'activating'
    monkeypatch.setattr(lifecycle, 'wait_gpu_release', race)
    result = lifecycle.restore()
    assert not result['verified'] and result['skipped_other_jobs'] == ['astra-pi-agentic-lease']
    assert calls == []


def test_failed_completion_cannot_claim_restoration(host, monkeypatch):
    def failure():
        raise TimeoutError('completion timeout')
    monkeypatch.setattr(lifecycle, 'completion', failure)
    with pytest.raises(TimeoutError, match='completion timeout'):
        lifecycle.restore()
