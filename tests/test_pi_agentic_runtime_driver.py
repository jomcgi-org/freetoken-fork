"""Focused checks for service recovery, lease EOF and geometry qualification."""

import importlib.util
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location('agentic_gate', Path(__file__).parents[1] / 'bench/pi-agentic-runtime-driver.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def test_stopping_an_absent_or_inactive_service_is_idempotent(monkeypatch):
    monkeypatch.setattr(gate, 'state', lambda _: {'ActiveState': 'inactive'})
    monkeypatch.setattr(gate, 'run', lambda *_args, **_kwargs: pytest.fail('inactive unit must not be stopped'))
    gate.stop(gate.SERVER)


def test_existing_original_service_survives_recovery(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(gate, 'R', tmp_path)
    monkeypatch.setattr(gate, 'state', lambda _: {'ActiveState': 'active'})
    monkeypatch.setattr(gate, 'stop', lambda unit: calls.append(unit))
    monkeypatch.setattr(gate, 'wait_gpu_release', lambda: pytest.fail('original GPU process is expected'))
    monkeypatch.setattr(gate, 'run', lambda *_args, **_kwargs: pytest.fail('must not restart a healthy original'))
    monkeypatch.setattr(gate, 'ready', lambda port: {'port': port})
    monkeypatch.setattr(gate, 'completion', lambda port: {'verified_port': port})
    result = gate.remote_main(SimpleNamespace(run_id='test', action='restore'))
    assert calls == [gate.SERVER]
    assert result['verified'] and result['completion']['verified_port'] == 8090
    assert json.loads((tmp_path / 'results/test/restoration.json').read_text())['verified']


def test_recovery_waits_for_gpu_before_starting_original(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(gate, 'R', tmp_path)
    monkeypatch.setattr(gate, 'state', lambda _: {'ActiveState': 'inactive'})
    monkeypatch.setattr(gate, 'stop', lambda unit: calls.append(('stop', unit)))
    monkeypatch.setattr(gate, 'wait_gpu_release', lambda: calls.append(('gpu',)))
    monkeypatch.setattr(gate, 'run', lambda *args, **kwargs: calls.append(args))
    monkeypatch.setattr(gate, 'ready', lambda port: calls.append(('ready', port)))
    monkeypatch.setattr(gate, 'completion', lambda port: calls.append(('completion', port)))
    result = gate.remote_main(SimpleNamespace(run_id='test', action='restore'))
    assert result['verified']
    assert calls[:3] == [('stop', gate.SERVER), ('gpu',), ('sudo', '-n', 'systemctl', 'start', 'freetoken-serve')]
    assert calls[-2:] == [('ready', 8090), ('completion', 8090)]


def test_failed_restoration_is_retained(monkeypatch, tmp_path):
    monkeypatch.setattr(gate, 'R', tmp_path)
    monkeypatch.setattr(gate, 'stop', lambda _: None)
    monkeypatch.setattr(gate, 'state', lambda _: {'ActiveState': 'active'})

    def fail(_):
        raise TimeoutError('original never became ready')

    monkeypatch.setattr(gate, 'ready', fail)
    with pytest.raises(TimeoutError):
        gate.remote_main(SimpleNamespace(run_id='test', action='restore'))
    result = json.loads((tmp_path / 'results/test/restoration.json').read_text())
    assert not result['verified'] and 'never became ready' in result['error']


@pytest.mark.parametrize('payload', [b'', b'ping\ndone\n', b'ping\nping\n'])
def test_lease_eof_and_batched_heartbeats_close_without_waiting(payload):
    read_fd, write_fd = os.pipe()
    with os.fdopen(read_fd, 'rb') as stream:
        os.write(write_fd, payload)
        os.close(write_fd)
        assert gate.hold_connection(stream, idle_s=0.05) == {'lease': 'closed'}


def test_missing_heartbeats_expire_lease():
    read_fd, write_fd = os.pipe()
    try:
        with os.fdopen(read_fd, 'rb') as stream:
            with pytest.raises(TimeoutError, match='renewing'):
                gate.hold_connection(stream, idle_s=0.01)
    finally:
        os.close(write_fd)


def test_geometry_ignores_log_timestamps_but_rejects_missing_fields():
    text = '\n'.join([
        'INFO MoE bank split residency: 26.44 GiB pinned + 37.02 GiB file-backed',
        'INFO MoE HOT expert residency: 2296 protected GPU rows',
        'INFO MoE activation dtype: bf16',
        'INFO --moe-cache-auto resolved moe_cache_size=3920 num_pages=1024',
        'INFO Allocating 65536 tokens for KV cache, K + V = 0.80 GiB',
        'INFO CPU MoE executor ready: threads=14 isa=avx512vnni(nvfp4-w4a8) max_tokens=1',
        'INFO Start capturing CUDA graphs with sizes: [1]',
        'INFO KV cache dtype: kv_dtype=fp8_e4m3',
    ])
    assert gate.geometry(text) == gate.geometry(text.replace('INFO', '[different timestamp] INFO'))
    with pytest.raises(AssertionError):
        gate.geometry(text.replace('max_tokens=1', 'max_tokens=4'))
    with pytest.raises(AssertionError):
        gate.geometry(text.replace('KV cache dtype:', 'missing dtype:'))
