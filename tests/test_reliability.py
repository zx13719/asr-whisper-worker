import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

import app
from batch_client import Client, JobError, atomic_json, chunks, validate_asr


def good():
    return {'text': 'hello', 'duration': 1, 'words': [{'word': 'hello', 'start': 0, 'end': 0.5}]}


@pytest.fixture
def fake_model(monkeypatch):
    counters = {'loads': 0, 'active': 0, 'peak': 0}
    def transcribe(*a, **kw):
        counters['active'] += 1
        counters['peak'] = max(counters['peak'], counters['active'])
        time.sleep(.02)
        counters['active'] -= 1
        return {'language': 'en', 'segments': [{'text': 'hello'}]}
    def load(*a, **kw):
        counters['loads'] += 1
        time.sleep(.02)
        return SimpleNamespace(transcribe=transcribe)
    wx = SimpleNamespace(load_model=load, load_audio=lambda _: np.zeros(16000),
                         load_align_model=lambda **kw: ('model', {}),
                         align=lambda *a, **kw: {'segments': [{'text': 'hello', 'words': good()['words']}]})
    monkeypatch.setitem(sys.modules, 'whisperx', wx)
    monkeypatch.setattr(app, '_whisper', None)
    monkeypatch.setattr(app, '_align', {})
    monkeypatch.setattr(app, 'fetch_audio', lambda *a: 'fake.wav')
    return counters, wx


def test_concurrent_requests_load_once_and_serialize_vad(fake_model):
    counters, _ = fake_model
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda i: app.transcribe_one({'id': str(i), 'language': 'en'}), range(16)))
    assert counters == {'loads': 1, 'active': 0, 'peak': 1}
    assert [r['id'] for r in results] == list(map(str, range(16)))
    assert all(validate_asr(r) for r in results)


def test_alignment_failure_is_not_success(fake_model):
    _, wx = fake_model
    def fail(*a, **kw):
        raise RuntimeError('alignment failed')
    wx.align = fail
    result = app.rp_handler({'input': {'items': [{'id': 'a', 'language': 'en'}]}})
    assert result['results'] == [{'id': 'a', 'error': 'alignment failed'}]


def test_health_not_ready_until_warmup(monkeypatch):
    app._ready.clear()
    monkeypatch.setattr(app, '_startup_error', None)
    assert app.ping().status_code == 204
    monkeypatch.setattr(app, '_startup_error', 'RuntimeError')
    assert app.ping().status_code == 503


def test_chunking_never_skips_or_duplicates():
    for n in [1, 4, 8, 40, 48, 70]:
        assert sum(chunks(list(range(n)), 4), []) == list(range(n))


@pytest.mark.parametrize('result', [{'error': '_lock'}, {}, {'text': 'a', 'duration': 1, 'words': []},
    {'text': 'a', 'duration': 1, 'words': [{'word': 'a', 'start': 0, 'end': 9}]}])
def test_reject_false_success(result):
    with pytest.raises(JobError):
        validate_asr(result)


def test_resume_polls_saved_job_without_resubmit(tmp_path):
    calls = []
    done = False
    def request(method, url, body=None):
        calls.append(method)
        if method == 'POST':
            return {'id': 'remote-job'}
        return {'status': 'COMPLETED', 'output': good()} if done else {'status': 'IN_PROGRESS'}
    client = Client(tmp_path, 'unused', request=request, timeout=0)
    with pytest.raises(JobError, match='saved for resume'):
        client.run('one', 'endpoint', 'queue', {'id': 'one'}, validate_asr)
    assert json.loads(next(tmp_path.glob('*.json')).read_text())['job_id'] == 'remote-job'
    done = True
    assert client.run('one', 'endpoint', 'queue', {'id': 'one'}, validate_asr)['text'] == 'hello'
    assert calls == ['POST', 'GET', 'GET']
    client.run('one', 'endpoint', 'queue', {'id': 'one'}, validate_asr)
    assert len(calls) == 3


def test_ambiguous_submission_cannot_double_charge_on_resume(tmp_path):
    calls = []
    def request(*a):
        calls.append(a)
        raise TimeoutError('response lost')
    client = Client(tmp_path, 'unused', request=request)
    with pytest.raises(TimeoutError):
        client.run('one', 'ep', 'queue', {}, validate_asr)
    with pytest.raises(JobError, match='outcome unknown'):
        client.run('one', 'ep', 'queue', {}, validate_asr)
    assert len(calls) == 1


def test_item_error_rejected_even_if_job_completed(tmp_path):
    def request(method, *a):
        return {'id': 'job'} if method == 'POST' else {'status': 'COMPLETED', 'output': {'error': '_lock'}}
    client = Client(tmp_path, 'unused', request=request)
    with pytest.raises(JobError):
        client.run('one', 'ep', 'queue', {}, validate_asr)
    assert json.loads(next(tmp_path.glob('*.json')).read_text())['status'] == 'INVALID_RESULT'


def test_http_received_response_is_not_resubmitted(tmp_path):
    client = Client(tmp_path, 'unused', request=lambda *a: good())
    client.run('one', 'ep', 'http', {}, validate_asr)
    path = next(tmp_path.glob('*.json'))
    state = json.loads(path.read_text()); state['status'] = 'RECEIVED'; atomic_json(path, state)
    client.request = lambda *a: pytest.fail('unexpected second POST')
    assert client.run('one', 'ep', 'http', {}, validate_asr)['text'] == 'hello'


def test_duplicate_concurrent_client_calls_submit_once(tmp_path):
    calls = []
    def request(method, *a):
        calls.append(method)
        time.sleep(.01)
        return {'id': 'job'} if method == 'POST' else {'status': 'COMPLETED', 'output': good()}
    client = Client(tmp_path, 'unused', request=request)
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda _: client.run('one', 'ep', 'queue', {}, validate_asr), range(6)))
    assert len(results) == 6
    assert calls == ['POST', 'GET']


def test_explicit_terminal_retry_preserves_previous_job(tmp_path):
    calls = []
    def request(method, *a):
        calls.append(method)
        return {'id': 'job'} if method == 'POST' else {'status': 'FAILED', 'error': 'OOM'}
    client = Client(tmp_path, 'unused', request=request, retry_failed=True)
    for _ in range(3):
        with pytest.raises(JobError, match='remote job FAILED'):
            client.run('one', 'ep', 'queue', {}, validate_asr)
    with pytest.raises(JobError, match='retry limit'):
        client.run('one', 'ep', 'queue', {}, validate_asr)
    assert calls.count('POST') == 3
    state = json.loads(next(tmp_path.glob('*.json')).read_text())
    assert len(state['attempts']) == 2
    assert state['attempts'][0]['job_id'] == 'job'
