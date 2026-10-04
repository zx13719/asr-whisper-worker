"""Durable per-item RunPod client. No ambiguous POST is automatically resubmitted."""
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
import urllib.error
import urllib.request


class JobError(RuntimeError):
    pass


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, ensure_ascii=False, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def chunks(items, size):
    if size < 1:
        raise ValueError('chunk size must be positive')
    return [items[i:i + size] for i in range(0, len(items), size)]


def validate_asr(result):
    if not isinstance(result, dict) or result.get('error'):
        raise JobError('ASR returned an error')
    if not isinstance(result.get('text'), str) or not result['text'].strip():
        raise JobError('ASR has no speech text')
    duration = result.get('duration')
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
        raise JobError('ASR duration is invalid')
    timed = 0
    last_start = -1
    for word in result.get('words') or []:
        start, end = word.get('start'), word.get('end')
        # Alignment may omit timing for punctuation/numerals; never fabricate it.
        if start is None and end is None:
            continue
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (start, end)):
            raise JobError('ASR timestamp is invalid')
        if not (0 <= start <= end <= duration + 0.25) or start < last_start:
            raise JobError('ASR timestamps are outside the audio or unordered')
        last_start = start
        timed += bool(end > start and str(word.get('word', '')).strip())
    if not timed:
        raise JobError('ASR has no usable timed words')
    return result


def validate_esr(result):
    if not isinstance(result, dict) or result.get('error') or not result.get('url'):
        raise JobError('ESR returned no valid output')
    if (result.get('width'), result.get('height')) != (720, 1280):
        raise JobError('ESR output is not 720x1280')
    frames = result.get('frames')
    if not isinstance(frames, int) or frames <= 0 or result.get('verified_frames') != frames:
        raise JobError('ESR output has not passed frame-count verification')
    return result


class Client:
    def __init__(self, directory, api_key, request=None, sleep=time.sleep, timeout=7200, retry_failed=False):
        self.directory = Path(directory)
        self.api_key = api_key
        self.request = request or self._request
        self.sleep = sleep
        self.timeout = timeout
        self.retry_failed = retry_failed

    def _request(self, method, url, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, method=method, headers={
            'Authorization': 'Bearer ' + self.api_key, 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=360 if '.api.runpod.ai/' in url else 120) as r:
                return json.load(r)
        except urllib.error.HTTPError as exc:
            # Never include request URLs/payloads/credentials in diagnostics.
            detail = exc.read(2000).decode(errors='replace').replace(self.api_key, '[REDACTED]')
            raise JobError(f'RunPod HTTP {exc.code}: {detail}') from None

    def run(self, key, endpoint, mode, payload, validate):
        if mode not in {'queue', 'http'}:
            raise ValueError('mode must be explicitly queue or http')
        identity = {'key': key, 'endpoint': endpoint, 'mode': mode, 'input': payload}
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        path = self.directory / (digest + '.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        # Locks both threads and separate resumed processes; each item has its own file.
        with open(path.with_suffix('.lock'), 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = json.loads(path.read_text()) if path.exists() else {**identity, 'status': 'NEW'}
            def save(**fields):
                state.update(fields)
                state['updated_at'] = time.time()
                atomic_json(path, state)
            if state['status'] == 'SUCCEEDED':
                return validate(state['result'])
            if state['status'] in {'SUBMITTING', 'UNKNOWN_SUBMISSION'} and not state.get('job_id'):
                raise JobError(f'{key}: submission outcome unknown; reconcile ledger {path.name} before retry')
            if state['status'] in {'FAILED', 'INVALID_RESULT'}:
                if not self.retry_failed:
                    raise JobError(f'{key}: previous failure in {path.name}; inspect then use --retry-failed')
                history = state.get('attempts', []) + [{k: v for k, v in state.items() if k != 'attempts'}]
                if len(history) >= 3:
                    raise JobError(f'{key}: retry limit reached; inspect {path.name}')
                state = {**identity, 'status': 'NEW', 'attempts': history}
                save()
            if state['status'] != 'RECEIVED' and not state.get('job_id'):
                save(status='SUBMITTING')
                try:
                    if mode == 'http':
                        result = self.request('POST', f'https://{endpoint}.api.runpod.ai/transcribe', payload)
                        save(status='RECEIVED', result=result)
                    else:
                        response = self.request('POST', f'https://api.runpod.ai/v2/{endpoint}/run', {'input': payload})
                        jid = response.get('id')
                        if not jid:
                            raise JobError('queue response has no job ID; check endpoint type')
                        save(status='SUBMITTED', job_id=jid)
                except Exception as exc:
                    save(status='UNKNOWN_SUBMISSION', error=str(exc))
                    raise
            if mode == 'queue' and state['status'] != 'RECEIVED':
                deadline = time.monotonic() + self.timeout
                while True:
                    try:
                        response = self.request('GET', f'https://api.runpod.ai/v2/{endpoint}/status/{state["job_id"]}')
                    except Exception as exc:
                        save(last_poll_error=str(exc))
                    else:
                        status = response.get('status')
                        save(remote_status=status)
                        if status == 'COMPLETED':
                            result = response.get('output')
                            save(status='RECEIVED', result=result)
                            break
                        if status in {'FAILED', 'CANCELLED', 'TIMED_OUT'}:
                            save(status='FAILED', error=response.get('error', status))
                            raise JobError(f'{key}: remote job {status}')
                    if time.monotonic() >= deadline:
                        raise JobError(f'{key}: polling deadline reached; job ID saved for resume')
                    self.sleep(5)
            result = state['result']
            try:
                validated = validate(result)
            except Exception as exc:
                save(status='INVALID_RESULT', error=str(exc))
                raise
            save(status='SUCCEEDED')
            return validated
