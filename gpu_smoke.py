"""Run inside the candidate GPU image: python gpu_smoke.py /path/to/speech.wav."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import app
from batch_client import validate_asr

if __name__ == '__main__':
    source = Path(sys.argv[1]).resolve()
    if not source.is_file():
        raise SystemExit('Provide a real speech sample')
    app.warmup()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda i: app.transcribe_one({
            'id': str(i), 'url': source.as_uri(), 'language': 'en', 'align': True}), range(8)))
    for i, result in enumerate(results):
        validate_asr(result)
        assert result['id'] == str(i)
    assert len({result['text'] for result in results}) == 1, 'concurrent transcripts differ'
    print('PASS: 8 real GPU transcriptions, concurrent callers, aligned words, stable text')
