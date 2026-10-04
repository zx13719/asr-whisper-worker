"""Resume-safe ESR/ASR batch runner. One video per remote job, explicit ASR mode."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path

from batch_client import Client, JobError, atomic_json, validate_asr, validate_esr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['asr', 'esr', 'both'], default='both')
    parser.add_argument('--asr-mode', choices=['queue', 'http'], default=os.getenv('RUNPOD_ASR_MODE'))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--retry-failed', action='store_true', help='Retry known terminal failures, at most 3 attempts; ambiguous submissions remain blocked')
    args = parser.parse_args()
    if args.stage != 'esr' and not args.asr_mode:
        parser.error('set --asr-mode or RUNPOD_ASR_MODE to match the deployed endpoint; no guessing')
    if args.workers < 1:
        parser.error('--workers must be positive')
    import boto3
    base = Path(os.environ['MCN_WORK'])
    public = os.environ['R2_PUBLIC_BASE_URL'].rstrip('/')
    bucket = os.environ['R2_BUCKET']
    prefix = os.getenv('MCN_RUNPOD_PREFIX', f'assets/production/{base.name}/runpod-v2').strip('/')
    s3 = boto3.client('s3', endpoint_url=os.environ['R2_ENDPOINT_URL'],
                      aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'],
                      aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'], region_name='auto')
    client = Client(base / 'runpod-ledger-v2', os.environ['RUNPOD_API_KEY'], retry_failed=args.retry_failed)
    paths = sorted((base / 'final').glob('*-motion-master.mp4'))
    if not paths:
        parser.error('no motion masters found')

    def process(path):
        key = path.name.removesuffix('-motion-master.mp4')
        hasher = hashlib.sha256()
        with path.open('rb') as source:
            for block in iter(lambda: source.read(1 << 20), b''):
                hasher.update(block)
        sha = hasher.hexdigest()
        input_key = f'{prefix}/inputs/{key}-{sha}.mp4'
        # Immutable key makes retrying an upload harmless and ties jobs to source bytes.
        try:
            s3.head_object(Bucket=bucket, Key=input_key)
        except s3.exceptions.ClientError as exc:
            if str(exc.response.get('Error', {}).get('Code')) not in {'404', 'NoSuchKey', 'NotFound'}:
                raise
            s3.upload_file(str(path), bucket, input_key, ExtraArgs={'ContentType': 'video/mp4'})
        url = f'{public}/{input_key}'
        result = {'id': key, 'source_sha256': sha}
        if args.stage in {'esr', 'both'}:
            output_key = f'{prefix}/esr/{key}-{sha}_720p.mp4'
            def check_esr(envelope):
                if not isinstance(envelope, dict) or envelope.get('ok') != 1 or len(envelope.get('results', [])) != 1:
                    raise JobError('ESR job did not return one successful item')
                item = validate_esr(envelope['results'][0])
                if item.get('key') != output_key or item.get('url') != f'{public}/{output_key}':
                    raise JobError('ESR output identity mismatch')
                head = s3.head_object(Bucket=bucket, Key=output_key)
                if head.get('ContentLength', 0) <= 0:
                    raise JobError('ESR uploaded object is empty')
                return item
            result['esr'] = client.run('esr:' + key, os.environ['RUNPOD_ESR_ENDPOINT_ID'], 'queue', {
                'videos': [{'url': url, 'key': output_key}], 'model': 'RealESRGAN_x2plus',
                'outscale': 1.5, 'frame_batch': int(os.getenv('ESR_FRAME_BATCH', '1')),
                'concurrency': 1, 'public_base': public}, check_esr)
        if args.stage in {'asr', 'both'}:
            def check_asr(item):
                validate_asr(item)
                if item.get('id') != key:
                    raise JobError('ASR output identity mismatch; deploy the updated worker')
                return item
            result['asr'] = client.run('asr:' + key, os.environ['RUNPOD_ASR_ENDPOINT_ID'], args.asr_mode,
                                      {'id': key, 'url': url, 'language': 'en', 'align': True}, check_asr)
            atomic_json(base / 'asr-runpod-v2' / f'{key}.json', result['asr'])
        return result

    completed, failures = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, path): path.stem for path in paths}
        for future in as_completed(futures):
            key = futures[future]
            try:
                completed.append(future.result())
                print('VALIDATED', key, flush=True)
            except Exception as exc:
                failures.append({'id': key, 'error': str(exc)})
                print('FAILED', key, str(exc)[:300], flush=True)
    atomic_json(base / f'runpod-{args.stage}-summary-v2.json', {
        'total': len(paths), 'completed': completed, 'failures': failures})
    print('VALIDATED', len(completed), '/', len(paths), flush=True)
    return bool(failures)


if __name__ == '__main__':
    raise SystemExit(main())
