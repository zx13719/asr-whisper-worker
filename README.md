# RunPod Load-Balancing ASR worker — large-v3 + aligner

`faster-whisper large-v3` (via WhisperX) + wav2vec2 forced alignment → word/char level
timestamps. FastAPI server on `$PORT` (80), concurrency via an async + thread-pool
bound on one shared GPU model.

## Endpoints
- `GET  /ping` — health (200)
- `GET  /info` — model/GPU info
- `POST /transcribe` — `{"url"|"key"|"audio_base64", "language"?, "align"?, "batch_size"?}`
- `POST /transcribe_batch` — `{"items":[{...}], "language"?}`

## Env
`WHISPER_MODEL=large-v3`, `COMPUTE_TYPE=float16`, `MAX_CONCURRENCY=4`,
`DEFAULT_LANGUAGE=zh`, optional `R2_*` for `{"key": ...}` inputs.

## Response
```json
{"language":"zh","duration":9.4,"text":"...","segments":[...],
 "words":[{"word":"你","start":0.12,"end":0.30,"score":0.91}, ...],
 "timing":{"asr":3.1,"align":0.6}}
```

## Reliability and rollout (v3)

The thread pool overlaps downloads, while one reentrant lock owns model initialization,
transcription/VAD and alignment. Do not remove this lock to increase GPU concurrency;
scale independent workers instead. `/ping` reports 204 while warming up, 503 after
startup failure and 200 only after CUDA/model warmup. Requested alignment failures
are returned as errors, never as an apparently successful unaligned transcript.

Match the endpoint type explicitly: queue endpoint → `SERVE_MODE=queue`; load-balancing
endpoint → `SERVE_MODE=http`. Keep existing mode until the endpoint type is verified.
For offline production batches prefer queue mode and one video per job.

`run_batch.py` is the maintained replacement for the ad-hoc `runpod_batch.py` and
`asr_lb*.py` scripts. Install boto3, set MCN_WORK and the existing RUNPOD/R2 env vars:

```sh
python run_batch.py --stage asr --asr-mode queue --workers 4
python run_batch.py --stage both --asr-mode queue --workers 4
```

It writes a separate `runpod-ledger-v2/`, `asr-runpod-v2/` and stage summary, preserving
old artifacts. R2 input/output keys include the source SHA-256. It does not import old
file-existence success markers. The updated ASR worker must be deployed first (it
returns the per-item `id`). Failures cause a nonzero process exit.

Poll timeouts resume the saved remote job ID. Ambiguous submission outcomes stop
instead of risking a duplicate charge: reconcile the job in RunPod and write its ID
and `SUBMITTED` status into that ledger entry. Known terminal failures can be retried
with `--retry-failed` after fixing the cause; previous attempts are retained and the
limit is three attempts. Never delete an unknown-submission entry to force a retry.

Base CUDA image digest and direct runtime dependency versions are pinned from the
previous successful build. Transitive dependencies and apt packages are not fully
locked. CI tests and builds PRs without publishing; main builds publish `v3` and
`sha-<commit>`. Deploy the immutable image digest returned by the build.

Before switching production, copy `gpu_smoke.py` and `batch_client.py` into the
candidate GPU container and run `python gpu_smoke.py /path/to/speech.wav`. Then run
one ESR/ASR pair and validate media and timestamps before a full batch. CPU regression
tests do not establish real GPU compatibility.

## Transcript-locked alignment (v4)

Both queue input and `/transcribe` accept `transcript` with `language` and `align:true`.
When supplied, recognition is bypassed: WhisperX aligns that exact text against the
full supplied audio. The response includes `mode:forced_alignment`, its UTF-8
`transcript_sha256`, item `id`, and image revision. Callers must verify these fields
before using the result for approved captions. An empty script or missing language
is rejected. Ordinary recognition remains supported without `transcript`.
