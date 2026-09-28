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
