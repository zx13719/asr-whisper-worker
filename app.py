"""RunPod Load-Balancing ASR worker.

large-v3 (faster-whisper via WhisperX) + wav2vec2 forced alignment (word/char timestamps).
Runs a FastAPI server on $PORT (default 80) with a /ping health check and real
concurrent I/O and serialized access to the shared GPU model.

Endpoints
  GET  /ping                         -> health (200)
  GET  /info                         -> loaded model info
  POST /transcribe                   -> {"url"|"audio_base64", language?, align?, word_timestamps?}
  POST /transcribe_batch             -> {"items":[{"url":..}|{"audio_base64":..}], language?}

Env
  WHISPER_MODEL   default "large-v3"
  COMPUTE_TYPE    default "float16"
  MAX_CONCURRENCY default 4 (I/O only; GPU/model access is serialized)
  DEFAULT_LANGUAGE (optional, e.g. "zh")
  R2_*            optional, to accept {"key": "..."} inputs from R2
"""
import os
import time
import base64
import asyncio
import math
import hashlib
from contextlib import asynccontextmanager
import tempfile
import threading
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

MODEL_NAME = os.environ.get("WHISPER_MODEL", "large-v3")
COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "float16")
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "4"))
DEFAULT_LANGUAGE = os.environ.get("DEFAULT_LANGUAGE") or None
DEVICE = "cuda"

_ready = threading.Event()
_startup_error = None


@asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(asyncio.to_thread(warmup))
    try:
        yield
    finally:
        await task


app = FastAPI(title="ASR large-v3 + aligner", lifespan=lifespan)

_executor = ThreadPoolExecutor(max_workers=max(2, MAX_CONCURRENCY))
# WhisperX/Silero keep mutable state. I/O can overlap, model calls cannot.
_model_lock = threading.RLock()

_whisper = None
_align = {}
_align_lock = threading.Lock()


def log(*a):
    print("[asr]", *a, flush=True)


def get_whisper():
    global _whisper
    with _model_lock:
        if _whisper is None:
            import whisperx
            t0 = time.time()
            log("loading whisper model", MODEL_NAME, COMPUTE_TYPE)
            _whisper = whisperx.load_model(MODEL_NAME, DEVICE, compute_type=COMPUTE_TYPE, vad_method="silero")
            log("whisper loaded in %.1fs" % (time.time() - t0))
        return _whisper


def get_align(language):
    if not language:
        return None
    with _model_lock, _align_lock:
        if language not in _align:
            import whisperx
            t0 = time.time()
            log("loading align model for", language)
            _align[language] = whisperx.load_align_model(language_code=language, device=DEVICE)
            log("align loaded for %s in %.1fs" % (language, time.time() - t0))
        return _align[language]


def _download(url, dst):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; RunPod-ASR/1.0)"}
    last = None
    for _ in range(4):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=300) as r, open(dst, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            if os.path.getsize(dst) > 0:
                return
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2)
    raise last if last else RuntimeError("download failed")


def _from_r2(key, dst):
    import boto3
    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )
    s3.download_file(os.environ["R2_BUCKET"], key, dst)


def fetch_audio(item, workdir):
    src = os.path.join(workdir, "in.bin")
    if item.get("url"):
        _download(item["url"], src)
    elif item.get("key"):
        _from_r2(item["key"], src)
    elif item.get("audio_base64"):
        with open(src, "wb") as f:
            f.write(base64.b64decode(item["audio_base64"]))
    else:
        raise ValueError("provide one of: url, key, audio_base64")
    wav = os.path.join(workdir, "in.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", src, "-ac", "1", "-ar", "16000", "-f", "wav", wav],
        check=True,
    )
    return wav


def transcribe_one(item):
    """Blocking: download/decode -> ASR -> forced alignment. Runs in a thread."""
    import whisperx

    language = item.get("language") or DEFAULT_LANGUAGE
    align = item.get("align", True)
    transcript = item.get("transcript")
    if transcript is not None and (not isinstance(transcript, str) or not transcript.strip()):
        raise ValueError("transcript must be a nonempty string")
    task = item.get("task", "transcribe")
    if task not in {"transcribe", "translate"}:
        raise ValueError("unsupported task")
    if transcript is not None and (not align or not language or task != "transcribe"):
        raise ValueError("forced alignment requires align=true, language and task=transcribe")
    if task == "translate" and align:
        raise ValueError("translated text cannot be force-aligned to source speech")

    with tempfile.TemporaryDirectory() as td:
        wav = fetch_audio(item, td)
        audio = whisperx.load_audio(wav)
        # NB: torchaudio/whisperx load_audio may emit float32 numpy
        duration = float(len(audio) / 16000)

        t0 = time.time()
        if transcript is not None:
            # Forced alignment must preserve the caller's approved script, not ASR text.
            result = {"language": language, "segments": [{"start": 0.0, "end": duration, "text": transcript}]}
        else:
            with _model_lock:
                model = get_whisper()
                result = model.transcribe(
                    audio, batch_size=int(item.get("batch_size", 16)),
                    language=language, task=task,
                )
        asr_time = time.time() - t0
        segments = result.get("segments", [])
        lang = result.get("language") or language

        align_time = 0.0
        if align:
            if not lang:
                raise RuntimeError("alignment requested but language is unknown")
            t1 = time.time()
            with _model_lock:
                am = get_align(lang)
                aligned = whisperx.align(
                    segments, am[0], am[1], audio, DEVICE,
                    return_char_alignments=bool(item.get("char_alignments", False)),
                )
            segments = aligned["segments"]
            align_time = time.time() - t1

        words = []
        for s in segments:
            for w in s.get("words", []) or []:
                words.append({
                    "word": w.get("word"),
                    "start": w.get("start"),
                    "end": w.get("end"),
                    "score": w.get("score"),
                })

        text = "".join((s.get("text") or "") for s in segments).strip()
        if align and text and not any(
            isinstance(w.get("start"), (int, float))
            and isinstance(w.get("end"), (int, float))
            and math.isfinite(w["start"]) and math.isfinite(w["end"])
            and 0 <= w["start"] < w["end"] <= duration + 0.25
            for w in words
        ):
            raise RuntimeError("alignment produced no valid word timestamps")
        return {
            "id": item.get("id"),
            "mode": "forced_alignment" if transcript is not None else "transcription",
            "transcript_sha256": hashlib.sha256(transcript.encode()).hexdigest() if transcript is not None else None,
            "revision": os.environ.get("IMAGE_REVISION", "unknown"),
            "language": lang,
            "duration": round(duration, 3),
            "text": text,
            "segments": segments,
            "words": words,
            "timing": {
                "asr": round(asr_time, 2),
                "align": round(align_time, 2),
            },
        }


def warmup():
    global _startup_error
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        with _model_lock:
            model = get_whisper()
            model.transcribe(np.zeros(16000, dtype=np.float32), batch_size=1,
                             language=DEFAULT_LANGUAGE or "en", task="transcribe")
            for language in {"en", DEFAULT_LANGUAGE} - {None}:
                get_align(language)
        _ready.set()
    except Exception as exc:
        _startup_error = type(exc).__name__
        log("warmup failed", repr(exc))
        raise


@app.get("/ping")
def ping():
    if _startup_error:
        return JSONResponse({"status": "failed", "error": _startup_error}, status_code=503)
    if not _ready.is_set():
        from fastapi.responses import Response
        return Response(status_code=204)
    return {"status": "ok"}


@app.get("/info")
def info():
    import torch
    return {
        "revision": os.environ.get("IMAGE_REVISION", "unknown"),
        "ready": _ready.is_set(),
        "model": MODEL_NAME,
        "compute_type": COMPUTE_TYPE,
        "max_concurrency": MAX_CONCURRENCY,
        "gpu_concurrency": 1,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
        "align_loaded": list(_align.keys()),
    }


@app.post("/transcribe")
async def transcribe(payload: dict):
    if not _ready.is_set():
        raise HTTPException(status_code=503, detail="model is not ready")
    loop = asyncio.get_event_loop()
    try:
        res = await loop.run_in_executor(_executor, transcribe_one, payload)
        return JSONResponse(res)
    except Exception as e:  # noqa: BLE001
        import traceback
        raise HTTPException(status_code=500, detail={"error": str(e), "tb": traceback.format_exc()[-1000:]})


@app.post("/transcribe_batch")
async def transcribe_batch(payload: dict):
    if not _ready.is_set():
        raise HTTPException(status_code=503, detail="model is not ready")
    items = payload.get("items") or []
    if not items:
        raise HTTPException(status_code=400, detail="items required")
    base = {k: v for k, v in payload.items() if k != "items"}
    merged = [{**base, **it} for it in items]
    loop = asyncio.get_event_loop()
    futs = [loop.run_in_executor(_executor, transcribe_one, m) for m in merged]
    results = await asyncio.gather(*futs, return_exceptions=True)
    out = []
    for item, r in zip(merged, results):
        if isinstance(r, Exception):
            out.append({"id": item.get("id"), "error": str(r)})
        else:
            out.append(r)
    return JSONResponse({"count": len(out), "results": out})


def rp_handler(job):
    """Queue-based endpoint entrypoint."""
    inp = job.get("input") or {}
    if inp.get("items"):
        items = inp["items"]
        base = {k: v for k, v in inp.items() if k != "items"}
        out = []
        for it in items:
            try:
                out.append(transcribe_one({**base, **it}))
            except Exception as e:  # noqa: BLE001
                out.append({"id": it.get("id"), "error": str(e)})
        return {"count": len(out), "results": out}
    return transcribe_one(inp)


if __name__ == "__main__":
    mode = os.environ.get("SERVE_MODE", "http")
    if mode not in {"queue", "http"}:
        raise ValueError("SERVE_MODE must be queue or http")
    if mode == "queue":
        import runpod
        warmup()
        runpod.serverless.start({"handler": rp_handler})
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "80")))
