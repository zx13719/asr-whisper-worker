"""RunPod Load-Balancing ASR worker.

large-v3 (faster-whisper via WhisperX) + wav2vec2 forced alignment (word/char timestamps).
Runs a FastAPI server on $PORT (default 80) with a /ping health check and real
concurrency (async + bounded thread pool sharing one GPU model).

Endpoints
  GET  /ping                         -> health (200)
  GET  /info                         -> loaded model info
  POST /transcribe                   -> {"url"|"audio_base64", language?, align?, word_timestamps?}
  POST /transcribe_batch             -> {"items":[{"url":..}|{"audio_base64":..}], language?}

Env
  WHISPER_MODEL   default "large-v3"
  COMPUTE_TYPE    default "float16"
  MAX_CONCURRENCY default 4
  DEFAULT_LANGUAGE (optional, e.g. "zh")
  R2_*            optional, to accept {"key": "..."} inputs from R2
"""
import os
import io
import sys
import time
import base64
import asyncio
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

app = FastAPI(title="ASR large-v3 + aligner")

_executor = ThreadPoolExecutor(max_workers=max(2, MAX_CONCURRENCY))
_gpu_sem = threading.Semaphore(MAX_CONCURRENCY)

_whisper = None
_align = {}
_align_lock = threading.Lock()


def log(*a):
    print("[asr]", *a, flush=True)


def get_whisper():
    global _whisper
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
    with _align_lock:
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
    import soundfile as sf
    import whisperx

    language = item.get("language") or DEFAULT_LANGUAGE
    align = item.get("align", True)
    word_timestamps = item.get("word_timestamps", True)

    with tempfile.TemporaryDirectory() as td:
        wav = fetch_audio(item, td)
        audio = whisperx.load_audio(wav)
        # NB: torchaudio/whisperx load_audio may emit float32 numpy
        duration = float(len(audio) / 16000)

        model = get_whisper()
        t0 = time.time()
        with _gpu_sem:
            result = model.transcribe(
                audio,
                batch_size=int(item.get("batch_size", 16)),
                language=language,
                task="transcribe",
                word_timestamps=word_timestamps,
            )
        asr_time = time.time() - t0
        segments = result.get("segments", [])
        lang = result.get("language") or language

        align_time = 0.0
        if align and lang:
            try:
                am = get_align(lang)
                if am is not None:
                    t1 = time.time()
                    with _gpu_sem:
                        aligned = whisperx.align(
                            segments, am[0], am[1], audio, DEVICE,
                            return_char_alignments=bool(item.get("char_alignments", False)),
                        )
                    segments = aligned.get("segments", segments)
                    align_time = time.time() - t1
            except Exception as e:  # noqa: BLE001
                log("align failed:", repr(e))

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
        return {
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


@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.get("/info")
def info():
    import torch
    return {
        "model": MODEL_NAME,
        "compute_type": COMPUTE_TYPE,
        "max_concurrency": MAX_CONCURRENCY,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
        "align_loaded": list(_align.keys()),
    }


@app.post("/transcribe")
async def transcribe(payload: dict):
    loop = asyncio.get_event_loop()
    try:
        res = await loop.run_in_executor(_executor, transcribe_one, payload)
        return JSONResponse(res)
    except Exception as e:  # noqa: BLE001
        import traceback
        raise HTTPException(status_code=500, detail={"error": str(e), "tb": traceback.format_exc()[-1000:]})


@app.post("/transcribe_batch")
async def transcribe_batch(payload: dict):
    items = payload.get("items") or []
    if not items:
        raise HTTPException(status_code=400, detail="items required")
    base = {k: v for k, v in payload.items() if k != "items"}
    merged = [{**base, **it} for it in items]
    loop = asyncio.get_event_loop()
    futs = [loop.run_in_executor(_executor, transcribe_one, m) for m in merged]
    results = await asyncio.gather(*futs, return_exceptions=True)
    out = []
    for r in results:
        if isinstance(r, Exception):
            out.append({"error": str(r)})
        else:
            out.append(r)
    return JSONResponse({"count": len(out), "results": out})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "80")))
