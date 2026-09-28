FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ffmpeg ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip setuptools wheel

# torch 2.8.0 + cu128 (Blackwell sm_120 + Ada sm_89)
RUN pip3 install --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

RUN pip3 install \
        "ctranslate2" "faster-whisper>=1.2.0" "numpy>=2.1.0" \
        "nltk>=3.9.1" "omegaconf>=2.3.0" "pandas>=2.2.3" \
        "huggingface-hub>=0.28.1" "transformers>=4.48.0" \
        torchcodec soundfile fastapi "uvicorn[standard]" python-multipart requests

# whisperx without deps (pyannote is lazy-loaded and not needed for align-only)
RUN pip3 install whisperx --no-deps

# make the pyannote VAD import optional (we only ever use silero)
RUN sed -i 's|^from whisperx.vads.pyannote import Pyannote as Pyannote|try:\n    from whisperx.vads.pyannote import Pyannote as Pyannote\nexcept Exception:\n    Pyannote = None|' \
    /usr/local/lib/python3.10/dist-packages/whisperx/vads/__init__.py \
    && cat /usr/local/lib/python3.10/dist-packages/whisperx/vads/__init__.py

# nltk data
RUN python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# pre-download ASR + alignment models so cold start doesn't fetch them
RUN python3 -c "import whisperx; whisperx.load_model('large-v3', device='cpu', compute_type='int8')"
RUN python3 -c "import whisperx; whisperx.load_align_model(language_code='zh', device='cpu')"
RUN python3 -c "import whisperx; whisperx.load_align_model(language_code='en', device='cpu')"

COPY app.py /app.py

EXPOSE 80
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-80} --workers 1 --timeout-keep-alive 75"]
