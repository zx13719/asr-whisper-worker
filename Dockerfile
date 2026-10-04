# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04@sha256:17e2934e1fa96152b14f78078bfbafd0f00f391df995dc6c641a720fce1202bb

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ffmpeg ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip setuptools wheel

# torch 2.8.0 + cu128 (Blackwell sm_120 + Ada sm_89)
RUN pip3 install --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

COPY requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt

# whisperx without deps (pyannote is only needed for diarization)
RUN pip3 install whisperx==3.8.6 --no-deps

# make pyannote imports optional — we only use silero VAD + wav2vec2 alignment
RUN sed -i 's|^from whisperx.vads.pyannote import Pyannote as Pyannote|try:\n    from whisperx.vads.pyannote import Pyannote as Pyannote\nexcept Exception:\n    Pyannote = None|' \
        /usr/local/lib/python3.10/dist-packages/whisperx/vads/__init__.py \
 && sed -i 's|^from pyannote.audio import Pipeline|try:\n    from pyannote.audio import Pipeline\nexcept Exception:\n    Pipeline = None|' \
        /usr/local/lib/python3.10/dist-packages/whisperx/diarize.py \
 && echo "--- vads/__init__.py ---" && cat /usr/local/lib/python3.10/dist-packages/whisperx/vads/__init__.py \
 && echo "--- diarize.py head ---" && head -8 /usr/local/lib/python3.10/dist-packages/whisperx/diarize.py

RUN python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# pre-download models so cold start doesn't fetch them
RUN python3 -c "import whisperx; whisperx.load_model('large-v3', device='cpu', compute_type='int8', vad_method='silero')"
RUN python3 -c "import whisperx; whisperx.load_align_model(language_code='zh', device='cpu')"
RUN python3 -c "import whisperx; whisperx.load_align_model(language_code='en', device='cpu')"

ARG VCS_REF=unknown
ENV IMAGE_REVISION=$VCS_REF
LABEL org.opencontainers.image.revision=$VCS_REF

COPY app.py /app.py

EXPOSE 80
CMD ["python3", "-u", "/app.py"]
