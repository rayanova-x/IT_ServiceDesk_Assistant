FROM python:3.11-slim

WORKDIR /app

# System deps for bcrypt/onnxruntime wheels -- kept minimal, no PyTorch
# in requirements.txt (see embed_and_store.py's docstring: fastembed's
# ONNX-only choice was deliberate, so the image stays small).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Set BEFORE the model pre-download step so its own output isn't
# buffered either -- see the fuller explanation below, near where the
# server's own env vars are set.
ENV PYTHONUNBUFFERED=1

# Reranker choice, set once here and reused by both the build-time
# pre-download step below and the real container boot: swaps the
# ~1.2GB-RSS default (BAAI/bge-reranker-base) for a ~290MB-RSS model, to
# fit Render's 512MB free-tier RAM cap -- see config.py's docstring on
# why this means the deployed app's confidence scores are NOT on the
# same scale as the calibrated 2.0 threshold / existing eval numbers
# (measured against the heavier default model, never re-run against
# this one).
ENV RERANKER_MODEL="Xenova/ms-marco-MiniLM-L-6-v2"

# Pre-download both ML models into the image at BUILD time (once per git
# push), not at container BOOT time (every restart, including Render
# free tier's spin-down/wake-up cycle). Previously both models downloaded
# fresh on every boot -- slow even once cached by Render's layer, and a
# real repeated cost every time the free tier sleeps and wakes. This
# RUN step shares the exact same MODEL_CACHE_DIR/RERANKER_MODEL the real
# boot sequence uses (see src/config.py), so what's cached here is
# actually what docker-entrypoint.sh finds already present -- not a
# separate, unused cache.
RUN python -c "\
import sys; sys.path.insert(0, 'src'); \
from fastembed import TextEmbedding; \
from fastembed.rerank.cross_encoder import TextCrossEncoder; \
from config import MODEL_CACHE_DIR, RERANKER_MODEL_NAME; \
TextEmbedding(model_name='BAAI/bge-small-en-v1.5', cache_dir=str(MODEL_CACHE_DIR)); \
TextCrossEncoder(model_name=RERANKER_MODEL_NAME, cache_dir=str(MODEL_CACHE_DIR)); \
print('Models pre-downloaded into image.')"

# Render's (and most container hosts') disk is ephemeral (wiped on every
# redeploy/restart) -- chroma_db/, local_model_cache/, and sessions.db are
# all rebuilt at container boot by docker-entrypoint.sh, not shipped in
# the image or expected to persist. Every path these write to is
# env-overridable (see src/config.py); CHROMA_DB_PATH/MODEL_CACHE_DIR/
# SESSION_DB_PATH all default to paths under /app, which IS this
# container's own ephemeral filesystem, so no extra env vars are needed
# for this to work.

# Render sets $PORT dynamically (default 10000) and requires binding
# 0.0.0.0, not a fixed port -- ui/server.py already reads both HOST and
# PORT from the environment (falling back to 127.0.0.1:8010 for local
# dev), so no hardcoded port here.
ENV HOST=0.0.0.0

RUN chmod +x docker-entrypoint.sh
ENTRYPOINT ["./docker-entrypoint.sh"]
