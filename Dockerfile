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

# Caps onnxruntime's thread pool at 1 -- the actual root cause of a real
# stuck deploy (2026-10-25): left unset, onnxruntime sizes its thread
# pool from os.cpu_count(), which reports the HOST machine's cores, not
# the container's cgroup CPU quota (Render free tier: 0.1 vCPU). The
# oversized thread pool doesn't just run slow, it HANGS -- threads
# spin-wait against a quota that never gives them real CPU time. A real
# deploy confirmed this: embed_and_store.py sat completely silent for
# 30+ seconds inside model.embed() with PYTHONUNBUFFERED=1 already set
# (ruling out a buffering explanation) before Render's port-scan timeout
# fired. See config.py's ONNX_NUM_THREADS docstring.
ENV ONNX_NUM_THREADS=1

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
from config import MODEL_CACHE_DIR, RERANKER_MODEL_NAME, ONNX_NUM_THREADS; \
TextEmbedding(model_name='BAAI/bge-small-en-v1.5', cache_dir=str(MODEL_CACHE_DIR), threads=ONNX_NUM_THREADS); \
TextCrossEncoder(model_name=RERANKER_MODEL_NAME, cache_dir=str(MODEL_CACHE_DIR), threads=ONNX_NUM_THREADS); \
print('Models pre-downloaded into image.')"

# Build the Chroma vector index HERE, at Docker BUILD time, not at
# container boot (2026-10-25, real deploy failure): a real deploy showed
# embedding 50/299 chunks took ~3m49s and 100/299 took another ~3m53s on
# Render's free-tier 0.1 vCPU -- all 299 chunks would take roughly
# 20+ minutes, but Render's port-scan timeout kills an unresponsive boot
# at ~5 minutes. threads=1 (above) ruled out a thread-pool hang; this is
# genuine CPU starvation, not a bug to tune away. A docker BUILD step has
# no such timeout, so running this once per git push (not once per
# container restart/wake-up) is the real fix, not a workaround.
#
# TRADE-OFF, accepted: the Chroma index is now baked into the image
# rather than rebuilt fresh every restart. This does NOT affect
# data/processed/*.jsonl (golden_set_v*.jsonl, eval_custom_cases.jsonl,
# etc.) -- those are still just files committed to git, read fresh by
# the app at runtime regardless of when the image was built. It only
# means a KB content change (editing data/chunks/*.jsonl) requires a new
# git push to take effect, not a server restart -- which was already
# true for a git-based deploy anyway.
RUN python src/embed_and_store.py

# Render's (and most container hosts') disk is ephemeral (wiped on every
# redeploy/restart), but that no longer matters for local_model_cache/ or
# chroma_db/ -- both are baked into the image at BUILD time (above), so a
# restart just reuses them unchanged, nothing to rebuild. Only
# sessions.db is reseeded at boot (docker-entrypoint.sh) -- cheap, and it
# genuinely needs to happen per-container since staff_accounts lives
# there. Every path here is still env-overridable (see src/config.py);
# CHROMA_DB_PATH/MODEL_CACHE_DIR/SESSION_DB_PATH all default to paths
# under /app, which IS this container's own filesystem, so no extra env
# vars are needed for this to work.

# Render sets $PORT dynamically (default 10000) and requires binding
# 0.0.0.0, not a fixed port -- ui/server.py already reads both HOST and
# PORT from the environment (falling back to 127.0.0.1:8010 for local
# dev), so no hardcoded port here.
ENV HOST=0.0.0.0

RUN chmod +x docker-entrypoint.sh
ENTRYPOINT ["./docker-entrypoint.sh"]
