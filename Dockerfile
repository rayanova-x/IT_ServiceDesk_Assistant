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
# dev), so no hardcoded port here. RERANKER_MODEL swaps the reranker for
# a ~290MB-RSS model instead of the ~1.2GB default, to fit Render's
# 512MB free-tier RAM cap -- see config.py's docstring on why this means
# the deployed app's confidence scores are NOT on the same scale as the
# calibrated 2.0 threshold / existing eval numbers (measured against the
# heavier default model, never re-run against this one).
ENV HOST=0.0.0.0
ENV RERANKER_MODEL="Xenova/ms-marco-MiniLM-L-6-v2"

RUN chmod +x docker-entrypoint.sh
ENTRYPOINT ["./docker-entrypoint.sh"]
