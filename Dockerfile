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

# Hugging Face Spaces' disk is ephemeral (wiped on every redeploy/restart)
# -- chroma_db/, local_model_cache/, and sessions.db are all rebuilt at
# container boot by docker-entrypoint.sh, not shipped in the image or
# expected to persist. Every path these write to is env-overridable
# (see src/config.py); CHROMA_DB_PATH/MODEL_CACHE_DIR/SESSION_DB_PATH
# all default to paths under /app, which IS this container's own
# ephemeral filesystem, so no extra env vars are needed for this to work.

# HF Spaces' Docker runtime expects the app to listen on 7860.
ENV HOST=0.0.0.0
ENV PORT=7860
EXPOSE 7860

RUN chmod +x docker-entrypoint.sh
ENTRYPOINT ["./docker-entrypoint.sh"]
