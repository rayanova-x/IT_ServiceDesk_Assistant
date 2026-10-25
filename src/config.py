"""
Single source of truth for every filesystem path this project writes
to or reads from at runtime (Chroma's data dir, the embedding/reranker
model cache, the SQLite session DB, the structured-log directory).

WHY THIS EXISTS: before this module, six different files
(embed_and_store.py, pipeline.py, eval_retrieval.py, calibrate_confidence.py,
inspect_chroma.py, visualize_embeddings.py) each independently computed
CHROMA_DB_PATH/MODEL_CACHE_DIR as `REPO_ROOT / "chroma_db"` etc. -- correct
today because they all happened to compute the same REPO_ROOT, but a
real risk of silent drift (one file's path moving out of sync with
another's) and, more importantly for a pilot that may move to a server
later: every path was HARDCODED relative to this one machine's folder
layout, with no way to point at a different data volume, a shared mount,
or an EFS path without editing code. "Move this to a server" should be a
config change, not a code change -- that's the whole point of reading
these from the environment, with the exact same local defaults as
before so nothing changes for anyone who doesn't set them.

Every path here can be overridden via an env var (set in .env, same
convention as GEMINI_API_KEY/JIRA_* already use) -- if unset, falls back
to the original hardcoded-relative-to-repo-root path, so existing local
setups are completely unaffected by this module's existence.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

SRC_DIR = Path(__file__).parent
REPO_ROOT = SRC_DIR.parent
load_dotenv(REPO_ROOT / ".env")


def _path_from_env(var_name: str, default: Path) -> Path:
    override = os.environ.get(var_name)
    return Path(override) if override else default


CHROMA_DB_PATH = _path_from_env("CHROMA_DB_PATH", REPO_ROOT / "chroma_db")
MODEL_CACHE_DIR = _path_from_env("MODEL_CACHE_DIR", REPO_ROOT / "local_model_cache")
SESSION_DB_PATH = _path_from_env("SESSION_DB_PATH", REPO_ROOT / "sessions.db")
LOG_DIR = _path_from_env("LOG_DIR", REPO_ROOT / "logs")
DATA_DIR = _path_from_env("DATA_DIR", REPO_ROOT / "data")
