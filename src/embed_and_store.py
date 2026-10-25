"""
Embed chunks (fastembed, fully local, ONNX-based -- no PyTorch) and store
them in Chroma.

fastembed + bge-small-en-v1.5 was chosen over sentence-transformers +
bge-base-en-v1.5: the PyTorch dependency didn't fit in the original dev
sandbox's disk space, and separately, both libraries have hit real Hugging
Face network access problems on the user's corporate machine. See
PROJECT_HANDOFF.md Section 1 and Section 7.

Install first:
    pip install fastembed chromadb
"""

import json
from fastembed import TextEmbedding
import chromadb

from config import CHROMA_DB_PATH, MODEL_CACHE_DIR, DATA_DIR, ONNX_NUM_THREADS

DATA_CHUNKS = DATA_DIR / "chunks"


# ---------------------------------------------------------------------------
# Step 1: Load the model
# ---------------------------------------------------------------------------
# bge-small-en-v1.5: local/open-source only, no third-party API (hard
# constraint). Runs fully on your own machine via ONNX -- first run
# downloads ~130MB, then it's cached locally and works offline after that.
#
# IMPORTANT: always pass cache_dir explicitly, pointing inside the project
# folder. fastembed's default cache location (OS Temp folder) was silently
# wiped between sessions on the user's Windows machine -- see
# PROJECT_HANDOFF.md Section 6, bug #4.
MODEL_NAME = "BAAI/bge-small-en-v1.5"
model = TextEmbedding(model_name=MODEL_NAME, cache_dir=str(MODEL_CACHE_DIR), threads=ONNX_NUM_THREADS)

# IMPORTANT, model-specific detail: BGE models expect an instruction prefix
# on QUERIES at search time ("Represent this sentence for searching
# relevant passages: ") but NOT on the documents/chunks you're indexing.
# We are indexing chunks here, so no prefix is needed in this script --
# but remember to add it later, in the retrieval script, to actual user
# queries. (Different model families have different conventions -- e.g.
# E5 models want "passage: " / "query: " prefixes on BOTH sides. Always
# check the specific model card before assuming.)


# ---------------------------------------------------------------------------
# Step 2: Load chunks from EVERY source connector
# ---------------------------------------------------------------------------
# Multi-source ingestion: each connector (chunk_confluence.py,
# ingest_error_code_spreadsheet.py, ...) normalizes its own source into
# the SAME canonical chunk schema and writes its own chunks file. This
# is the one place those independent connectors converge -- matching
# the real production pattern (researched, not assumed -- see
# PROJECT_HANDOFF.md): per-source connectors stay separate because each
# source has different auth/structure/pagination, but everything is
# normalized to one schema before embedding so retrieval doesn't need
# to know which connector a chunk came from. source_type on each chunk
# (set by its connector) keeps that origin traceable downstream anyway.
SOURCE_FILES = [
    "chunks_from_confluence.jsonl",
    "chunks_from_error_codes.jsonl",
    "chunks_from_slack_threads.jsonl",
]

chunks = []
for filename in SOURCE_FILES:
    path = DATA_CHUNKS / filename
    if not path.exists():
        print(f"WARNING: {filename} not found, skipping (run its connector script first)")
        continue
    with open(path, encoding="utf-8") as f:
        source_chunks = [json.loads(line) for line in f]
    print(f"Loaded {len(source_chunks)} chunks from {filename}")
    chunks.extend(source_chunks)

print(f"Total: {len(chunks)} chunks from {len(SOURCE_FILES)} source connectors")


# ---------------------------------------------------------------------------
# Step 3: Embed the TEXT only -- never the metadata
# ---------------------------------------------------------------------------
# model.embed() returns a generator and batches internally when given a
# list, so we pass all chunk texts at once rather than looping one at a
# time (much faster).
texts = [c["text"] for c in chunks]

import numpy as np
# Explicit per-item progress print (2026-10-25, deployment diagnosis): on
# a CPU-starved free-tier host (e.g. Render's 0.1 vCPU), this step can
# take far longer than it does locally -- previously this printed NOTHING
# until all 299 chunks finished, which was indistinguishable in the
# deploy log from a hung/crashed process. Printing every 50 chunks proves
# forward progress is actually happening, not just a cosmetic nicety.
embeddings = []
for i, e in enumerate(model.embed(texts, batch_size=32), start=1):
    embeddings.append(np.asarray(e))
    if i % 50 == 0 or i == len(texts):
        print(f"  embedded {i}/{len(texts)} chunks...")
# BGE ONNX models from fastembed are already L2-normalized, but we
# normalize explicitly so cosine similarity behaves correctly regardless.
embeddings = [e / np.linalg.norm(e) for e in embeddings]
# embeddings is now a list of 384-dim vectors, one per chunk, in the same
# order as `texts` / `chunks`.


# ---------------------------------------------------------------------------
# Step 4: Prepare metadata for Chroma
# ---------------------------------------------------------------------------
# Chroma metadata values must be simple scalars (str/int/float/bool) -- no
# lists, no nested dicts. `labels` and `restricted_to` are lists on our
# chunks, so we join them into comma-separated strings here.
def prepare_metadata(chunk: dict) -> dict:
    return {
        "page_id": chunk["page_id"],
        "title": chunk["title"],
        "category": chunk["category"] or "none",
        "status": chunk["status"],
        "version_number": chunk["version_number"],
        "last_updated": chunk["last_updated"],
        "labels": ",".join(chunk["labels"]) if chunk["labels"] else "none",
        "restricted_to": ",".join(chunk["restricted_to"]) if chunk["restricted_to"] else "none",
        "header_path": chunk["header_path"],
        "token_count": chunk["token_count"],
        "secondary_split": chunk["secondary_split"],
        "source_type": chunk.get("source_type", "unknown"),
    }

metadatas = [prepare_metadata(c) for c in chunks]
ids = [c["chunk_id"] for c in chunks]
documents = texts  # Chroma stores the raw text too, so query results can return it directly


# ---------------------------------------------------------------------------
# Step 5: Store in Chroma
# ---------------------------------------------------------------------------
# PersistentClient writes to disk at the given path, so the collection
# survives between runs -- you don't need to re-embed every time.
client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))

# DELETE-then-recreate, not get_or_create -- real bug found 2026-09-21:
# get_or_create_collection().add() only UPSERTS by chunk_id. When a
# source connector's content changes such that some chunk_ids simply
# stop existing (e.g. error_code_lookup.csv went from 15 rows to 11,
# cutting 7 out-of-scope rows -- see ingest_error_code_spreadsheet.py's
# "SCOPE DISCIPLINE" note), the OLD chunk_ids for the removed rows are
# never touched by add() and stay in Chroma forever as stale, still-
# retrievable garbage. Verified directly: after the CSV was cut from 15
# to 11 rows and this script re-run, Chroma still held all 15 -- the 4
# updated rows upserted correctly, but the 7 removed rows (printer/
# Windows-Update/Outlook/badge-reader/MDM content that was cut for being
# out of project scope) silently remained queryable. This project's
# corpus is small enough that a full delete-and-rebuild on every run is
# the simple, correct fix -- a real production system with millions of
# chunks would need proper tombstone/delta handling per source instead
# (see PROJECT_HANDOFF.md's multi-source ingestion research), but that's
# a real added complexity this project's scale doesn't need yet.
try:
    client.delete_collection("it_helpdesk_kb")
    print("Deleted existing 'it_helpdesk_kb' collection before rebuild.")
except Exception:
    pass  # didn't exist yet -- fine, get_or_create below handles first run

# "hnsw:space": "cosine" tells Chroma to use cosine similarity for search,
# which matches how bge-small-en-v1.5 (and most modern embedding models)
# are designed to be compared.
collection = client.get_or_create_collection(
    name="it_helpdesk_kb",
    metadata={"hnsw:space": "cosine"},
)

# Chroma's add() has a practical batch size limit on some backends -- add
# in batches of 100 to be safe on large corpora.
BATCH_SIZE = 100
for i in range(0, len(chunks), BATCH_SIZE):
    collection.add(
        ids=ids[i:i + BATCH_SIZE],
        embeddings=[e.tolist() for e in embeddings[i:i + BATCH_SIZE]],
        documents=documents[i:i + BATCH_SIZE],
        metadatas=metadatas[i:i + BATCH_SIZE],
    )

print(f"Stored {collection.count()} chunks in Chroma collection 'it_helpdesk_kb'")


# ---------------------------------------------------------------------------
# Step 6: Quick sanity check -- embed a test query and search
# ---------------------------------------------------------------------------
test_query = "my VPN keeps disconnecting"

# Per BGE's convention: add the instruction prefix to the QUERY only.
query_prefix = "Represent this sentence for searching relevant passages: "
query_embedding = next(model.embed([query_prefix + test_query]))
query_embedding = query_embedding / np.linalg.norm(query_embedding)

results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=5,
    where={"status": "current"},  # metadata FILTER -- not part of the embedding itself
)

print(f"\nTest query: '{test_query}'")
for doc_id, title, header in zip(results["ids"][0], 
                                   [m["title"] for m in results["metadatas"][0]],
                                   [m["header_path"] for m in results["metadatas"][0]]):
    print(f"  [{doc_id}] {title} -- {header}")
