"""
Run this on YOUR machine (needs your real Chroma collection + fastembed
model). This directly tests the open question from the BM25 debugging:
does dense search actually rescue "VPN keeps disconnecting" the way
hybrid retrieval is supposed to?

Requires retrieval_pipeline.py in the same folder.
"""

import sys
from pathlib import Path

TESTS_DIR = Path(__file__).parent
PROJECT_ROOT = TESTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import chromadb
from fastembed import TextEmbedding
from retrieval_pipeline import bm25_search, reciprocal_rank_fusion, CHUNKS

CHROMA_DB_PATH = PROJECT_ROOT / "chroma_db"
MODEL_CACHE_DIR = PROJECT_ROOT / "local_model_cache"

client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
collection = client.get_collection("it_helpdesk_kb")
model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", cache_dir=str(MODEL_CACHE_DIR))


def dense_search(query: str, top_k: int = 20, category: str | None = None):
    query_prefix = "Represent this sentence for searching relevant passages: "
    query_embedding = list(model.embed([query_prefix + query]))[0]
    where_filter = {"category": category} if category else None
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        where=where_filter,
    )
    return results["ids"][0]  # just the ranked chunk_ids


# ---------------------------------------------------------------------------
# The exact query that broke BM25 alone
# ---------------------------------------------------------------------------
query = "VPN keeps disconnecting"
error_codes = ["0x8007042B"]

print("=== BM25 alone (already known to fail on this query) ===")
bm25_results = bm25_search(query, error_codes=error_codes)
bm25_ids = [r["chunk_id"] for r in bm25_results]
for r in bm25_results[:5]:
    print(f"  {r['score']:.2f}  [{r['chunk_id']}] {r['chunk']['title']}")

print("\n=== Dense alone ===")
dense_ids = dense_search(query)
for chunk_id in dense_ids[:5]:
    title = next(c["title"] for c in CHUNKS if c["chunk_id"] == chunk_id)
    print(f"  [{chunk_id}] {title}")

print("\n=== Hybrid: RRF fusion of both ===")
fused = reciprocal_rank_fusion([bm25_ids, dense_ids])
for chunk_id, score in fused[:5]:
    title = next(c["title"] for c in CHUNKS if c["chunk_id"] == chunk_id)
    print(f"  {score:.4f}  [{chunk_id}] {title}")

print("\n--- The question this answers: does the fused list put a genuine")
print("VPN-troubleshooting page ahead of the VPN-access-request pages")
print("that dominated BM25 alone? Compare the top result here against")
print("the BM25-alone list above.")
