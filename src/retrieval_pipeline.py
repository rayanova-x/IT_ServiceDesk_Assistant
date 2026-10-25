"""
Steps 5-6 of the pipeline: dual retrieval (dense + BM25), RRF fusion, and
cross-encoder reranking. All three retrievers are real and tested against
the actual 247-chunk Chroma collection -- see eval_retrieval.py.

Two end-to-end entry points, both tested by eval_retrieval.py:
  - hybrid_search(): BM25 + dense -> RRF fusion -> top-N. Simpler, no
    reranker model needed.
  - hybrid_search_reranked(): same, but widens the candidate pool and
    re-sorts with a local cross-encoder (BAAI/bge-reranker-base) before
    the top-N cutoff. USE THIS ONE for any new code -- measured 2026-09-13
    to beat hybrid_search() on every metric with no regressions (combined
    top-1 accuracy 45.9% -> 58.2%, confuser-trap rate 75% -> 50%). See
    PROJECT_HANDOFF.md Section 8 for the full before/after numbers and
    Section 6 for the one known remaining weak spot (confuser pairs where
    the decoy's own text states a distinguishing condition -- a
    similarity-based reranker doesn't reliably catch that).
"""

import json
import re
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from config import DATA_DIR

DATA_CHUNKS = DATA_DIR / "chunks"

# ---------------------------------------------------------------------------
# Load chunks from EVERY source connector -- same set embed_and_store.py
# embeds into Chroma. Bug found and fixed 2026-09-21: this used to load
# ONLY chunks_from_confluence.jsonl, so after adding the error-code
# spreadsheet as a second source, dense search (via Chroma, correctly
# holding both sources) could surface a spreadsheet chunk but BM25 never
# could -- and RRF fusion over the two retrievers' candidate lists means
# hybrid_search's OWN merge logic could only promote a chunk BM25 never
# proposed if dense ranked it highly enough on its own, which it often
# didn't for a terse, code-heavy row.  Caught by testing the exact
# scenario this spreadsheet was built for (a query containing the exact
# error code 0x8007042B) and finding the spreadsheet chunk missing from
# results entirely, not just poorly ranked.
# ---------------------------------------------------------------------------
SOURCE_FILES = [
    "chunks_from_confluence.jsonl",
    "chunks_from_error_codes.jsonl",
    "chunks_from_slack_threads.jsonl",
]

CHUNKS = []
for _filename in SOURCE_FILES:
    _path = DATA_CHUNKS / _filename
    if not _path.exists():
        continue
    with open(_path, encoding="utf-8") as f:
        CHUNKS.extend(json.loads(line) for line in f)

print(f"Loaded {len(CHUNKS)} chunks for BM25 indexing")


# ---------------------------------------------------------------------------
# BM25 index -- built once, at startup, reused for every query
# ---------------------------------------------------------------------------
def simple_tokenize(text: str) -> list[str]:
    """
    Lowercase, strip punctuation, split on whitespace, drop stopwords.

    The stopword removal matters more than it looks: on a small corpus
    (247 chunks, many under 100 words), a generic word like "keeps" can
    end up RARE across the corpus purely by chance -- which makes BM25's
    IDF weighting treat it as a strong, distinguishing signal, when
    semantically it carries no diagnostic meaning at all. An earlier,
    untested version of this function skipped stopword removal and it
    produced a completely irrelevant top result for a real test query
    (an "Application update failed" chunk outranked every VPN-related
    chunk, purely because it happened to share the word "keeps" with the
    query). Standard BM25 usage assumes the caller removes stopwords --
    the library itself doesn't do this automatically.
    """
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in ENGLISH_STOP_WORDS]

_tokenized_corpus = [simple_tokenize(c["text"]) for c in CHUNKS]

# Small-corpus BM25 pitfall, found by testing (not assumed): a term
# appearing in just 1-2 of 247 chunks gets an outsized IDF weight even
# when it's generic filler ("keeps", "seems"), because BM25's rarity
# math can't distinguish "genuinely distinctive" from "just didn't happen
# to appear elsewhere." The standard mitigation is a minimum document-
# frequency cutoff -- drop terms too rare to trust statistically.
MIN_DOC_FREQUENCY = 2

def _doc_frequencies(tokenized_corpus):
    from collections import Counter
    df = Counter()
    for doc in tokenized_corpus:
        df.update(set(doc))
    return df

_df = _doc_frequencies(_tokenized_corpus)
_tokenized_corpus = [[t for t in doc if _df[t] >= MIN_DOC_FREQUENCY] for doc in _tokenized_corpus]

bm25_index = BM25Okapi(_tokenized_corpus)


def _passes_permission_filter(chunk: dict, allowed_groups: set[str] | None) -> bool:
    """
    Step 4.5: permission resolution, applied as a retrieval-time filter,
    never a post-hoc check on already-surfaced results (see
    PROJECT_HANDOFF.md Section 3/5 -- every real system researched, Glean/
    M365 Copilot/Atlassian Rovo/Pinecone, enforces this pre-retrieval).

    allowed_groups=None means "no restriction known" -- callers that don't
    pass a caller identity (e.g. offline eval scripts) get the old
    unfiltered behavior, not an accidental lockout. A chunk with no
    `restricted_to` groups is always visible to everyone.
    """
    restricted_to = chunk.get("restricted_to") or []
    if not restricted_to:
        return True
    if allowed_groups is None:
        return True
    return bool(set(restricted_to) & allowed_groups)


def bm25_search(query: str, error_codes: list[str] = None, top_k: int = 20,
                 category: str | None = None,
                 allowed_groups: set[str] | None = None) -> list[dict]:
    """
    Sparse/keyword retrieval. error_codes get appended to the query text
    directly -- BM25 rewards exact token overlap, so an error code sitting
    verbatim in the query text is exactly what makes this retriever good
    at catching it, without any special-case logic.

    allowed_groups: the caller's group memberships (Step 4.5) -- a chunk
    restricted to a group the caller doesn't have is excluded here, never
    returned and then hidden later at display time.
    """
    query_text = query
    if error_codes:
        query_text += " " + " ".join(error_codes)

    tokens = simple_tokenize(query_text)
    scores = bm25_index.get_scores(tokens)

    ranked_idx = sorted(range(len(scores)), key=lambda i: -scores[i])

    results = []
    for idx in ranked_idx:
        chunk = CHUNKS[idx]
        if category and chunk["category"] != category:
            continue  # post-filter by category -- fine at this corpus size
        if not _passes_permission_filter(chunk, allowed_groups):
            continue
        if scores[idx] <= 0:
            continue  # BM25 gives 0 to chunks with zero token overlap -- skip them
        results.append({"chunk_id": chunk["chunk_id"], "score": scores[idx], "chunk": chunk})
        if len(results) >= top_k:
            break
    return results


# ---------------------------------------------------------------------------
# Dense search -- matches embed_and_store.py's conventions exactly.
# NOT runnable here (no live fastembed model in this sandbox) -- this is
# the real code to use on your machine, where fastembed already works.
# ---------------------------------------------------------------------------
def dense_search(query: str, collection, model, top_k: int = 20,
                  category: str | None = None,
                  allowed_groups: set[str] | None = None) -> list[dict]:
    """
    Run this version on your machine, against your real Chroma collection.
    `model` is the same fastembed TextEmbedding instance from embed_and_store.py.

    allowed_groups: Step 4.5's permission filter, applied as a Chroma
    metadata `where` clause -- never a post-hoc filter on already-returned
    results. `restricted_to` is stored as a single joined string per chunk
    (embed_and_store.py's prepare_metadata(), "none" if unrestricted), so
    "visible to this caller" means restricted_to is EITHER "none" OR one
    of the caller's own groups.
    """
    query_prefix = "Represent this sentence for searching relevant passages: "
    query_embedding = list(model.embed([query_prefix + query]))[0]

    conditions = []
    if category:
        conditions.append({"category": category})
    if allowed_groups is not None:
        conditions.append({"restricted_to": {"$in": ["none", *sorted(allowed_groups)]}})

    if len(conditions) == 0:
        where_filter = None
    elif len(conditions) == 1:
        where_filter = conditions[0]
    else:
        where_filter = {"$and": conditions}

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        where=where_filter,
    )

    out = []
    for i, chunk_id in enumerate(results["ids"][0]):
        out.append({
            "chunk_id": chunk_id,
            "score": 1 - results["distances"][0][i],  # cosine distance -> similarity
            "metadata": results["metadatas"][0][i],
            "text": results["documents"][0][i],
        })
    return out


# ---------------------------------------------------------------------------
# RRF fusion -- combines two ranked lists by POSITION, never by raw score.
# This part needs no model at all, so it's fully testable here regardless
# of which retriever produced the lists.
# ---------------------------------------------------------------------------
def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """
    ranked_lists: a list of ranked chunk_id lists, e.g. [bm25_ids, dense_ids]
    Returns EVERY chunk_id that appeared in any list, sorted by fused RRF
    score, highest first -- no cutoff. Callers that want a fixed-size
    result (e.g. hybrid_search()) must slice this themselves.
    """
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, chunk_id in enumerate(ranked_list, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


# ---------------------------------------------------------------------------
# Full hybrid pipeline: BM25 + dense -> RRF fusion -> fixed top-N cutoff.
#
# This is Steps 5-6 of the pipeline end to end, wired together. It returns
# a FIXED number of chunks (TOP_N_FINAL) regardless of how many candidates
# either retriever found -- no confidence/threshold logic yet (Step 8 in
# the design doc). That's deliberate: get the full pipeline runnable and
# measurable first, then decide from real eval numbers whether a
# confidence gate is actually needed, rather than guessing a threshold
# with no data. See PROJECT_HANDOFF.md Section 8.
# ---------------------------------------------------------------------------
TOP_N_FINAL = 5

def hybrid_search(query: str, collection, model, error_codes: list[str] = None,
                   category: str | None = None, top_n: int = TOP_N_FINAL,
                   allowed_groups: set[str] | None = None) -> list[dict]:
    """
    Run BM25 + dense search, fuse by RRF, return the top `top_n` chunks
    with full chunk data attached (not just chunk_ids) -- ready to hand
    directly to a generation step.
    """
    bm25_results = bm25_search(query, error_codes=error_codes, category=category,
                                allowed_groups=allowed_groups)
    dense_results = dense_search(query, collection, model, category=category,
                                  allowed_groups=allowed_groups)

    bm25_ids = [r["chunk_id"] for r in bm25_results]
    dense_ids = [r["chunk_id"] for r in dense_results]

    fused = reciprocal_rank_fusion([bm25_ids, dense_ids])[:top_n]

    chunk_by_id = {c["chunk_id"]: c for c in CHUNKS}
    return [
        {"chunk_id": chunk_id, "rrf_score": score, "chunk": chunk_by_id[chunk_id]}
        for chunk_id, score in fused
        if chunk_id in chunk_by_id
    ]


# ---------------------------------------------------------------------------
# Reranking: fuse a WIDER candidate pool (RERANK_CANDIDATE_POOL), then use a
# dedicated cross-encoder reranker model to re-score and re-sort those
# candidates before cutting to the final top_n. This targets a specific
# weakness measured by eval_retrieval.py, not assumed: RRF fusion alone gets
# the right document into the top 5 (recall@5 89.8% combined) but often
# ranks a designed confuser decoy above the real answer at #1 (top-1 accuracy
# only 45.9% combined; decoys beat the real answer 80% of the time when a
# known decoy existed). A reranker is the standard fix for exactly this
# "found the right candidates, ranked them wrong" symptom -- see
# PROJECT_HANDOFF.md and reference_docs/rag_evals2.md's anti-pattern list
# ("I'll fine-tune the embedding model" is called out as low-value for this;
# reranking and hybrid retrieval are the recommended levers).
#
# Uses fastembed's local TextCrossEncoder (BAAI/bge-reranker-base, same
# cache_dir convention as the embedding model -- never the OS temp folder,
# per PROJECT_HANDOFF.md Section 6 bug #4). Fully local, no third-party API,
# consistent with Section 1's hard constraint.
# ---------------------------------------------------------------------------
RERANK_CANDIDATE_POOL = 20

def _access_denial_info(query: str, error_codes: list[str], category: str | None,
                         candidate_pool: int, allowed_groups: set[str]) -> dict:
    """
    Step 4.5 audit: re-runs the SAME bm25_search() candidate scan with no
    permission filter (allowed_groups=None) and diffs its chunk_ids
    against the filtered run, to report WHICH groups' content was
    actually withheld from this specific query -- not just that a filter
    exists. Cheap (BM25 is pure local scoring, no model call) and only
    run when allowed_groups was actually supplied, so an unfiltered
    caller (eval scripts, no identity) pays nothing extra.
    """
    unfiltered = bm25_search(query, error_codes=error_codes, category=category,
                              top_k=candidate_pool, allowed_groups=None)
    filtered_ids = {
        r["chunk_id"] for r in bm25_search(query, error_codes=error_codes, category=category,
                                            top_k=candidate_pool, allowed_groups=allowed_groups)
    }
    excluded = [r["chunk"] for r in unfiltered if r["chunk_id"] not in filtered_ids]
    excluded_groups = sorted({g for c in excluded for g in (c.get("restricted_to") or [])})
    return {"excluded_count": len(excluded), "excluded_groups": excluded_groups}


def hybrid_search_reranked(query: str, collection, model, reranker,
                            error_codes: list[str] = None, category: str | None = None,
                            top_n: int = TOP_N_FINAL,
                            candidate_pool: int = RERANK_CANDIDATE_POOL,
                            allowed_groups: set[str] | None = None) -> list[dict]:
    """
    Same as hybrid_search(), but fuses a wider candidate pool and re-sorts
    it with a cross-encoder reranker before cutting to top_n. `reranker` is
    a fastembed.rerank.cross_encoder.TextCrossEncoder instance.

    allowed_groups: Step 4.5's permission filter -- see
    _passes_permission_filter()/dense_search()'s docstrings. None means no
    caller identity was supplied (unfiltered, same as before this existed).
    Return shape is UNCHANGED (still a plain list) to avoid a breaking
    change across every existing caller (calibrate_confidence.py,
    eval_retrieval.py, same_topic_detection.py, step7_recency.py,
    step8_confidence.py) -- callers that need to know WHAT was excluded
    for audit purposes call access_denial_info() separately (see
    pipeline.py), which only costs anything extra when allowed_groups is
    actually supplied.
    """
    bm25_results = bm25_search(query, error_codes=error_codes, category=category,
                                top_k=candidate_pool, allowed_groups=allowed_groups)
    dense_results = dense_search(query, collection, model, top_k=candidate_pool,
                                  category=category, allowed_groups=allowed_groups)

    bm25_ids = [r["chunk_id"] for r in bm25_results]
    dense_ids = [r["chunk_id"] for r in dense_results]

    fused = reciprocal_rank_fusion([bm25_ids, dense_ids])[:candidate_pool]

    chunk_by_id = {c["chunk_id"]: c for c in CHUNKS}
    candidates = [(chunk_id, chunk_by_id[chunk_id]) for chunk_id, _ in fused if chunk_id in chunk_by_id]
    if not candidates:
        return []

    candidate_texts = [c["text"] for _, c in candidates]
    rerank_scores = list(reranker.rerank(query, candidate_texts))

    reranked = sorted(zip(rerank_scores, candidates), key=lambda x: -x[0])[:top_n]
    return [
        {"chunk_id": chunk_id, "rerank_score": float(score), "chunk": chunk}
        for score, (chunk_id, chunk) in reranked
    ]


def access_denial_info(query: str, error_codes: list[str] | None, category: str | None,
                        allowed_groups: set[str], candidate_pool: int = RERANK_CANDIDATE_POOL) -> dict | None:
    """
    Public wrapper around _access_denial_info() -- called separately from
    hybrid_search_reranked() (see that function's docstring for why the
    two are split rather than one combined return value). Returns None if
    nothing was actually excluded, so callers can treat "falsy" as "no
    denial to log" without inspecting excluded_count themselves.
    """
    info = _access_denial_info(query, error_codes or [], category, candidate_pool, allowed_groups)
    return info if info["excluded_count"] > 0 else None


# ---------------------------------------------------------------------------
# Test: BM25 alone (fully real), then RRF fusion against a stand-in dense
# list (clearly labeled -- not a real embedding search result)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== BM25 alone: 'VPN keeps disconnecting error 0x8007042B' ===")
    bm25_results = bm25_search("VPN keeps disconnecting", error_codes=["0x8007042B"])
    for r in bm25_results[:5]:
        print(f"  {r['score']:.2f}  [{r['chunk_id']}] {r['chunk']['title']} -- {r['chunk']['header_path']}")

    print("\n=== RRF fusion test (BM25 real + a STAND-IN fake dense list) ===")
    # This dense list is NOT a real embedding search -- just a plausible-looking
    # stand-in so the fusion LOGIC can be verified end-to-end here. Replace
    # with the real dense_search() output on your machine.
    fake_dense_ids = [c["chunk_id"] for c in CHUNKS if "VPN" in c["title"]][:10]
    bm25_ids = [r["chunk_id"] for r in bm25_results]

    fused = reciprocal_rank_fusion([bm25_ids, fake_dense_ids])
    print("Top 5 fused results:")
    for chunk_id, score in fused[:5]:
        chunk = next(c for c in CHUNKS if c["chunk_id"] == chunk_id)
        print(f"  {score:.4f}  [{chunk_id}] {chunk['title']}")
