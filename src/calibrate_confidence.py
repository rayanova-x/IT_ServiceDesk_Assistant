"""
Step 8 groundwork: build a real calibration curve for the reranker's
score, per the sourced guidance (claudecertificationguide.com 5-5,
human-review-calibration, verified 2026-09-14): "Calibration requires
labelled validation sets... run the model, compare its confidence scores
to actual accuracy, and build a calibration curve."

This is the reranker's OWN computed relevance score (rerank_score from
hybrid_search_reranked() -- see retrieval_pipeline.py), NOT an LLM
self-reporting how confident it feels in words. Fully local -- no Gemini
call, no API quota spent -- since reranking runs on-machine.

For every real eval query (50 ticket-based + 62 edge-case, same set used
by eval_retrieval.py), records the TOP result's rerank_score alongside
whether that top result was actually correct. The resulting score/
correctness pairs are what a threshold gets chosen from -- e.g. "results
scoring below X are correct only Y% of the time" -- rather than guessing
a cutoff with no evidence.

Run: python src/calibrate_confidence.py
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
import sys  # noqa: E402
sys.path.insert(0, str(PROJECT_ROOT))

import chromadb  # noqa: E402
import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from fastembed import TextEmbedding  # noqa: E402
from fastembed.rerank.cross_encoder import TextCrossEncoder  # noqa: E402

from retrieval_pipeline import hybrid_search_reranked  # noqa: E402
from config import CHROMA_DB_PATH, MODEL_CACHE_DIR, DATA_DIR, REPO_ROOT  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"


def load_confluence_lookup():
    with open(DATA_PROCESSED / "confluence_export.jsonl", encoding="utf-8") as f:
        pages = [json.loads(line) for line in f]
    with open(DATA_RAW / "rag_corpus_full.jsonl", encoding="utf-8") as f:
        docs = [json.loads(line) for line in f]
    return {
        d["subcategory"]: p["id"]
        for p, d in zip(pages, docs)
        if d["doc_type"] == "current_sop"
    }


def build_ticket_queries(subcat_to_page: dict) -> list[dict]:
    tickets = pd.read_csv(DATA_RAW / "enriched_tickets.csv")
    sample = tickets.groupby("Subcategory").first().reset_index()
    queries = []
    for _, row in sample.iterrows():
        expected = subcat_to_page.get(row["Subcategory"])
        if expected is None:
            continue
        queries.append({
            "query_id": f"TICKET-{row['Subcategory']}",
            "query": row["Description"],
            "expected_page_id": expected,
            "unanswerable": False,
        })
    return queries


def load_edge_case_queries() -> list[dict]:
    with open(DATA_PROCESSED / "eval_edge_case_queries.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return [
        r for r in rows
        if not (r["letter"] == "G" and r["query_id"] != "G01")
    ]


def main():
    print("Loading models and Chroma collection...")
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    collection = client.get_collection("it_helpdesk_kb")
    model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", cache_dir=str(MODEL_CACHE_DIR))
    reranker = TextCrossEncoder(model_name="BAAI/bge-reranker-base", cache_dir=str(MODEL_CACHE_DIR))

    subcat_to_page = load_confluence_lookup()
    ticket_queries = build_ticket_queries(subcat_to_page)
    edge_queries = load_edge_case_queries()
    all_queries = ticket_queries + edge_queries
    print(f"Running {len(all_queries)} queries ({len(ticket_queries)} ticket, {len(edge_queries)} edge-case)")

    records = []
    for i, row in enumerate(all_queries, start=1):
        expected = row["expected_page_id"]
        expected_list = expected if isinstance(expected, list) else [expected]
        expected_list = [e for e in expected_list if e is not None]

        results = hybrid_search_reranked(row["query"], collection, model, reranker, top_n=5)
        if not results:
            top_score = None
            top_correct = False
            top_category = None
        else:
            top_score = results[0]["rerank_score"]
            top_correct = results[0]["chunk"]["page_id"] in expected_list
            # Category of the TOP-RANKED result itself, not the expected
            # answer's category -- this is what the calibration curve is
            # actually conditioning on (the score Step 8 sees at decision
            # time), and it's already on the chunk from reranking, so no
            # separate join/lookup is needed. For an incorrect top result,
            # this correctly reflects "what category did the system
            # confidently-but-wrongly guess into," which is the more
            # useful slice for catching a category-specific confuser
            # problem (see ADR-003) than the expected answer's category
            # would be.
            top_category = results[0]["chunk"].get("category")

        records.append({
            "query_id": row["query_id"],
            "unanswerable": row["unanswerable"],
            "top_score": top_score,
            "top_correct": top_correct,
            "category": top_category,
        })
        if i % 20 == 0:
            print(f"  {i}/{len(all_queries)} done")

    out_path = DATA_PROCESSED / "confidence_calibration.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(records)} records to {out_path}")

    # Quick summary: bucket by score range, show correctness rate per bucket.
    scored = [r for r in records if r["top_score"] is not None and not r["unanswerable"]]
    scored.sort(key=lambda r: r["top_score"])
    print(f"\n{len(scored)} answerable queries with a score (excludes {len(records)-len(scored)} unanswerable/no-result)")

    buckets = [(-100, -5), (-5, -2), (-2, 0), (0, 2), (2, 5), (5, 100)]
    print("\nScore range -> count, correct, accuracy")
    for lo, hi in buckets:
        in_bucket = [r for r in scored if lo <= r["top_score"] < hi]
        if not in_bucket:
            continue
        correct = sum(1 for r in in_bucket if r["top_correct"])
        print(f"  [{lo:>5}, {hi:>5}): n={len(in_bucket):3d}  correct={correct:3d}  accuracy={correct/len(in_bucket):.1%}")

    # Per-category breakdown, added 2026-10-15 -- the aggregate curve above
    # was the ONLY calibration view this project had until now, which begs
    # the question an architecture review raised directly: is a single
    # global 2.0 threshold actually valid across categories, or is it
    # dominated by whichever category happens to contain the worst
    # confuser pairs in this specific 112-query sample? This slices the
    # SAME threshold (2.0, unchanged) against each category's own
    # above/below-threshold accuracy, so a category where 2.0 is too loose
    # (or needlessly strict) becomes visible instead of averaged away.
    print("\n--- Per-category accuracy at the current threshold (2.0) ---")
    from collections import defaultdict
    by_category = defaultdict(list)
    for r in scored:
        by_category[r["category"] or "(none -- empty result set)"].append(r)

    print(f"{'Category':<34} {'n':>4}  {'below 2.0 acc':>14}  {'at/above 2.0 acc':>17}")
    for category in sorted(by_category):
        rows = by_category[category]
        below = [r for r in rows if r["top_score"] < 2.0]
        above = [r for r in rows if r["top_score"] >= 2.0]
        below_acc = f"{sum(r['top_correct'] for r in below)/len(below):.1%} (n={len(below)})" if below else "n/a"
        above_acc = f"{sum(r['top_correct'] for r in above)/len(above):.1%} (n={len(above)})" if above else "n/a"
        print(f"{category:<34} {len(rows):>4}  {below_acc:>14}  {above_acc:>17}")


if __name__ == "__main__":
    main()
