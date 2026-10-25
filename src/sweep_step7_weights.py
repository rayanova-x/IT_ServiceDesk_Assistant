"""
Sweeps multiple RECENCY_WEIGHT values against the cached raw data from
capture_step7_raw_data.py -- ZERO API calls, pure local math, so this can
be re-run as many times as needed while tuning without spending any
Gemini quota (see that script's docstring for why this split exists).

Grounded in the real published finding (arXiv 2509.19376, "Freshness and
the Limits of Heuristic Trend Detection in Temporal RAG", read 2026-09-14):
their formula is score = alpha*relevance + (1-alpha)*recency, and even
their "conservative" default (alpha=0.7, i.e. recency gets 30% weight)
needed CORPUS-SPECIFIC tuning to have any effect on real data -- a fixed
weight that works on one corpus can "collapse to 0.00" effect on another.
So this sweep tests a real range of weights on OUR actual data rather
than assuming any single value transfers from another domain, and goes
into it expecting a real ceiling per that paper's finding that recency
"cannot rescue" a document that's a poor semantic match to begin with --
success here means a measurable, non-zero effect on close competitors,
not a dramatic fix for the confuser-pair problem (a relevance issue, not
a freshness issue).

Run: python src/sweep_step7_weights.py
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
import sys  # noqa: E402
sys.path.insert(0, str(PROJECT_ROOT))

from step7_recency import recency_score  # noqa: E402

REPO_ROOT = PROJECT_ROOT.parent
DATA_PROCESSED = REPO_ROOT / "data" / "processed"

WEIGHTS_TO_TEST = [1.0, 2.0, 3.0, 4.0, 5.0, 8.0]


def apply_weight(search_results: list[dict], wants_latest: bool, weight: float) -> list[dict]:
    if not wants_latest or not search_results:
        return search_results
    scored = []
    for r in search_results:
        rscore = recency_score(r["last_updated"], r.get("category"))
        combined = r["rerank_score"] + weight * rscore
        scored.append({**r, "combined_score": combined})
    return sorted(scored, key=lambda r: -r["combined_score"])


def hit_rank(results: list[dict], expected_list: list[str]) -> int | None:
    for rank, r in enumerate(results, start=1):
        if r["page_id"] in expected_list:
            return rank
    return None


def main():
    capture_path = DATA_PROCESSED / "step7_raw_capture.jsonl"
    if not capture_path.exists():
        print(f"No captured data found at {capture_path}. Run capture_step7_raw_data.py first.")
        return

    records = []
    with open(capture_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    print(f"Loaded {len(records)} captured queries ({sum(1 for r in records if r['wants_latest'])} with wants_latest=True)")

    baseline_rank = {r["query_id"]: hit_rank(r["search_results"], r["expected_page_ids"]) for r in records}
    baseline_top1 = sum(1 for qid, rank in baseline_rank.items() if rank == 1)
    baseline_recall = sum(1 for rank in baseline_rank.values() if rank is not None)
    n = len(records)
    print(f"\nBaseline (RECENCY_WEIGHT=0, i.e. no Step 7): "
          f"top1={baseline_top1}/{n} ({baseline_top1/n:.1%})  "
          f"recall@5={baseline_recall}/{n} ({baseline_recall/n:.1%})")

    print("\n=== Weight sweep ===")
    for weight in WEIGHTS_TO_TEST:
        top1 = 0
        recall = 0
        changed_queries = []
        for r in records:
            reranked = apply_weight(r["search_results"], r["wants_latest"], weight)
            rank = hit_rank(reranked, r["expected_page_ids"])
            if rank == 1:
                top1 += 1
            if rank is not None:
                recall += 1
            if rank != baseline_rank[r["query_id"]]:
                changed_queries.append((r["query_id"], baseline_rank[r["query_id"]], rank))

        print(f"\nweight={weight}: top1={top1}/{n} ({top1/n:.1%})  recall@5={recall}/{n} ({recall/n:.1%})")
        if changed_queries:
            for qid, before, after in changed_queries:
                print(f"    {qid}: rank {before} -> {after}")
        else:
            print("    (no change from baseline)")


if __name__ == "__main__":
    main()
