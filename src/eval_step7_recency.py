"""
Before/after comparison for Step 7 (recency rerank) on a stratified
sample, per explicit user request: build Step 7, then measure real
impact rather than assuming it helps.

Sample = every query specifically designed to test recency/staleness
(explicit "current"/"latest" wording, or a known silently-stale
duplicate page) + a general stratified sample across the other letter
categories, to confirm Step 7 doesn't regress anything unrelated.

For each query: run retrieval WITHOUT Step 7 (baseline, what every prior
eval measured) and WITH Step 7 applied (apply_recency_rerank(), only
actually changes anything when wants_latest=True), compare hit_rank and
top-1 correctness.

Run: python src/eval_step7_recency.py
"""

import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
import sys  # noqa: E402
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import PipelineClients, run_pipeline  # noqa: E402
from step7_recency import apply_recency_rerank  # noqa: E402

REPO_ROOT = PROJECT_ROOT.parent
DATA_PROCESSED = REPO_ROOT / "data" / "processed"

# Every query specifically designed to test recency/staleness.
RECENCY_TARGETED = ["D03", "F03", "F05", "F07", "F09", "G06"]

# General stratified sample across other letters, to check for regressions.
GENERAL_SAMPLE = ["A01", "A02", "B02", "B06", "C03", "C05", "E01", "E10", "F10"]


def hit_rank(results: list[dict], expected_list: list[str]) -> int | None:
    for rank, r in enumerate(results, start=1):
        if r["chunk"]["page_id"] in expected_list:
            return rank
    return None


def main():
    with open(DATA_PROCESSED / "eval_edge_case_queries.jsonl", encoding="utf-8") as f:
        rows = {r["query_id"]: r for r in (json.loads(line) for line in f)}

    sample_ids = RECENCY_TARGETED + GENERAL_SAMPLE
    print(f"Running {len(sample_ids)} queries ({len(RECENCY_TARGETED)} recency-targeted, "
          f"{len(GENERAL_SAMPLE)} general)")

    print("Building pipeline clients (loading models)...")
    clients = PipelineClients.build()

    results = []
    for i, qid in enumerate(sample_ids, start=1):
        row = rows[qid]
        expected = row["expected_page_id"]
        expected_list = expected if isinstance(expected, list) else [expected]
        expected_list = [e for e in expected_list if e is not None]

        for attempt in range(4):
            try:
                outcome = run_pipeline(row["query"], clients)
                break
            except Exception as e:
                if attempt == 3:
                    outcome = None
                    break
                wait = 10 * (attempt + 1)
                print(f"  ({qid}: {type(e).__name__}, retrying in {wait}s...)")
                time.sleep(wait)

        if outcome is None or outcome.stopped_at != "ok" or not outcome.results:
            print(f"[{i}/{len(sample_ids)}] {qid:6s} SKIPPED (stopped_at={outcome.stopped_at if outcome else 'FAILED'})")
            time.sleep(15)
            continue

        r = outcome.results[0]
        before_rank = hit_rank(r.search_results, expected_list)
        before_top1 = before_rank == 1

        reranked = apply_recency_rerank(r.search_results, outcome.wants_latest)
        after_rank = hit_rank(reranked, expected_list)
        after_top1 = after_rank == 1

        changed = before_rank != after_rank
        line = (f"[{i}/{len(sample_ids)}] {qid:6s} wants_latest={outcome.wants_latest!s:5s} "
                f"before_rank={str(before_rank):4s} after_rank={str(after_rank):4s} "
                f"{'CHANGED' if changed else ''}")
        print(line)

        results.append({
            "query_id": qid, "targeted": qid in RECENCY_TARGETED,
            "wants_latest": outcome.wants_latest,
            "before_rank": before_rank, "after_rank": after_rank,
            "before_top1": before_top1, "after_top1": after_top1,
        })
        time.sleep(15)

    out_path = DATA_PROCESSED / "step7_before_after_results.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(results)} results to {out_path}")

    print("\n=== SUMMARY ===")
    n = len(results)
    before_recall = sum(1 for r in results if r["before_rank"] is not None) / n if n else 0
    after_recall = sum(1 for r in results if r["after_rank"] is not None) / n if n else 0
    before_top1 = sum(1 for r in results if r["before_top1"]) / n if n else 0
    after_top1 = sum(1 for r in results if r["after_top1"]) / n if n else 0
    print(f"Recall@5:      before={before_recall:.1%}  after={after_recall:.1%}")
    print(f"Top-1 accuracy: before={before_top1:.1%}  after={after_top1:.1%}")

    changed = [r for r in results if r["before_rank"] != r["after_rank"]]
    print(f"\nQueries where Step 7 changed the ranking: {len(changed)}/{n}")
    for r in changed:
        print(f"  {r['query_id']}: rank {r['before_rank']} -> {r['after_rank']} (wants_latest={r['wants_latest']})")

    regressions = [r for r in results if r["before_top1"] and not r["after_top1"]]
    if regressions:
        print(f"\nREGRESSIONS (was top-1, now isn't): {[r['query_id'] for r in regressions]}")
    else:
        print("\nNo regressions (no query that was top-1 became not-top-1).")


if __name__ == "__main__":
    main()
