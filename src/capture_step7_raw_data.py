"""
Captures the RAW search_results (with real rerank scores + full chunk
metadata) + wants_latest for the Step 7 stratified sample, ONE TIME,
via the real pipeline (needs Gemini for routing/understanding -- Steps
2/3 -- but retrieval/reranking themselves, Steps 5/6.5, are fully local).

WHY THIS EXISTS (2026-09-14): the first Step 7 before/after test
(eval_step7_recency.py) only saved final ranks, not the underlying
search_results -- so re-testing a different RECENCY_WEIGHT value would
have required re-running the whole pipeline (and re-spending Gemini
quota) for every single weight tried. This script separates the
EXPENSIVE part (Gemini calls, run once) from the CHEAP part (weight
tuning, pure math on cached data, run as many times as needed for free)
-- see sweep_step7_weights.py, which reads this script's output and
needs zero API calls.

Run: python src/capture_step7_raw_data.py
"""

import json
import time
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
import sys  # noqa: E402
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import PipelineClients, run_pipeline  # noqa: E402

REPO_ROOT = PROJECT_ROOT.parent
DATA_PROCESSED = REPO_ROOT / "data" / "processed"

RECENCY_TARGETED = ["D03", "F03", "F05", "F07", "F09", "G06"]
GENERAL_SAMPLE = ["A01", "A02", "B02", "B06", "C03", "C05", "E01", "E10", "F10"]


def main():
    with open(DATA_PROCESSED / "eval_edge_case_queries.jsonl", encoding="utf-8") as f:
        rows = {r["query_id"]: r for r in (json.loads(line) for line in f)}

    sample_ids = RECENCY_TARGETED + GENERAL_SAMPLE
    print(f"Capturing raw data for {len(sample_ids)} queries (Gemini calls happen ONCE here)")

    print("Building pipeline clients (loading models)...")
    clients = PipelineClients.build()

    out_path = DATA_PROCESSED / "step7_raw_capture.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for i, qid in enumerate(sample_ids, start=1):
            row = rows[qid]
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
            expected = row["expected_page_id"]
            expected_list = expected if isinstance(expected, list) else [expected]
            expected_list = [e for e in expected_list if e is not None]

            record = {
                "query_id": qid,
                "targeted": qid in RECENCY_TARGETED,
                "wants_latest": outcome.wants_latest,
                "expected_page_ids": expected_list,
                "search_results": [
                    {
                        "chunk_id": sr["chunk_id"],
                        "rerank_score": sr["rerank_score"],
                        "page_id": sr["chunk"]["page_id"],
                        "title": sr["chunk"]["title"],
                        "last_updated": sr["chunk"]["last_updated"],
                        "category": sr["chunk"]["category"],
                    }
                    for sr in r.search_results
                ],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(sample_ids)}] {qid:6s} captured, wants_latest={outcome.wants_latest}, "
                  f"{len(record['search_results'])} results")
            time.sleep(15)

    print(f"\nSaved raw capture to {out_path}")
    print("Now run sweep_step7_weights.py to test different RECENCY_WEIGHT values -- no API calls needed.")


if __name__ == "__main__":
    main()
