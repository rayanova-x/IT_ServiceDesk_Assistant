"""
Phase 3 experiment: does the hard category filter help or hurt retrieval?

At serving time the router's one-word category is a HARD filter on both the
dense and BM25 search (pipeline.py). It was never measured against unfiltered
search. This runs the same answerable queries three ways, locally (no Gemini):

  none    unfiltered search                       -- what removing the filter gives
  oracle  filtered by the TRUE category of the    -- the best the filter can ever do
          expected page (perfect router)
  wrong   filtered by a deterministically chosen  -- the damage when the router errs
          WRONG category

DECISION RULE (fixed before running, see the plan):
  oracle - none top-1 < 3 points  -> remove the hard filter (category stays a
                                     label for logging / per-category thresholds)
  oracle - none top-1 >= 3 points -> keep the filter, but also run unfiltered
                                     and prefer it when its top rerank score
                                     beats the filtered one by a margin chosen
                                     on dev

Mirrors serving: regex-extracted error codes are passed to the search, and
confidence is checked with step8's real (per-category) thresholds. Pinned to
BAAI/bge-reranker-base like eval_retrieval.py, because those thresholds were
calibrated on it.

Default split is dev. Run on heldout only to confirm a decision already made.

Run: python src/eval_category_filter.py [--split dev|heldout|all]
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))

import chromadb  # noqa: E402
from fastembed import TextEmbedding  # noqa: E402
from fastembed.rerank.cross_encoder import TextCrossEncoder  # noqa: E402

import eval_retrieval as er  # noqa: E402
import eval_split  # noqa: E402
from config import CHROMA_DB_PATH, MODEL_CACHE_DIR  # noqa: E402
from query_parsing import parse_query  # noqa: E402
from retrieval_pipeline import CHUNKS, hybrid_search_reranked  # noqa: E402
from step3_understanding import CATEGORY_VALUES  # noqa: E402
from step8_confidence import check_confidence  # noqa: E402

REAL_CATEGORIES = sorted(c for c in CATEGORY_VALUES if c != "unclear")
DECISION_MARGIN_POINTS = 3.0
MODES = ("none", "oracle", "wrong")


def _wrong_category(true_cat: str, query_id: str) -> str:
    others = [c for c in REAL_CATEGORIES if c != true_cat]
    return others[int(hashlib.sha256(query_id.encode()).hexdigest(), 16) % len(others)]


def _run_mode(row: dict, category: str | None, collection, model, reranker) -> dict:
    stash = {}

    def search_fn(q: str):
        results = hybrid_search_reranked(
            q, collection, model, reranker,
            error_codes=parse_query(q).error_codes, category=category, top_n=5,
        )
        stash["results"] = results
        return results

    scored = er.score_query(row, search_fn)
    conf = check_confidence(stash["results"], category=category)
    return {
        "hit_rank": scored["hit_rank"],
        "top_1_correct": scored["top_1_correct"],
        "recall_at_5": scored["recall_at_5"],
        "mrr": scored["mrr"],
        "top_score": conf.top_score,
        "confident": conf.confident,
        "wrong_but_confident": bool(conf.confident and not scored["top_1_correct"]),
    }


def _summary(rows: list[dict], mode: str) -> dict:
    n = len(rows)
    m = [r[mode] for r in rows]
    return {
        "n": n,
        "recall@5": sum(x["recall_at_5"] for x in m) / n,
        "top-1": sum(x["top_1_correct"] for x in m) / n,
        "mrr": sum(x["mrr"] for x in m) / n,
        "confident": sum(x["confident"] for x in m) / n,
        "wrong_but_confident": sum(x["wrong_but_confident"] for x in m),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Hard category filter vs unfiltered retrieval.")
    eval_split.add_split_argument(ap)
    args = ap.parse_args()

    print("Loading models and Chroma collection...")
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    collection = client.get_collection("it_helpdesk_kb")
    model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", cache_dir=str(MODEL_CACHE_DIR))
    reranker = TextCrossEncoder(model_name="BAAI/bge-reranker-base", cache_dir=str(MODEL_CACHE_DIR))

    subcat_to_page = er.load_confluence_lookup()
    edge, _blocked = er.load_edge_case_queries()
    every = (er.build_ticket_queries(subcat_to_page) + edge
             + er.load_error_code_queries() + er.load_slack_thread_queries())
    queries = eval_split.select(every, args.split)

    page_category = {c["page_id"]: c["category"] for c in CHUNKS}
    rows = []
    skipped = 0
    for q in queries:
        expected = q["expected_page_id"]
        expected_list = [e for e in (expected if isinstance(expected, list) else [expected]) if e is not None]
        true_cat = page_category.get(expected_list[0]) if expected_list else None
        if q["unanswerable"] or true_cat is None:
            skipped += 1  # no true category to filter by -> can't compare modes fairly
            continue
        cats = {"none": None, "oracle": true_cat, "wrong": _wrong_category(true_cat, q["query_id"])}
        row = {"query_id": q["query_id"], "true_category": true_cat, "wrong_category": cats["wrong"]}
        for mode in MODES:
            row[mode] = _run_mode(q, cats[mode], collection, model, reranker)
        rows.append(row)
        if len(rows) % 20 == 0:
            print(f"  {len(rows)} done...")

    print(f"\nSplit '{args.split}': {len(rows)} answerable queries compared ({skipped} skipped: unanswerable or no true category)")
    print(f"\n{'mode':8} {'recall@5':>9} {'top-1':>7} {'MRR':>6} {'confident':>10} {'wrong+confident':>16}")
    summaries = {}
    for mode in MODES:
        s = _summary(rows, mode)
        summaries[mode] = s
        print(f"{mode:8} {s['recall@5']:>9.1%} {s['top-1']:>7.1%} {s['mrr']:>6.3f} {s['confident']:>10.1%} {s['wrong_but_confident']:>16d}")

    helps = [r["query_id"] for r in rows if r["oracle"]["top_1_correct"] and not r["none"]["top_1_correct"]]
    hurts = [r["query_id"] for r in rows if r["none"]["top_1_correct"] and not r["oracle"]["top_1_correct"]]
    broke = [r["query_id"] for r in rows if r["none"]["recall_at_5"] and not r["wrong"]["recall_at_5"]]
    print(f"\nPerfect filter FIXES top-1 on {len(helps)}: {helps}")
    print(f"Perfect filter BREAKS top-1 on {len(hurts)}: {hurts}")
    print(f"A WRONG category loses a top-5 hit that unfiltered had on {len(broke)} of {len(rows)} queries: {broke}")

    delta = (summaries["oracle"]["top-1"] - summaries["none"]["top-1"]) * 100
    print(f"\noracle - none top-1: {delta:+.1f} points (threshold {DECISION_MARGIN_POINTS} points)")
    if delta < DECISION_MARGIN_POINTS:
        print("DECISION: remove the hard filter (a perfect router barely beats no filter, so a real router can only lose).")
    else:
        print("DECISION: keep the filter but verify against an unfiltered search (margin to be chosen on dev).")
    print(f"Error bars are wide at n={len(rows)}; one query is {100/len(rows):.1f} points.")

    out = er.DATA_PROCESSED / eval_split.out_name("category_filter_experiment.jsonl", args.split)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Per-query results saved to {out}")


if __name__ == "__main__":
    main()
