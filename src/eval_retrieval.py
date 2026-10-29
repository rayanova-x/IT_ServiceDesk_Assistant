"""
Retrieval-only eval harness: does hybrid_search() actually return the right
document? Follows the methodology from PROJECT_HANDOFF.md Section 8 step 3,
Anthropic's own contextual-retrieval eval writeup (recall@k as the primary
metric, tested per query, not blended into one number), and the two
reference_docs files -- decompose the RAG system and measure retrieval
BEFORE generation exists at all.

Two query sources, scored separately then combined:
  1. 50 real employee ticket descriptions (one per subcategory) --
     data/raw/enriched_tickets.csv. Realistic content, but all clean,
     well-formed text (see PROJECT_HANDOFF.md -- the ticket generator never
     produced messy/edge-case text).
  2. 67 hand-written edge-case queries -- data/processed/eval_edge_case_queries.jsonl.
     Covers the messy query types (A-G in the handoff) that real tickets
     don't. 5 of these (G02-G06) are multi-turn conversation tests that
     need Step 0 (not yet built) to make sense -- they are EXCLUDED from
     scoring and reported separately as blocked, not silently dropped.

Metrics computed (per Anthropic's methodology + rag_evals.md/rag_evals2.md):
  - Recall@5: is the expected page among the top 5 hybrid_search() results?
  - Top-1 accuracy: is the expected page the #1 result?
  - MRR: 1/rank of the first correct hit, 0 if not found in top 5.
  - Hit Rate@5 (added 2026-09-18, per rag_evals.md 4.1's metric table):
    binary per-query "was at least one relevant chunk in the top 5" --
    HONEST NOTE: this eval set labels exactly ONE expected_page_id per
    query (occasionally a short list of interchangeable alternates, see
    expected_list below), so Hit Rate@5 is mathematically IDENTICAL to
    Recall@5 here (recall of a single relevant item in-or-out of top-k
    IS the hit/miss binary). Computed and reported as its own named
    field anyway, since rag_evals.md treats them as distinct metrics and
    a reader comparing this project against that reference shouldn't
    have to re-derive the equivalence themselves -- but this is NOT a
    second independent signal, and the summary says so explicitly.
  - Precision@5 (added 2026-09-18): of the 5 retrieved chunks, how many
    are relevant -- with exactly one relevant page per query, this is
    1/5 on a hit and 0/5 on a miss (Precision@5 = Recall@5 / 5 with a
    single-relevant-doc eval set). Reported for completeness against
    rag_evals.md's metric table, same honesty caveat as Hit Rate above.
  - NDCG@5 (added 2026-09-18): unlike the two metrics above, this is
    NOT a trivial rescaling of Recall@5 -- it's position-sensitive via
    log2 discounting (1/log2(rank+1), normalized against the ideal
    ranking's DCG of 1.0 since there's one relevant doc), so a hit at
    rank 1 (NDCG=1.0) and a hit at rank 5 (NDCG=1/log2(6)=~0.39) score
    differently even though both count identically toward Recall@5/Hit
    Rate@5. This is the metric in rag_evals.md's table that actually
    rewards ranking QUALITY, not just presence in the top-k.
  - Confuser-trap rate: for queries with a known confuser_trap_page_id,
    how often does the DECOY outrank the real answer at #1? This is the
    metric standard recall/MRR numbers are blind to (a decoy in the top 5
    still counts as "recall achieved" even if it beat the real answer).
  - Unanswerable handling: for queries marked unanswerable=True, there is
    no generation step yet to test "did it say I don't know" -- this
    script only reports whether a confusable/superseded/no-KB-match page
    got retrieved at all, as a proxy signal for later.

NOT implemented (deliberately, not an oversight): Context Relevance and
Citation Accuracy from rag_evals.md 4.2/rag_evals2.md Layer 2 are left
out of THIS script -- Context Relevance is conventionally LLM-judged
(same category as Faithfulness/Answer Relevance, already covered by
judge_generation_eval.py, not duplicated here), and Citation Accuracy
needs a real generated answer + its citations to check, which doesn't
exist at the retrieval-only stage this script tests. See
capture_generation_eval_data.py / judge_generation_eval.py for where
citation accuracy is actually computed (deterministically, no LLM call --
see citation_accuracy() below, wired into the live per-turn logging path
in session_store.py instead of a new offline script, since the underlying
data -- a turn's citations vs. its answer text -- only exists post-generation).

Run: python src/eval_retrieval.py
"""

import argparse
import json
import math
import sys
from pathlib import Path

import chromadb
import pandas as pd
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

SRC_DIR = Path(__file__).parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from retrieval_pipeline import hybrid_search, hybrid_search_reranked  # noqa: E402
from config import CHROMA_DB_PATH, MODEL_CACHE_DIR, DATA_DIR  # noqa: E402
import eval_split  # noqa: E402

DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"


def load_confluence_lookup():
    with open(DATA_PROCESSED / "confluence_export.jsonl", encoding="utf-8") as f:
        pages = [json.loads(line) for line in f]
    with open(DATA_RAW / "rag_corpus_full.jsonl", encoding="utf-8") as f:
        docs = [json.loads(line) for line in f]
    subcat_to_page = {
        d["subcategory"]: p["id"]
        for p, d in zip(pages, docs)
        if d["doc_type"] == "current_sop"
    }
    return subcat_to_page


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
            "source": "ticket",
            "query": row["Description"],
            "expected_page_id": expected,
            "confuser_trap_page_id": None,
            "unanswerable": False,
        })
    return queries


def load_edge_case_queries() -> tuple[list[dict], list[dict]]:
    with open(DATA_PROCESSED / "eval_edge_case_queries.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    scorable, blocked = [], []
    for r in rows:
        if r["letter"] == "G" and r["query_id"] != "G01":
            blocked.append(r)
            continue
        scorable.append({
            "query_id": r["query_id"],
            "source": "edge_case",
            "query": r["query"],
            "expected_page_id": r["expected_page_id"],
            "confuser_trap_page_id": r.get("confuser_trap_page_id"),
            "unanswerable": r["unanswerable"],
        })
    return scorable, blocked


def load_error_code_queries() -> list[dict]:
    """
    2026-09-21: eval coverage for the error-code spreadsheet source
    (ingest_error_code_spreadsheet.py) -- 22 exact-code queries (2 per
    code, 11 codes) plus 3 deliberately ambiguous queries that should
    still favor the general Confluence SOP page over any spreadsheet
    row. Built AFTER finding the original 3-query/1-code coverage was
    too thin to support any real retrieval-quality claim -- see
    PROJECT_HANDOFF.md's "Multi-source ingestion" writeup for why.
    """
    with open(DATA_PROCESSED / "eval_error_code_queries.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return [
        {
            "query_id": r["query_id"],
            "source": "error_code",
            "query": r["query"],
            "expected_page_id": r["expected_page_id"],
            "confuser_trap_page_id": None,
            "unanswerable": r["unanswerable"],
        }
        for r in rows
    ]


def load_slack_thread_queries() -> list[dict]:
    """2026-09-22: eval coverage for the Slack-thread source
    (ingest_slack_threads.py) -- 2 queries per thread (natural phrasing
    + a peer-question phrasing), one per thread's expected page_id."""
    with open(DATA_PROCESSED / "eval_slack_thread_queries.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return [
        {
            "query_id": r["query_id"],
            "source": "slack_thread",
            "query": r["query"],
            "expected_page_id": r["expected_page_id"],
            "confuser_trap_page_id": None,
            "unanswerable": r["unanswerable"],
        }
        for r in rows
    ]


def score_query(query_row: dict, search_fn) -> dict:
    query_text = query_row["query"]
    expected = query_row["expected_page_id"]

    results = search_fn(query_text)
    retrieved_page_ids = [r["chunk"]["page_id"] for r in results]

    expected_list = expected if isinstance(expected, list) else [expected]
    expected_list = [e for e in expected_list if e is not None]

    hit_rank = None
    for rank, pid in enumerate(retrieved_page_ids, start=1):
        if pid in expected_list:
            hit_rank = rank
            break

    recall_at_5 = hit_rank is not None
    top_1_correct = retrieved_page_ids[0] in expected_list if retrieved_page_ids else False
    mrr = (1.0 / hit_rank) if hit_rank else 0.0

    # See module docstring: with exactly one relevant page per query,
    # hit_rate_at_5 and precision_at_5 are DERIVED from recall_at_5/
    # hit_rank, not independent signals -- computed anyway to report
    # against rag_evals.md's named metrics honestly (see docstring).
    hit_rate_at_5 = recall_at_5
    n_retrieved = len(retrieved_page_ids)
    precision_at_5 = (1 / n_retrieved) if (recall_at_5 and n_retrieved) else 0.0
    # NDCG@5: DCG of this ranking (log2-discounted single relevant hit)
    # over the ideal ranking's DCG (relevant hit at rank 1 -> 1/log2(2)=1.0).
    # This DOES vary independently of recall/hit-rate -- see docstring.
    ndcg_at_5 = (1.0 / math.log2(hit_rank + 1)) if hit_rank else 0.0

    trap = query_row.get("confuser_trap_page_id")
    trap_beat_real = False
    if trap and retrieved_page_ids:
        trap_rank = retrieved_page_ids.index(trap) + 1 if trap in retrieved_page_ids else None
        if trap_rank is not None and (hit_rank is None or trap_rank < hit_rank):
            trap_beat_real = True

    return {
        "query_id": query_row["query_id"],
        "source": query_row["source"],
        "unanswerable": query_row["unanswerable"],
        "expected_page_id": expected,
        "confuser_trap_page_id": trap,
        "retrieved_page_ids": retrieved_page_ids,
        "hit_rank": hit_rank,
        "recall_at_5": recall_at_5,
        "top_1_correct": top_1_correct,
        "mrr": mrr,
        "hit_rate_at_5": hit_rate_at_5,
        "precision_at_5": precision_at_5,
        "ndcg_at_5": ndcg_at_5,
        "trap_beat_real": trap_beat_real,
    }


def summarize(results: list[dict], label: str):
    scorable = [r for r in results if not r["unanswerable"]]
    n = len(scorable)
    if n == 0:
        print(f"\n=== {label}: no scorable queries ===")
        return

    recall_5 = sum(r["recall_at_5"] for r in scorable) / n
    top_1 = sum(r["top_1_correct"] for r in scorable) / n
    mrr = sum(r["mrr"] for r in scorable) / n
    hit_rate_5 = sum(r["hit_rate_at_5"] for r in scorable) / n
    precision_5 = sum(r["precision_at_5"] for r in scorable) / n
    ndcg_5 = sum(r["ndcg_at_5"] for r in scorable) / n

    trap_queries = [r for r in scorable if r["confuser_trap_page_id"]]
    trap_rate = (
        sum(r["trap_beat_real"] for r in trap_queries) / len(trap_queries)
        if trap_queries else None
    )

    print(f"\n=== {label} (n={n}) ===")
    print(f"  Recall@5:      {recall_5:.1%}")
    print(f"  Top-1 accuracy: {top_1:.1%}")
    print(f"  MRR:           {mrr:.3f}")
    print(f"  Hit Rate@5:    {hit_rate_5:.1%}  (identical to Recall@5 -- single relevant doc per query, see docstring)")
    print(f"  Precision@5:   {precision_5:.1%}  (= Recall@5 / 5, single relevant doc per query)")
    print(f"  NDCG@5:        {ndcg_5:.3f}  (position-sensitive, NOT derived from the above)")
    if trap_rate is not None:
        print(f"  Confuser-trap rate (decoy beat real answer, n={len(trap_queries)}): {trap_rate:.1%}")

    misses = [r for r in scorable if not r["recall_at_5"]]
    if misses:
        print(f"  Missed entirely (not in top 5): {[r['query_id'] for r in misses]}")

    traps_won = [r for r in scorable if r["trap_beat_real"]]
    if traps_won:
        print(f"  Decoy beat real answer: {[r['query_id'] for r in traps_won]}")


def run_pass(label: str, queries: list[dict], search_fn, out_filename: str):
    all_results = [score_query(q, search_fn) for q in queries]
    ticket_results = [r for r in all_results if r["source"] == "ticket"]
    edge_results = [r for r in all_results if r["source"] == "edge_case"]
    error_code_results = [r for r in all_results if r["source"] == "error_code"]
    slack_results = [r for r in all_results if r["source"] == "slack_thread"]

    print(f"\n{'#' * 70}\n# {label}\n{'#' * 70}")
    summarize(ticket_results, "Ticket-based queries (clean, real-world text)")
    summarize(edge_results, "Edge-case queries (messy/adversarial text)")
    if error_code_results:
        summarize(error_code_results, "Error-code spreadsheet queries (second ingestion source)")
    if slack_results:
        summarize(slack_results, "Slack-thread queries (third ingestion source)")
    summarize(all_results, "COMBINED")

    out_path = DATA_PROCESSED / out_filename
    with open(out_path, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nFull per-query results saved to {out_path}")
    return all_results


def main():
    ap = argparse.ArgumentParser(description="Retrieval-only eval over a frozen dev/heldout split.")
    # Default is dev: tune against dev, run heldout once per finished change.
    # Non-"all" runs write split-suffixed files so they never overwrite the
    # "all" outputs the dashboard reads (see eval_split.out_name).
    eval_split.add_split_argument(ap)
    args = ap.parse_args()

    print("Loading models and Chroma collection...")
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    collection = client.get_collection("it_helpdesk_kb")
    model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", cache_dir=str(MODEL_CACHE_DIR))
    # Deliberately hardcoded, NOT config.RERANKER_MODEL_NAME: this script
    # defines what every golden-set/retrieval-eval number in this project
    # MEANS. A deployment can run a lighter reranker for RAM reasons (see
    # config.py's docstring), but the eval harness itself must stay pinned
    # to the model those existing numbers were actually measured against,
    # or re-running this would silently produce numbers that look
    # comparable but aren't.
    reranker = TextCrossEncoder(model_name="BAAI/bge-reranker-base", cache_dir=str(MODEL_CACHE_DIR))

    subcat_to_page = load_confluence_lookup()
    ticket_queries = build_ticket_queries(subcat_to_page)
    edge_queries, blocked_queries = load_edge_case_queries()
    error_code_queries = load_error_code_queries()
    slack_thread_queries = load_slack_thread_queries()
    every_query = ticket_queries + edge_queries + error_code_queries + slack_thread_queries

    print(f"Loaded {len(ticket_queries)} ticket-based queries, "
          f"{len(edge_queries)} scorable edge-case queries, "
          f"{len(error_code_queries)} error-code spreadsheet queries, "
          f"{len(slack_thread_queries)} Slack-thread queries, "
          f"{len(blocked_queries)} blocked (Step 0 not built).")

    all_queries = eval_split.select(every_query, args.split)
    print(f"Split '{args.split}': running {len(all_queries)} of {len(every_query)} queries.")

    baseline_fn = lambda q: hybrid_search(q, collection, model, top_n=5)
    reranked_fn = lambda q: hybrid_search_reranked(q, collection, model, reranker, top_n=5)

    run_pass(f"BASELINE [{args.split}]: hybrid_search (RRF fusion only, no reranker)",
              all_queries, baseline_fn, eval_split.out_name("retrieval_eval_results.jsonl", args.split))
    run_pass(f"RERANKED [{args.split}]: hybrid_search_reranked (BAAI/bge-reranker-base)",
              all_queries, reranked_fn, eval_split.out_name("retrieval_eval_results_reranked.jsonl", args.split))

    if blocked_queries:
        print(f"\n=== BLOCKED (need Step 0 conversation contextualization, not yet built) ===")
        for r in blocked_queries:
            print(f"  {r['query_id']}: {r['subtype']}")


if __name__ == "__main__":
    main()
