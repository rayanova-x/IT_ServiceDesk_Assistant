"""
Full end-to-end pipeline eval: runs every real eval query through
run_pipeline() (Steps 1->2->3->4->5) + generate_answer() (Step 9), not
just retrieval in isolation. Checks the WHOLE chain, since a bug in how
one step hands its output to the next wouldn't show up in per-step tests
(see the F05/F09 eval-label bugs, both found only via full pipeline runs
-- PROJECT_HANDOFF.md Section 8).

Two things this checks that eval_retrieval.py does NOT:
  1. Routing correctness: off-topic queries should stop at Step 2, never
     reach retrieval/generation.
  2. Generation-mode sanity: low_confidence results should ALWAYS produce
     mode="candidates_only" (never a synthesized answer -- code-enforced
     in step9_generation.py, verified here it actually holds across real
     queries, not just the hand-picked smoke test).
  3. Citation completeness: every citation must carry last_updated/status
     (Layer 2 staleness disclosure must never silently go missing).

Retrieval correctness (does the right page_id show up) is scored the
same way as eval_retrieval.py, against the same expected_page_id/
confuser_trap_page_id ground truth -- but running through the REAL
pipeline (Step 3's Gemini call rewrites the query and picks a category,
Step 4 decides the filter) rather than calling hybrid_search directly on
the raw query text. This can differ from eval_retrieval.py's numbers,
since retrieval now runs on a Gemini-rewritten query + a possibly
category-filtered search, not the raw query unfiltered.

Run: python src/eval_pipeline_e2e.py
"""

import json
import re
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
import sys  # noqa: E402
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import PipelineClients, run_pipeline  # noqa: E402
from step9_generation import generate_answer  # noqa: E402
import eval_split  # noqa: E402

REPO_ROOT = PROJECT_ROOT.parent
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
DATA_RAW = REPO_ROOT / "data" / "raw"


def load_eval_queries():
    with open(DATA_PROCESSED / "eval_edge_case_queries.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    scorable = [r for r in rows if not (r["letter"] == "G" and r["query_id"] != "G01")]
    blocked = [r for r in rows if r["letter"] == "G" and r["query_id"] != "G01"]
    return scorable, blocked


def _base_metrics(row: dict) -> dict:
    """A consistent, JSON-serializable metrics dict every check_query()
    branch fills in, so downstream summary code never has to guard for
    missing keys depending on which branch a query happened to take."""
    return {
        "query_id": row["query_id"],
        "letter": row["letter"],
        "unanswerable": row["unanswerable"],
        "issues": [],
        "stopped_at": None,
        "hit_rank": None,
        "trap_beat_real": None,  # None = no trap defined for this query
        "modes": [],  # generation mode per sub-query, e.g. ["synthesized"]
    }


def check_query(row: dict, clients: PipelineClients) -> dict:
    expected = row["expected_page_id"]
    trap = row.get("confuser_trap_page_id")
    expected_list = expected if isinstance(expected, list) else [expected]
    expected_list = [e for e in expected_list if e is not None]

    metrics = _base_metrics(row)
    issues = metrics["issues"]

    outcome = run_pipeline(row["query"], clients)
    metrics["stopped_at"] = outcome.stopped_at

    # Off-topic routing check: E06 is our one known off-topic query.
    if row["subtype"] == "off_topic_guardrail":
        if outcome.stopped_at != "off_topic":
            issues.append(f"expected off_topic routing, got stopped_at={outcome.stopped_at!r}")
        return metrics

    if outcome.stopped_at != "ok":
        if not row["unanswerable"]:
            issues.append(f"unexpectedly stopped at {outcome.stopped_at!r} for an answerable query")
        return metrics

    hit_rank = None
    trap_beat_real = False if trap else None
    for r in outcome.results:
        retrieved_ids = [sr["chunk"]["page_id"] for sr in r.search_results]
        for rank, pid in enumerate(retrieved_ids, start=1):
            if pid in expected_list and hit_rank is None:
                hit_rank = rank
        if trap and trap in retrieved_ids:
            trap_rank = retrieved_ids.index(trap) + 1
            if hit_rank is None or trap_rank < hit_rank:
                trap_beat_real = True

        gen = generate_answer(r.sub_query, r.search_results, r.low_confidence, client=clients.gemini)
        metrics["modes"].append(gen.mode)

        # Generation-mode sanity check: low_confidence must NEVER produce
        # a synthesized answer -- this must hold in code, not just in the
        # smoke test.
        if r.low_confidence and gen.mode != "candidates_only":
            issues.append(f"low_confidence=True but generation mode was {gen.mode!r}, not candidates_only")

        # Citation completeness: every citation must carry last_updated/status.
        for c in gen.citations:
            if not c.last_updated or not c.status:
                issues.append(f"citation for page {c.page_id} missing last_updated/status")

    metrics["hit_rank"] = hit_rank
    metrics["trap_beat_real"] = trap_beat_real
    return metrics


def _metrics_path(out_filename: str) -> Path:
    """
    STRUCTURED METRICS (added 2026-09-14 after realizing the plain-text
    log only ever recorded OK/ISSUES per query, never the actual
    hit_rank/recall data needed to report a real combined-N accuracy
    number -- the only numbers that existed were partial, printed to the
    console for whichever SUBSET happened to run in one invocation (e.g.
    18 of 62 after a quota-driven resume), never the true full-set
    number. From now on: every run appends one JSONL record per query
    with everything needed to recompute recall@5/top-1/trap-rate across
    ALL queries ever recorded, not just the current invocation's subset --
    see _write_summary(), which reads this file, not in-memory state.
    """
    return DATA_PROCESSED / out_filename.replace(".txt", "_metrics.jsonl")


def _load_already_done(out_path: Path) -> set[str]:
    """
    RESUMABLE RUNS (added 2026-09-14 after hitting Gemini's free-tier
    DAILY quota -- 500 requests/day -- partway through a 62-query run;
    the per-minute pacing/retry fixes don't help against a daily cap, so
    the run has to be split across days without re-spending quota on
    already-completed queries). Parses query_ids already recorded in a
    prior (possibly partial) results file so main() can skip them.
    """
    done = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                m = re.match(r"\[\d+/\d+\]\s+(\S+)\s+", line)
                if not m:
                    continue
                # A quota/API failure ("FAILED after N attempts") is NOT
                # a genuine result -- don't skip it on resume, it needs
                # to actually run once the daily quota resets.
                if "FAILED after" in line:
                    continue
                done.add(m.group(1))
    return done


def main(query_ids: list[str] | None = None, out_filename: str = "eval_pipeline_e2e_results.txt",
         resume: bool = True, split: str = "all"):
    """
    split: "dev" / "heldout" / "all" -- see eval_split.py. The function
    default stays "all" so existing programmatic callers are unchanged; the
    CLI below defaults to "dev". Non-"all" runs write split-suffixed result
    files, so each split keeps its own resumable progress.

    query_ids: if given, only run these specific query_ids (for a fast
    stratified-sample test before committing to the full 62). Each query
    does up to 3 Gemini calls (routing, understanding, generation) against
    the free tier's 15 requests/minute limit -- pace conservatively
    (~15s between queries, i.e. ~4 calls/minute effective) rather than
    the 4.5s used for single-call scripts like step2_batch_test.py, since
    the SDK's own internal retry-with-backoff on 429s can silently stack
    up multi-minute delays per query if paced too aggressively (see
    PROJECT_HANDOFF.md Section 6 for this lesson, found 2026-09-14).

    resume: if True (default), skip query_ids already recorded in
    out_filename from a prior run -- needed because the free tier's
    DAILY quota (500 req/day) can be exhausted mid-run, and re-running
    already-done queries the next day would waste quota re-testing what
    already passed.
    """
    scorable, blocked = load_eval_queries()
    scorable = eval_split.select(scorable, split)
    out_filename = eval_split.out_name(out_filename, split)
    if query_ids:
        scorable = [r for r in scorable if r["query_id"] in query_ids]

    out_path = DATA_PROCESSED / out_filename
    already_done = _load_already_done(out_path) if resume else set()
    if already_done:
        print(f"Resuming: {len(already_done)} queries already done in {out_path.name}, skipping those.")
        scorable = [r for r in scorable if r["query_id"] not in already_done]

    print(f"Running {len(scorable)} queries, {len(blocked)} blocked (Step 0 not built)")
    if not scorable:
        print("Nothing left to run.")
        return

    print("Building pipeline clients (loading models)...")
    clients = PipelineClients.build()

    metrics_path = _metrics_path(out_filename)
    text_mode = "a" if already_done else "w"
    metrics_mode = "a" if already_done else "w"
    with open(out_path, text_mode, encoding="utf-8") as out, \
         open(metrics_path, metrics_mode, encoding="utf-8") as mout:
        for i, row in enumerate(scorable, start=1):
            t0 = time.time()
            # Retry the WHOLE query (not just one call) on transient API
            # errors (429 rate limit, 503 server unavailable) -- a
            # multi-call query (routing + understanding + generation) can
            # fail partway through, and re-running the whole thing is
            # simpler and safer than trying to resume mid-query. A single
            # 503 blip previously crashed the entire 62-query run at
            # query 4 (2026-09-14) -- don't let one transient error lose
            # 20+ minutes of prior progress.
            for attempt in range(4):
                try:
                    result = check_query(row, clients)
                    break
                except Exception as e:
                    if attempt == 3:
                        result = _base_metrics(row)
                        result["issues"] = [f"FAILED after 4 attempts: {e}"]
                        break
                    wait = 10 * (attempt + 1)
                    print(f"  ({row['query_id']}: {type(e).__name__}, retrying in {wait}s...)", flush=True)
                    time.sleep(wait)
            elapsed = time.time() - t0
            status = "OK" if not result["issues"] else f"ISSUES: {result['issues']}"
            line = f"[{i}/{len(scorable)}] {result['query_id']:6s} {status}  ({elapsed:.1f}s)"
            print(line, flush=True)
            out.write(line + "\n")
            out.flush()
            # Structured metrics record -- see _metrics_path() docstring.
            # A quota/API failure is marked failed=True and excluded from
            # the summary's scoring (it isn't a real result), but IS
            # excluded from _load_already_done() too so it gets re-run.
            record = dict(result)
            record["failed"] = "FAILED after" in str(result["issues"])
            mout.write(json.dumps(record, ensure_ascii=False) + "\n")
            mout.flush()
            time.sleep(15)

    _print_summary(metrics_path)


def _print_summary(metrics_path: Path):
    """
    Computes and prints the summary from EVERY record ever saved to
    metrics_path, not just the current invocation's subset -- this is
    what makes the reported numbers correct after a quota-driven resume
    (e.g. 44 queries run today, 18 tomorrow: the summary after tomorrow's
    run must reflect all 62, not just tomorrow's 18). Keeps only the
    LAST record per query_id (a re-run after a quota failure produces a
    second, newer record for the same query_id).
    """
    if not metrics_path.exists():
        print("No metrics file found.")
        return

    latest_by_qid = {}
    with open(metrics_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("failed"):
                continue  # not a real result, don't let it shadow a later real one
            latest_by_qid[rec["query_id"]] = rec

    all_results = list(latest_by_qid.values())
    print("\n=== SUMMARY (all queries ever recorded in this metrics file) ===")
    total = len(all_results)
    with_issues = [r for r in all_results if r["issues"]]
    print(f"Total queries: {total}")
    print(f"Queries with issues: {len(with_issues)}")
    for r in with_issues:
        print(f"  {r['query_id']}: {r['issues']}")

    scored = [r for r in all_results if r["stopped_at"] == "ok" and not r["unanswerable"]]
    recall5 = sum(1 for r in scored if r["hit_rank"] is not None) / len(scored) if scored else 0
    top1 = sum(1 for r in scored if r["hit_rank"] == 1) / len(scored) if scored else 0
    trap_defined = [r for r in scored if r["trap_beat_real"] is not None]
    trap_won = [r for r in trap_defined if r["trap_beat_real"]]
    print(f"\nRecall@5 (through full pipeline, category-filtered where applicable): {recall5:.1%} (n={len(scored)})")
    print(f"Top-1 accuracy: {top1:.1%} (n={len(scored)})")
    if trap_defined:
        print(f"Confuser-trap rate: {len(trap_won)}/{len(trap_defined)} decoy beat real answer: {[r['query_id'] for r in trap_won]}")


STRATIFIED_SAMPLE = [
    "A01", "A02", "B02", "B06", "C03", "C05", "D05", "D04",
    "E10", "E01", "F10", "F03", "G01",
]

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="fast 13-query stratified smoke run")
    eval_split.add_split_argument(ap, default=None)
    args = ap.parse_args()
    # --sample is a hand-picked smoke test across all splits; keep it that way
    # unless --split is given explicitly. A full run defaults to dev.
    split = args.split or ("all" if args.sample else "dev")
    if args.sample:
        main(query_ids=STRATIFIED_SAMPLE, out_filename="eval_pipeline_e2e_sample_results.txt", split=split)
    else:
        main(split=split)
