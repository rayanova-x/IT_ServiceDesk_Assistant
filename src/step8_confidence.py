"""
Step 8 of the pipeline: confidence threshold check + knowledge-gap
logging. Programmatic, no LLM call -- uses the reranker's OWN computed
relevance score (rerank_score from hybrid_search_reranked()), never an
LLM self-reporting confidence in words (see PROJECT_HANDOFF.md Section 5:
self-rated confidence is unreliable, confirmed across multiple sources).

THRESHOLD IS CALIBRATED, NOT GUESSED. Built 2026-09-14 by running all 112
real eval queries (50 ticket + 62 edge-case) through the real reranker and
recording each top result's score against whether it was actually
correct (src/calibrate_confidence.py, data/processed/confidence_calibration.jsonl).
Per the sourced guidance (claudecertificationguide.com 5-5,
human-review-calibration): "run the model, compare its confidence scores
to actual accuracy, and build a calibration curve."

Real result (100 answerable queries with a score):
  score <  0.0: ~45-50% accuracy (n=32)
  score 0.0-2.0: ~38% accuracy (n=21) -- WORSE than the negative band,
    not a smooth curve. Traced to our known confuser-pair weakness
    (VPN-topic queries specifically) landing moderate, not low, scores
    while still being wrong -- see PROJECT_HANDOFF.md Section 8.
  score >= 2.0: 75-80% accuracy (n=47)
There is no reliable accuracy gradient below 2.0 -- scores in [-2, 2) are
NOT meaningfully more trustworthy than scores below -2. So the threshold
is a single cut at 2.0, not a finer-grained scale, because the data
doesn't support finer distinctions below that point.

PER-CATEGORY THRESHOLDS, added 2026-10-15 -- A direct architecture-review
question ("shouldn't calibration be done per-document-type, not
aggregate?") was correct and led to re-slicing the SAME 112-query
calibration run by category (src/calibrate_confidence.py now records
each query's top-ranked category, not just its score). The honest
result, from scanning EVERY candidate threshold per category (not just
2.0), not just re-checking 2.0 in isolation:
  - Software & Applications Support (n=22): the reranker score is a
    genuinely strong, real signal here -- accuracy below ~3.0 is
    17-25%, accuracy at/above it jumps to 70%, a real ~50-point gap
    that holds across a range of nearby candidate cutoffs, not a
    cherry-picked single point. This category gets its OWN threshold.
  - Every other category (Network Infrastructure, Access & Identity
    Management, Hardware & Device Support, Security Operations): the
    gap between below/above-threshold accuracy stayed small, flat, or
    in Security Operations' case INVERTED (lower scores were MORE
    accurate than higher ones) across every threshold tested. Forcing
    a distinct number for these would be fitting noise to look more
    sophisticated, not real calibration -- they correctly keep the
    global 2.0 default, which the data does support as the right
    fallback for them specifically (their own curves don't contradict
    it, they just don't improve on it either).
  - Onboarding & Provisioning, IT Policies & Procedures, Facilities &
    Regional IT: n=1-2 each in this calibration run -- far too thin to
    derive anything from. These also fall back to the global default,
    explicitly because there isn't enough data yet, not because their
    own data was checked and found flat.
This is NOT "every category has been calibrated" -- it's "the one
category where the data actually supports a different number has one,
and the mechanism exists to add more as real usage data accumulates for
the others." See ADR-003/ADR-005 for the full writeup.

This is a SEPARATE, independent low-confidence signal from Step 4's
ambiguity gate (category-level ambiguity). A query can have an
unambiguous, correctly-identified category (Step 4 confident) and still
get weak search results (Step 8 not confident) -- e.g. the category is
right but no document in the KB is actually a good match. Both signals
feed the same downstream behavior (Step 9: show candidates, don't
synthesize) via PipelineResult.low_confidence, which is set to True if
EITHER Step 4 OR Step 8 says so.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from config import DATA_DIR

PROJECT_ROOT = Path(__file__).parent
DATA_PROCESSED = DATA_DIR / "processed"

CONFIDENCE_THRESHOLD = 2.0  # the global fallback -- see module docstring for the calibration data

# Per-category overrides -- ONLY for a category whose own calibration
# curve showed a real, non-noisy accuracy gap at some cutoff (see module
# docstring's "PER-CATEGORY THRESHOLDS" section). A category absent from
# this dict is not an oversight -- it means its own data was checked and
# did not support a number different from CONFIDENCE_THRESHOLD, or there
# wasn't enough data yet to check at all. Never add an entry here without
# the same kind of real before/after accuracy numbers Software &
# Applications Support has -- an unjustified per-category number would be
# strictly worse than the honest global fallback.
CATEGORY_THRESHOLDS: dict[str, float] = {
    "Software & Applications Support": 3.0,  # real curve: ~17-25% below, ~70% at/above -- see calibrate_confidence.py
}


@dataclass
class ConfidenceCheck:
    confident: bool
    top_score: float | None  # None if there were zero results at all
    threshold_used: float  # the actual cutoff applied -- category-specific or the global fallback


def _threshold_for(category: str | None) -> float:
    if category is None:
        return CONFIDENCE_THRESHOLD
    return CATEGORY_THRESHOLDS.get(category, CONFIDENCE_THRESHOLD)


def check_confidence(search_results: list[dict], category: str | None = None) -> ConfidenceCheck:
    """
    search_results: the `search_results` list from a PipelineResult, i.e.
    output of hybrid_search_reranked() -- [{"chunk_id", "rerank_score",
    "chunk"}, ...], already sorted best-first.

    category: Step 4's resolved search_category for this result set (None
    for an ambiguous/unfiltered search, which correctly falls back to the
    global threshold -- an ambiguous query has no single category whose
    calibration curve would even apply). Optional and defaults to None so
    every pre-existing call site that doesn't pass it keeps behaving
    exactly as before this existed.
    """
    threshold = _threshold_for(category)
    if not search_results:
        return ConfidenceCheck(confident=False, top_score=None, threshold_used=threshold)
    top_score = search_results[0]["rerank_score"]
    return ConfidenceCheck(confident=top_score >= threshold, top_score=top_score, threshold_used=threshold)


def log_knowledge_gap(
    query: str,
    sub_query: str,
    category: str | None,
    top_score: float | None,
    reason: str,
    log_path: Path | None = None,
    threshold_used: float | None = None,
) -> None:
    """
    Append-only JSONL log of queries the system couldn't confidently
    answer -- per the sourced agent-assist research (Section 3, Step 2):
    Zendesk/Forethought/Ada all treat a low-confidence/no-match result as
    a CONTENT GAP for the KB owner to review later, not just a dead end
    for the current user. This is a real, useful project feature in its
    own right (a "what does our KB not cover well" report), not just a
    fallback path.

    threshold_used: the actual cutoff this gap was measured against --
    recorded explicitly as of per-category thresholds (2026-10-15) since
    a logged top_score of 2.5 means something different depending on
    whether it was compared against the global 2.0 or Software & Apps
    Support's 3.0. Optional/defaults to None so this stays backward
    compatible with any caller that hasn't been updated to pass it.
    """
    if log_path is None:
        log_path = DATA_PROCESSED / "knowledge_gaps.jsonl"
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "sub_query": sub_query,
        "category": category,
        "top_score": top_score,
        "reason": reason,  # "no_results" | "low_confidence"
        "threshold_used": threshold_used,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))
    from pipeline import PipelineClients, run_pipeline

    print("Building pipeline clients...")
    clients = PipelineClients.build()

    test_messages = [
        "My VPN keeps disconnecting right after I log in, error 0x8007042B.",  # expect confident
        "My VPN keeps disconnecting right after I log in.",  # known confuser case, expect NOT confident
        "What time does the cafeteria close?",  # off-topic, never reaches Step 8
    ]

    # NOTE: run_pipeline() already runs check_confidence() and
    # log_knowledge_gap() internally (see pipeline.py) -- this smoke test
    # just reads the already-computed PipelineResult.low_confidence, it
    # does NOT call either function again itself. An earlier version of
    # this smoke test re-called both, which double-logged every gap to
    # knowledge_gaps.jsonl -- found and fixed 2026-09-14.
    print("\nStep 8 confidence check -- smoke test\n")
    for msg in test_messages:
        outcome = run_pipeline(msg, clients)
        if outcome.stopped_at != "ok":
            print(f"MESSAGE: {msg}\n  stopped_at={outcome.stopped_at}\n")
            continue
        for r in outcome.results:
            print(f"MESSAGE: {msg}")
            print(f"  sub_query={r.sub_query!r} low_confidence={r.low_confidence}")
            if r.low_confidence:
                print("  (already logged as a knowledge gap inside run_pipeline, if Step 8 was the cause)")
            print()
