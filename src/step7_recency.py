"""
Step 7 of the pipeline: conditional recency rerank. Only runs if
wants_latest=True (detected by Step 3, verified to survive Step 0's
conversation rewrite too -- see the G06 test case). Programmatic, no LLM
call -- blends a recency score (from version.when, per PROJECT_HANDOFF.md
Section 5: NEVER history.createdDate) additively into the existing
rerank score, never replacing it -- a very recent but irrelevant document
must not outrank a genuinely relevant older one.

PER-CATEGORY HALF-LIVES: the original design (PROJECT_HANDOFF.md Section
3) calls for per-category staleness thresholds, not one universal rate.
We have no real usage data to CALIBRATE category-specific rates from (see
Section 5 -- Step 8's threshold WAS calibrated from 112 real queries;
this one is not, by explicit user decision to still build per-category
rates now, using reasoned assumptions about real-world IT domain change
velocity rather than a single universal curve). Assumptions, not
measurements:
  - Security Operations (90d): threats/policies evolve fastest.
  - Software & Applications Support (120d): app/version updates frequent.
  - Network Infrastructure, Access & Identity Management (180d): infra/
    policy changes periodically.
  - IT Policies & Procedures, Onboarding & Provisioning (270d): formal
    policy, revised less often.
  - Hardware & Device Support, Facilities & Regional IT (365d): slowest-
    changing, troubleshooting/facilities info is fairly timeless.
Sanity-checked against the real chunk age range in this corpus (44-1122
days, median 240) -- these half-lives produce a meaningful score spread
across that range, not values so large/small they're irrelevant (see
conversation history for the checked numbers).
"""

import json
from datetime import date, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
REPO_ROOT = PROJECT_ROOT.parent

CATEGORY_HALF_LIFE_DAYS = {
    "Security Operations": 90,
    "Software & Applications Support": 120,
    "Network Infrastructure": 180,
    "Access & Identity Management": 180,
    "IT Policies & Procedures": 270,
    "Onboarding & Provisioning": 270,
    "Hardware & Device Support": 365,
    "Facilities & Regional IT": 365,
}
DEFAULT_HALF_LIFE_DAYS = 180  # fallback for an unrecognized/missing category

# How much weight the recency bonus carries relative to the rerank score.
# Additive, not replacement (Section 3): a recency score of up to 1.0
# (brand new) is scaled down before adding, so a stale-but-relevant
# document isn't automatically buried by a fresh-but-irrelevant one.
# This value is a reasoned starting point, not calibrated -- flag for
# recalibration once the before/after eval (this session) shows real
# impact.
RECENCY_WEIGHT = 1.0


def recency_score(last_updated: str, category: str | None, today: date | None = None) -> float:
    """
    Returns a value in (0, 1], 1.0 = updated today, decaying by half
    every CATEGORY_HALF_LIFE_DAYS[category] days.
    """
    if today is None:
        today = date.today()
    updated_date = datetime.strptime(last_updated, "%Y-%m-%d").date()
    age_days = max(0, (today - updated_date).days)
    half_life = CATEGORY_HALF_LIFE_DAYS.get(category, DEFAULT_HALF_LIFE_DAYS)
    return 0.5 ** (age_days / half_life)


def apply_recency_rerank(search_results: list[dict], wants_latest: bool,
                          today: date | None = None) -> list[dict]:
    """
    search_results: output of hybrid_search_reranked() -- [{"chunk_id",
    "rerank_score", "chunk"}, ...], already sorted best-first by
    rerank_score alone.

    If wants_latest is False, returns search_results UNCHANGED (Step 7 is
    conditional -- see PROJECT_HANDOFF.md Section 3: "only if
    wants_latest=True").

    If True, adds a "combined_score" field to each result
    (rerank_score + RECENCY_WEIGHT * recency_score) and re-sorts by it.
    The original rerank_score is preserved, not overwritten -- additive,
    not replacement.
    """
    if not wants_latest or not search_results:
        return search_results

    scored = []
    for r in search_results:
        chunk = r["chunk"]
        rscore = recency_score(chunk["last_updated"], chunk.get("category"), today=today)
        combined = r["rerank_score"] + RECENCY_WEIGHT * rscore
        scored.append({**r, "recency_score": rscore, "combined_score": combined})

    return sorted(scored, key=lambda r: -r["combined_score"])


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))
    from pipeline import PipelineClients, run_pipeline

    print("Building pipeline clients...")
    clients = PipelineClients.build()

    test_messages = [
        ("What's the current, up to date procedure for handling a certificate access denial?", True),
        ("How do I fix a certificate access denial?", False),
    ]

    print("\nStep 7 recency rerank -- smoke test\n")
    for msg, expect_latest in test_messages:
        outcome = run_pipeline(msg, clients)
        if outcome.stopped_at != "ok":
            continue
        for r in outcome.results:
            print(f"MESSAGE: {msg}")
            print(f"  wants_latest={outcome.wants_latest} (expected {expect_latest})")
            before = [(sr["chunk"]["page_id"], round(sr["rerank_score"], 2)) for sr in r.search_results]
            reranked = apply_recency_rerank(r.search_results, outcome.wants_latest)
            after = [(sr["chunk"]["page_id"], round(sr.get("combined_score", sr["rerank_score"]), 2)) for sr in reranked]
            print(f"  before: {before}")
            print(f"  after:  {after}")
            print()
