"""
Detects when a retrieval result set contains a SAME-TOPIC pair at
different ages -- one current document and an older, possibly silently
stale duplicate of the same underlying topic -- so Step 9 can surface
both explicitly rather than relying on prose alone to mention it.

WHY THIS EXISTS (2026-09-17): investigated exactly how many such pairs
exist in this KB before building anything (see PROJECT_HANDOFF.md
Section 8/3) -- 10 of 205 documents are "unmarked_stale" duplicates of a
current document on the exact same topic, and a real local test found
8 of those 10 topics ALREADY co-occur in the same top-5 search result
today (often adjacent, sometimes with the STALE one outranking the
current one). This isn't a rare edge case for this corpus -- it's the
common case for this specific document relationship.

DETECTION SIGNAL: same `title`, different `page_id`, both present in one
search result set. Verified safe (src/scan investigation, 2026-09-17):
exactly the 20 known same-topic pairs in this corpus (10 unmarked_stale +
10 labeled_archived) share an identical title with their current
counterpart; the deliberately-different "confuser pair" documents (e.g.
"Network SSL VPN failure" vs. "VPN client hangs on connect") always have
DIFFERENT titles, so this signal does not accidentally fire on them.
Deliberately does NOT use the eval-only `superseded_by_page_true` field
(data/processed/eval_only_ground_truth.jsonl) -- that field must never
feed retrieval/generation, per this project's hard constraint (see
PROJECT_HANDOFF.md Section 1). Title-matching uses only data already
present in the retrievable corpus.

SOURCED DESIGN DECISION on how detected pairs are surfaced: per
claudecertificationguide.com's "Information Provenance & Multi-Source
Synthesis" page (fetched and quoted 2026-09-17), conflicting/differently-
dated sources must never be silently resolved to one value, AND must be
represented as a DISCRETE STRUCTURED FIELD with date prominent -- not
"buried in prose" inside the generated answer text. This is why
same_topic_pairs feeds a new `superseded_alternatives` field on
GenerationResult (step9_generation.py), separate from `answer_text` and
`citations`, rather than only relying on the existing prompt instruction
that tells the model to mention the older version in prose (that
instruction still exists and still helps the read of the prose itself --
this adds the structured field the sourced guidance says prose alone
doesn't satisfy).

CROSS-SOURCE AUTHORITY, NOT JUST RECENCY (added 2026-09-22, real bug
found and fixed): this detector was built and safety-verified (see
DETECTION SIGNAL above) against Confluence-only title collisions --
two versions of the SAME kind of document (an official SOP), where
"newer wins" is the correct rule. After adding the error-code
spreadsheet and Slack-thread sources, a real case surfaced: a Slack
thread and a Confluence page both titled "Plugin not loading" got
compared by pure date, and since the Slack thread was freshly authored,
it was labeled "current" and the official Confluence SOP was labeled
"superseded" -- which is misleading. An unreviewed peer chat message
didn't actually supersede a reviewed SOP; it's a different KIND of
source that happens to share a title and post-date it. Researched the
right fix (not guessed): real production RAG systems handle this by
tagging source type/authority as structured metadata and baking an
authority tier into the ranker, so "an audited record always outranks a
draft" rather than defaulting to raw recency across source kinds
(source: multiple production-RAG engineering writeups on source-aware
retrieval, fetched 2026-09-22 while investigating this exact bug).

Fix, in two parts:
  1. SOURCE_AUTHORITY_TIER below encodes this project's real authority
     ordering: Confluence (reviewed SOP) and the error-code spreadsheet
     (reviewed reference data) are tier 1 -- genuine "current work
     product," a same-topic pair WITHIN or ACROSS these two tier-1
     sources is still compared by recency, since that's the original,
     correct "silently stale duplicate SOP" case this detector was
     built for. Slack threads are tier 2 -- unreviewed, informal, never
     eligible to be reported as "superseding" a tier-1 source, no
     matter its date; the reverse (a tier-1 doc superseding an old
     Slack thread) also doesn't fire, because a chat thread isn't a
     "current SOP" to supersede in the first place -- it's simply
     excluded from this pairing logic entirely, see below.
  2. find_same_topic_pairs() now only forms a pair between chunks in
     the SAME tier -- this directly fixes the exact case above (a tier-2
     Slack thread and a tier-1 Confluence page sharing a title no
     longer form a pair at all) while leaving the original tier-1-vs-
     tier-1 stale-duplicate-SOP detection completely unchanged.
"""

from dataclasses import dataclass

# See "CROSS-SOURCE AUTHORITY, NOT JUST RECENCY" above. Lower tier
# number = higher authority. Only same-tier chunks are ever compared
# against each other -- see find_same_topic_pairs().
SOURCE_AUTHORITY_TIER = {
    "confluence": 1,
    "spreadsheet": 1,
    "slack_thread": 2,
}
DEFAULT_TIER = 1  # unrecognized/missing source_type -- treat as tier-1
# rather than silently matching everything (fails toward the ORIGINAL,
# narrower behavior, not toward over-matching across unknown sources).


@dataclass
class SameTopicPair:
    current_page_id: str
    current_title: str
    current_last_updated: str
    older_page_id: str
    older_last_updated: str


def find_same_topic_pairs(search_results: list[dict]) -> list[SameTopicPair]:
    """
    search_results: the list from hybrid_search_reranked() /
    PipelineResult.search_results -- [{"chunk_id", "rerank_score",
    "chunk": {...}}, ...]. Groups by title, and for any title with 2+
    DISTINCT page_ids present, returns the newer as `current` and each
    older one as a separate pair (older = smaller last_updated date
    string, safe since all dates in this corpus are ISO YYYY-MM-DD).

    Returns an empty list (the common case -- most search results don't
    contain a same-topic pair) if no title has 2+ distinct page_ids.
    """
    by_title_tier: dict[tuple[str, int], dict[str, dict]] = {}
    for r in search_results:
        chunk = r["chunk"]
        title = chunk["title"]
        page_id = chunk["page_id"]
        tier = SOURCE_AUTHORITY_TIER.get(chunk.get("source_type"), DEFAULT_TIER)
        # Grouping key is (title, tier) NOT just title -- see
        # "CROSS-SOURCE AUTHORITY, NOT JUST RECENCY" in the module
        # docstring. A tier-2 (Slack thread) and tier-1 (Confluence/
        # spreadsheet) chunk sharing a title never land in the same
        # group, so they never form a pair -- only same-tier title
        # collisions (the original, safety-verified case) do.
        # A title can appear multiple times (multiple chunks of the same
        # page) -- keep only one representative chunk per page_id.
        by_title_tier.setdefault((title, tier), {})[page_id] = chunk

    pairs: list[SameTopicPair] = []
    for (title, tier), chunks_by_page in by_title_tier.items():
        if len(chunks_by_page) < 2:
            continue
        ordered = sorted(chunks_by_page.values(), key=lambda c: c["last_updated"], reverse=True)
        newest = ordered[0]
        for older in ordered[1:]:
            pairs.append(SameTopicPair(
                current_page_id=newest["page_id"],
                current_title=newest["title"],
                current_last_updated=newest["last_updated"],
                older_page_id=older["page_id"],
                older_last_updated=older["last_updated"],
            ))
    return pairs
