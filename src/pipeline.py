"""
Wires Steps 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 8 into one callable end-to-end
pipeline: raw user message in, ranked search results out (with a
confidence flag from Step 8, and now multi-turn support from Step 0,
added 2026-09-14).

Each step already exists as an independently tested function (see
PROJECT_HANDOFF.md Section 4, "Query understanding (Steps 0-4)",
"Retrieval", and step8_confidence.py). This module is ONLY the wiring --
no new decision logic, no new prompts.

Step 0 is OPTIONAL and backward-compatible: run_pipeline() takes an
optional `conversation_state` parameter. If None (the default, and what
every test before 2026-09-14 used), behavior is unchanged from before --
Step 0 never runs, raw_message is used as-is. If a ConversationState is
passed, Step 0 runs first and its standalone_message feeds Step 1
onward instead of the raw message. Step 0 itself decides internally
whether it actually needs to call Gemini (skips entirely on a message
with no real prior context -- see step0_context.py).

For a compound/bundled message, Step 3 can return `rewritten_query` as a
list of sub-questions instead of one string (see step3_understanding.py).
This pipeline runs retrieval ONCE PER sub-question and returns one
PipelineResult per sub-question, since each is a genuinely separate
search -- there is no single "top 5" that makes sense across two
unrelated problems bundled into one message.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from google import genai

PROJECT_ROOT = Path(__file__).parent
import sys  # noqa: E402
sys.path.insert(0, str(PROJECT_ROOT))

from step0_context import ConversationState  # noqa: E402
from query_parsing import parse_query  # noqa: E402
from step_route_or_search import route_or_search  # noqa: E402
from step4_ambiguity_gate import apply_ambiguity_gate  # noqa: E402
from step7_recency import apply_recency_rerank  # noqa: E402
from step8_confidence import check_confidence, log_knowledge_gap  # noqa: E402
from retrieval_pipeline import access_denial_info, hybrid_search_reranked  # noqa: E402
from live_events import StageTimer  # noqa: E402
from config import CHROMA_DB_PATH, MODEL_CACHE_DIR, REPO_ROOT, RERANKER_MODEL_NAME, ONNX_NUM_THREADS  # noqa: E402

load_dotenv(REPO_ROOT / ".env")


@dataclass
class PipelineClients:
    """Bundle of the expensive-to-construct clients/models this pipeline
    needs, so callers build them once and reuse across many queries
    instead of reloading a model per call."""
    gemini: genai.Client
    embedding_model: TextEmbedding
    reranker: TextCrossEncoder
    chroma_collection: "chromadb.Collection"

    @classmethod
    def build(cls) -> "PipelineClients":
        gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        embedding_model = TextEmbedding(
            model_name="BAAI/bge-small-en-v1.5", cache_dir=str(MODEL_CACHE_DIR), threads=ONNX_NUM_THREADS
        )
        reranker = TextCrossEncoder(
            model_name=RERANKER_MODEL_NAME, cache_dir=str(MODEL_CACHE_DIR), threads=ONNX_NUM_THREADS
        )
        client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        collection = client.get_collection("it_helpdesk_kb")
        return cls(gemini, embedding_model, reranker, collection)


@dataclass
class PipelineResult:
    sub_query: str  # the specific rewritten (sub-)question this result answers
    category_used: str | None  # None if the search was unfiltered (ambiguous/unclear)
    low_confidence: bool
    search_results: list[dict] = field(default_factory=list)
    access_denial: dict | None = None  # Step 4.5 audit -- see retrieval_pipeline.access_denial_info()
    # WHY low_confidence is True, added 2026-10-24 so the UI can tell the
    # staffer something more useful than a generic "not confident"
    # caveat. Two genuinely different reasons previously collapsed into
    # one boolean: "ambiguous_category" (Step 4 couldn't tell which of
    # the 8 categories this belongs to, so the search was widened
    # unfiltered) vs "low_score" (the category WAS clear, but the best
    # match's calibrated reranker score still fell below threshold --
    # found a plausible category, just not confident any single article
    # in it is the right one). None when low_confidence is False.
    low_confidence_reason: str | None = None  # "ambiguous_category" | "low_score" | None
    confidence_top_score: float | None = None  # Step 8's confidence.top_score, for the user-facing message
    confidence_threshold: float | None = None  # Step 8's confidence.threshold_used


@dataclass
class PipelineOutcome:
    """Top-level result of running the pipeline on one raw message."""
    # "declined" | "ok" -- where the pipeline ended. SIMPLIFIED 2026-10-24:
    # the old design had 3 separate stop reasons (off_topic/conversational/
    # ok), each produced by its own classification call. The new single
    # tool-calling reasoning step (step_route_or_search.py) naturally only
    # distinguishes "replied directly" vs "called the search tool" -- it
    # does NOT separately tag WHY it chose not to search (greeting vs
    # off-topic vs identity question), and researched before dropping that
    # distinction rather than assumed: Anthropic's own tool-design guidance
    # explicitly warns against wrapping/asking for classification work
    # that could be handled by a single pass of internal reasoning, and
    # real agent-observability practice tracks tool-call frequency / the
    # fraction of requests completed without a tool call, not a detailed
    # reason taxonomy for why a tool wasn't called. Forcing the model to
    # also self-tag a reason would reintroduce exactly the kind of
    # secondary judgment call that caused this session's repeated routing
    # bugs, for a label that's informational only, never safety-critical.
    stopped_at: str
    error_codes: list[str] = field(default_factory=list)
    wants_latest: bool = False
    results: list[PipelineResult] = field(default_factory=list)
    stage_latency_ms: dict = field(default_factory=dict)  # see live_events.StageTimer
    # Only set when stopped_at == "declined" -- the direct reply text to
    # show the staffer for that turn, generated by the same single
    # reasoning call that decided not to search.
    message: str | None = None


def run_pipeline(raw_message: str, clients: PipelineClients,
                  conversation_state: ConversationState | None = None,
                  allowed_groups: set[str] | None = None,
                  message_source: str = "user_message") -> PipelineOutcome:
    """
    Run Steps 0 -> 1 -> 2 -> 3 -> 4 -> 4.5 -> 5 -> 6.5 -> 7 -> 8. See
    module docstring for the conversation_state (Step 0) and compound-query
    (list rewritten_query) handling.

    allowed_groups: Step 4.5, permission resolution -- the caller's group
    memberships, applied as a PRE-retrieval filter inside
    hybrid_search_reranked() (never a post-hoc filter on already-surfaced
    results). None means no caller identity was supplied -- retrieval is
    unfiltered, matching every pre-existing call site/test (no restricted
    content was ever being returned to the EVAL harnesses before this
    existed, since none of them pass a real identity; a live caller like
    chat_service.py should always pass the IT staffer's real groups).

    message_source: which delimiting tag (see prompt_delimiting.py) Step
    2/3 wrap `raw_message` in -- "user_message" (default) for anything the
    IT staffer typed themselves, "ticket" for a Jira ticket's seed content
    (chat_service.py's start_or_resume_ticket_session()). Distinct tag
    names matter here because a ticket's description/comments can
    originate from the END EMPLOYEE who filed it, not the staffer -- a
    third untrusted-content source alongside retrieved KB chunks
    (ADR-005) and the staffer's own typed messages, closed 2026-10-24
    after a direct review question asked whether ticket content gets the
    same injection protection. Only matters on a ticket's FIRST message
    (Step 0 is skipped then, no prior context yet) -- every later message
    in that session, including this same seed text once Step 0 has
    rewritten/stored it, is already-processed conversation history, not
    fresh untrusted input, so the tag distinction is moot past turn 0.
    """
    timer = StageTimer()

    # Step 1: regex pre-parsing, no LLM -- kept as a safety net alongside
    # the model's own error-code extraction (unioned per SearchCall
    # below), same reasoning as before: a pure pattern match can't
    # misread the prompt, so it catches codes a model might miss.
    parsed = parse_query(raw_message)
    timer.mark("step1_parsing")

    # Record the user's turn in conversation memory BEFORE the routing
    # call -- case_facts extraction (via parse_query() inside add_turn())
    # happens regardless of what the routing/search decision turns out to
    # be, same as before. Render current_turns as plain text for
    # route_or_search()'s own prompt (replaces step0_context.py's
    # separate rewrite call -- this module does its own contextualization
    # internally, reasoning over the real conversation directly rather
    # than a pre-rewritten proxy of it).
    conversation_text = ""
    if conversation_state is not None:
        if conversation_state.current_turns:
            conversation_text = "\n".join(
                f"{t['role']}: {t['text']}" for t in conversation_state.current_turns
            )
        conversation_state.add_turn("user", raw_message)

    # Steps 0+2+3 REPLACED 2026-10-24 by ONE tool-calling reasoning call
    # (step_route_or_search.py) -- see that module's docstring for the
    # full rationale (repeated live-UI failures from 3 blind sequential
    # gates disagreeing with each other). It decides, in one pass: reply
    # directly (greeting/off-topic/identity question/already-known-fact),
    # or call search_knowledge_base once per distinct issue.
    route = route_or_search(conversation_text, raw_message, client=clients.gemini,
                             message_source=message_source)
    timer.mark("route_or_search")

    if not route.search_calls:
        # Direct reply -- no search needed (replaces the old off_topic/
        # conversational stop paths, now a single "declined" outcome; see
        # PipelineOutcome.stopped_at's own comment for why the 3-way split
        # was dropped rather than preserved). The assistant turn itself is
        # recorded by the CALLER (chat_service.py's send_message(), which
        # already does this for any stopped_at != "ok") -- not here, to
        # avoid double-adding it.
        reply = route.direct_reply or "I'm not sure how to respond to that -- could you rephrase?"
        return PipelineOutcome(stopped_at="declined", stage_latency_ms=timer.marks, message=reply)

    # Step 3's old error_codes may miss codes Step 1 already found via
    # regex (or vice versa) -- union them rather than trusting only one
    # source, same reasoning as before, now applied per search call.
    all_error_codes_overall: set[str] = set(parsed.error_codes)
    wants_latest_overall = False

    results = []
    for call in route.search_calls:
        call_error_codes = sorted(set(parsed.error_codes) | set(call.error_codes))
        all_error_codes_overall |= set(call.error_codes)
        wants_latest_overall = wants_latest_overall or call.wants_latest

        # Now that category is known for this call, file this turn's
        # provisionally-bucketed case_facts under their real category
        # bucket -- same SEQUENCING NOTE reasoning as before (category
        # isn't known until this call returns). A no-op if
        # conversation_state is None or category is "unclear".
        if conversation_state is not None:
            conversation_state.file_facts_for_category(call.category)

        # Step 4: ambiguity gate -- UNCHANGED, now takes the category
        # string directly rather than an UnderstandingResult object (see
        # step4_ambiguity_gate.py's own 2026-10-24 simplification note).
        gate = apply_ambiguity_gate(call.category)

        search_results = hybrid_search_reranked(
            call.query,
            clients.chroma_collection,
            clients.embedding_model,
            clients.reranker,
            error_codes=call_error_codes,
            category=gate.search_category,
            allowed_groups=allowed_groups,
        )
        timer.mark("retrieval_rerank")  # Steps 5/6/6.5, summed across calls if compound

        # Step 4.5 audit: only costs anything (one extra local BM25 scan,
        # no model call) when a real caller identity was supplied.
        denial = (
            access_denial_info(call.query, call_error_codes, gate.search_category, allowed_groups)
            if allowed_groups is not None else None
        )

        # Step 7: conditional recency rerank -- UNCHANGED. No-op unless
        # THIS call's wants_latest=True (now per-call, a real improvement
        # over the old single message-wide flag: a compound message can
        # have one sub-issue wanting the latest version and another not).
        search_results = apply_recency_rerank(search_results, call.wants_latest)
        timer.mark("step7_recency")

        # Step 8: confidence threshold -- UNCHANGED, still the
        # calibrated, non-LLM-self-reported check. INDEPENDENT of Step
        # 4's ambiguity gate (see original comment, still true): a clear
        # category can still get weak search results.
        confidence = check_confidence(search_results, category=gate.search_category)
        category_used = gate.search_category
        if category_used is not None and not confidence.confident:
            # The category is the router's one-line guess and a hard filter: a
            # question about a policy or a neighbouring topic can live in
            # another category. Retry once across all categories and keep it
            # only if it clears the same confidence bar.
            wide = apply_recency_rerank(
                hybrid_search_reranked(
                    call.query, clients.chroma_collection, clients.embedding_model,
                    clients.reranker, error_codes=call_error_codes, category=None,
                    allowed_groups=allowed_groups,
                ),
                call.wants_latest,
            )
            wide_confidence = check_confidence(wide, category=None)
            if wide_confidence.confident:
                search_results, confidence, category_used = wide, wide_confidence, None
        timer.mark("step8_confidence")
        low_confidence = gate.low_confidence or not confidence.confident
        # Which signal actually caused it, for the user-facing message --
        # gate.low_confidence (ambiguous/unclear category) checked FIRST
        # since it's the more fundamental "couldn't even tell the
        # category" case; a category-clear-but-low-score result only
        # gets "low_score" if the category gate itself was fine.
        low_confidence_reason = None
        if gate.low_confidence:
            low_confidence_reason = "ambiguous_category"
        elif not confidence.confident:
            low_confidence_reason = "low_score"
        if not confidence.confident:
            log_knowledge_gap(
                query=raw_message,
                sub_query=call.query,
                category=gate.search_category,
                top_score=confidence.top_score,
                reason="no_results" if confidence.top_score is None else "low_confidence",
                threshold_used=confidence.threshold_used,
            )

        results.append(PipelineResult(
            sub_query=call.query,
            category_used=category_used,
            low_confidence=low_confidence,
            search_results=search_results,
            access_denial=denial,
            low_confidence_reason=low_confidence_reason,
            confidence_top_score=confidence.top_score,
            confidence_threshold=confidence.threshold_used,
        ))

    results = _merge_same_question(results, [c.same_question for c in route.search_calls])

    return PipelineOutcome(
        stopped_at="ok",
        error_codes=sorted(all_error_codes_overall),
        wants_latest=wants_latest_overall,
        results=results,
        stage_latency_ms=timer.marks,
    )


def _merge_same_question(results: list[PipelineResult], same_flags: list[bool]) -> list[PipelineResult]:
    """
    Search calls flagged same_question are parts of ONE question (e.g. the
    two sides of a comparison): pool their excerpts into a single result so
    it is answered together, not as separate turns each seeing half the
    evidence. Only confident parts are pooled (a weak part would add
    decoy-prone noise); if every part is weak the merged result is weak too.
    """
    group = [r for r, same in zip(results, same_flags) if same]
    if len(group) < 2:
        return results
    low = all(r.low_confidence for r in group)
    lead = next((r for r in group if not r.low_confidence), group[0])
    pooled, seen = [], set()
    for r in (group if low else [r for r in group if not r.low_confidence]):
        for sr in r.search_results:
            if sr["chunk_id"] not in seen:
                seen.add(sr["chunk_id"])
                pooled.append(sr)
    merged = PipelineResult(
        sub_query=" + ".join(r.sub_query for r in group),
        category_used=lead.category_used,
        low_confidence=low,
        search_results=pooled,
        access_denial=next((r.access_denial for r in group if r.access_denial), None),
        low_confidence_reason=lead.low_confidence_reason if low else None,
        confidence_top_score=lead.confidence_top_score,
        confidence_threshold=lead.confidence_threshold,
    )
    out, placed = [], False
    for r, same in zip(results, same_flags):
        if not same:
            out.append(r)
        elif not placed:
            out.append(merged)
            placed = True
    return out


if __name__ == "__main__":
    print("Building pipeline clients (loading models, this takes a moment)...")
    clients = PipelineClients.build()

    test_messages = [
        "My VPN keeps disconnecting right after I log in, error 0x8007042B.",
        "What time does the cafeteria on the 4th floor close today?",
        "My screen just went black.",
        "My VPN and my email are both broken today -- VPN disconnects right after I log in, and separately Outlook won't sync any new mail.",
    ]

    print("\nPipeline smoke test\n")
    for msg in test_messages:
        outcome = run_pipeline(msg, clients)
        print(f"MESSAGE: {msg}")
        print(f"  stopped_at: {outcome.stopped_at}")
        if outcome.stopped_at == "ok":
            print(f"  error_codes: {outcome.error_codes}  wants_latest: {outcome.wants_latest}")
            for r in outcome.results:
                print(f"  sub_query: {r.sub_query!r}")
                print(f"    category_used: {r.category_used!r}  low_confidence: {r.low_confidence}")
                for sr in r.search_results[:3]:
                    print(f"    - [{sr['chunk']['page_id']}] {sr['chunk']['title']}")
        print()
