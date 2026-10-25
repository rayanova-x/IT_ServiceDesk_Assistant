# IT Helpdesk RAG Assistant — Project Handoff

**Purpose:** A portfolio project demonstrating a production-realistic agentic
RAG system for IT helpdesk support — persistent knowledge grounding, hybrid
retrieval, staleness/confusable-document handling, and query understanding.
Modeled after real patterns from Walmart/Amazon-style agentic shopping
assistants and Atlassian Rovo, adapted to an IT-support domain.

**WHO THE USER IS — confirmed 2026-09-13, read this before touching
routing/escalation/confidence logic.** This is an **agent-assist tool for
IT SUPPORT STAFF**, not a chatbot for the confused employee. IT staff
paste in a ticket (or type their own query) to find the right internal
KB article/procedure to resolve someone else's issue. This is why
`enriched_tickets.csv` and several hand-written eval queries look like
pasted ticket dumps (`Ticket #`/`Priority:`/`Category:` fields) — that's
IT-staff workflow, not employee phrasing. **An employee-facing chatbot
with its own ticket-raising flow is explicitly scoped for LATER, not
now.** This corrected a real wrong assumption baked into the original
architecture (an "escalate to a human" intent that made no sense once
the human is already the one using the tool) — see Section 3, Steps 2/4/8.

This document is the single source of truth for picking this project back
up. Read it fully before writing any new code — several early design
choices exist specifically *because* of bugs or dead ends found during
development, and re-doing them differently will likely reintroduce solved
problems.

---

## 1. Hard constraints — do not violate these

- **Embeddings: local/open-source ONLY, no third-party API.** Currently
  using `fastembed` with `BAAI/bge-small-en-v1.5` (ONNX-based, no PyTorch).
  `sentence-transformers` was tried first and rejected — its PyTorch
  dependency didn't fit in the original dev sandbox's disk space, and
  separately, both libraries have hit real Hugging Face network access
  problems on the user's corporate machine (see Section 6).
- **LLM reasoning (query understanding, generation): Gemini 2.5 Flash via
  Google AI Studio API key ONLY.** **Changed 2026-09-13** — originally
  GPT-4.1-mini via Azure OpenAI (confirmed explicitly by the user at
  project start), switched to Gemini per explicit user request. Setup:
  `GEMINI_API_KEY` + `GEMINI_MODEL` in a project-root `.env` file
  (gitignored — verified 2026-09-13 the repo had zero commits at the
  time, so no leak risk from the switch). Uses the `google-genai` SDK.
  **Known model-availability gotcha**: `gemini-2.5-flash` was found to be
  retired ("no longer available to new users" — the API's own 404 error
  explicitly recommends `gemini-3.6-flash` as of 2026-09-13). Actually in
  use: **`gemini-3.5-flash-lite`**, confirmed available via
  `client.models.list()` and smoke-tested successfully with
  schema-enforced JSON output (intent classification + category enum +
  error-code extraction + boolean flag, all correctly parsed in one
  call). Gemini's model lineup moves fast — re-verify with
  `client.models.list()` before assuming a model name discussed in an
  older conversation is still valid, rather than hard-coding trust in any
  specific version string.
  **Free-tier rate limit, confirmed 2026-09-13**: 15 requests/minute for
  `gemini-3.5-flash-lite`. Any script calling Step 2/3 in a loop over many
  queries needs pacing (~4.5s between calls was sufficient in testing) and
  429-retry handling, or it will hit `RESOURCE_EXHAUSTED` errors. Also
  learned the hard way: never pipe a long-running incremental script
  through `grep`/`tail` without `--line-buffered` — it buffers ALL output
  until the whole pipeline exits, making a perfectly healthy slow run look
  completely stuck. Write results to a flushed file or print with
  `flush=True` instead, and read the file/task output directly.
- **Vector store: Chroma.** Confirmed, in use throughout.
- **Never let `doc_type`, `confuses_with`, `supersedes`, `superseded_by`
  fields leak into the retrievable corpus or chunks.** These exist ONLY in
  `eval_only_ground_truth.jsonl`, used for scoring retrieval after the
  fact — never as a retrieval-time filter or chunk metadata field. This
  was enforced deliberately and verified multiple times (see Section 6).

---

## 2. Current status at a glance

| Component | Status |
|---|---|
| Ticket + employee data | ✅ Done |
| Knowledge base (Confluence-realistic) | ✅ Done, verified clean |
| Chunking | ✅ Done, tested (247 chunks) |
| Embedding + Chroma storage | ✅ Done, confirmed working on user's machine |
| Chroma inspection/visualization tools | ✅ Done |
| BM25 retrieval | ✅ Done, real bugs found and fixed |
| RRF fusion | ✅ Logic verified |
| Dense + hybrid retrieval, end-to-end | ✅ Done, blocker resolved 2026-09-09 — hybrid beats BM25-alone but does NOT yet beat confuser decoys at rank #1, see Section 7 |
| Reranking (cross-encoder) | ✅ Done 2026-09-13 — `hybrid_search_reranked()` is now the recommended entry point, measured improvement on every metric, see Section 8 |
| Retrieval eval harness (`src/eval_retrieval.py`) | ✅ Done 2026-09-13 — 98 scorable queries (50 real ticket + 48 hand-written edge-case), 5 blocked pending Step 0 (Step 0 is now done as of 2026-09-14 — see `tests/test_step0_context.py` for where those 5 got scored instead; `eval_retrieval.py` itself was never updated to use Step 0 since it deliberately tests raw retrieval, not the full pipeline) |
| Project reorg (`src/`, `data/{raw,processed,chunks}/`, `tests/`, `.venv`) | ✅ Done 2026-09-09 |
| Step 1 (regex pre-parsing) | ✅ Done and tested 2026-09-13, `src/query_parsing.py` + `tests/test_query_parsing.py` |
| Step 2 (routing) | ✅ Done and tested 2026-09-13, `src/step2_routing.py`, 100% correct on all 62 real eval queries. REVISED same day: dropped `escalate` intent after correcting the user-model assumption (IT staff use this tool, not the employee directly) — now 2 intents (`off_topic`/`genuine_question`), re-verified 100% correct after the change |
| Step 3 (structured understanding) | ✅ Done and tested 2026-09-13, `src/step3_understanding.py`, 92% category accuracy (44/48, remaining 4 are defensible alternate judgments not clear errors), 100% on error codes and wants_latest |
| Step 4 (ambiguity gate) | ✅ Done and tested 2026-09-13, `src/step4_ambiguity_gate.py` — pure decision logic, no LLM call. Tested against all 62 real Step 3 outputs (57 clean-category, 5 ambiguous), 0 exceptions, all correct. One known gap: the "category=unclear but ambiguous=False" combination is logically correct and covered by a hand-built smoke test, but has zero real examples in the current 62-query eval set |
| Pipeline wiring (Steps 0→1→2→3→4→5→8) | ✅ Done 2026-09-14, `src/pipeline.py`, `run_pipeline()`. Multi-turn support via optional `conversation_state` param (see Step 0's row below) |
| Generation (Step 9) | ✅ Done and tested 2026-09-14, `src/step9_generation.py`. Confident results get a synthesized Gemini answer + citations; low_confidence/ambiguous results skip synthesis entirely (code-enforced) and show raw ranked candidates with a caveat instead, per sourced guidance against forcing synthesis on ambiguous data — see Section 5. **Revised same day** with 3 improvements sourced directly from Anthropic's own official docs (citation verification with retraction, XML-tagged prompt structure, dates visible at generation time) — the last one caught and fixed a REAL bug where conflicting current-vs-stale-duplicate procedures were presented as equally valid, verified against a real example before and after the fix |
| Staleness disclosure in generation (Layer 2) | ✅ Done 2026-09-14 as part of Step 9 — every citation unconditionally carries `last_updated`/`status`; `staleness_disclosure` computed on `GenerationResult` itself (not left to `format_for_display()`) so the guarantee travels with the data regardless of how it's later rendered |
| Full end-to-end pipeline eval (Steps 1-9) | ✅ Done 2026-09-15 (re-run against the FIXED corpus, post-heading-merge-fix, completed after a quota-reset resume — see Section 6). **62/62 complete, 0 issues, real numbers**: Recall@5 **88.0%**, top-1 accuracy **66.0%**, **confuser-trap rate 0/4 (all 4 known traps now resolve correctly through the full pipeline)** — a real, measured improvement over the pre-fix run. Old pre-fix results archived to `data/processed/pre_headingfix_archive/` for comparison, not deleted |
| Generation-quality eval (faithfulness/relevance) | 🔄 IN PROGRESS 2026-09-17, re-run against the FIXED corpus. **1/6 synthesized sub-results scored so far (E02: faithfulness 5, relevance 5), 5/6 PENDING** (D03, E10, F02, F09, G05 — all blocked on `gemini-3.6-flash`'s exhausted daily quota, see below). **0 contract violations across all 22 sub-results** (confidence-gating + staleness-citation both held). Old pre-fix run (faithfulness 5.00/5, relevance 5.00/5, n=7, self-graded) archived to `data/processed/pre_headingfix_archive/`. **Judge methodology, finalized 2026-09-17 after real testing** (see `judge_generation_eval.py` module docstring for the full story): every non-lite Gemini model (`gemini-3.6-flash`, `gemini-3.5-flash`) turned out to have only a 20 req/day free-tier quota, confirmed directly via the 429 error's own text, not assumed — nowhere near enough to independently judge every sub-result. Settled on a HYBRID split (`is_complex()`): straightforward sub-results (1-2 source docs, no self-reported gap) are self-graded by `GEMINI_JUDGE_MODEL` (same model as generation, `gemini-3.5-flash-lite`, 500/day quota); complex ones (3+ source docs, or the answer claims the context doesn't fully answer the query) are independently judged by `GEMINI_STRONG_JUDGE_MODEL` (`gemini-3.6-flash`, 20/day) — spending the scarce independent-judge budget where self-grading is riskiest rather than spreading it thin or abandoning independent judging entirely. **Script made resumable/crash-safe 2026-09-17** after a real incident: the first run silently lost 2 already-judged rows (D03, E02) when a mid-loop 429 crashed the script before its end-of-run bulk write — fixed by writing each row to disk incrementally, and by catching a per-row judge failure (marking it `PENDING`, continuing to the next row) instead of crashing the whole pass. To resume once `gemini-3.6-flash`'s quota resets (midnight Pacific): `python src/judge_generation_eval.py` — it will skip the 1 already-scored row and retry only the 5 PENDING ones |
| Heading-merge chunking bug (content-loss root cause) | ✅ Found and fixed 2026-09-14, `src/build_confluence_export.py`'s `strip_cross_references()`. **15 of 205 documents were silently losing their "Overview"/disambiguating sentence** during export (heading + body text merged into one `<h4>` tag with no blank line between them in the source). Fixed, corpus regenerated (247 -> 283 chunks), re-embedded. **Real before/after (112 real queries, local-only eval): top-1 58.0% (was 52.7%), MRR 0.669 (was 0.630), VPN-query top-1 40.0% (was 30.0%)** — net positive but NOT a clean fix: 7/30 VPN queries improved, 7/30 got worse, and the 2 tracked confuser-trap wins (C11, E10) were unchanged. See Section 8's "Heading-merge chunking bug" writeup for full numbers and the honest caveat |
| Conversation contextualization (Step 0) | ✅ Done and tested 2026-09-14, `src/step0_context.py`, wired into `pipeline.py` as an optional backward-compatible parameter. Two-tier state (case facts never summarized + compressible narrative), built as designed from the start. **Extended 2026-09-17 (two changes)**: (1) `case_facts` is now CATEGORY-KEYED (was flat), so two unrelated issues in one session stay in separate fact buckets — found via an interview-style question review, verified live through the real pipeline, no regressions; (2) added 4 few-shot examples to the rewrite prompt (was pure prose rules) — proactive hardening per another interview-style question's finding that prose-only rules degrade over long context. **Real result: G02-G06 suite now 5/5 correct (was 4/5)** — the previous miss (G03) now scores hit_rank=5, a genuine improvement, though still a weak pass against the documented confuser-pair weakness. See Section 4's detailed writeup |
| Recency reranking (Step 7) | ✅ Built and tested 2026-09-14, `src/step7_recency.py`. **Honest result: 0 ranking changes on the 15-query sample tested, at any weight 1.0-8.0x** — traced to the sample lacking genuine cross-document competition, not broken math. Kept as-is; the real fix was the "show multiple same-topic docs at different ages" feature — see below, done 2026-09-17 |
| Same-topic-different-age detection | ✅ Done 2026-09-17, `src/same_topic_detection.py` + `step9_generation.py`'s new `superseded_alternatives` structured field. **10/205 KB docs are silently-stale duplicates on the same topic as a current doc; 8/10 already co-occur in a real top-5 result today** (checked, not assumed) — confirmed the feature was worth building before writing detection code. Sourced design (claudecertificationguide.com, fetched directly): conflicting/dated sources need a DISCRETE STRUCTURED FIELD, not just prose — this revised the original plan mid-build. **Real test: 8/10 topics correctly detected through the actual pipeline**, including one case with 2 independent pairs in one result set, kept correctly separate. See Section 8's "Same-topic-different-age detection" writeup |
| Confidence threshold (Step 8) | ✅ Done and tested 2026-09-14, `src/step8_confidence.py`, wired into `pipeline.py`. Threshold (2.0) calibrated from real data (`src/calibrate_confidence.py`, 112 queries), not guessed. **Honest finding**: only 47/100 answerable calibration queries scored confident — the confuser-pair weakness means well over half of real queries currently land in the knowledge-gap log. Real feature, not just a fallback — see Section 8 |
| Eval harness against ground truth | ✅ Done — `src/eval_retrieval.py` (retrieval), `src/eval_pipeline_e2e.py` (full pipeline), `src/judge_generation_eval.py` (generation quality) — all scored against `data/processed/eval_only_ground_truth.jsonl` |

---

## 3. Architecture — the full intended pipeline

**Revised 2026-09-13** after researching how real enterprise internal-KB
chatbots are built (Atlassian Rovo, Microsoft 365 Copilot Retrieval API,
Glean, Pinecone's access-control guide) and reading Anthropic's "Building
Effective Agents" plus the claudecertificationguide.com CCAR-F curriculum
(19 pages across agentic architecture, prompt engineering, and context
management). This is a **workflow, not an agent**, in Anthropic's own
terms — a fixed, predictable sequence, not an LLM dynamically directing
its own process. That distinction argues for fewer, well-chosen LLM calls
per Anthropic's explicit guidance: "add complexity only when it
demonstrably improves outcomes." Changes from the original design are
called out inline with the reasoning.

```
[User message: text, user_id, department/group memberships, conversation_id]
   |
STEP 0: Conversation contextualization -- SMALL, FAST, DEDICATED call
   -> only runs if conversation_id has prior turns
   -> input: last N raw turns (sliding window / summarized older turns,
      not full raw history -- see memory note below)
   -> output: standalone_message: str
   -> kept as its OWN call, not merged into Step 3 below: a real
      production pattern (a customer-support "Contextualizer" layer) runs
      this on every single turn and keeps it deliberately lightweight
      (p95 latency tracked, <100ms) -- folding it into the heavier
      structured-extraction call would make every simple first-turn
      message pay that heavier cost too.
   |
   [MEMORY NOTE, not yet built]: don't summarize the whole conversation
   into one prose blob. Keep a separate, NEVER-summarized "case facts"
   block (ticket IDs, error codes, asset tags, dates, employee IDs)
   carried forward verbatim every turn, alongside a summarized/windowed
   narrative for general flow. Naive summarization silently destroys
   exact values -- e.g. "$247.83 for order #8891 placed March 3rd"
   degrading to "customer wants a refund for a recent order." Step 1's
   regex-extracted entities are exactly what belongs in this facts block.
   |
STEP 1: Regex pre-parsing (no LLM, instant, free)
   -> strip ticket#/priority/employee-name (routing info, not search-relevant)
   -> if explicit "Category:" field present, trust it directly
   -> regex-extract fixed-format identifiers (error codes, IPs, hostnames,
      file paths) -> confirmed_entities (feeds the case-facts block above)
   |
STEP 2: Intent/routing classification -- OWN call/function, SEPARATE from
Step 3. **Un-merged 2026-09-13** (had briefly been merged into Step 3 on
2026-09-13 for "simpler is better" reasons -- reversed the same day once
the user flagged that routing specifically has real potential to grow
into agentic decision-making later, e.g. the model dynamically choosing
among tools/actions rather than picking one of 4 fixed enum values).
Keeping routing separate now avoids having to restructure Step 3's
extraction logic later just to upgrade Step 2 -- extraction (category,
error codes, rewritten query) is stable regardless of how agentic routing
eventually becomes; routing is the part most likely to change shape.
Cheap/fast call (or classifier), Gemini gemini-3.5-flash-lite for now.
**`canned_answer` intent DROPPED 2026-09-13** — it was named in the
original design with no actual list of canned questions/answers ever
defined behind it (a dead code path with nothing to route to). Skipped
rather than built with an empty placeholder list, to avoid wiring up a
category with no real content. Add it back if/when a real canned-Q&A
list exists to route to.

**`escalate` intent DROPPED 2026-09-13 — CORRECTS A WRONG ASSUMPTION,
not a refinement.** This whole architecture (through Section 3's
original draft) was written assuming the chatbot talks directly to a
confused employee. **Actual scope, confirmed by the user 2026-09-13: the
user of this tool is IT SUPPORT STAFF**, who paste in a ticket or type
their own query to find the right internal KB article/procedure to
resolve someone else's issue. (This is why `enriched_tickets.csv` and
the `A` category of hand-written eval queries look like pasted ticket
dumps with `Ticket #`/`Priority:`/`Category:` fields — that's IT staff
workflow, not employee phrasing. An employee-facing chatbot with its own
ticket-raising flow is scoped for LATER, not now.) An IT staffer is never
going to ask this tool to "connect me to a human" — they ARE the human.
Researched 2026-09-13 how real agent-assist / internal-copilot tools
built for support staff (not end customers) actually handle this — Zendesk
agent copilot, Dynamics 365 Customer Service "Smart assist", Forethought
Discover, Ada — and the pattern is consistent and well-documented: NONE
of them have an "escalate the AI's decision" concept. Instead:
  - Low confidence / ambiguous -> show RANKED CANDIDATES with a
    confidence indicator each, let the staffer's own expertise pick —
    do NOT interrupt with a clarifying-question loop. Interrupting an
    expert user's workflow has a real cost these products evidently
    avoid; asking a chat-style clarifying question is the exception used
    in conversational UIs, not the default for a search/suggestion panel.
  - Genuine zero-match -> an honest "no confident match found" state,
    AND log the query as a KNOWLEDGE GAP for later KB curation (Zendesk's
    "Coverage" metric, Forethought Discover, Ada's "gap detection ->
    prioritized content backlog" are all explicit examples of this
    two-track pattern: real-time honest non-answer + async gap log for
    whoever owns the KB). This is a genuinely more useful feature than a
    fake escalation path — see Step 8.
   Output: intent: Literal["off_topic", "genuine_question"]
   -> off_topic -> decline, END
   -> genuine_question -> CONTINUE to Step 3
   |
STEP 3: Structured query understanding -- OWN call, ONE Gemini call
(gemini-3.5-flash-lite, see Section 1), schema-enforced. Only runs if
Step 2 returned genuine_question -- off-topic/escalated/canned messages
never pay for this call.
   Output fields:
     - rewritten_query: str (or list[str] if compound/bundled issues)
     - category: Literal[8 REAL category strings, PLUS an explicit
       "unclear"/other fallback] | None  <-- MUST be enum, not free
       string (Section 6), but must NOT be a closed set with no escape
       hatch either -- forcing a required pick from 8 fixed values
       pressures the model to fabricate a plausible-but-wrong category
       when the query genuinely doesn't fit one. The 8 real values
       (verified directly against confluence_export.jsonl's ancestors[0]
       .title field, 2026-09-09): "IT Policies & Procedures", "Facilities
       & Regional IT", "Onboarding & Provisioning", "Hardware & Device
       Support", "Security Operations", "Software & Applications Support",
       "Network Infrastructure", "Access & Identity Management".
     - error_codes: list[str] (free text, no enum possible; optional/
       nullable -- don't force a value when the query doesn't mention one)
     - wants_latest: bool (recency signal-word detection)
     - ambiguous: bool
   |
STEP 4: Ambiguity gate (programmatic check, no LLM). **REDESIGNED
2026-09-13** — the original "ask ONE clarifying question, loop back"
design assumed a chat interface talking to a confused employee. For an
IT-staff tool, per the same agent-assist research as Step 2: don't
interrupt with a question-and-answer loop at all.
   -> ambiguous=True (regardless of category) -> do NOT block or loop.
      Proceed to retrieval with category=None (i.e. unfiltered by
      category) so Step 5 casts a wider net, and surface this in the UI
      as a lower-confidence/ambiguous result set — Step 9 should show
      the staffer several ranked candidates rather than confidently
      picking one, exactly the confuser-pair case Section 8's
      confuser-trap-rate metric measures. The staffer's own expertise
      resolves the ambiguity, not a bot Q&A loop.
   -> otherwise CONTINUE with category set, as normal.
   |
STEP 4.5: Permission resolution (no LLM, instant) -- NEW, was missing
from the original design. Confirmed 2026-09-13 this is NOT optional:
147 of 247 chunks (60%) carry a `restricted_to` group requirement
(118 it-department-group, 29 security-team-group). Every real system
researched (Glean, Microsoft 365 Copilot, Atlassian Rovo, Pinecone's
access-control guide) enforces permissions as a PRE-retrieval filter,
never a post-hoc check on already-surfaced results.
   -> user's group memberships -> allowed_groups set
   |
STEP 5: Dual retrieval (parallel), PRE-FILTERED by permissions
   - Dense: embed rewritten_query (bge-small + BGE query prefix) -> Chroma
     query with where={"category":..., "restricted_to": in allowed_groups}
     -> top 20
   - Sparse (BM25): rank_bm25 over chunk texts; error_codes appended to
     BM25 query text; category + restricted_to post-filtered -> top 20
   |
STEP 6: RRF fusion (rank-based, k=60, never raw score)
   |
STEP 6.5: Reranking (cross-encoder) -- ADDED AND MEASURED 2026-09-13, not
in the original design. Widen candidate pool to 20 -> BAAI/bge-reranker-base
(local, fastembed) re-scores -> re-sort -> cut to top 5. Directly targets
confuser discrimination: measured to cut the decoy-beats-real-answer rate
from 75% to 50% (see Section 8's eval results) while improving every
other metric with zero regressions. `hybrid_search_reranked()` in
`retrieval_pipeline.py` is the recommended function for this.
   |
STEP 7: Conditional recency rerank. BUILT AND TESTED 2026-09-14 --
`src/step7_recency.py`, `apply_recency_rerank()`. Only runs if
wants_latest=True (Step 3, verified to survive Step 0's rewrite too --
G06). Blends the reranker's score with a recency score from
version.when (per-category half-life decay -- see Section 5, NEVER
history.createdDate), additive not replacement: a very recent but
irrelevant document must not outrank a genuinely relevant older one.
   -> **HONEST FINDING, real measured result, not guessed**: on the
      15-query stratified sample (6 recency-targeted + 9 general),
      Step 7 produced ZERO ranking changes at ANY tested weight (1.0
      through 8.0x) and zero regressions. Root cause, confirmed by
      inspecting the actual captured scores directly (not assumed): the
      3 real wants_latest=True test queries didn't have genuine
      cross-document competition for recency to act on -- two had
      near-identical dates across their top candidates (so recency
      carries zero distinguishing signal), and the third's correct
      answer was already winning by a wide relevance margin before any
      recency boost. This matches a real published finding (arXiv
      2509.19376, "Freshness and the Limits of Heuristic Trend Detection
      in Temporal RAG", read 2026-09-14): a fixed recency-decay formula
      is corpus/query-composition-sensitive and "cannot rescue" cases
      where relevance, not freshness, is the actual problem -- their own
      published numbers show the same effect can "collapse to 0.00" on
      real data. Our 3-query sample was too small/narrow to properly
      exercise this mechanism either way -- this is NOT evidence Step 7
      is broken, only that this specific sample didn't test it well.
      **Decision: keep the code as-is (RECENCY_WEIGHT=1.0), do not chase
      further tuning without a bigger/better-targeted sample.**
   -> **DESIGN DIRECTION FOR LATER, not yet built**: when wants_latest=True
      surfaces multiple DIFFERENT documents (not just multiple chunks of
      the same page) on the same/similar topic at genuinely different
      ages, do NOT silently re-rank one above the other via the recency
      score -- per the sourced guidance (Section 5, information-provenance:
      "arbitrarily selecting one value... presents a false certainty";
      explicitly cautions against assuming the later date is "the more
      reliable reflection"), show BOTH/ALL of them with their dates
      visible, the same pattern already used for Step 4/8's
      low_confidence candidates_only mode. This resolves the real tension
      between Step 7 (silently boosts by recency) and that guidance
      (never silently prefer newer) -- explicit user decision 2026-09-14.
      Not yet implemented: needs same-topic-different-document detection,
      which doesn't exist yet.
   |
STEP 8: Confidence threshold check (programmatic, NOT the LLM's own
self-reported confidence). Multiple independent sources converge on
calling LLM self-reported confidence scores unreliable for this: models
are often confidently wrong on hard cases and needlessly unsure on easy
ones. Build this from a CALIBRATED signal instead -- e.g. the reranker
score's actual relationship to correctness on a labeled validation set
(the eval harness in Section 8 is exactly this labeled set), not a
1-10 self-rating from a generation call. Also distinguish two different
failure modes that need different handling: "retrieval backend failed/
timed out" (transient, retry) vs. "retrieval succeeded but found nothing
above threshold" (valid empty result, don't retry).
   -> top result below calibrated threshold, or zero results after the
      permission filter -> **REDESIGNED 2026-09-13, "escalate" dropped
      (see Step 2)**: return an honest "no confident match found" state
      to the IT staffer (show the best available partial matches if any
      exist, clearly labeled low-confidence, rather than nothing) AND
      log the query as a KNOWLEDGE GAP (query text, timestamp, category
      if any, why it failed) to a separate store for later KB curation —
      the real-time honest non-answer + async gap-log pattern used by
      Zendesk/Forethought/Ada. This is a genuinely useful project feature
      in its own right (a "what does our KB not cover well" report), not
      just a fallback. Not yet built — needs a simple append-only log
      (even a JSONL file to start) plus whatever surfaces it later.
   |
STEP 9: Top-N chunks -> generation (Gemini, gemini-3.5-flash-lite). DONE
2026-09-14, `src/step9_generation.py`, `generate_answer()`.
   Context assembled from: system instructions, current date (grounds
   "latest"/recency questions), and retrieved chunks carried as
   STRUCTURED objects (chunk text + page_id + title + last_updated +
   status) -- not flattened into one text blob, so citations can be
   attached reliably. (Step 0's case-facts memory block, now built --
   see `src/step0_context.py` -- feeds Step 0's own rewrite prompt, not
   Step 9's generation prompt directly; Step 9 still only sees the
   already-rewritten standalone message plus retrieved chunks, not the
   raw case_facts dict itself. Revisit if a future case shows Step 9
   itself needs direct access to exact values beyond what survives the
   Step 0 rewrite.)
   -> low_confidence=True (from Step 4) -> **NEVER call Gemini for
      synthesis** -- this is code-enforced, not a prompt instruction.
      Return the caveat + raw ranked candidates instead. Researched
      2026-09-14 whether to still generate hedged prose here instead:
      the sourced guidance (Section 5, information-provenance) is
      explicit that arbitrarily collapsing ambiguous/conflicting data
      into one answer "destroys information and presents a false
      certainty" -- preserve and present the raw attributed options
      instead. This is a stronger guarantee than trusting generated
      prose to accurately self-report its own uncertainty, which has
      the same reliability problem as self-rated confidence scores.
   -> low_confidence=False -> generate a short synthesized answer using
      ONLY the provided excerpts (explicitly instructed not to fill gaps
      with general knowledge). Verified 2026-09-14: when the top-ranked
      chunk was actually the wrong confuser decoy (200060, "VPN client
      hangs on connect" -- a different symptom than what the query
      described), the model correctly said the excerpts didn't contain
      the needed resolution rather than fabricating one. A live
      demonstration of the documented confuser-pair weakness (Section 8)
      surfacing safely rather than convincingly wrong.
   -> ALWAYS disclose each source's last-updated date/status, in BOTH
      modes, regardless of wants_latest (Layer 2, Section 5) -- enforced
      in `format_for_display()` by always rendering citation metadata,
      not left to the generation prompt to remember to include.
   -> For confuser-pair / compare-two-documents queries, AND for any
      Step 4 ambiguous=True case: when multiple retrieved documents
      genuinely disagree, overlap, or are close in rank, do NOT silently
      pick one -- present the top candidates with their distinguishing
      context (e.g. different scenario, different version) and let the
      IT staffer (the actual user, see Step 2) choose, rather than
      forcing a single confident-sounding answer that risks being wrong.
      (Currently implemented via the low_confidence branch above; a
      finer-grained "these two specific documents disagree" detection
      within an otherwise-confident result set is not yet built.)
   -> **REVISED 2026-09-14, sourced from Anthropic's own official
      guidance** ("Reduce hallucinations",
      platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-hallucinations,
      and the Claude Cookbook's RAG guide, both verified directly, not
      via secondary summary). Three changes to `SYSTEM_PROMPT`/
      `_build_prompt()`:
      1. **Citation verification with retraction** -- after drafting,
         the model checks each claim against a supporting quote and
         removes anything unsupported, marking removals with `[]`. Done
         SILENTLY (output only the final verified answer, not the
         verification process itself -- found this needed an explicit
         instruction, the model showed its work by default otherwise).
      2. **XML-tagged `<query>`/`<documents>` structure** replacing plain
         text concatenation, and a stricter "100% sure" faithfulness bar
         + no-preamble instruction -- both taken directly from the Claude
         Cookbook's RAG template, which measurably moved end-to-end
         accuracy 71%->81% in Anthropic's own published eval (not just
         claimed, their real number).
      3. **`last_updated` moved INTO each document's tag**, not just the
         citation list appended after generation. **Real bug found and
         fixed via this exact change**: tested against the
         "Suspicious contractor compromise" query (page 200012 current
         vs. 200088, its silently-stale 2023 duplicate, unmarked_stale
         tier -- Section 8) and the model presented BOTH conflicting
         procedures as equally valid, since it had no way to know one
         was 3 years older while writing -- the date only existed in
         metadata added AFTER generation. Fixed by including
         `last_updated` in the document tag itself, plus an explicit
         rule: when excerpts describe DIFFERENT/conflicting procedures
         for the same topic, prioritize the most recent one and note the
         older alternative exists rather than presenting both as equally
         current. Re-tested against the same query after the fix --
         confirmed correct (prioritizes 2026-07 procedure, flags the
         2023 one as superseded). This is a genuinely different, more
         targeted mechanism than Step 7's recency reranking (which
         changes ORDER of results before generation) -- this fixes what
         the model does with conflicting information it's ALREADY been
         given, at generation time.
```

---

## 4. File inventory

**Reorganized 2026-09-09** into `data/`, `src/`, `tests/`, `notebooks/`
(previously everything was flat in the project root). Paths below are
current. Every script resolves its paths relative to its own file
location via `pathlib.Path(__file__)`, not the working directory it's
launched from.

Also as of 2026-09-09: a project-local `.venv/` (Python 3.11, installed
via winget since neither pre-existing Python on the machine fit — see
Section 7) holds all dependencies; `requirements.txt` is frozen from it.
`local_model_cache/` and `chroma_db/` are created at runtime and gitignored.

### Data — `data/raw/`
- `enriched_tickets.csv` — 5,000 sampled/enriched IT tickets, 50 real
  subcategories, natural-language descriptions, linked to employees via
  `Employee_ID`, 28 seeded recurring-issue pairs for memory testing.
  Verified 2026-09-09: all 50 ticket `Subcategory` values match a KB
  `current_sop` subcategory 1:1 — tickets can be used directly as eval
  queries with unambiguous ground truth. Known quirk (harmless, unused by
  the pipeline): `Status`/`Date_Resolved` are inconsistent on ~2,800 rows
  (e.g. `Resolved` with no resolution date, or a resolution date while
  still `Open`) — realistic ticket-system noise, not a bug worth fixing
  since nothing in the pipeline reads either field yet. ~31% of
  `Description` values are exact duplicates (template generation) — sample
  per-subcategory for eval, not purely at random.
- `employees.csv` — 2,200 synthetic employees. Clean, no issues found.

### Knowledge base
- `data/raw/rag_corpus_full.jsonl` — FIRST-DRAFT corpus (205 docs: 50 real
  SOPs + 30 confusable + 25 superseded + 100 noise, 15 SOPs are
  long/branching). Has flat `doc_type`/`confuses_with`/`superseded_by`
  fields — this format is a **build-time input only**, never the
  retrievable corpus.
- `src/superseded_content.py` — source data for the 25 outdated
  procedures' original (honest, non-self-aware) text. Imported by the
  build script.
- `src/build_confluence_export.py` — transforms `rag_corpus_full.jsonl`
  into the realistic export. **Run this, don't hand-edit the output
  JSONL.** Reads `data/raw/rag_corpus_full.jsonl` +
  `src/superseded_content.py`, writes `data/processed/confluence_export.jsonl`
  + `data/processed/eval_only_ground_truth.jsonl`. Contains
  `derive_created_date()` (added 2026-09-09, see Section 5) for the
  createdDate/version.when divergence fix.
- `data/processed/confluence_export.jsonl` — **the actual retrievable
  corpus.** 205 Confluence-API-shaped pages: `id`, `status`
  (current/trashed only, per real Confluence semantics), `version`
  (`when`, `number`), `history` (`createdDate`), `ancestors` (page-tree,
  used to derive category — **8 real category values**, not 5, see
  Section 3), `metadata.labels`, `metadata.properties`,
  `restrictions.read` (group-based, Security pages locked to
  `security-team-group`).
  - Staleness modeled in 3 tiers among the 25 superseded docs: 5 `trashed`,
    10 `deprecated`-labeled + moved to an Archive sub-page, **10 with NO
    signal at all** — `status: current`, `active-sop` label, only an old
    `version.when` and low `version.number` give it away. This last tier
    is the hard, realistic case query-understanding alone cannot solve.
  - Verified 0 forbidden-field leaks (`doc_type`/`confuses_with`/
    `supersedes`/`superseded_by`) and 0 residual cross-reference leaks in
    body text, directly, 2026-09-09.
- `data/processed/eval_only_ground_truth.jsonl` — the answer key (80
  records: 30 confusable + 25 superseded + 25 corresponding current-doc
  records; staleness tiers confirmed 5/10/10). **Never fed to retrieval.
  Only used to score results after the fact.**
- `data/processed/chroma_contents.csv` — debug export from
  `inspect_chroma.py`, gitignored.

### Chunking
- `src/chunk_confluence.py` — the current chunker (HTML-aware, cuts on
  `<h4>`, `<h3>` treated as the page's own title not a boundary).
  **Excludes `status=="trashed"` pages BEFORE chunking** — this mirrors
  real Confluence API default behavior, not an arbitrary choice. Derives
  `category` from `ancestors[0]['title']` (no flat field exists).
  Carries `status`, `version_number`, `last_updated`, `labels`,
  `restricted_to` per chunk. Verified: 0 forbidden fields ever attached.
  Reads `data/processed/confluence_export.jsonl`, writes
  `data/chunks/chunks_from_confluence.jsonl`.
- `data/chunks/chunks_from_confluence.jsonl` — output: 200 retrievable
  pages -> 247 chunks. Token counts 27-218, mean ~99.
- *(Deprecated, do not use)*: `chunk_corpus.py`,
  `chunking_walkthrough.ipynb` — earlier/educational versions, superseded
  by the above. Only `notebooks/chunking_confluence_walkthrough.ipynb`
  currently exists on disk of the deprecated set.

### Embedding + storage — `src/`
- `embed_and_store.py` — embeds chunk **text only** (never metadata) with
  `bge-small-en-v1.5` via `fastembed`, stores in Chroma (`./chroma_db` at
  the project root, collection `it_helpdesk_kb`, cosine similarity). List-
  type metadata (`labels`, `restricted_to`) serialized to comma-joined
  strings — Chroma metadata only accepts scalars. **Done and tested
  2026-09-09** — migrated off an earlier sentence-transformers/
  bge-base-en-v1.5 draft to match Section 1's hard constraint; passes
  `cache_dir` pointing at `local_model_cache/` inside the project root
  explicitly (see Section 6, bug #4). Reads
  `data/chunks/chunks_from_confluence.jsonl`.
  - BGE convention: query embeddings need the prefix `"Represent this
    sentence for searching relevant passages: "`; indexed chunk text does
    NOT get this prefix. Different model families (E5, etc.) have
    different conventions — check the model card if this changes.
- `inspect_chroma.py` — readable dump of stored chunks + CSV export to
  `data/processed/chroma_contents.csv`.
- `visualize_embeddings.py` — PCA-reduced 2D plots of the embedding space,
  colored by category and by status. Saves to `outputs/` (gitignored).

### Query understanding (Steps 1-4)
- `src/query_parsing.py` — Step 1, regex pre-parsing, no LLM. Strips
  ticket#/priority/category/subcategory fields (including pipe-separated
  "Priority: X | Category: Y" on one line, and "Description:" label
  prefixes) into `cleaned_text`; extracts error codes (`0x...` hex),
  IPs, hostnames, file paths, and inline ticket numbers into separate
  fields. `tests/test_query_parsing.py` hand-verifies 16 cases against
  real text plus sweeps all 67 queries for false positives — 0 found.
- `src/step2_routing.py` — Step 2, one Gemini call, own function
  (deliberately kept separate from Step 3 — see Section 5, routing may
  grow agentic later). Classifies into `off_topic` / `genuine_question`
  only — **`escalate` and `canned_answer` intents were both dropped**,
  see Section 3 and the top-of-file user-model note. 2 few-shot examples
  with stated reasoning, forced JSON schema output.
  `tests/` verified 100% correct on all 62 real eval queries, both before
  and after the `escalate` removal.
- `src/step3_understanding.py` — Step 3, one Gemini call, only runs if
  Step 2 returns `genuine_question`. Outputs `rewritten_query` (str or
  list[str] for compound/bundled issues), `category` (8 real values +
  "unclear" fallback — never a closed set, see Section 5), `error_codes`,
  `wants_latest`, `ambiguous`. 5 few-shot examples with reasoning, forced
  schema, plus post-parse validation (`_validate()`) checking category is
  a real allowed value and error codes match the expected hex format —
  schema enforcement only guarantees shape, not content correctness.
  `tests/test_step3_understanding.py`: 92% category accuracy (44/48; the
  4 disagreements are defensible alternate judgments, not clear errors —
  see the test's docstring for the "unanswerable ≠ ambiguous" lesson
  learned while building this test), 100% on error codes and
  `wants_latest`.
- `src/step4_ambiguity_gate.py` — Step 4, pure decision logic, no LLM
  call (Step 3 already produced the signals it needs). `ambiguous=True`
  or `category="unclear"` -> widen the search (no category filter),
  `low_confidence=True`; otherwise pass the real category through as the
  search filter. **REDESIGNED 2026-09-13**: no longer asks a clarifying
  question and loops — see Section 3, this tool is used by IT staff, not
  a confused employee needing to be interviewed. Tested against all 62
  real Step 3 outputs (no new API calls needed), 0 exceptions, all
  correct. One coverage gap: the "unclear but not ambiguous" combination
  has 0 real examples in the current eval set, though the logic is
  covered by a hand-built smoke test.
- `src/pipeline.py` — wires Steps 1→2→3→4→5 into one callable function,
  built 2026-09-14. `PipelineClients.build()` constructs the Gemini
  client, embedding model, reranker, and Chroma collection once (reuse
  across many calls, these are expensive to load); `run_pipeline(raw_message,
  clients)` runs the full chain and returns a `PipelineOutcome`. Handles
  Step 3's compound-query case (`rewritten_query` as a list) by running
  retrieval once per sub-question, since each is a genuinely separate
  search. Unions Step 1's regex-extracted error codes with Step 3's
  LLM-extracted ones rather than trusting only one source. **Known
  simplification**: Step 4's ambiguity gate is applied once per whole
  message, not per sub-question — a compound message with one clear
  issue and one ambiguous issue isn't distinguished yet. Tested against
  real queries end-to-end; found and fixed 2 more eval-label bugs (F05,
  F09 — see "Retrieval eval results" below) in the process.
- `src/eval_pipeline_e2e.py` — full end-to-end eval (Steps 1-9 through
  the real `run_pipeline()` + `generate_answer()`, not just retrieval in
  isolation), built 2026-09-14. Checks 3 things `eval_retrieval.py`
  doesn't: routing correctness (off-topic stops before retrieval),
  generation-mode sanity (`low_confidence=True` must NEVER produce a
  synthesized answer — verified this holds on real queries, not just the
  smoke test), and citation completeness (every citation carries
  `last_updated`/`status`). **RESUMABLE** (`_load_already_done()`) — a
  quota-failed query is never marked done, so a partial run can be safely
  re-invoked later without re-spending quota on queries that already
  passed. Writes to `data/processed/eval_pipeline_e2e_results.txt`
  (append mode on resume). Run `python src/eval_pipeline_e2e.py --sample`
  for a fast 13-query stratified check first; plain `python
  src/eval_pipeline_e2e.py` for the full 62.
  **STATUS: DONE. 62/62 real queries complete, 0 issues found**, split
  across two days due to the daily quota (44 on 2026-09-14, remaining 18
  the following morning after the quota reset). Full results:
  `data/processed/eval_pipeline_e2e_results.txt`.

  Three real bugs found and fixed while building/running this:
  1. The `google-genai` SDK has its OWN internal retry-with-backoff on
     429s, independent of any retry loop in calling code — pacing too
     aggressively (4.5s, fine for single-call scripts) let those internal
     retries silently compound into 10+ minutes per query on a 3-call
     chain. Fixed with 15s pacing between queries.
  2. A transient `503` crashed the entire run with no retry at query 4
     the first time. Fixed by wrapping each query in its own 4-attempt
     retry (10s/20s/30s backoff) so one transient blip doesn't lose the
     whole run's progress.
  3. **The free-tier quota resets at midnight PACIFIC TIME, not the
     local machine's midnight, and not exactly 24h after first use.**
     The local system clock rolling over to a new calendar day does NOT
     mean the quota has reset — confirmed by hitting the same 429 wall
     again even after the local date changed, because Pacific midnight
     hadn't arrived yet. Worse: right at the actual Pacific-midnight
     boundary, enforcement was briefly inconsistent — isolated single
     test calls succeeded while a resumed multi-query run still hit 429s
     for several more queries before genuinely clearing. **Lesson: when
     resuming after a daily-quota wait, don't trust one successful test
     call as proof the quota is open — if a resumed batch run still
     fails, wait a bit longer past the theoretical reset time rather
     than assuming there's a new bug.**

  **4th gap found AFTER the 62/62 run completed, fixed same day**: the
  script only ever saved a plain-text `OK`/`ISSUES` line per query — the
  actual `hit_rank`/`trap_beat_real`/generation-`mode` data needed to
  report a real recall@5/top-1 number existed only in memory during each
  run, printed to console for whichever SUBSET happened to run that
  invocation (e.g. the summary after a quota-driven resume only covered
  the 18 queries in that resume, not the true combined 62). **Fixed**:
  every run now also appends one JSONL record per query to
  `<name>_metrics.jsonl` (`_metrics_path()`), and the summary
  (`_print_summary()`) is computed by reading and deduplicating (keep
  latest per query_id, skip `failed=True` records) EVERY record ever
  saved to that file — not just the current invocation's in-memory list.
  This means the reported numbers are always correct across partial/
  resumed/multi-day runs. **Going forward: every pipeline change should
  be re-tested with this script and the resulting recall@5/top-1/
  confuser-trap numbers recorded here, the same discipline already used
  for `eval_retrieval.py`** — don't rely on "0 issues found" alone, that
  only means no crashes/contract violations, not that accuracy held.
- `src/step9_generation.py` — Step 9, generation, built 2026-09-14.
  `generate_answer(sub_query, search_results, low_confidence, client)`
  returns a `GenerationResult` with `mode` = `"synthesized"` (Gemini
  wrote an answer, citations attached) or `"candidates_only"`
  (low_confidence=True or zero results — no Gemini call, just a caveat +
  the raw ranked candidates). `format_for_display()` renders either mode
  to plain text, always including every source's `last_updated`/`status`
  unconditionally (Layer 2 staleness disclosure, Section 5). See Section
  3's Step 9 entry for the sourced reasoning behind never synthesizing on
  low-confidence results, and the real confuser-pair example caught while
  testing this (the model correctly refused to fabricate an answer from
  a mismatched top-ranked decoy chunk). **Extended 2026-09-17** with
  `superseded_alternatives` (a new structured field, populated in BOTH
  modes) — see `src/same_topic_detection.py` and Section 8's "Same-topic-
  different-age detection" writeup.
- `src/same_topic_detection.py` — `find_same_topic_pairs()`, built
  2026-09-17. Pure local function (no Gemini, no DB), groups a
  search-result set by title and flags any title with 2+ distinct
  page_ids as a same-topic-different-age pair. See Section 8's writeup
  for why this signal is safe (doesn't leak the eval-only
  `superseded_by_page_true` field) and how often it actually fires.
- `src/calibrate_confidence.py` — builds the calibration data Step 8's
  threshold is based on, built 2026-09-14. Fully LOCAL, no Gemini call
  (reranking runs on-machine) — runs all 112 real eval queries (50
  ticket + 62 edge-case) through `hybrid_search_reranked()` directly on
  raw query text, records each top result's `rerank_score` against
  whether it was actually correct. Saves
  `data/processed/confidence_calibration.jsonl` and prints a bucketed
  score-vs-accuracy table. Per the sourced guidance
  (claudecertificationguide.com 5-5, human-review-calibration, verified
  2026-09-14): "run the model, compare its confidence scores to actual
  accuracy, and build a calibration curve" — this is that curve, built
  from real data, not a guessed number.
  **Real result**: score < 0: ~45-50% accuracy; score 0-2: ~38% accuracy
  (WORSE than below 0 — traced to the known VPN confuser-pair weakness
  landing moderate, not low, scores while still wrong); score >= 2:
  75-80% accuracy. No reliable gradient below 2.0, so the threshold is a
  single cut there, not a finer scale.
- `src/step8_confidence.py` — Step 8, confidence threshold + knowledge-gap
  log, built and wired into `pipeline.py` 2026-09-14. `check_confidence()`
  compares a result set's top `rerank_score` against `CONFIDENCE_THRESHOLD
  = 2.0` (see `calibrate_confidence.py` above — NOT guessed). No LLM call
  — uses the reranker's own computed score, never a model self-reporting
  confidence in words. `log_knowledge_gap()` appends to
  `data/processed/knowledge_gaps.jsonl` (timestamp, query, sub_query,
  category, top_score, reason) whenever a result is below threshold or
  empty — per the real agent-assist research (Section 3, Step 2):
  Zendesk/Forethought/Ada all treat a low-confidence result as a content
  gap for the KB owner to review, not just a dead end. This is a genuine
  project feature ("what does our KB not cover well"), not just a
  fallback. `PipelineResult.low_confidence` is now `True` if EITHER
  Step 4 (category ambiguity) OR Step 8 (weak search results) says so —
  two independent signals feeding the same Step 9 behavior.
  **Bug found and fixed while smoke-testing**: the first version of this
  file's own `__main__` smoke test called `run_pipeline()` (which already
  runs Step 8 internally) and then redundantly re-ran
  `check_confidence()`/`log_knowledge_gap()` a second time on the same
  result, double-logging every gap. Fixed by having the smoke test read
  `PipelineResult.low_confidence` directly instead of recomputing it.
  **Honest finding from real testing**: on the 13-query stratified sample
  run through the full pipeline, 10 of 13 queries produced at least one
  knowledge-gap log entry. This matches the calibration data (only 47/100
  scored confident) — well over half of real queries currently fall
  below the confidence threshold, mostly driven by the documented
  confuser-pair weakness, not a bug in Step 8 itself.
- `src/step0_context.py` — Step 0, conversation contextualization, built
  2026-09-14. **Two-tier conversation state** (`ConversationState`),
  built as designed from the start rather than a simpler flat-list
  version, per explicit user direction: (1) `case_facts` — a dict of
  exact values (error codes, ticket numbers, IPs, hostnames, file paths,
  confirmed category) that NEVER gets summarized or dropped, fed by Step
  1's regex extraction on every user turn; (2) two levels of narrative
  that DO compress over time — `current_turns` (raw, unsummarized,
  live-session-only) and `past_sessions` (a list of short summary
  strings, one per closed prior session — `close_session()` compresses
  `current_turns` into one summary and discards the raw turns, while
  `case_facts` migrate to the permanent dict untouched first). Matches
  the memory-management guidance in Section 5 (naive single-blob
  summarization silently destroys exact values).
  `contextualize(state, new_message, client)` skips the Gemini call
  entirely if `state.has_context()` is False (genuine first message,
  matching the "never pay this call's cost on a simple standalone
  message" design goal from Section 3) — verified this actually skips in
  testing, not just claimed.
  **Bug found and fixed while smoke-testing**: `add_turn()` originally
  only appended to `current_turns` without extracting case facts from
  it — so a fact mentioned in an earlier turn (via `add_turn`) rather
  than the final message passed to `contextualize()` was silently
  missing from `case_facts`. Fixed by having `add_turn()` run Step 1's
  extraction on every user turn as it's added.
  **Wired into `pipeline.py` as an OPTIONAL, backward-compatible
  parameter**: `run_pipeline(raw_message, clients, conversation_state=None)`
  — if `None` (unchanged default, what all pre-2026-09-14 tests used),
  Step 0 never runs and behavior is identical to before. If a
  `ConversationState` is passed, Step 0 runs first and its
  `standalone_message` feeds Step 1 onward instead of the raw message.
  **Tested against all 5 real multi-turn eval queries** (G02-G06 — these
  were BLOCKED and unscored until Step 0 existed, see Section 8), via
  `tests/test_step0_context.py`, which builds a real `ConversationState`
  from each query's `prior_turn`/`current_turn`/`prior_session_summary`/
  `turn_1..3` encoding and runs it through the actual `run_pipeline()`.
  **Result: 4/5 correct.**
    - G02 (real follow-up needing prior context): correct, hit_rank=1.
    - G03 (resuming from a summarized PAST SESSION, not raw turns):
      Step 0's rewrite was clean and reasonable ("How to resolve ongoing
      VPN disconnection issues"), but retrieval missed entirely (not in
      top 5) — traced directly to the ALREADY-DOCUMENTED confuser-pair
      weakness (Section 8): the decoy page 200060 won with a rerank
      score of 2.36 (above Step 8's 2.0 confidence threshold, so this
      would show as a CONFIDENT wrong answer, not even flagged
      low-confidence — a real, concerning combination worth noting).
      This is a retrieval limitation, not a Step 0 bug.
    - G04 (3 fragmented messages merged into one question): correct,
      hit_rank=5.
    - G05 (follow-up that explicitly ABANDONS the prior topic): correct
      — cleanly dropped old VPN context rather than dragging it into the
      new MFA-lockout question.
    - G06 (recency intent surfaces ONLY in the follow-up, not the
      original question): correct, AND verified the `wants_latest`
      signal survived Step 0's rewrite into Step 3's structured output
      (`wants_latest=True` on the final outcome) — this was a specific
      risk worth checking, not just assuming it carried through.
  **DESIGN NOTE for any future extension, not yet a live bug (2026-09-17)**:
  `case_facts` is currently populated ONLY from Step 1's regex extraction
  (error codes, ticket numbers, IPs, hostnames, file paths) — values that
  are exact identifiers and genuinely never go stale (an error code
  mentioned 3 turns ago is still that same error code). This design is
  SAFE only because nothing currently writes a LIVE SYSTEM STATE value
  (e.g. a ticket's current status/resolution-ETA from an external lookup)
  into `case_facts`. If a future extension ever adds a tool call that
  looks up live status and wants to remember it across turns, it must
  NOT be stored in `case_facts` the same way — `case_facts` is
  never-summarized AND never-refreshed, so a stale "Status: PENDING"
  looked up hours ago would sit there permanently, exactly the
  stale-tool-result bug pattern described in a real interview-style
  question the user surfaced (ExamHeist Q175, IT/billing-support agent
  scenario): resuming a session with an old tool result in context
  caused the agent to reference outdated data even after a fresh tool
  call returned different information; the credited fix was to carry
  forward only a compressed narrative summary of what happened, and
  re-run any live lookups fresh rather than trusting cached tool output
  from history. This matches `close_session()`'s existing design for
  narrative (compress, don't carry raw history forward) but is a
  DIFFERENT rule than `case_facts` follows (never expire) — the two must
  not be conflated. Any live-status field added in the future belongs in
  a third bucket: fetched fresh every session, never cached in
  `case_facts`, never treated as a permanent fact.
  **CATEGORY-KEYED case_facts, built 2026-09-17**: found a related, real
  gap independently while reviewing a second interview-style question the
  user surfaced (ExamHeist Q64, multi-issue support session) — the OLD
  `case_facts` was one FLAT dict, so a ticket number for issue A and one
  for a genuinely unrelated issue B in the same session landed in the
  same undifferentiated list, with no way to tell which belonged to
  which. That question's own credited answer didn't specify a fix this
  detailed (it only established "structured facts in a separate layer,"
  already true here) — the category-keying below is this project's own
  extension beyond what the question itself called for, confirmed with
  the user before building. Restructured `case_facts` to be keyed by
  CATEGORY (Step 3's 8-value taxonomy) instead of flat — see
  `step0_context.py`'s "CATEGORY-KEYED case_facts" docstring section for
  the full design, including the real sequencing problem this surfaced
  (category isn't known until Step 3 runs, AFTER Step 0/Step 1 already
  extracted a turn's facts) and its fix: a provisional `_general` bucket
  plus `file_facts_for_category()`, called from `pipeline.py` right after
  `understand_query()` resolves the real category. `all_facts_flat()`
  gives Step 0's own prompt-builder the flattened view it needs (it
  doesn't care about issue boundaries, only "everything known so far").
  **Tested live through the real pipeline** (not just simulated): two
  genuinely unrelated messages in one session (a VPN error, then an
  unrelated MFA lockout) correctly landed in two separate category
  buckets (`Network Infrastructure` / `Access & Identity Management`).
  Re-ran the existing G02-G06 regression suite afterward — same 4/5
  result as before, no regressions from the restructuring.
  **Few-shot examples added 2026-09-17**: `SYSTEM_PROMPT` was pure prose
  rules with zero examples, unlike Step 3's proven 5-example pattern.
  Another interview-style question the user surfaced (ExamHeist Q74)
  finds that long prose-only rule blocks lose effectiveness as
  conversation history grows, while few-shot examples "strongly anchor
  behavior." Not something our own testing had caught failing (the
  G02-G06 eval only exercises 1-2-turn histories, nowhere near Q74's
  12+-turn scenario) — added proactively since the fix was small and the
  pattern already measurably worked for Step 3. Added
  `STEP0_FEW_SHOT_EXAMPLES` (4 examples, one per prompt rule, each a REAL
  scenario pulled from the project's own G02/G03/G04/G05 eval cases with
  a stated reason — not invented). **Real result, not assumed**: re-ran
  the G02-G06 suite with the examples in place — **5/5 correct**, up
  from 4/5. G03 (previously the one miss, hit_rank=None) now scores
  hit_rank=5 — a genuine improvement in the rewrite's phrasing (compare
  the two real rewrites: old = "How to resolve ongoing VPN disconnection
  issues", new = "What is the troubleshooting procedure for the VPN
  disconnect issue?", closer to the added examples' phrasing pattern).
  Honest caveat: hit_rank=5 is a WEAK pass, barely inside the top 5 and
  still losing ground to the documented confuser-pair decoy — real
  improvement, not a full fix of that underlying weakness.
  **DESIGN NOTE, documented only, NOT built (2026-09-17)**: a third
  interview-style question (ExamHeist Q71) surfaced a fact TYPE
  `case_facts` doesn't handle — a persistent STANDING RULE/constraint
  (its example: "this character never lies"), distinct from both an
  atomic identifier and a narrative summary. Checked the real eval data
  for anything that would actually need this (a plausible IT-support
  analogue would be "this account is VIP, route with high priority") —
  found nothing exercising it; `Priority: High` in the real eval data is
  a one-time per-message field, not a persistent constraint. Deliberately
  NOT building a speculative bucket for a need the project's own data
  doesn't demonstrate — documented in `step0_context.py`'s module
  docstring so the gap is known if a real trigger case surfaces later
  (e.g. a future ticket-system integration flagging VIP accounts).
- `src/step7_recency.py` — Step 7, conditional recency rerank, built
  2026-09-14. `recency_score()` computes a 0-1 value from `version.when`
  using PER-CATEGORY half-lives (90-365 days, reasoned assumptions about
  real-world IT domain change velocity, NOT calibrated from data like
  Step 8's threshold was — explicit user decision to build per-category
  as originally designed anyway). `apply_recency_rerank()` adds
  `RECENCY_WEIGHT * recency_score` to the existing rerank_score and
  re-sorts — additive, never a replacement; returns results UNCHANGED if
  `wants_latest=False`.
  **Measured, not assumed, to have zero effect at the current weight
  (1.0) — kept anyway, see Section 3's Step 7 entry for the full
  honest writeup and the real published source (arXiv 2509.19376) this
  result matches.**
- `src/capture_step7_raw_data.py` + `src/sweep_step7_weights.py` — split
  into two scripts specifically to avoid re-spending Gemini quota while
  tuning Step 7's weight: the capture script runs the real pipeline ONCE
  (needs Gemini for Steps 2/3 routing+understanding) and saves the full
  `search_results` (real rerank scores + chunk metadata) to
  `data/processed/step7_raw_capture.jsonl`; the sweep script re-scores
  that cached data against multiple `RECENCY_WEIGHT` values with ZERO
  API calls (retrieval/reranking themselves are fully local — same
  principle as `calibrate_confidence.py`). Tested weights 1.0 through
  8.0x, all produced identical results to baseline on the 15-query
  sample. If revisiting Step 7 later, re-run the sweep script freely
  (free) before considering another capture run (costs quota).

### Retrieval
- `src/retrieval_pipeline.py` — BM25 index + `bm25_search()` +
  `reciprocal_rank_fusion()`. **Real bugs found and fixed here — read
  the docstrings before modifying**: stopword removal AND a minimum-
  document-frequency filter (drop terms appearing in <2 of 247 chunks)
  were both necessary; without them, generic words that happen to be rare
  in this specific small corpus (e.g. "keeps," appeared in 1/247 chunks)
  get scored as if they were highly distinctive. Also includes a
  `dense_search()` function matching `embed_and_store.py`'s conventions.
  Reads `data/chunks/chunks_from_confluence.jsonl`.
  - `hybrid_search()` — BM25 + dense -> RRF fusion -> fixed top-5. No
    confidence/threshold logic yet (Step 8), deliberately — see Section 5.
  - `hybrid_search_reranked()` — **the recommended entry point for new
    code**, added 2026-09-13. Same as above but widens the candidate pool
    to 20 and re-sorts with a local cross-encoder (`BAAI/bge-reranker-base`,
    via `fastembed.rerank.cross_encoder.TextCrossEncoder`, same
    project-local `cache_dir` convention as the embedding model) before
    the top-5 cutoff. Measured to beat `hybrid_search()` on every metric,
    see Section 8's eval results.
- `tests/test_hybrid_rescue.py` — the verification script for whether
  dense/hybrid actually rescues a query BM25 fails on. **Run 2026-09-09,
  partial pass** — hybrid beats BM25-alone (real answer reaches top-5) but
  still loses rank #1 to a designed confuser decoy. See Section 7.
- `data/processed/eval_edge_case_queries.jsonl` — 67 hand-written queries
  (62 scorable + 5 blocked on Step 0) covering messy query types real
  tickets don't have, built 2026-09-09 after discovering
  `enriched_tickets.csv` is 100% clean, template-generated text with zero
  examples of raw logs, typos, compound issues, off-topic questions, etc.
  Every row cross-checked against real page IDs and ground truth —
  multiple mislabelings were found and fixed during verification (see
  Section 8's eval results for the lesson learned).
- `src/eval_retrieval.py` — the retrieval eval harness, built 2026-09-13.
  Runs both `hybrid_search()` and `hybrid_search_reranked()` against all
  98 scorable queries (50 ticket + 48 edge-case; 5 blocked), computing
  Recall@5, top-1 accuracy, MRR, and confuser-trap rate. Writes full
  per-query results to `data/processed/retrieval_eval_results.jsonl` and
  `..._reranked.jsonl`. Run: `python src/eval_retrieval.py`. **Slow** —
  the reranker pass does ~2,240 individual (query, candidate) scoring
  calls (98 queries x ~20 candidates x 2 variants), CPU-only on this
  machine (no `onnxruntime-gpu` installed; the GTX 1650 here only has 4GB
  memory and CUDA 11.7, judged not worth the setup effort for an
  occasional eval run — revisit if eval re-runs become frequent).

### UI, session storage, and production monitoring — `ui/`, `src/`

- `src/session_store.py` — durable SQLite session/turn storage, built
  2026-09-18. `sessions` / `turns` / `citations` / `superseded_alternatives`
  tables + `monitoring_summary()`. No new dependency (`sqlite3` is
  stdlib). DB file: `sessions.db` at repo root (gitignored). See Section
  8's "Chat UI, session storage, and production monitoring" writeup for
  the full sourced design rationale.
- `src/chat_service.py` — `send_message()`, wires the real pipeline
  (Steps 0-9) to `session_store.py`. Pure wiring, no new decision logic.
  Fixed a real `used_step0` logging-order bug while smoke-testing — see
  Section 8's writeup.
- `ui/server.py` — FastAPI backend (`/api/messages`, `/api/sessions`,
  `/api/sessions/{id}/turns`, `/api/sessions/{id}/close`,
  `/api/monitoring`). `PipelineClients` built once at startup via
  FastAPI's `lifespan` context manager. Run: `python ui/server.py`, then
  open `http://localhost:8010` (moved off the default 8000 — already in
  use by another process on this machine).
- `ui/static/index.html` — the chat frontend (vanilla HTML/CSS/JS, no
  build step, no framework) — session sidebar, message bubbles,
  low-confidence/candidates-only badges, staleness disclosure, and
  superseded-alternative callouts, all reading directly from Step 9's
  real structured `GenerationResult` fields via the API.

---

## 5. Key design decisions and why (don't relitigate these)

- **This is currently a workflow, not an agent (Anthropic's own
  terminology) — but Step 2 (routing) specifically is expected to grow
  agentic capability later.** Anthropic's "Building Effective Agents"
  guidance (add complexity only when it demonstrably improves outcomes;
  optimize single calls before building elaborate chains) is why Step 0
  stays a small dedicated call rather than being folded into Step 3.
  Steps 2 and 3 were briefly merged into one call on 2026-09-13, then
  **un-merged the same day**: the user flagged that Step 2 (routing) has
  real potential to become agentic later (the model dynamically choosing
  among actions/tools, not just picking one of 4 fixed enum values) —
  keeping it as its own call/function now avoids restructuring Step 3's
  extraction logic later just to upgrade Step 2. Extraction is stable
  regardless of how agentic routing becomes; routing is the part most
  likely to change shape. Don't re-merge these without re-checking
  whether Step 2 still needs standalone room to grow.
- **Permission filtering happens BEFORE retrieval, never after.**
  Confirmed 2026-09-13 across every real system researched (Glean,
  Microsoft 365 Copilot's Graph-permission-scoped Retrieval API,
  Atlassian Rovo, Pinecone's access-control engineering guide): tag
  content with the group/role allowed to see it at ingest time (already
  done — `restricted_to` on every chunk), then apply it as a metadata
  filter during the vector/BM25 query itself. Never let the model see
  restricted content and rely on it to withhold it in the answer. 147 of
  247 chunks (60%) carry a restriction — this is a real requirement, not
  a nice-to-have, and was a genuine gap in the pipeline until Step 4.5
  was added.
- **Confidence thresholds must be calibrated against a labeled eval set,
  never the LLM's own self-reported confidence score.** Multiple
  independent sources (claudecertificationguide.com's escalation,
  multi-pass-review, and human-review-calibration pages, read
  2026-09-13) converge on this: self-rated confidence is poorly
  calibrated — models are often confidently wrong on hard cases and
  needlessly unsure on easy ones. Build Step 8's threshold from the
  reranker score's measured relationship to actual correctness on the
  eval harness (Section 8), not a 1-10 self-rating from a generation call.
- **Conversation memory splits into two parts, not one summarization
  strategy.** A separate, NEVER-summarized "case facts" block (ticket
  IDs, error codes, dates, asset tags — exactly what Step 1's regex
  extracts) carried forward verbatim every turn, alongside a
  summarized/windowed narrative for general conversational flow. Naive
  single-blob summarization silently destroys exact values (a concrete
  example found: "$247.83 for order #8891 placed March 3rd" degrading to
  "customer wants a refund for a recent order").
- **When two documents are ambiguous or genuinely disagree, don't
  silently pick one — surface both.** Applies directly to confuser
  pairs and compare-two-documents queries: acknowledge both retrieved
  documents, cite both, and state the distinguishing context (different
  scenario, different version) rather than the system quietly guessing
  which one is "more correct." This is a generation-layer answer to the
  same problem the confuser-trap-rate metric measures at the retrieval
  layer (Section 8).
- **Metadata filters use `Literal`/enum types, never free strings.** A
  generic `category: str` field lets the LLM output "Networking" when the
  real value is "Network Infrastructure" — silent filter failure, zero
  results, no error. Enum-constrained structured output (grammar-
  constrained sampling, confirmed in both Anthropic's and AWS's own
  documentation) makes this mechanically impossible, not just less likely.
- **Error codes are NOT metadata filters.** They're open-vocabulary, so
  they go into the BM25 query text as search terms instead — BM25 rewards
  exact token matches, which is what identifiers need; embeddings don't
  meaningfully distinguish one code-like string from another.
- **RRF fuses by rank position, never raw score.** BM25 and cosine
  similarity live on incomparable scales; averaging them directly is
  meaningless.
- **Staleness cannot be solved by retrieval ranking alone.** The 10
  "unmarked stale" documents were deliberately built with zero retrievable
  signal. The real fix is a two-layer approach: Step 7 (query-triggered
  recency rerank) catches the case where the user's wording signals
  recency intent; **Layer 2 (always-on, in generation) — done 2026-09-14,
  `src/step9_generation.py`'s unconditional `staleness_disclosure`** —
  is required for the more common case — a plain question with no
  recency wording, where staleness must be disclosed regardless. Standard
  RAG eval metrics (faithfulness, groundedness) are structurally blind to
  this — a stale document can score perfectly on both. A dedicated
  staleness metric is required, using the `staleness_tier` field already
  present in `eval_only_ground_truth.jsonl`.
- **Staleness must be computed from `version.when` (last-updated), NEVER
  `history.createdDate`.** A page created long ago but revised recently
  is current; the reverse is stale. **RESOLVED 2026-09-09** (was a known
  data limitation): `build_confluence_export.py` now has a
  `derive_created_date(when_date, version_number)` helper that offsets
  `createdDate` earlier than `version.when` by 30-120 random days per
  revision beyond the first. Pages with `version_number == 1` (the
  superseded/unmarked-stale tier — created once, never touched) correctly
  keep `createdDate == version.when`, since that IS the realistic case for
  them. Verified after regenerating from scratch: 156/205 pages now
  diverge, the 49 remaining matches are exactly the `version_number == 1`
  pages, and 0 pages have `createdDate` after `version.when`. This means
  staleness/recency logic can now actually be tested against a real
  divergence instead of two fields that happened to always agree.
- **Trashed pages are excluded before chunking, not filtered at query
  time.** This matches real Confluence API default behavior (content
  search defaults to `status=current` only) — it's not an arbitrary
  convenience filter.

---

## 6. Bugs found during development — read before touching related code

1. **Cross-reference leaks in generated SOP text.** Early versions of the
   long-branching articles' "Common misdiagnoses" sections literally named
   their confusable partner document by exact title ("see 'X' instead"),
   handing away the deliberately-hidden pairing. Fixed via
   `strip_cross_references()` in `build_confluence_export.py` — operates
   at sentence/bullet granularity, case-insensitive. Multiple narrower
   regex attempts failed to catch all grammatical variants; verified
   exhaustively with case-insensitive full-corpus search before trusting
   it clean.
2. **Output-file patches vs. pipeline fixes are not the same thing.**
   Several early fixes were applied directly to `confluence_export.jsonl`
   without being folded back into `build_confluence_export.py` — meaning
   the script itself didn't reproduce the clean state if re-run. Always
   verify a fix by regenerating from scratch in a clean folder, not by
   re-checking the already-patched output file.
3. **BM25 on a small corpus needs stopword removal AND a minimum document-
   frequency filter**, not just one or the other. See `retrieval_pipeline.py`
   docstrings for the exact failure case this fixes.
4. **`fastembed`'s default cache location (OS Temp folder) is not
   reliable** — it was silently wiped between sessions on the user's
   Windows machine (likely IT policy or disk cleanup). Always pass an
   explicit `cache_dir` pointing inside the project folder, consistently,
   in every script that instantiates `TextEmbedding`.
5. **Every `open()` call reading/writing the JSONL corpus files needs
   `encoding="utf-8"` explicitly, on Windows.** `build_confluence_export.py`
   writes with `json.dumps(..., ensure_ascii=False)` (correct, to keep real
   unicode characters like em-dashes readable) but the `open()` calls
   themselves had no `encoding=` argument, so Python fell back to the
   Windows system codepage instead of UTF-8 on both the write and read
   side. This surfaced as a `UnicodeDecodeError` when re-reading the file
   after regenerating it. Fixed across every script that touches
   `confluence_export.jsonl`, `eval_only_ground_truth.jsonl`,
   `rag_corpus_full.jsonl`, and `chunks_from_confluence.jsonl`
   (`build_confluence_export.py`, `chunk_confluence.py`,
   `embed_and_store.py`, `retrieval_pipeline.py`) — always pass
   `encoding="utf-8"` explicitly on every `open()` touching these files,
   never rely on the platform default.

---

## 7. Former blocker — RESOLVED 2026-09-09 (kept for history)

**Problem (no longer in effect):** `fastembed` could not download
`BAAI/bge-small-en-v1.5` on the user's corporate Windows machine. Timeline
of failures: SSL certificate verification error (resolved via
`pip install pip-system-certs`) -> `Host not in allowlist: huggingface.co`
type block -> a `403 Forbidden` even attempting a browser-adjacent path.
The model DID work once, briefly, then was lost because it was cached in
the OS Temp folder and got wiped.

**Resolution:** On 2026-09-09, `huggingface.co` and its model-file CDN
were both directly reachable again (tested with raw `urllib.request`
before touching any project code) — the corporate network restriction is
no longer in effect, for reasons unknown (could change back). Set up a
dedicated Python 3.11 venv (`.venv/`, installed via winget since neither
of the machine's existing Python installs — 3.14 and Anaconda 3.9 — was a
good fit), installed `fastembed` + `chromadb` with no PyTorch dependency
needed, and downloaded `bge-small-en-v1.5` (~65MB) directly into
`./local_model_cache/` (project-local, per bug #4 in Section 6 — never the
OS temp folder). `embed_and_store.py` now uses this working setup.

**If this blocker recurs** (network access revoked again), the original
options are still valid, roughly in order of likely success:
1. Ask a teammate in the same working group if they already have the
   model files locally — fastest fix if available.
2. Ask IT/infra whether an internal Artifactory/Nexus mirror already
   proxies Hugging Face — a much easier approval than "unblock
   huggingface.co" outright.
3. Download the model folder via a non-corporate network (phone hotspot,
   home connection) and transfer via USB/internal file share.
4. Once files are obtained by any means: point `fastembed` at the local
   folder directly, and **always pass `cache_dir` pointing inside the
   project folder explicitly** — never rely on the default cache location.

**`test_hybrid_rescue.py` has been run — PARTIAL success, corrected
2026-09-09 after an initial misread.** BM25 alone fails badly on "VPN
keeps disconnecting": it surfaces irrelevant "VPN Access Request
Procedure" pages at the top, with the real answer nowhere near top-5.
Dense search and RRF-fused hybrid both improve on this — the real correct
SOP (page `200046`, "Network SSL VPN failure") makes it into the top 5 —
but **neither correctly ranks it #1**. Rank #1 in both dense-alone and
hybrid goes to page `200060`, "VPN client hangs on connect" — which
`eval_only_ground_truth.jsonl` identifies as a `confusable` decoy
specifically built to be mistaken for `200046`. (An earlier pass through
this test misread `200060` ranking first as the correct answer, purely
from its title sounding right, without checking the actual ground truth
file — a reminder to always verify a retrieval result's `page_id` against
`eval_only_ground_truth.jsonl`, never just eyeball the title.) So the
honest conclusion: hybrid retrieval measurably helps over BM25-alone
(real answer moves from "not in top 10" to "top 5"), but does NOT yet
solve confuser discrimination — the real answer still loses the #1 spot
to its designed decoy. This is exactly the kind of case the confuser-pair
eval metric (Section 8, step 3) needs to catch and quantify, not something
Steps 0-4 (query understanding) are likely to fix on their own.

---

## 8. Recommended order of remaining work

1. ~~Unblock the model download~~ — done 2026-09-09, see Section 7.
2. ~~Run `test_hybrid_rescue.py`~~ — done 2026-09-09, partial result: hybrid
   beats BM25-alone but still loses to a confuser decoy at rank #1, see
   Section 7.
3. ~~Build a minimal eval harness~~ — done 2026-09-13, see "Retrieval eval
   results" subsection below for the full numbers and methodology.
4. ~~Add a reranker~~ — done 2026-09-13, see below. Directly targeted the
   confuser-discrimination weakness the eval harness measured; use
   `hybrid_search_reranked()` for all new code (see module docstring in
   `retrieval_pipeline.py`).
5. ~~Build Steps 1-4~~ — done 2026-09-13. `src/query_parsing.py` (Step 1),
   `src/step2_routing.py` (Step 2), `src/step3_understanding.py` (Step 3),
   `src/step4_ambiguity_gate.py` (Step 4), each with a real test in
   `tests/` against the 62-query eval set (no new eval queries needed —
   the existing set already covered this well).
   ~~Not yet wired together~~ — **done 2026-09-14**: `src/pipeline.py`
   wires Steps 1→2→3→4→5 into one callable function (`run_pipeline()`),
   tested against real queries end-to-end. Found and fixed 2 more
   eval-label bugs (F05, F09) while testing this — see the "Retrieval
   eval results" 2026-09-14 update above.
   ~~Step 0 (conversation contextualization)~~ — **done 2026-09-14**,
   `src/step0_context.py`, two-tier conversation state built as designed
   (never-summarized case facts + compressible narrative). Wired into
   `pipeline.py` as an optional `conversation_state` parameter, tested
   against the 5 previously-blocked G02-G06 queries (4/5 correct, see
   Section 4's writeup). These 5 queries are still not picked up by
   `eval_retrieval.py` itself (that script deliberately tests raw
   retrieval only, not the full multi-turn pipeline) — they're scored by
   `tests/test_step0_context.py` instead.
6. ~~Build Step 7 (recency rerank)~~ — done 2026-09-14, `src/step7_recency.py`.
   Measured ZERO ranking effect on the 15-query sample tested (weights
   1.0-8.0x, no change) — traced to the sample lacking genuine
   cross-document competition, matching a real published finding on
   this exact limitation (see Section 3/4). Kept as-is; NOT re-tuned
   further without a bigger sample, to avoid spending more quota chasing
   a result the sample can't actually measure. The real fix for the
   underlying need (surface staleness fairly when multiple same-topic
   documents exist at different ages) is now planned as a "show
   multiple candidates" design extension, not a bigger recency weight —
   see Section 3.
7. ~~Build Step 8 (confidence threshold)~~ — done 2026-09-14, see
   Section 4. Threshold (2.0) calibrated from 112 real queries, not
   guessed. Unanswerable queries (13 in `eval_edge_case_queries.jsonl`)
   don't get a dedicated eval pass yet — worth deciding once more
   real-world usage data exists on what "correctly declining to answer"
   should be measured against.
8. ~~Design and build the generation step~~ — done 2026-09-14,
   `src/step9_generation.py`. Citation back to source page: done.
   Staleness disclosure (Layer 2): done, unconditional. Graceful
   "I don't know" when confidence is low: done via the `candidates_only`
   mode, now gated on EITHER Step 4's or Step 8's low_confidence signal
   (both wired in as of 2026-09-14).
9. ~~Re-run the full eval harness after Step 7 is built~~ — done
   2026-09-14, see "Retrieval eval results" below (Step 7 itself measured
   zero effect on its own sample, already documented in item 6 above).
10. ~~Build a generation-quality eval~~ — done 2026-09-14, see
    "Generation-quality eval results" below. Faithfulness/groundedness +
    answer relevance (the RAG triad's generation half,
    `reference_docs/rag_evals.md` 4.2-4.3) scored via LLM-as-judge on a
    20-query stratified sample, plus 2 deterministic contract checks
    (confidence-gating, staleness-citation). Found and fixed a real bug
    IN THE JUDGE ITSELF while running this (not in Step 9) — see below.
11. ~~Build a real UI, session persistence, and production-style
    logging~~ — done 2026-09-18, see the "Chat UI, session storage, and
    production monitoring" writeup below.

### Chat UI, session storage, and production monitoring (2026-09-18)

Built the tool's first real interactive interface (previously only
reachable via test scripts) plus durable session storage and production-
style per-turn logging, per explicit user request: "build a ui for the
chatbot... every conversation and every session should be tracked and
stored appropriately and must also be evaluated as well. properly every
convo logged like in production systems."

**Checked Anthropic's own docs first**, per that same request. Fetched
`anthropic.com/engineering/demystifying-evals-for-ai-agents` and
`.../multi-agent-research-system` directly (2026-09-18). Two concrete,
usable findings:
- Their own production system explicitly monitors **"agent decision
  patterns and interaction structures... without monitoring the contents
  of individual conversations, to maintain user privacy."** Shaped the
  core design decision below: log the SAME deterministic signals every
  eval script in this project already computes (category, confidence
  score, mode, whether a knowledge gap or same-topic pair fired) on
  every real turn automatically, rather than adding a new LLM-judge call
  on every live conversation — that stays a deliberate, curated,
  quota-conscious eval-sample activity (`judge_generation_eval.py`), not
  a live-traffic cost. This was also an explicit user decision when
  offered the choice.
- Anthropic's own term for the unit worth capturing is a **"transcript/
  trace/trajectory"** — the full decision trail, not just the final
  answer. Mirrored in the `turns` table's shape below (one row per
  sub-result, carrying the pipeline's actual decisions: category used,
  confidence, mode, citations, superseded alternatives), not just a
  chat-log's raw text.
- **Honest gap**: neither Anthropic doc gives a concrete logging SCHEMA
  or a PII/retention framework — checked directly, not assumed absent.
  The actual table shapes below are reasoned from first principles, not
  sourced, and are flagged as such in `session_store.py`'s own docstring.

**Storage — SQLite, not JSONL** (explicit user decision, offered as a
choice): every prior eval script in this project used JSONL, appropriate
for a fixed offline eval set; live conversation data needs to be
queried by session/date/category/confidence, which a real embedded
database supports and flat files don't. `src/session_store.py` — 5
tables (`sessions`, `turns`, `citations`, `superseded_alternatives`,
plus indexes), no new dependency (`sqlite3` is stdlib). Includes
`monitoring_summary()` — the production-monitoring aggregate query
(low-confidence rate, knowledge-gap rate, same-topic-pair rate,
category distribution), the same metrics `eval_pipeline_e2e.py`/
`calibrate_confidence.py` already compute on the curated eval set, now
computable over REAL logged traffic instead.

**Wiring — `src/chat_service.py`**: `send_message(session_id, raw_message,
conversation_state, clients)` runs the real pipeline (Steps 0-9) and
logs every resulting turn — no new pipeline/business logic, pure wiring,
same convention as `pipeline.py` itself. **Real bug found and fixed
while smoke-testing this module**: `used_step0` was originally computed
by calling `conversation_state.has_context()` AFTER `run_pipeline()`
already ran — but `run_pipeline()` calls `add_turn()` internally as a
side effect, so by the time the check ran, the just-added message always
made `has_context()` return True, even on a genuine first message with
no real prior context. Fixed by capturing `has_context()` BEFORE calling
`run_pipeline()`. Verified against real data: turn 1 (genuine first
message) now correctly logs `used_step0=0`; turn 2 (a real follow-up)
correctly logs `used_step0=1`.

**UI — `ui/server.py` (FastAPI) + `ui/static/index.html`**: a local web
app, Claude-Desktop-style layout (session sidebar, message bubbles,
composer). `PipelineClients` built ONCE at server startup (FastAPI
`lifespan`, not the deprecated `on_event` pattern) and reused across
every request, matching `PipelineClients`' own docstring intent.
`ConversationState` objects live in an in-memory dict keyed by
`session_id` for the server process's lifetime — NOT persisted across a
restart; `session_store.py`'s SQLite database is the durable record, the
in-memory dict is only Step 0's live working state within one server
run (a documented limitation, not an oversight — a real multi-instance
deployment would need Redis or similar, out of scope for a local,
single-user tool). UI surfaces the low_confidence/candidates_only
badges, the caveat box, superseded-alternative callouts, and the
unconditional staleness disclosure — every structured signal Step 9
computes, not just the answer text.

**Port note**: originally 8000, but something else on this machine was
already listening there (checked via `netstat`, did not investigate or
kill it — not this project's process to touch) — moved to 8010 instead.

**Tested end-to-end via real HTTP calls, not just unit-level**: posted a
real first message through `/api/messages` (correct low-confidence
handling, correct staleness disclosure), then a real follow-up in the
same session (Step 0 correctly carried the VPN topic forward into the
rewrite), then confirmed `/api/sessions` and `/api/monitoring` both
reflect the real logged data. Full round trip verified working, not
assumed from unit tests alone.

Run: `python ui/server.py`, open `http://localhost:8010`.

### Retrieval eval results (2026-09-13, numbers refreshed 2026-09-14)

Built `src/eval_retrieval.py`, which runs every query through the real
Chroma collection + BM25 index and scores against known-correct page IDs
— no stand-ins. Two query sources, scored separately then combined:
- 50 real employee ticket descriptions (`data/raw/enriched_tickets.csv`,
  one per subcategory) — realistic content, but all clean, well-formed
  text (the ticket generator never produced messy/edge-case text).
- 62 hand-written edge-case queries (`data/processed/eval_edge_case_queries.jsonl`),
  50 scorable + 12 unanswerable-by-design, covering the messy query types
  (A-G map, see conversation history) real tickets don't have — typos,
  code-mixed language, pasted error logs, compound issues, off-topic
  questions, confuser-pair traps, etc. 5 more (G02-G06) are multi-turn
  conversation tests **blocked on Step 0** (not yet built) and excluded
  from scoring, reported separately.

**Metrics**: Recall@5, top-1 accuracy, MRR, and a project-specific
**confuser-trap rate** — for queries with a known decoy document
(`confuser_trap_page_id` in the eval set), how often does the decoy
outrank the real answer at #1. Standard recall/MRR are blind to this: a
decoy in the top 5 alongside the real answer still counts as "recall
achieved" even though a system reading only the #1 result would confidently
hand the user the wrong document.

**2026-09-14 correction**: found and fixed 2 more mislabeled eval rows
(F05, F09) while testing `src/pipeline.py` end-to-end — both were marked
`unanswerable: True` because their topic's ONLY KB page was `trashed`,
without checking whether a newer current replacement existed for the
same topic. It did, for both (every trashed-tier topic in
`eval_only_ground_truth.jsonl` has a real `superseded_by_page_true`
replacement — checked systematically, no other instances of this found).
Numbers below are the re-run after that fix (n moved from 48/98 to
50/100 scorable edge-case/combined queries).

| Metric | Baseline (`hybrid_search`, RRF only) | Reranked (`hybrid_search_reranked`, BAAI/bge-reranker-base) |
|---|---|---|
| Recall@5 (combined, n=100) | 91.0% | **93.0%** |
| Top-1 accuracy (combined) | 47.0% | **59.0%** |
| MRR (combined) | 0.649 | **0.706** |
| Confuser-trap rate (n=4 known traps) | 75.0% (3 of 4 decoys won) | **50.0%** (2 of 4) |
| Ticket-query top-1 | 66.0% | **76.0%** |
| Edge-case-query top-1 (n=50) | 28.0% | **42.0%** |

**Adding the reranker improved every single metric with zero regressions.**
Full per-query results: `data/processed/retrieval_eval_results.jsonl`
(baseline) and `..._reranked.jsonl` (reranked).

**Remaining known weakness (2 of 4 confuser cases, both the same pair)**:
queries about "black screen at boot" still sometimes rank the decoy page
200056 ("Graphics driver crash causing black screen") above the real
answer 200036 ("Laptop screen black screen") — even though 200056's own
body text explicitly says its symptom happens *"rather than at boot"* and
is app-triggered. The reranker matches on topic-level similarity (black
screen, graphics, crash) without reasoning about that stated distinguishing
condition. This is a real limitation of similarity-based reranking, not a
bug — an LLM-based check that can actually read and reason about the
distinguishing text (rather than another layer of similarity scoring) is
the likely next lever if this needs to improve further, but hasn't been
tried. Don't spend time re-tuning the embedding model for this — per
`reference_docs/rag_evals2.md`'s anti-pattern list, that's specifically
called out as low-value for a ranking (not recall) problem like this one.

**CORRECTION 2026-09-14, important caveat on the confuser-trap metric
above**: investigated the 2 remaining confuser-trap cases (C11, E10)
directly through the REAL full pipeline (`run_pipeline()`, including
Step 3's query rewrite) rather than trusting the raw-retrieval-only
number, and found NEITHER is actually a live problem:
- **E10** (code-mixed Hindi/English): `eval_retrieval.py` feeds raw,
  untranslated query text straight to search, by design (it deliberately
  tests the retrieval layer in isolation). But the real pipeline runs
  Step 3 first, which rewrites code-mixed text into clean English before
  any search happens. Ran E10's exact raw text through `run_pipeline()`:
  Step 3 correctly rewrote it to "Laptop screen goes completely black
  after booting, nothing is visible," and with that text the real answer
  (200036) wins clearly — decoy 200056 doesn't even make the top 5.
  **Not a real problem** — it's an artifact of testing one layer on text
  a real user's message is never actually sent to search as.
- **C11** ("...my laptop's graphics driver seems to be crashing and
  causing a black screen" — the query itself uses the DECOY's own
  vocabulary): in the real pipeline, 200036 (the correct answer) does
  win — decoy 200056 doesn't appear in the top 5 either. But the winning
  score is thin (0.97), so Step 8's confidence check correctly flags
  `low_confidence=True` and the system shows ranked candidates instead
  of a synthesized answer, rather than confidently repeating the query's
  own likely-wrong self-diagnosis back. **Working as designed**, not a
  bug — this is Step 8 correctly hedging on a genuinely tricky case
  where the person misdiagnosed their own issue.

**Lesson**: a metric from a single-layer eval (retrieval tested in
isolation) can look like a live failure that doesn't actually survive
contact with the full pipeline — always check a "confirmed" weakness
against `run_pipeline()` end-to-end before spending more effort chasing
a fix for it. The retrieval-only confuser-trap number above is still
useful for tracking the retrieval LAYER's own behavior in isolation, but
"decoy beat real answer" in that number does not automatically mean the
real tool gives a wrong or unhedged answer to a real user.

**A labeling lesson worth remembering**: while building and re-verifying
`eval_edge_case_queries.jsonl`, multiple hand-written rows were initially
mislabeled by mechanically applying the `confuses_with_subcat` ground-truth
mapping without checking the actual chunk text against the query (e.g.
query C09 was corrected 2026-09-13 after discovering the "decoy" it was
tested against was actually the more accurate answer to that specific
query's wording). Always cross-check a hand-written expected answer
against the real chunk text, not just the ground-truth relationship label.

### Heading-merge chunking bug: found, fixed, measured (2026-09-14)

Investigated the confuser-pair retrieval weakness directly (rather than
tuning the reranker further blind) by reading real failing examples by
hand. The VPN pair (200046 "Network SSL VPN failure" vs. 200060 "VPN
client hangs on connect") looked at first like ordinary vocabulary
overlap between two similar documents. Reading 200046's actual chunks
found something more specific: **it had no "Symptoms"/"Overview"
sentence anywhere in its retrievable text** — every chunk jumped
straight into a branching diagnostic procedure, never stating in plain
language what the problem actually looks like. Every other reliably-
retrieved document in the KB opens with exactly that kind of sentence.

Traced to the actual source: `data/raw/rag_corpus_full.jsonl`'s markdown
body DOES contain the sentence ("Covers SSL VPN client connections that
fail explicitly... rather than hanging indefinitely"), immediately after
a `### Overview` heading with NO blank line in between (a common style
in this corpus). `strip_cross_references()` (`src/build_confluence_export.py`,
removes any sentence naming another doc by title, so the retrievable
corpus can't leak `confuses_with` relationships) splits into paragraphs
on blank lines only — so the heading line and the paragraph below it got
treated as ONE paragraph, joined with spaces after the cross-reference
sentence was stripped, and the result still started with `### `.
`to_storage_format()` then wrapped that ENTIRE joined line — heading
word plus the real disambiguating sentence — into a single `<h4>` tag.
Section headings become chunk TITLES during chunking
(`split_into_sections()`, `src/chunk_confluence.py`), not retrievable
body text, so the one sentence meant to prevent exactly this kind of
confusion was silently discarded before it ever reached the embedding
model.

**Scanned all 205 source documents** for the same pattern
(`src/scan_heading_bug.py`, output saved to
`data/processed/heading_bug_scan.jsonl`): **15 documents affected, 60
heading instances total, and all 15 lost their "Overview" section** —
the section written specifically to state each document's distinguishing
symptom. This is a systemic bug, not a one-off — it affects a large
fraction of the documents most likely to need disambiguating language in
the first place (certificate policy, session timeout, laptop black
screen, VPN failure, brute force login, and 10 others).

**Fix**: `strip_cross_references()` now treats any `## `/`### ` heading
line as its own forced paragraph boundary (via a regex substitution that
inserts blank-line markers around every heading line before the
existing blank-line-based split runs), so a heading can never be merged
with the text that follows it regardless of the source markdown's
spacing. Verified in isolation first (heading now emits as its own
`<h4>`, body sentence survives as a separate `<p>`, cross-reference
sentence still correctly stripped), then regenerated the full pipeline:
`build_confluence_export.py` -> `chunk_confluence.py` (247 -> 283 chunks,
the previously-swallowed sentences now produce real content) ->
`embed_and_store.py` (283 chunks re-embedded into Chroma). Re-scanned the
regenerated export: 0 documents with a suspiciously long `<h4>` tag
remain (was 15).

**Real before/after, `src/eval_retrieval.py` (local only, no Gemini, same
112 real queries)**:

| Metric | Before | After |
| :--- | :--- | :--- |
| Top-1 accuracy (combined) | 52.7% | **58.0%** |
| MRR (combined) | 0.630 | **0.669** |
| Recall@5 (combined) | 83.0% | 82.1% (flat) |
| VPN-query top-1 (n=30, expected=200046) | 30.0% (9/30) | **40.0% (12/30)** |

**Honest finding — net positive, not a clean fix**: of the 30 VPN
queries, 14 changed rank at all; 7 improved, 7 got WORSE. Restoring the
missing content shifted the balance of evidence across the whole index,
and while it net-helped, it wasn't uniformly good for every query. More
importantly: the three specific confuser-trap wins tracked before this
fix (C11, F02, E10 — see "Retrieval eval results" above) were
**UNCHANGED** — the decoy still beats the real answer on C11 and E10
exactly as before. So this fix improved general ranking quality but did
NOT resolve the specific designed confuser pairs that motivated the
investigation — for those, restoring the missing Overview sentence
either isn't disambiguating enough on its own, or the reranker isn't
weighting it strongly enough yet. **The confuser-trap rate itself
remains the single most significant unresolved weakness in the
pipeline** — this fix was worth doing (it was a real, systemic content-
loss bug affecting 15 documents) but is not the full answer to it.

Backed up pre-fix results to
`data/processed/retrieval_eval_results_reranked_BEFORE_headingfix.jsonl`
for reference.

### Generation-quality eval results (2026-09-14)

Built to close the last open item from Section 8: retrieval accuracy and
pipeline-contract correctness (routing, mode enforcement, citation
completeness) were already formally evaled, but Step 9's SYNTHESIZED
ANSWER TEXT itself had only ever been spot-checked manually. Per
`reference_docs/rag_evals.md`/`rag_evals2.md`, this is the "generation"
half of the RAG evaluation triad: Faithfulness/Groundedness and Answer
Relevance, scored via LLM-as-judge on a labeled set, tracked as real
numbers (not vibes).

**Scripts** (same capture/judge split used for Step 7, to isolate the
expensive Gemini-calling part from the cheap, freely-repeatable part):
- `data/processed/eval_generation_sample.jsonl` — 20 queries stratified
  from `eval_edge_case_queries.jsonl`, picked to stress generation
  specifically: unanswerable cases (must not fabricate), compound/bundled
  queries, the F07 stale-duplicate case (the exact bug fixed earlier
  2026-09-14 in Step 9 itself — included as a direct regression check),
  trashed-page cases (F05/F09), recency-intent cases (D03/G06), and 4
  multi-turn Step 0 cases (G01/G03/G05/G06).
- `src/capture_generation_eval_data.py` — runs the REAL pipeline +
  `generate_answer()` once per query (costs Gemini quota: routing +
  understanding + generation calls), saves query/context/answer/mode to
  `data/processed/generation_eval_capture.jsonl`. Run once.
- `src/judge_generation_eval.py` — reads the capture file, scores each
  `synthesized` answer via LLM-as-judge (Faithfulness 1-5, Answer
  Relevance 1-5, separate judge calls per rag_evals2.md's design so a
  drop in one doesn't hide in the other's score), plus 2 DETERMINISTIC
  contract checks that don't need a judge because ground truth is already
  known: (a) `low_confidence=True` must never produce `mode=synthesized`
  (Step 9's own contract), (b) a cited page with `staleness_tier` ==
  `unmarked_stale` must have a same-topic CURRENT page also cited
  alongside it (the only way Step 9's prompt is supposed to surface a
  stale duplicate — as a disclosed aside, never alone). No new generation
  calls happen in this script — it can be re-run for free while tuning
  the judge prompt itself, which is exactly what happened (see below).

**Real numbers (final, 2026-09-14)**: 20 queries, 22 sub-results (2
queries had compound/bundled sub-questions). Only 7/22 sub-results
reached `mode=synthesized` — the rest correctly landed in
`candidates_only` (see "Honest finding" below).

| Metric | Result |
| :--- | :--- |
| Faithfulness (1-5, n=7 synthesized) | **mean 5.00** — scores: [5,5,5,5,5,5,5] |
| Answer relevance (1-5, n=7 synthesized) | **mean 5.00** — scores: [5,5,5,5,5,5,5] |
| Confidence-gating contract violations | **0/22** |
| Staleness-citation contract violations | **0/22** |

**A real bug was found and fixed — in the JUDGE, not in Step 9.** The
first judge run scored F07 (the contractor-compromise / stale-duplicate
query) faithfulness=1/5, reasoning "introduces external information...
about which procedure was more recent." This looked like a regression of
the exact bug fixed earlier the same day. Investigated by manually
reading the captured context and answer before trusting the score:
Step 9's answer was actually correct — it cited the 2026-07 current
procedure as primary and correctly flagged the 2023-11 page as an older,
superseded alternative. The judge scored it as fabrication because
`judge_generation_eval.py`'s own context-formatting function
(`judge_faithfulness()`) only passed each excerpt's `title` and `text` to
the judge, NOT its `last_updated` date — so the judge had no way to
verify a "this is more recent than that" claim it was shown. This is the
identical class of bug fixed in Step 9 itself hours earlier (date only
available where the code assembling context forgot to include it), now
found one layer up in the eval harness that was supposed to be checking
Step 9's work. Fixed by (a) including `last_updated` in each excerpt
shown to the judge, and (b) adding an explicit prompt line telling the
judge that a correctly-ordered "more recent than" claim, backed by the
shown dates, counts as grounded, not as added external information. Also
tightened the deterministic staleness check the same way — it now
verifies a current page was actually cited alongside any stale one,
rather than just flagging the stale citation for manual review. Re-ran
the judge pass only (zero new generation calls, same capture file) and
F07 scored 5/5; the one other imperfect score (F02, 3/5, a laptop
power-on confuser-pair case) also came back 5/5 — both were spot-checked
by hand afterward and the 5/5 scores hold up on manual read.
**Lesson**: an eval harness's own scoring code is just as capable of the
same class of bug as the system under test, and deserves the same
"read the actual data before trusting the number" discipline — a
suspicious score is a prompt to go verify by hand, not just re-run and
hope, and not just accept the first number either way.

**Honest finding — most of this sample did NOT reach synthesis**: 15/22
sub-results landed in `candidates_only` mode, including A01 (the
"clean," canonical VPN query that retrieval alone handles at ~90%+
recall). This is the same confuser-pair confidence weakness already
documented in Section 2's Step 8 status row and calibrated in
`src/calibrate_confidence.py` (only 47/100 real queries scored confident
enough to synthesize) — now confirmed to propagate all the way to the
generation layer, not just retrieval. It means the faithfulness/relevance
numbers above (5.00/5.00) describe a smaller n=7 than the 20-query sample
suggests, and the more consequential number in this eval run is arguably
the 15/22 `candidates_only` rate itself: the system is conservative about
when it will let Gemini write prose at all, which is the intended design
(Section 5: never synthesize on a low-confidence result set) working
as built, not a flaw in Step 9 — but it does mean the confuser-pair
retrieval weakness remains the single highest-leverage unresolved issue
in the whole pipeline, more than any further generation-prompt tuning
would be.

---

### Same-topic-different-age detection (2026-09-17)

Built the "show multiple same-topic documents at different ages" feature
decided earlier ("if same topic but diff dated files, we will show
both"), after first checking real data rather than guessing at scope.

**How many such pairs actually exist**: queried
`eval_only_ground_truth.jsonl` directly. Of 205 KB documents, 25 are
`superseded` (have a real current replacement); 5 of those are `trashed`
(excluded from the retrievable corpus, can never co-occur with their
replacement in a real search — don't count). Of the remaining 20
retrievable same-topic pairs, 10 are `labeled_archived` (flagged/
findable via metadata) and **10 are `unmarked_stale`** — silently
outdated, indistinguishable from current except by comparing dates. The
10 unmarked_stale pairs are the genuinely hard case this feature exists
for.

**How often they actually co-occur in a real result set — checked, not
assumed**: ran all 10 topics through `hybrid_search_reranked()` locally
(no Gemini needed) before building anything. **8 of 10 topics already
surface BOTH the stale and current document in the same top-5 result**,
often adjacent (ranks 1-2), and for 2 of them (`Unauthorized login
attempt`, `Keyboard malfunction`) the STALE one currently outranks the
current one. This is not a rare edge case for this corpus — it's the
common outcome for this exact document relationship, confirming the
feature was worth building.

**Detection signal, verified safe**: same `title`, different `page_id`,
both present in one result set. Checked this doesn't false-positive on
the deliberately-different "confuser pair" documents (e.g. "Network SSL
VPN failure" vs. "VPN client hangs on connect" — always have DIFFERENT
titles) — confirmed exactly the 20 known same-topic pairs share an
identical title with their counterpart, nothing else in the corpus does.
Deliberately does NOT use the eval-only `superseded_by_page_true` field
— that must never feed retrieval/generation, per Section 1's hard
constraint — title-matching uses only data already in the retrievable
corpus. `src/same_topic_detection.py`, `find_same_topic_pairs()`.

**Sourced design decision on HOW to surface a detected pair**: checked
Anthropic/community research first rather than picking an approach by
instinct — general RAG research on multi-document conflicts didn't give
concrete UX guidance, but `claudecertificationguide.com`'s "Information
Provenance & Multi-Source Synthesis" page (fetched and quoted directly
2026-09-17) did: conflicting/differently-dated sources must never be
silently resolved to one value, AND must be represented as a **discrete
structured field with date prominent — explicitly "not buried in
prose."** This directly revised the plan: the original instinct (rely on
Step 9's existing prompt instruction, which already tells the model to
mention the older version in a sentence) is exactly the pattern this
source calls insufficient. Kept that prompt instruction (it still helps
the read of the prose) but added a NEW structured field,
`GenerationResult.superseded_alternatives` (`step9_generation.py`) —
computed directly from `find_same_topic_pairs()`, independent of
whatever the model's prose says, wired into `generate_answer()` for
BOTH the `synthesized` and `candidates_only` paths (a same-topic pair is
a property of the search results, not of whether synthesis happened),
and rendered in `format_for_display()`.

**Tested end-to-end against all 10 real topics through the REAL
pipeline** (not just local retrieval) — real numbers: **8/10 correctly
detected**, matching the local pre-check almost exactly. One topic
(`app_crash_uninstall`) correctly detected TWO independent pairs in one
result set (a coincidence of two unrelated "Application crash on X"
topics both appearing together — verified the detector kept them
correctly separate, not conflated). The 2 non-detections were both
genuine retrieval-ranking outcomes, not detector bugs — checked each by
hand: `unauthorized_login`'s real search for that phrasing didn't
surface either "Unauthorized login attempt" document at all (pulled
different, plausible security topics instead); the detector correctly
reported 0 pairs because there wasn't one present in that specific
result set.

### Multi-source ingestion: error-code spreadsheet + BM25 gap found and fixed (2026-09-21)

Added a second, structurally different ingestion source per explicit user
request ("show that our pipeline can ingest documents from multisources,
to show diversity of data and ingestion strategies") — deliberately NOT
another prose SOP source (would just replicate what Confluence already
covers), but a genuinely different shape: `data/raw/error_code_lookup.csv`,
15 rows of `error_code, system, short_cause, fix_summary, category`,
covering real error codes including `0x8007042B` (already referenced in
real eval queries A05/C01/D04, and never stated by its exact code string
anywhere in the existing prose KB — verified by grep before building,
not assumed).

**Researched the right architecture before building** (per explicit user
request — "whats the right strategy in production in real systems," not
guessed): real multi-source RAG ingestion uses one CONNECTOR script per
source (different auth/pagination/structure per source), each normalizing
into the same canonical chunk schema, converging only at the embedding
step — not one script with per-source branches. Source: Unstructured.io's
connector/canonical-schema model and Informatica's enterprise RAG
ingestion writeup (both fetched directly 2026-09-21, not paraphrased from
memory). Built accordingly:
- `src/ingest_error_code_spreadsheet.py` — new, independent connector.
  One spreadsheet ROW = one chunk (not sub-split — a row is already one
  complete, atomic fact, unlike a multi-section SOP page). Writes
  `data/chunks/chunks_from_error_codes.jsonl` in the SAME schema
  `chunk_confluence.py` produces, plus a `source_type` field
  (`"spreadsheet"` vs. `"confluence"`) so origin stays traceable after
  merging — the "preserve source metadata through every stage" principle
  from the same research.
- `embed_and_store.py` updated to load and combine BOTH source files
  before embedding (was hardcoded to `chunks_from_confluence.jsonl`
  only) — this is the real convergence point, matching the researched
  pattern exactly.

**Real bug found and fixed while testing this** (not assumed away):
`retrieval_pipeline.py`'s BM25 index was ALSO hardcoded to load only
`chunks_from_confluence.jsonl`, completely independent of what
`embed_and_store.py` put in Chroma. Effect: dense search (via Chroma)
could see the new spreadsheet chunks, but BM25/sparse search never could
— and since `hybrid_search`'s RRF fusion only promotes what at least one
retriever actually proposed, a terse, code-heavy spreadsheet row (exactly
the content type BM25's exact-token-match strength is best suited for)
was silently invisible to half the retrieval system. Caught by testing
the exact scenario the spreadsheet was built for — a query containing
the literal string "0x8007042B" — and finding the matching spreadsheet
chunk missing from top-5 results entirely (not just poorly ranked) at
BOTH the baseline and reranked stage. Fixed by loading BM25's corpus from
every source file, same convention as the `embed_and_store.py` fix.

**Verified after the fix, real queries against the live 298-chunk Chroma
collection** (283 Confluence + 15 spreadsheet):
- `"Getting error 0x8007042B every time I try to connect to VPN."` →
  reranked #1 is the exact spreadsheet row (`900000`, score 4.895,
  the only positive score in the result set) — was previously absent
  from the top 5 entirely, even at baseline.
- `"What does error code 0x8007042B actually mean?"` → same, reranked
  #1 is `900000` (score 3.133), correctly beating the general-purpose
  "Network SSL VPN failure" prose SOP that doesn't name the code.

This is the retrieval-quality question this addition was specifically
designed to test (does an exact-code query correctly prefer a precise
lookup-table hit over a general diagnostic prose page it would otherwise
partially match on) — confirmed working, but only after finding and
fixing a real cross-retriever consistency bug, not by construction.

**Full retrieval eval re-run after the fix** (`eval_retrieval.py`, same
100-query set as before — none of these queries target the spreadsheet
content itself, this measures collateral effect on existing coverage).
HONEST RESULT, not a clean win: reranked/COMBINED Recall@5 91.0% (was
92.0%), Top-1 63.0% (was 65.0%), MRR 0.736 (was 0.750) — a small but real
regression, checked by hand per this project's own standing rule (verify
a suspicious eval delta against raw per-query data before trusting or
dismissing it, not just the summary number).

**What's actually happening, verified per-query, not assumed:** 2 queries
newly miss their expected page (`A09`, `B01`) that didn't before; 1
previously-missing query (`D04`) is still missing. Root cause is a real,
explicable retrieval tradeoff, not a bug: `A09`'s raw text is literally
"TLS Error: TLS handshake failed," and the spreadsheet has a row for
exactly that — "TLS handshake failure between client and gateway"
(`0x80090326`) — which now legitimately, correctly outranks the general
"Network SSL VPN failure" SOP page this eval set happens to have labeled
as A09's expected answer. Same story for `B01` ("VPN error" — generic
enough that terse, precise error-code rows now compete directly with the
general SOP for the top slots). This is NOT the BM25 bug re-appearing —
it's the expected, honest cost of adding more specific, genuinely
relevant content: it sometimes wins a ranking slot away from the
"designated correct" answer on a query general enough that BOTH are
defensible answers. Two queries DID newly succeed for the same reason in
the other direction (`A04`, `E03`, no longer missing). Net: -1 query
combined. Not swept under the rug — a real, small precision/recall
tradeoff from genuine multi-source diversity, exactly the kind of effect
a real production system would need to monitor for (see the observability
dashboard's category/rate tracking) rather than assume away.

### Scope correction + proper eval coverage for the error-code source (2026-09-21, same day)

Two follow-up problems surfaced when the user asked how the spreadsheet
source was actually evaluated — both real, both worth the honest record.

**Problem 1 — scope drift, caught by explicit user request to cross-check,
not caught by me proactively.** Of the original 15 spreadsheet rows, 7
referenced systems this project has ZERO other coverage for (printers,
Windows Update, Outlook/Exchange, badge readers, MDM) — content that
looked individually plausible but didn't fit this project's actual real
ticket taxonomy (`data/raw/enriched_tickets.csv`'s real `Subcategory`
values, checked directly). 3 more rows were "adjacent but not exact"
(a Windows-login stale-credential code, a disk SMART warning, an SSO
lockout code — each close to a real subcategory but describing a
different failure mode). Cut all 10, kept the 4 clean VPN rows, added 7
NEW rows for real subcategories that had zero prior error-code coverage
(License expired, Application deadlock, Application crash on deploy,
DHCP failure, Duplicate IP detected, Latency spikes, Switch port
failure) — each cross-checked against its real Confluence SOP page
first (page exists, doesn't already state a code) before being added.
Final: 11 rows, every one defensible. Added a `subcategory` column to
`error_code_lookup.csv` and a `cross_check_subcategories()` function in
`ingest_error_code_spreadsheet.py` that now runs EVERY time the
connector runs and fails loudly if a row's subcategory isn't real —
this check was missing the first time, this makes it structural instead
of a one-off manual review.

**Problem 2 — real Chroma staleness bug, found while rebuilding.** After
cutting the CSV from 15 to 11 rows and re-running `embed_and_store.py`,
Chroma still held all 15 chunks — `get_or_create_collection().add()`
only upserts by ID, so the 7 removed rows' old chunk IDs (`900004`
through `900014` minus the kept ones) were never deleted and stayed
silently queryable. Fixed by having `embed_and_store.py` delete and
fully recreate the collection on every run rather than incrementally
upsert — correct and simple at this project's scale (294 chunks);
explicitly noted in the script that a real production system with
millions of chunks would need proper per-source tombstone/delta
handling instead, which is real added complexity this project doesn't
need yet. Verified directly: `collection.get()` filtered to the
900xxx page-id range now returns exactly the 11 expected chunks, zero
stale entries.

**Real eval coverage gap found and fixed.** Before this, the ONLY
evidence for "does exact-error-code retrieval work" was 3 queries, all
for the SAME single code (`0x8007042B`, phrased 3 ways) — checked by
hand at the user's request and confirmed too thin to support any real
retrieval-quality claim. Built `data/processed/eval_error_code_queries.jsonl`:
22 exact-code queries (2 natural phrasings per code × 11 codes) plus 3
deliberately AMBIGUOUS queries with no code stated (which should still
favor the general Confluence SOP page, not a spreadsheet row) — one of
which (`EC24`) deliberately reuses the exact text of `B09`, the specific
regression query found earlier, as a direct re-test. Wired into
`eval_retrieval.py` as a third query source (`load_error_code_queries()`),
reported as its own labeled section alongside the existing ticket/
edge-case sections, not blended in silently.

**Full eval re-run, real numbers, on the corrected 11-row spreadsource
(294 total chunks)**:
- Error-code queries alone (n=25, reranked): Recall@5 96.0%, Top-1
  accuracy 88.0%, MRR 0.913, NDCG@5 0.925 — the first real, broad-based
  evidence for this capability (vs. the earlier 3-query/1-code sample).
- `EC24` (the B09 regression re-test): now hits at rank 3 — the specific
  earlier regression is GONE after scoping the spreadsheet to real,
  in-scope content. Checked by hand, not just via the summary number.
- `B01` and `D04` (the two originally-found regressions): STILL missing
  — a real, persistent tradeoff, not fixed by the scope correction, and
  said so plainly rather than declared resolved.
- `EC23` ("VPN error, not sure what's wrong" — the most generic possible
  VPN query, written specifically to test this): NEWLY misses in this
  expanded set — confirms the B01/D04 pattern (vague VPN queries losing
  to spreadsheet rows) is a real, repeatable phenomenon, not a one-off.
- COMBINED across all 125 queries (ticket + edge-case + error-code,
  reranked): Recall@5 96.0%, Top-1 68.0%, MRR 0.787, NDCG@5 0.830 —
  numbers moved because the query MIX changed (25 new, generally
  easier, exact-code queries added), not because retrieval quality on
  the ORIGINAL 100 queries changed — the original 100's own numbers are
  the ones in the prior subsection above and are what the -1 net
  regression figure describes.

Net honest position: the error-code retrieval capability is now real,
broad-based evidence, not a proof-of-concept demo. The tradeoff it
introduces on generic/ambiguous VPN-adjacent queries is real, small, and
now specifically monitored for via `EC23`/`B01`/`D04` rather than
discovered by accident — which is itself the actual point of building
proper eval coverage before trusting a capability.

### Third ingestion source: Slack threads via Anthropic's Contextual Retrieval (2026-09-22)

Added a THIRD, again structurally different source, per explicit user
request to explore "communication archives" (Slack/Teams-style informal
peer troubleshooting) as a multi-source ingestion example, with the
explicit instruction to research the right approach first, including
checking Anthropic's own docs.

**Researched before building, not guessed.** General research (Snyk's
own "Slack threads to structured knowledge" RAG writeup, Paragon's
Slack-ingestion guide, both fetched directly) converged on: treat a
THREAD as the retrievable unit, not a single message — a bare message
like "yep that was it, updated it, loads fine now" has no standalone
meaning. Anthropic's own engineering blog
(anthropic.com/engineering/contextual-retrieval, fetched directly, plus
the Claude Cookbook's contextual-embeddings implementation guide) turned
out to be a direct, better-fit solution to exactly this problem:
**Contextual Retrieval** — before embedding a chunk, have an LLM
generate a short (50-100 token) description that SITUATES the chunk
within its full parent document, and prepend that description to the
chunk rather than embedding the bare chunk alone. Anthropic's own
measured numbers: 35% fewer retrieval failures from contextual
embeddings alone, 49% combined with contextual BM25, 67% combined with
reranking too — tested across codebases/fiction/ArXiv/science papers,
NOT conversational data specifically (stated honestly, not oversold —
applying it to Slack threads here is a reasoned extension, not a
repeated claim from Anthropic's own writeup).

**Real, stated difference from Anthropic's reference implementation**:
their Cookbook example uses Claude with explicit prompt caching
(`cache_control: {"type": "ephemeral"}`) to keep the per-chunk cost down
when many chunks share one large parent document. This project's LLM
provider is Gemini (a hard constraint from project inception — see
Section 1), which has a different caching mechanism, and at this
dataset's scale (4 threads) the caching optimization isn't worth the
added complexity. The CORE technique — one LLM call per chunk, given the
full parent document, to generate a situating description before
embedding — is implemented via the existing Gemini client; the specific
cost-optimization Anthropic paired it with is not, and that's stated
directly in `ingest_slack_threads.py`'s own docstring rather than
silently dropped or misrepresented as identical to Anthropic's exact
implementation.

**Built**: `data/raw/slack_threads.json` — 4 simulated (not a live Slack
API pull — same honesty convention as the Confluence corpus) but
realistic multi-person triage-channel threads, each with genuine noise
(emojis, an unrelated interruption from a third person, casual
phrasing) surrounding one real problem-to-resolution exchange. Topics
picked from real subcategories with NO existing spreadsheet coverage
(Plugin not loading, Session timeout, Laptop screen blue tint, External
HDD not mounting), each cross-checked against its real Confluence page's
actual diagnostic content first so the thread's "what fixed it" stays
consistent with, not contradicting, the official SOP.
`src/ingest_slack_threads.py` — new connector: extracts the
problem-through-resolution exchange from each thread (deterministic
keyword heuristic, not an LLM call — reserving the LLM call for the part
that needs actual judgment), then calls Gemini once per thread to
generate the situating context, prepends it, and writes
`chunks_from_slack_threads.jsonl` in the same canonical schema as the
other two connectors (`source_type: "slack_thread"`, plus a
`related_page_id` cross-reference to the linked Confluence page).
Same `cross_check_*` scope-discipline pattern as the spreadsheet
connector — fails loudly if a thread's subcategory or related page
isn't real, not a one-off manual check.

**Verified real, both ways**: for all 4 thread topics, querying the
system with the natural problem description retrieves BOTH the Slack
thread AND its linked Confluence page together in the top 5 (checked
directly against the live 298-chunk collection) — confirming sources
aren't siloed, one retrieval pass searches across all three at once.
Unrelated queries (onboarding, contractor compromise) correctly surface
ZERO Slack-thread results — the new source doesn't pollute irrelevant
queries.

**Full eval, built the same way as the spreadsheet's** (not spot-checks
alone): `data/processed/eval_slack_thread_queries.jsonl`, 8 queries (2
phrasings — natural description + peer-question — per thread), wired
into `eval_retrieval.py` as a 4th labeled pass. Real numbers (n=133
total across all four sources, reranked): Slack-thread queries alone hit
100% Recall@5, NDCG@5 0.862; Top-1 accuracy 62.5% (5/8) — the 3 misses
are NOT wrong answers, they're cases where the reranker preferred the
official Confluence SOP page over the linked Slack thread for the same
underlying question (both are genuinely correct answers; checked by
hand, this is a ranking preference between two valid sources, not a
retrieval failure).

**Same honest collateral-effect check as the spreadsheet source, done
this time on ticket-based queries specifically**: of 15 ticket-based
queries that lost their #1 ranking spot in this run, 5 are directly
attributable to the two new sources (`TICKET-Application crash on
deploy`, `TICKET-Application deadlock` — spreadsheet; `TICKET-External
HDD not mounting`, `TICKET-Laptop screen blue tint`, `TICKET-Plugin not
loading` — Slack threads) — in EVERY one of those 5 cases, checked by
hand, the new top-ranked chunk is a directly, specifically relevant
answer to that exact ticket (the Slack thread literally resolves that
exact case), not a wrong or irrelevant result crowding out a correct
one. This is the SAME real tradeoff pattern documented in the
spreadsheet writeup above, now also showing up from a second new
source — a second data point that this is a real, recurring, honest
property of adding more specific, genuinely relevant content to a
retrieval system with only one "designated correct" answer per eval
query, not a bug specific to either source.

### Cross-source authority bug: found, root-caused, fixed in TWO places (2026-09-22, same day)

While using the live chat UI to demo the new Slack-thread source, a real
generation output for "my plugin just stopped loading" said the official
Confluence "Plugin not loading" SOP page was **superseded by** the new
Slack thread — because the Slack thread was freshly authored today and
therefore had a newer `last_updated` date. This is wrong: an informal,
unreviewed peer chat message doesn't actually supersede a reviewed
official procedure just by being newer.

**Root-caused properly before fixing anything** (initial hypothesis was
WRONG, caught by testing directly rather than assumed): first suspected
`same_topic_detection.py`'s `find_same_topic_pairs()` (the structured
same-topic-pair detector) was the cause, since it groups documents by
title and compares dates. Direct test
(`find_same_topic_pairs(r.search_results)` on the real query) returned
`[]` — it never fired, because the Slack thread's title
("Slack thread: Plugin not loading (#it-helpdesk-triage)") and the
Confluence page's title ("Plugin not loading") are different STRINGS,
so the detector's exact-title-match never grouped them in the first
place. **The real cause was the GENERATION PROMPT itself**
(`step9_generation.py`'s `SYSTEM_PROMPT`): its existing instruction
("if two excerpts describe different procedures for the same topic,
prefer the one with the most recent last_updated date, note the older
one exists") has no concept of source authority — Gemini saw both
documents were about the same underlying topic (even with different
titles, the topic overlap was obvious from content) and correctly
followed its literal instruction: prefer the newer-dated one, flag the
older as superseded. The model did exactly what it was told; what it
was told was incomplete.

**Researched the right fix, not guessed** (per the same research
already done for the Slack-thread source's design): production RAG
systems handle cross-source conflicts by tagging source authority as
structured metadata and baking an authority tier into resolution logic
— "an audited record always outranks a draft" — rather than defaulting
to raw recency across fundamentally different kinds of sources.

**Fixed in TWO places, since both really do have the same blind spot,
even though only one actually fired in this specific case:**
1. `same_topic_detection.py` — added `SOURCE_AUTHORITY_TIER`
   (`confluence`/`spreadsheet` = tier 1, `slack_thread` = tier 2).
   `find_same_topic_pairs()` now groups by `(title, tier)` instead of
   just `title`, so a tier-2 and tier-1 chunk can never form a
   "superseded" pair even if they DID someday share an exact title
   (didn't fire in THIS case, but was a real latent bug in the
   detector's own logic, found by writing the fix and confirming the
   original title-based grouping had no tier awareness at all -- worth
   fixing even though it wasn't the actual cause here, since the exact-
   title-match dodge is fragile and a future Slack thread or ingestion
   change could easily produce a real title collision).
2. `step9_generation.py` — this was the ACTUAL cause. Added
   `source_type` to each `<document>` tag in the prompt (previously
   only `title`/`last_updated` were visible to the model — same fix
   PATTERN as the 2026-09-14 `last_updated`-in-the-tag fix documented
   above: the model can't follow an instruction about information it
   can't see at generation time). Rewrote the conflict-resolution rule:
   same-tier conflicts still resolve by recency (unchanged, this is the
   original correct stale-duplicate-SOP behavior); a tier-1-vs-tier-2
   conflict ALWAYS prefers the official source as primary, regardless
   of dates, with the informal source allowed only as corroborating
   context, never as "replacing" the official one.

**Verified the fix, real query, real output**: re-ran the exact
"my plugin just stopped loading" query end-to-end. New answer correctly
cites the official Confluence procedure as primary for every step, and
mentions the Slack thread only as corroboration ("a Slack thread also
notes this resolves silent load failures caused by version mismatches")
— no "superseded" language anywhere. Confirmed the genuine, original
stale-duplicate-SOP case still works unchanged (`find_same_topic_pairs`
on the real "Suspicious contractor compromise" query, tier-1-vs-tier-1,
still correctly detects and reports the real 2023 stale duplicate).
Full retrieval eval re-run to confirm neither fix touched retrieval
scoring (only generation prompt content and the detector's grouping
key changed, not ranking) — see this run's numbers alongside the
Slack-thread writeup above.

## 9. Open questions not yet resolved in conversation

Both items originally listed here (eval timing/strategy, generation step
design) are long since resolved and built — see Section 8's work order
and the eval-results subsections. Current genuinely open items:

- ~~"Show multiple same-topic documents at different ages" feature~~ —
  **done 2026-09-17**, see the "Same-topic-different-age detection"
  writeup below.
- **Generation-quality eval, LLM-as-judge completion** — paused
  2026-09-17 by explicit user request ("lets keep the llm as judge part
  for later"). 5 of 6 synthesized sub-results are marked PENDING in
  `data/processed/generation_eval_judged.jsonl`, blocked on
  `gemini-3.6-flash`'s 20/day quota — resume any time with
  `python src/judge_generation_eval.py` (auto-retries only the pending
  rows, no wasted quota on the 17 already-scored/non-applicable ones).
- Unanswerable-query handling (13 queries in `eval_edge_case_queries.jsonl`)
  doesn't have a dedicated eval pass yet — worth deciding once more
  real-world usage data exists on what "correctly declining to answer"
  should be measured against (Section 8, item 7).
