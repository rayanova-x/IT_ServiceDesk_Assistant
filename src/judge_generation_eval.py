"""
Scores the captured generation output (generation_eval_capture.jsonl) on
the "generation" half of the RAG evaluation triad
(reference_docs/rag_evals.md 4.2/4.3, rag_evals2.md Layer 2):

  1. Faithfulness / Groundedness (1-5, LLM-as-judge) -- is every claim in
     the answer actually supported by the retrieved context? This is the
     anti-hallucination metric both reference docs call out as the core
     generation check.
  2. Answer Relevance (1-5, LLM-as-judge) -- does the answer actually
     address what the user asked?
  3. Contract checks (deterministic, no judge needed -- we already know
     ground truth): never synthesize when low_confidence=True (candidates
     were spot-checked before, never formally counted); never cite a
     trashed or silently-stale page as if current, cross-checked against
     data/processed/eval_only_ground_truth.jsonl's doc_type_true/
     staleness_tier fields.

Per rag_evals2.md's judge design (score 1-5 via LLM-as-judge, "first pass"
+ human spot-check for high-stakes), each judge call gets ONLY the query,
context, and answer -- no ground truth page ID -- since faithfulness and
relevance are properties of (query, context, answer) alone, not of
whether retrieval happened to find the "right" document. That's already
measured separately by eval_retrieval.py.

JUDGE MODEL is a HYBRID split, decided 2026-09-15 after direct testing
ruled out a single always-independent judge as impractical on the free
tier: EVERY non-lite Gemini model tried (gemini-3.6-flash, then
gemini-3.5-flash as a fallback) turned out to cap at only 20 free-tier
requests/day (confirmed via the 429 error's own message text --
"generate_content_free_tier_requests, limit: 20" -- not assumed), while
only the "-lite" tier (gemini-3.5-flash-lite, also the generation model)
carries the standard 500/day allowance. 20/day can't cover judging every
sub-result independently (12+ calls per pass, no headroom for retries or
iterating on the judge prompt) -- but it's enough to spend selectively.

So: is_complex() below routes each sub-result to ONE of two judges --
  - GEMINI_JUDGE_MODEL (defaults to gemini-3.5-flash-lite, SAME as the
    generator): used for straightforward sub-results -- a direct answer
    grounded in 1-2 source documents, no self-reported gap. Self-grading
    is a real, named weakness (a model is more likely to rate its own
    reasoning as acceptable and miss its own blind spots than an
    independent judge would -- see rag_evals2.md's own reference script,
    which uses a separate provider for exactly this reason), but an
    acceptable tradeoff for the easy cases given the quota reality.
  - GEMINI_STRONG_JUDGE_MODEL (defaults to gemini-3.6-flash, A DIFFERENT
    model from the generator): reserved for complex sub-results -- the
    answer claims the context has a gap, or synthesizes across 3+
    distinct source documents. These are exactly the cases where
    self-grading is riskiest (checking a self-reported "I don't have
    enough info" claim, or judging cross-document prioritization), so
    the scarce ~20/day independent-judge budget is spent where it
    matters most rather than spread thin or abandoned entirely.
Kept on the same provider (Gemini) rather than adding a second paid
provider purely for judging, per user decision 2026-09-14.

Reads generation_eval_capture.jsonl (from capture_generation_eval_data.py)
-- NO new pipeline/generation calls happen here, only judge calls (2 per
answerable sub-result: faithfulness + relevance). This script CAN be
re-run to tune the judge prompt without re-spending quota on generation
itself, same capture/sweep split used for Step 7.

Run: python src/judge_generation_eval.py
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from google import genai

PROJECT_ROOT = Path(__file__).parent
REPO_ROOT = PROJECT_ROOT.parent
load_dotenv(REPO_ROOT / ".env")

DATA_PROCESSED = REPO_ROOT / "data" / "processed"

FAITHFULNESS_JUDGE_PROMPT = """You are grading an internal IT-helpdesk assistant's answer for \
faithfulness to its source material. You are NOT judging whether the answer is a good answer \
overall -- ONLY whether every claim in it is actually supported by the provided context excerpts.

Each excerpt below is labeled with its own last_updated date. If the answer says one excerpt's \
procedure is more recent than, supersedes, or is preferred over another, that IS a supported \
claim as long as the last_updated dates shown actually support that ordering -- this is not \
external/added information, it's a correct read of metadata that was given to you. Only mark a \
claim unsupported if it isn't backed by the excerpt text OR the given last_updated dates.

<context>
{context}
</context>

<answer>
{answer}
</answer>

Score 1-5:
5 = every claim in the answer is directly supported by the context (text or dates), nothing added
4 = answer is essentially grounded, at most one minor unsupported detail
3 = mostly grounded but at least one claim goes beyond what the context supports
2 = significant unsupported content mixed with grounded content
1 = the answer is substantially fabricated or contradicts the context

Respond with ONLY a JSON object, no other text: {{"score": <int 1-5>, "reasoning": "<one sentence>"}}"""

RELEVANCE_JUDGE_PROMPT = """You are grading an internal IT-helpdesk assistant's answer for \
relevance to the user's query. You are NOT judging factual accuracy -- ONLY whether the answer \
addresses what was actually asked.

<query>
{query}
</query>

<answer>
{answer}
</answer>

Score 1-5:
5 = directly and fully addresses the query
4 = addresses the query with minor drift or missing a small piece
3 = partially addresses the query, notable gaps or tangents
2 = mostly misses the query's actual question
1 = does not address the query at all

Respond with ONLY a JSON object, no other text: {{"score": <int 1-5>, "reasoning": "<one sentence>"}}"""


def _call_judge(client, model, prompt) -> dict:
    for attempt in range(4):
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            text = response.text.strip()
            if text.startswith("```"):
                text = text.strip("`")
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text.strip())
        except Exception as e:
            if attempt == 3:
                raise
            wait = 10 * (attempt + 1)
            print(f"    (judge call {type(e).__name__}, retrying in {wait}s...)")
            time.sleep(wait)


def judge_faithfulness(client, model, context_chunks: list[dict], answer: str) -> dict:
    # last_updated must be included here -- the answer is allowed (per
    # Step 9's own prompt, step9_generation.py) to state which excerpt is
    # more recent and note that an older, superseded procedure exists.
    # Without the date visible to the judge too, a correct "X supersedes
    # the older Y" claim looks like fabricated/unsupported information --
    # this is the exact same class of bug fixed in Step 9 itself
    # 2026-09-14 (date only in the post-hoc citation list, not where the
    # model/judge needed it). Found via this eval run: F07 scored
    # faithfulness=1 with reasoning "introduces external information
    # entirely absent from the provided context" about which procedure
    # was more recent -- the dates WERE in the real context (Step 9 saw
    # them), just not shown to this judge. Judge prompt bug, not a Step 9
    # regression -- confirmed by manually reading the captured context.
    context = "\n\n".join(
        f"[{c['title']}, last_updated={c['last_updated']}]\n{c['text']}" for c in context_chunks
    )
    prompt = FAITHFULNESS_JUDGE_PROMPT.format(context=context, answer=answer)
    return _call_judge(client, model, prompt)


def judge_relevance(client, model, query: str, answer: str) -> dict:
    prompt = RELEVANCE_JUDGE_PROMPT.format(query=query, answer=answer)
    return _call_judge(client, model, prompt)


def load_ground_truth() -> dict:
    gt = {}
    with open(DATA_PROCESSED / "eval_only_ground_truth.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            gt[row["page_id"]] = row
    return gt


def check_staleness_contract(sub_result: dict, ground_truth: dict) -> list[str]:
    """Deterministic check: did the answer cite a trashed page at all
    (never acceptable -- trashed means removed from service, not just
    outdated), or cite an unmarked-stale page WITHOUT a same-topic
    current page also being cited (the only acceptable way an
    unmarked-stale page appears is as a disclosed "older procedure also
    exists" aside alongside the current one -- see step9_generation.py's
    prompt rule). Returns a list of violation strings (empty if clean).

    NOTE: this can't verify the DISCLOSURE WORDING itself (only an LLM
    read of the answer text could) -- it verifies the necessary
    precondition, that the current version was cited alongside it. A
    stale page cited ALONE, with no current counterpart, is flagged."""
    violations = []
    if sub_result["mode"] != "synthesized":
        return violations
    cited_titles_current = set()
    cited = []
    for page_id in sub_result["citations"]:
        gt_row = ground_truth.get(page_id)
        if gt_row is None:
            continue
        cited.append((page_id, gt_row))
        if gt_row.get("doc_type_true") not in ("trashed",) and gt_row.get("staleness_tier") != "unmarked_stale":
            cited_titles_current.add(gt_row.get("subcategory"))

    for page_id, gt_row in cited:
        doc_type = gt_row.get("doc_type_true")
        tier = gt_row.get("staleness_tier")
        if doc_type == "trashed" or tier == "trashed":
            violations.append(f"cited TRASHED page {page_id} in a synthesized answer")
        elif tier == "unmarked_stale":
            same_topic_current_cited = gt_row.get("subcategory") in cited_titles_current
            if not same_topic_current_cited:
                violations.append(
                    f"cited UNMARKED-STALE page {page_id} with NO same-topic current page also "
                    f"cited -- likely presented as the primary/only answer"
                )
    return violations


GAP_PHRASES = ("do not fully answer", "does not fully answer", "do not contain",
               "does not contain", "do not provide", "does not provide")


def is_complex(sub_result: dict) -> bool:
    """
    A sub-result is COMPLEX (routed to the independent judge,
    GEMINI_STRONG_JUDGE_MODEL) if either:
      (a) the answer itself claims the context has a GAP (doesn't fully
          answer/contain/provide) -- judging whether a self-reported gap
          is actually correct requires reading the source material
          skeptically, exactly the kind of check a model is least
          reliable at running on its own output; or
      (b) the answer draws on 3+ DISTINCT source pages -- cross-document
          synthesis/prioritization (which excerpt to lead with, how to
          reconcile them) is a harder faithfulness judgment than
          restating one document's content.
    Everything else (a direct answer grounded in 1-2 documents, no
    self-reported gap) is judged by the SAME model that generated it
    (GEMINI_JUDGE_MODEL) -- straightforward enough that self-grading is
    an acceptable, quota-conscious tradeoff. This split exists because
    every genuinely independent Gemini model available on the free tier
    caps at 20 requests/day (found 2026-09-15 testing gemini-3.6-flash
    and gemini-3.5-flash directly) -- nowhere near enough to judge every
    sub-result independently, so the scarce independent-judge budget is
    spent only where self-grading is riskiest.
    """
    answer_lower = sub_result["answer_text"].lower()
    if any(phrase in answer_lower for phrase in GAP_PHRASES):
        return True
    n_distinct_pages = len(set(c["page_id"] for c in sub_result["context_chunks"]))
    return n_distinct_pages >= 3


def check_confidence_contract(sub_result: dict) -> list[str]:
    """Deterministic check: low_confidence=True must NEVER produce a
    synthesized (Gemini-written) answer -- see step9_generation.py
    module docstring."""
    if sub_result["low_confidence"] and sub_result["mode"] == "synthesized":
        return ["CONTRACT VIOLATION: low_confidence=True but mode=synthesized (should be candidates_only)"]
    return []


def main():
    capture_path = DATA_PROCESSED / "generation_eval_capture.jsonl"
    if not capture_path.exists():
        print(f"No captured data found at {capture_path}. Run capture_generation_eval_data.py first.")
        return

    records = [json.loads(line) for line in open(capture_path, encoding="utf-8")]
    ground_truth = load_ground_truth()

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    self_model = os.environ.get("GEMINI_JUDGE_MODEL", os.environ["GEMINI_MODEL"])
    strong_model = os.environ.get("GEMINI_STRONG_JUDGE_MODEL", "gemini-3.6-flash")

    # Resumable: gemini-3.6-flash's 20/day quota can run out mid-pass, and
    # a crash used to lose every row judged so far (all_rows was only
    # written to disk at the very end) -- found the hard way 2026-09-17
    # when a 429 mid-loop discarded 2 real judged rows (D03, E02).
    # already_judged keys on (query_id, sub_query) since that's stable
    # across re-runs of this script against the same capture file.
    out_path = DATA_PROCESSED / "generation_eval_judged.jsonl"
    already_judged: dict[tuple[str, str], dict] = {}
    stale_pending_rows: list[dict] = []  # kept in all_rows for the record, but re-attempted below
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                key = (row["query_id"], row["sub_query"])
                if row.get("pending"):
                    # A row marked PENDING (judge call failed, e.g. quota
                    # exhausted) must be RETRIED on the next run, not
                    # silently treated as done -- do not add it to
                    # already_judged. It'll be re-written (as either a
                    # real result or PENDING again) when the loop reaches
                    # it, so don't carry the stale copy into all_rows either.
                    stale_pending_rows.append(row)
                    continue
                already_judged[key] = row
        n_pending = len(stale_pending_rows)
        print(f"Resuming: {len(already_judged)} sub-result(s) already judged in {out_path.name}, "
              f"skipping those" + (f" ({n_pending} PENDING row(s) will be retried)" if n_pending else "") + ".\n")

    all_rows = list(already_judged.values())
    # Re-open fresh (not append) if there were pending rows to retry --
    # otherwise a retried row would be written a SECOND time alongside
    # its stale PENDING copy. Rewrite the file from already_judged (the
    # clean, non-pending rows) instead; the loop below re-appends both
    # the pending rows it retries and any new rows.
    if stale_pending_rows:
        with open(out_path, "w", encoding="utf-8") as f:
            for row in already_judged.values():
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    out_f = open(out_path, "a", encoding="utf-8")

    print(f"Judging {len(records)} captured queries -- straightforward sub-results judged by "
          f"{self_model!r} (same as generation, self-grading), complex ones by {strong_model!r} "
          f"(independent, quota-limited to ~20/day) -- see is_complex() for the split rule...\n")

    for i, rec in enumerate(records, start=1):
        qid = rec["query_id"]
        if not rec["sub_results"]:
            print(f"[{i}/{len(records)}] {qid:6s} no sub_results (stopped_at={rec['stopped_at']}) -- skipped")
            continue

        for sub in rec["sub_results"]:
            if (qid, sub["sub_query"]) in already_judged:
                print(f"[{i}/{len(records)}] {qid:6s} already judged, skipping")
                continue
            row = {
                "query_id": qid,
                "sub_query": sub["sub_query"],
                "mode": sub["mode"],
                "low_confidence": sub["low_confidence"],
                "contract_violations": [],
            }
            row["contract_violations"] += check_confidence_contract(sub)
            row["contract_violations"] += check_staleness_contract(sub, ground_truth)

            if sub["mode"] == "synthesized":
                complex_case = is_complex(sub)
                judge_model = strong_model if complex_case else self_model
                row["judge_model"] = judge_model
                row["complex"] = complex_case
                try:
                    faith = judge_faithfulness(client, judge_model, sub["context_chunks"], sub["answer_text"])
                    time.sleep(4)
                    rel = judge_relevance(client, judge_model, sub["sub_query"], sub["answer_text"])
                    time.sleep(4)
                    row["faithfulness_score"] = faith["score"]
                    row["faithfulness_reasoning"] = faith["reasoning"]
                    row["relevance_score"] = rel["score"]
                    row["relevance_reasoning"] = rel["reasoning"]
                except Exception as e:
                    # A quota-exhausted judge model (gemini-3.6-flash's
                    # 20/day cap especially) must not crash the whole
                    # pass -- other rows using the OTHER judge model may
                    # still be gettable. Mark this row PENDING (not
                    # written as a permanent null) so a future re-run's
                    # already_judged skip does NOT treat it as done --
                    # see the "pending" check below, which excludes these
                    # from already_judged on reload.
                    print(f"    ({qid} judge call failed on {judge_model!r}: {type(e).__name__}, "
                          f"marking PENDING, continuing to next row)")
                    row["faithfulness_score"] = None
                    row["relevance_score"] = None
                    row["pending"] = True
            else:
                row["faithfulness_score"] = None
                row["relevance_score"] = None

            flag = " CONTRACT VIOLATION" if row["contract_violations"] else ""
            judge_tag = f" judge={row['judge_model']}" if sub["mode"] == "synthesized" else ""
            print(f"[{i}/{len(records)}] {qid:6s} mode={sub['mode']:16s} "
                  f"faith={row['faithfulness_score']} rel={row['relevance_score']}{judge_tag}{flag}")
            all_rows.append(row)
            # Write incrementally, not just at the very end -- a mid-run
            # 429/crash (e.g. gemini-3.6-flash's 20/day cap) used to lose
            # every row judged so far. See the resumable-load comment above.
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()

    out_f.close()
    print(f"\nSaved {len(all_rows)} judged rows total to {out_path}")

    all_synthesized = [r for r in all_rows if r["mode"] == "synthesized"]
    pending = [r for r in all_synthesized if r.get("pending")]
    synthesized = [r for r in all_synthesized if not r.get("pending")]  # scored, complete rows only
    candidates_only = [r for r in all_rows if r["mode"] == "candidates_only"]
    n = len(synthesized)

    print("\n=== SUMMARY ===")
    print(f"Total sub-results: {len(all_rows)}  (synthesized={len(all_synthesized)}, "
          f"of which scored={n} and PENDING={len(pending)}, candidates_only={len(candidates_only)})")
    if pending:
        print(f"PENDING (judge call failed, e.g. quota exhausted -- re-run this script to retry): "
              f"{[r['query_id'] for r in pending]}")
    if n:
        avg_faith = sum(r["faithfulness_score"] for r in synthesized) / n
        avg_rel = sum(r["relevance_score"] for r in synthesized) / n
        print(f"Faithfulness (1-5), ALL: mean={avg_faith:.2f}  scores={[r['faithfulness_score'] for r in synthesized]}")
        print(f"Answer relevance (1-5), ALL: mean={avg_rel:.2f}  scores={[r['relevance_score'] for r in synthesized]}")

        # Break down by which judge scored it -- mixing self-graded and
        # independently-graded scores into one mean would hide whether
        # the independent judge agrees with self-grading or not, which
        # is the whole point of splitting them (see is_complex()).
        self_graded = [r for r in synthesized if not r.get("complex")]
        strong_graded = [r for r in synthesized if r.get("complex")]
        if self_graded:
            sg_faith = sum(r["faithfulness_score"] for r in self_graded) / len(self_graded)
            print(f"  Self-graded (n={len(self_graded)}, straightforward): faithfulness mean={sg_faith:.2f} "
                  f"-- {[r['query_id'] for r in self_graded]}")
        if strong_graded:
            st_faith = sum(r["faithfulness_score"] for r in strong_graded) / len(strong_graded)
            print(f"  Independently-graded (n={len(strong_graded)}, complex): faithfulness mean={st_faith:.2f} "
                  f"-- {[r['query_id'] for r in strong_graded]}")

        low_faith = [r for r in synthesized if r["faithfulness_score"] <= 3]
        if low_faith:
            print(f"\nLow-faithfulness (<=3) answers to review: {[r['query_id'] for r in low_faith]}")

    all_violations = [(r["query_id"], v) for r in all_rows for v in r["contract_violations"]]
    if all_violations:
        print(f"\nCONTRACT VIOLATIONS ({len(all_violations)}):")
        for qid, v in all_violations:
            print(f"  {qid}: {v}")
    else:
        print("\nNo contract violations (confidence-gating and staleness-citation rules both held).")

    summary_path = DATA_PROCESSED / "generation_eval_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "run_at": datetime.now().isoformat(),
            "self_judge_model": self_model,
            "strong_judge_model": strong_model,
            "generation_model": os.environ["GEMINI_MODEL"],
            "n_synthesized": n,
            "n_pending": len(pending),
            "n_candidates_only": len(candidates_only),
            "n_self_graded": len(self_graded),
            "n_independently_graded": len(strong_graded),
            "mean_faithfulness": round(sum(r["faithfulness_score"] for r in synthesized) / n, 3) if n else None,
            "mean_relevance": round(sum(r["relevance_score"] for r in synthesized) / n, 3) if n else None,
            "mean_faithfulness_self_graded": round(sum(r["faithfulness_score"] for r in self_graded) / len(self_graded), 3) if self_graded else None,
            "mean_faithfulness_independently_graded": round(sum(r["faithfulness_score"] for r in strong_graded) / len(strong_graded), 3) if strong_graded else None,
            "contract_violations": len(all_violations),
        }, f, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
