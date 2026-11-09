"""
Step 10 of the pipeline: post-generation verification + output gate --
the piece the note's reference architecture (Claim Parser -> Citation
Verifier + Grounding Judge -> Output Gate) has and this project didn't:
a CODE-OWNED check that runs AFTER generation and can withhold output,
independent of whatever the generator itself might claim about its own
answer.

PER-UNIT, NOT PER-ANSWER (redesigned 2026-11-09): step9_generation.py's
generate_draft() now returns a GenerationDraft of typed AnswerUnit
objects (diagnosis/step/warning), each carrying its OWN citations,
instead of one free-prose answer_text. This module verifies ONE unit at
a time for the same reason that redesign happened: real-time token
streaming shows text to the browser as it's produced, so by the time a
whole-answer verifier could see the complete text, it's already on
screen and a "verification failed" can no longer actually withhold
anything. Verifying per-unit lets the caller (chat_service.py) commit
(and only commit) units that already passed, before any of that unit's
text has reached the browser -- see AnswerUnit's own docstring in
step9_generation.py for the full reasoning.

Two independent signals per unit, same as before:

  1. citation_ids_valid (deterministic, free, no LLM call): every page_id
     a unit cites must be among the page_ids actually retrieved for this
     query. No LLM needed -- exactly the kind of condition the eval
     guidance says code should verify directly rather than asking a
     model to self-report.

  2. groundedness (LLM-as-judge, BINARY, not the 1-5 holistic score
     judge_generation_eval.py uses offline). A live per-unit gate needs
     a hard pass/fail for THAT UNIT's text specifically, not an
     aggregate score across a whole answer.

unit_can_release = citation_ids_valid AND groundedness_pass. A failing
unit must never be shown -- not relabeled, not shown-with-a-disclaimer,
dropped.

verify_draft() runs every unit's two checks IN PARALLEL (a
ThreadPoolExecutor, since this project's Gemini calls are synchronous
throughout -- see gemini_retry.py) rather than one at a time, since
doing N independent judge calls sequentially would multiply this step's
latency by N for no reason; the checks don't depend on each other.
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from google import genai

PROJECT_ROOT = Path(__file__).parent
REPO_ROOT = PROJECT_ROOT.parent
load_dotenv(REPO_ROOT / ".env")

sys.path.insert(0, str(PROJECT_ROOT))
from gemini_retry import call_with_retry  # noqa: E402

GROUNDEDNESS_JUDGE_PROMPT = """You are checking ONE claim from an internal IT-helpdesk \
assistant's answer for groundedness. Decide if the claim below is directly supported by the \
context excerpts -- text content, or the last_updated/source_type metadata shown in each \
excerpt's tag.

Do NOT judge overall answer quality -- ONLY whether this specific claim is supported.

<context>
{context}
</context>

<claim>
{claim}
</claim>

Respond with ONLY a JSON object, no other text: {{"supported": <true|false>, "reason": "<one sentence>"}}"""


@dataclass
class UnitVerdict:
    """Verification outcome for ONE AnswerUnit (step9_generation.py)."""
    unit_type: str
    text: str
    can_release: bool
    citation_ids_valid: bool
    invalid_citation_ids: list[str] = field(default_factory=list)
    groundedness_pass: bool = True
    judge_reason: str | None = None
    judge_error: str | None = None  # set if the judge call itself failed -- see check_groundedness()


@dataclass
class DraftVerification:
    """Result of verifying every unit in a GenerationDraft. released_units
    preserves the draft's original unit_type ordering (diagnosis, then
    steps, then warnings) -- chat_service.py commits them in this order."""
    released_units: list  # list[step9_generation.AnswerUnit] -- the ones that passed
    verdicts: list[UnitVerdict]  # every unit's verdict, released or not, same order as the draft

    @property
    def dropped_count(self) -> int:
        return sum(1 for v in self.verdicts if not v.can_release)


def check_citation_ids(citation_page_ids: list[str], retrieved_page_ids: set[str]) -> tuple[bool, list[str]]:
    """
    Deterministic: every cited page_id must be one of the page_ids
    actually retrieved for this query (search_results, NOT the model's
    own claims about what it used). No LLM call.
    """
    invalid = sorted(set(citation_page_ids) - retrieved_page_ids)
    return (len(invalid) == 0, invalid)


def _parse_judge_response(text: str) -> tuple[bool, str | None]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    parsed = json.loads(text.strip())
    return bool(parsed["supported"]), parsed.get("reason")


def check_groundedness(
    client: genai.Client, context: str, claim_text: str,
) -> tuple[bool, str | None, str | None]:
    """
    Independent per-unit groundedness judge -- a SEPARATE model call from
    generation, given only the retrieved context and this one unit's
    text, returning a binary verdict for just that unit.

    On a judge-call failure (network/parsing), FAILS OPEN to "pass" with
    judge_error set. A judge outage must not silently convert every live
    unit into a withheld one; the caller can inspect judge_error to see
    verification didn't actually run, rather than trusting a false pass
    blindly.
    """
    model = os.environ["GEMINI_MODEL"]
    prompt = GROUNDEDNESS_JUDGE_PROMPT.format(context=context, claim=claim_text)
    try:
        response = call_with_retry(
            "step10_verification", client.models.generate_content, model=model, contents=prompt
        )
        supported, reason = _parse_judge_response(response.text)
        return supported, reason, None
    except Exception as e:  # noqa: BLE001 -- see docstring: fail open, report the error, don't crash the turn
        return True, None, f"{type(e).__name__}: {e}"


def verify_unit(unit, retrieved_page_ids: set[str], context: str, client: genai.Client) -> UnitVerdict:
    """Verifies ONE AnswerUnit (step9_generation.py) -- both checks, combined
    into unit_can_release. `context` is the shared formatted excerpt text
    (built once per draft by verify_draft(), not re-built per unit)."""
    citation_ids_valid, invalid_citation_ids = check_citation_ids(unit.citations, retrieved_page_ids)
    groundedness_pass, judge_reason, judge_error = check_groundedness(client, context, unit.text)
    return UnitVerdict(
        unit_type=unit.unit_type,
        text=unit.text,
        can_release=citation_ids_valid and groundedness_pass,
        citation_ids_valid=citation_ids_valid,
        invalid_citation_ids=invalid_citation_ids,
        groundedness_pass=groundedness_pass,
        judge_reason=judge_reason,
        judge_error=judge_error,
    )


def verify_draft(draft, client: genai.Client | None = None, max_workers: int = 4) -> DraftVerification:
    """
    Verifies every unit in a GenerationDraft (step9_generation.py) IN
    PARALLEL -- see module docstring for why (independent checks,
    sequential judge calls would multiply latency by unit count for no
    benefit). Returns released_units in the SAME order the draft had
    them (diagnosis, steps, warnings), filtered to only the ones that
    passed -- chat_service.py commits them in this order as each is
    confirmed released.
    """
    if not draft.units:
        return DraftVerification(released_units=[], verdicts=[])

    if client is None:
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    retrieved_page_ids = {sr["chunk"]["page_id"] for sr in draft.search_results}
    context = "\n\n".join(
        f"[{sr['chunk']['title']}, last_updated={sr['chunk']['last_updated']}]\n{sr['chunk']['text']}"
        for sr in draft.search_results
    )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        verdicts = list(pool.map(
            lambda u: verify_unit(u, retrieved_page_ids, context, client), draft.units
        ))

    released_units = [u for u, v in zip(draft.units, verdicts) if v.can_release]
    return DraftVerification(released_units=released_units, verdicts=verdicts)


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT))
    from pipeline import PipelineClients, run_pipeline
    from step9_generation import GenerationResult, generate_draft, render_committed_units

    print("Building pipeline clients...")
    clients = PipelineClients.build()

    test_messages = [
        "My VPN keeps disconnecting right after I log in, error 0x8007042B.",
        "My screen just went black.",
    ]

    print("\nStep 10 verification -- smoke test\n")
    for msg in test_messages:
        outcome = run_pipeline(msg, clients)
        if outcome.stopped_at != "ok":
            print(f"MESSAGE: {msg}\n  (pipeline stopped at: {outcome.stopped_at})\n")
            continue
        for r in outcome.results:
            draft = generate_draft(r.sub_query, r.search_results, r.low_confidence, client=clients.gemini)
            print(f"--- sub_query: {r.sub_query!r} ---")
            if isinstance(draft, GenerationResult):
                print(f"  (mode={draft.mode} -- no units to verify)\n")
                continue
            verification = verify_draft(draft, client=clients.gemini)
            for v in verification.verdicts:
                status = "RELEASED" if v.can_release else "DROPPED"
                print(f"  [{status}] ({v.unit_type}) {v.text}")
                if not v.citation_ids_valid:
                    print(f"      invalid_citation_ids={v.invalid_citation_ids}")
                if not v.groundedness_pass:
                    print(f"      groundedness_fail: {v.judge_reason}")
                if v.judge_error:
                    print(f"      judge_error={v.judge_error}")
            final = render_committed_units(
                verification.released_units, draft.search_results, draft.superseded_alternatives,
                verification.dropped_count,
            )
            print(f"  FINAL mode={final.mode} caveat={final.caveat!r}")
            print()
