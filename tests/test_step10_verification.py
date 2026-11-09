"""
Real correctness test for Step 10 (post-generation verification/output
gate -- see src/step10_verification.py, src/step9_generation.py's
AnswerUnit/GenerationDraft). Covers the DETERMINISTIC parts with no live
Gemini call (check_citation_ids(), the partial-commit rendering in
render_committed_units()) -- fast, free, deterministic, same convention
as test_permission_filter.py/test_error_handling.py. A live judge-call
smoke test already exists as step10_verification.py's own __main__
block; this file does not duplicate that cost.

Checks:
  1. check_citation_ids() rejects a page_id that wasn't actually
     retrieved, and accepts one that was -- the one check in this module
     that needs no LLM call at all.
  2. verify_draft(), with the groundedness judge FAKED to always pass,
     correctly releases every unit when citations are all valid, and
     drops ONLY the unit(s) with an invalid citation -- never discarding
     a sibling unit that was fine.
  3. render_committed_units() on a mix of released/dropped units shows
     the released ones normally and adds a caveat noting how many were
     withheld -- never silently drops the note, never retracts a
     released unit.
  4. render_committed_units() with ZERO released units degrades to
     candidates_only (the all-failed case), same shape as the existing
     low_confidence fallback.

Run: python tests/test_step10_verification.py
"""

import sys
from pathlib import Path
from unittest.mock import patch

TESTS_DIR = Path(__file__).parent
PROJECT_ROOT = TESTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from step9_generation import AnswerUnit, GenerationDraft  # noqa: E402
from step10_verification import check_citation_ids, verify_draft  # noqa: E402
from step9_generation import render_committed_units  # noqa: E402

FAKE_CHUNK_A = {
    "chunk_id": "200046-C00", "page_id": "200046", "title": "Network SSL VPN failure",
    "last_updated": "2026-07-01", "status": "current", "text": "VPN excerpt text", "header_path": "VPN",
}
FAKE_CHUNK_B = {
    "chunk_id": "200036-C00", "page_id": "200036", "title": "Laptop screen black screen",
    "last_updated": "2026-01-02", "status": "current", "text": "black screen excerpt text", "header_path": "Display",
}
FAKE_SEARCH_RESULTS = [
    {"chunk_id": "200046-C00", "rerank_score": 3.0, "chunk": FAKE_CHUNK_A},
    {"chunk_id": "200036-C00", "rerank_score": 2.5, "chunk": FAKE_CHUNK_B},
]
RETRIEVED_PAGE_IDS = {"200046", "200036"}


def check_citation_ids_rejects_unretrieved(errors: list) -> None:
    valid, invalid = check_citation_ids(["200046"], RETRIEVED_PAGE_IDS)
    if not valid or invalid:
        errors.append(f"expected a real retrieved page_id to validate clean, got valid={valid} invalid={invalid}")

    valid, invalid = check_citation_ids(["200046", "999999"], RETRIEVED_PAGE_IDS)
    if valid or invalid != ["999999"]:
        errors.append(f"expected exactly ['999999'] flagged invalid, got valid={valid} invalid={invalid}")

    valid, invalid = check_citation_ids([], RETRIEVED_PAGE_IDS)
    if not valid or invalid:
        errors.append(f"expected an empty citation list to validate clean (nothing to check), got valid={valid} invalid={invalid}")


def _fake_groundedness_always_pass(client, context, claim_text):
    return True, "fake pass", None


def check_verify_draft_drops_only_the_bad_citation_unit(errors: list) -> None:
    """With groundedness FAKED to always pass, the only thing that can
    fail a unit is check_citation_ids() -- isolates Step 10's citation
    gate from the judge call, which the live smoke test already covers
    separately."""
    units = [
        AnswerUnit(unit_type="diagnosis", text="Likely a tunnel negotiation failure.", citations=["200046"]),
        AnswerUnit(unit_type="step", text="Update the VPN client.", citations=["200046"]),
        AnswerUnit(unit_type="step", text="This cites a page that was never retrieved.", citations=["999999"]),
        AnswerUnit(unit_type="warning", text="Don't modify the registry.", citations=["200036"]),
    ]
    draft = GenerationDraft(units=units, search_results=FAKE_SEARCH_RESULTS)

    with patch("step10_verification.check_groundedness", side_effect=_fake_groundedness_always_pass):
        verification = verify_draft(draft, client=object())

    if verification.dropped_count != 1:
        errors.append(f"expected exactly 1 dropped unit (the bad citation), got {verification.dropped_count}")
    released_texts = [u.text for u in verification.released_units]
    if "This cites a page that was never retrieved." in released_texts:
        errors.append("the unit with an invalid citation was released -- it must be dropped")
    if "Update the VPN client." not in released_texts:
        errors.append("a sibling unit with a VALID citation was incorrectly dropped alongside the bad one")
    if len(verification.released_units) != 3:
        errors.append(f"expected 3 of 4 units released, got {len(verification.released_units)}")


def check_render_committed_units_partial(errors: list) -> None:
    released = [
        AnswerUnit(unit_type="diagnosis", text="Likely a tunnel negotiation failure.", citations=["200046"]),
        AnswerUnit(unit_type="step", text="Update the VPN client.", citations=["200046"]),
    ]
    gen = render_committed_units(released, FAKE_SEARCH_RESULTS, [], dropped_count=1)

    if gen.mode != "synthesized":
        errors.append(f"expected mode='synthesized' when at least one unit was released, got {gen.mode!r}")
    if "Update the VPN client." not in (gen.answer_text or ""):
        errors.append("a released step did not appear in the rendered answer_text")
    if not gen.caveat or "1" not in gen.caveat:
        errors.append(f"expected a caveat noting exactly 1 withheld recommendation, got {gen.caveat!r}")
    if len(gen.citations) != 1 or gen.citations[0].page_id != "200046":
        errors.append(f"expected exactly one citation (200046, from the released units only), got {gen.citations}")


def check_render_committed_units_all_dropped(errors: list) -> None:
    gen = render_committed_units([], FAKE_SEARCH_RESULTS, [], dropped_count=3)

    if gen.mode != "candidates_only":
        errors.append(f"expected mode='candidates_only' when every unit was dropped, got {gen.mode!r}")
    if gen.answer_text:
        errors.append(f"expected no answer_text when every unit was dropped, got {gen.answer_text!r}")
    if len(gen.citations) != len(FAKE_SEARCH_RESULTS):
        errors.append(
            f"expected the raw search_results surfaced as candidates_only citations "
            f"({len(FAKE_SEARCH_RESULTS)}), got {len(gen.citations)}"
        )


def main() -> bool:
    checks = [
        ("check_citation_ids rejects an unretrieved page_id", check_citation_ids_rejects_unretrieved),
        ("verify_draft drops only the unit with an invalid citation", check_verify_draft_drops_only_the_bad_citation_unit),
        ("render_committed_units shows a partial-withholding caveat", check_render_committed_units_partial),
        ("render_committed_units degrades to candidates_only when all units fail", check_render_committed_units_all_dropped),
    ]

    errors: list = []
    for name, fn in checks:
        before = len(errors)
        try:
            fn(errors)
        except Exception as e:
            errors.append(f"{name}: unexpected exception {e!r}")
        status = "FAIL" if len(errors) > before else "ok"
        print(f"  [{status}] {name}")

    print()
    if errors:
        print(f"{len(errors)} PROBLEMS FOUND:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("All Step 10 verification checks passed.")
    return len(errors) == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
