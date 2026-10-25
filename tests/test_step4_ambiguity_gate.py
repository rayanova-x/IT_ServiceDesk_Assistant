"""
Real correctness test for Step 4 (src/step4_ambiguity_gate.py) against
the 62 real Step 3 outputs already collected in
data/processed/step3_batch_test_results.txt -- no new API calls needed,
Step 4 has no LLM call of its own.

Checks:
  - Every ambiguous=True case gets search_category=None, low_confidence=True.
  - Every clean category (not "unclear") with ambiguous=False gets that
    exact category as the filter, low_confidence=False.
  - Every category="unclear" case (regardless of ambiguous) widens the
    search (search_category=None).
  - No exceptions across all 62 real cases.

Run: python tests/test_step4_ambiguity_gate.py
"""

import re
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).parent
PROJECT_ROOT = TESTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from step3_understanding import UnderstandingResult  # noqa: E402
from step4_ambiguity_gate import apply_ambiguity_gate  # noqa: E402

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

LINE_PATTERN = re.compile(
    r"(\S+)\s+category='([^']*)'\s+error_codes=(\[[^\]]*\])\s+"
    r"wants_latest=(\w+)\s+ambiguous=(\w+)"
)


def load_step3_results() -> list[tuple[str, UnderstandingResult]]:
    results = []
    with open(DATA_PROCESSED / "step3_batch_test_results.txt", encoding="utf-8") as f:
        for line in f:
            m = LINE_PATTERN.match(line)
            if not m:
                continue
            qid, category, error_codes_str, wants_latest, ambiguous = m.groups()
            understanding = UnderstandingResult(
                rewritten_query="",  # not needed for this test
                category=category,
                error_codes=eval(error_codes_str),
                wants_latest=wants_latest == "True",
                ambiguous=ambiguous == "True",
            )
            results.append((qid, understanding))
    return results


def main():
    results = load_step3_results()
    print(f"Loaded {len(results)} real Step 3 outputs\n")

    errors = []
    ambiguous_count, unclear_not_ambiguous_count, clean_count = 0, 0, 0

    for qid, understanding in results:
        try:
            # apply_ambiguity_gate() takes a bare category string as of
            # 2026-10-24 (step_route_or_search.py replaced Steps 0/2/3 and
            # doesn't produce a separate `ambiguous` boolean -- see
            # step4_ambiguity_gate.py's own simplification note). This
            # test still exercises real historical Step 3 outputs, so the
            # ambiguous=True cases are passed through as category="unclear"
            # here, matching how they'd be collapsed under the new
            # single-field contract -- both already produced identical
            # GateDecision output before this change (verified by this
            # same test, which checked them against the same expected
            # result), so this is a like-for-like substitution, not a
            # test weakening.
            effective_category = "unclear" if understanding.ambiguous else understanding.category
            decision = apply_ambiguity_gate(effective_category)
        except Exception as e:
            errors.append(f"{qid}: EXCEPTION {e}")
            continue

        if understanding.ambiguous:
            ambiguous_count += 1
            if decision.search_category is not None or not decision.low_confidence:
                errors.append(
                    f"{qid}: ambiguous=True should give search_category=None, "
                    f"low_confidence=True -- got {decision}"
                )
        elif understanding.category == "unclear":
            unclear_not_ambiguous_count += 1
            if decision.search_category is not None or not decision.low_confidence:
                errors.append(
                    f"{qid}: category=unclear should widen search regardless of "
                    f"ambiguous flag -- got {decision}"
                )
        else:
            clean_count += 1
            if decision.search_category != understanding.category or decision.low_confidence:
                errors.append(
                    f"{qid}: clean category {understanding.category!r} should pass "
                    f"through as search_category with low_confidence=False -- got {decision}"
                )

    print(f"Breakdown: {clean_count} clean-category, {ambiguous_count} ambiguous, "
          f"{unclear_not_ambiguous_count} unclear-but-not-ambiguous")
    print()
    if errors:
        print(f"{len(errors)} PROBLEMS FOUND:")
        for e in errors:
            print(f"  {e}")
    else:
        print(f"All {len(results)} cases behaved correctly, 0 exceptions.")

    return len(errors) == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
