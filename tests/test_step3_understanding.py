"""
Real correctness test for Step 3 (src/step3_understanding.py) against the
full eval_edge_case_queries.jsonl set. Expected category is derived from
each query's known expected_page_id (mapped through confluence_export.jsonl's
real ancestors[0].title), not guessed -- the same "check against real data,
not the label" discipline used for Step 1 and the retrieval eval.

Multi-turn (G02-G06 except G01) and multi-target (expected_page_id is a
list) queries are excluded from category scoring -- Step 3 outputs ONE
category per call, so a query mapping to two different-category documents
(e.g. D06's compare-two-documents case) has no single right answer to
score against.

LESSON LEARNED 2026-09-13 (first draft of this test had a real bug):
`unanswerable=True` in eval_edge_case_queries.jsonl means "no exact KB
DOCUMENT exists for this" -- it does NOT mean "the category is
ambiguous." Most unanswerable queries (e.g. "are we allowed to use a
personal VPN" -- clearly IT Policies, just no document covers that exact
policy) have a perfectly clear topic category; Step 3's job is to
categorize the TOPIC, not to know whether a matching document exists
(that's retrieval's problem downstream). The first draft of
EXPECTED_UNCLEAR_CATEGORY wrongly included every unanswerable=True query,
which produced 9 false "failures" against a genuinely correct Step 3.
Only include a query here if the MESSAGE ITSELF is ambiguous across
categories, independent of whether a KB document exists for it.

Run: python tests/test_step3_understanding.py
"""

import json
import sys
import time
from pathlib import Path

TESTS_DIR = Path(__file__).parent
PROJECT_ROOT = TESTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from step3_understanding import understand_query, CATEGORY_VALUES  # noqa: E402
from query_parsing import parse_query  # noqa: E402
from google import genai  # noqa: E402
import os  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# Queries where error_codes should be non-empty, and what they should contain.
EXPECTED_ERROR_CODES = {
    "A05": ["0x8007042B"],
    "C01": ["0x8007042B"],
    "D04": ["0x8007042B"],
}

# Queries where wants_latest should be True. Corrected 2026-09-13: F07
# ("What's the CURRENT procedure...") was originally missing from this
# set -- an oversight in building expectations, not a Step 3 bug. Always
# re-check the actual query text for the system prompt's own trigger
# words ("current", "latest", "up to date") rather than relying on memory
# of which queries were discussed earlier in conversation.
EXPECTED_WANTS_LATEST = {"D03", "G06", "F07"}

# Queries genuinely AMBIGUOUS ACROSS CATEGORIES -- NOT the same thing as
# "unanswerable" (no exact KB document exists). Corrected 2026-09-13:
# the original list wrongly conflated eval_only's unanswerable=True flag
# with category ambiguity -- most unanswerable queries (D07, D08, F04,
# F05, F08, F09, etc.) have a perfectly clear TOPIC category, they just
# lack a matching document, which is retrieval's problem, not Step 3's.
# Only genuinely cross-category-ambiguous queries belong here.
EXPECTED_UNCLEAR_CATEGORY = {"B02", "E01", "E06"}


def main():
    with open(DATA_PROCESSED / "confluence_export.jsonl", encoding="utf-8") as f:
        pages = [json.loads(line) for line in f]
    page_to_category = {
        p["id"]: p["ancestors"][0]["title"] if p["ancestors"] else None for p in pages
    }

    with open(DATA_PROCESSED / "eval_edge_case_queries.jsonl", encoding="utf-8") as f:
        rows = {}
        for line in f:
            r = json.loads(line)
            rows[r["query_id"]] = r

    # Build expected category per scorable query.
    expected_category = {}
    scorable_ids = []
    for qid, r in rows.items():
        if r["letter"] == "G" and qid != "G01":
            continue  # multi-turn, not real standalone text
        eid = r["expected_page_id"]
        if qid in EXPECTED_UNCLEAR_CATEGORY:
            expected_category[qid] = "unclear"
            scorable_ids.append(qid)
        elif isinstance(eid, str):
            expected_category[qid] = page_to_category.get(eid)
            scorable_ids.append(qid)
        # lists (multi-target) and other unlisted None cases: no single
        # expected category, skip category scoring but still run the call
        # to check it doesn't crash.

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    results = []
    out_path = DATA_PROCESSED / "step3_batch_test_results.txt"
    with open(out_path, "w", encoding="utf-8") as out:
        for i, qid in enumerate(sorted(rows.keys()), start=1):
            r = rows[qid]
            if r["letter"] == "G" and qid != "G01":
                continue
            parsed = parse_query(r["query"])
            try:
                result = understand_query(parsed.cleaned_text, client=client)
            except Exception as e:
                line = f"[{i}] {qid}: EXCEPTION {e}"
                print(line, flush=True)
                out.write(line + "\n")
                out.flush()
                time.sleep(4.5)
                continue

            line = (
                f"{qid:6s} category={result.category!r:35s} "
                f"error_codes={result.error_codes} wants_latest={result.wants_latest} "
                f"ambiguous={result.ambiguous} warnings={result.validation_warnings}"
            )
            print(line, flush=True)
            out.write(line + "\n")
            out.flush()
            results.append((qid, result))
            time.sleep(4.5)

    # Score
    print("\n--- Category correctness (only for queries with a known single expected category) ---")
    cat_correct, cat_total, cat_fails = 0, 0, []
    for qid, result in results:
        if qid in expected_category:
            cat_total += 1
            if result.category == expected_category[qid]:
                cat_correct += 1
            else:
                cat_fails.append((qid, expected_category[qid], result.category))
    print(f"{cat_correct}/{cat_total} correct")
    for qid, exp, got in cat_fails:
        print(f"  FAIL {qid}: expected {exp!r}, got {got!r}")

    print("\n--- error_codes correctness ---")
    ec_correct, ec_total, ec_fails = 0, 0, []
    for qid, result in results:
        if qid in EXPECTED_ERROR_CODES:
            ec_total += 1
            if sorted(result.error_codes) == sorted(EXPECTED_ERROR_CODES[qid]):
                ec_correct += 1
            else:
                ec_fails.append((qid, EXPECTED_ERROR_CODES[qid], result.error_codes))
    print(f"{ec_correct}/{ec_total} correct")
    for qid, exp, got in ec_fails:
        print(f"  FAIL {qid}: expected {exp}, got {got}")

    print("\n--- False-positive error_codes (queries with NO expected code but one was extracted) ---")
    fp_count = 0
    for qid, result in results:
        if qid not in EXPECTED_ERROR_CODES and result.error_codes:
            fp_count += 1
            print(f"  {qid}: unexpectedly extracted {result.error_codes}")
    if fp_count == 0:
        print("  (none)")

    print("\n--- wants_latest correctness ---")
    wl_correct, wl_total, wl_fails = 0, 0, []
    for qid, result in results:
        expected = qid in EXPECTED_WANTS_LATEST
        wl_total += 1
        if result.wants_latest == expected:
            wl_correct += 1
        else:
            wl_fails.append((qid, expected, result.wants_latest))
    print(f"{wl_correct}/{wl_total} correct")
    for qid, exp, got in wl_fails:
        print(f"  FAIL {qid}: expected {exp}, got {got}")

    print("\n--- Any validation warnings raised ---")
    warn_count = 0
    for qid, result in results:
        if result.validation_warnings:
            warn_count += 1
            print(f"  {qid}: {result.validation_warnings}")
    if warn_count == 0:
        print("  (none)")


if __name__ == "__main__":
    main()
