"""
Real correctness test for Step 1 (src/query_parsing.py), not just a
smoke test. For every query in data/processed/eval_edge_case_queries.jsonl
where I hand-verified (by reading the actual text) what SHOULD be
extracted, check parse_query() against that expectation and report exact
pass/fail -- including false positives (extracted something that isn't
really there) and false negatives (missed something that is).

Run: python tests/test_query_parsing.py
"""

import json
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).parent
PROJECT_ROOT = TESTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from query_parsing import parse_query  # noqa: E402

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# Hand-verified expected extractions, read directly from each query's
# actual text -- not assumed from the subtype label. Only fields with a
# real expectation are listed; omitted fields default to "expect empty".
EXPECTED = {
    "A01": {"confirmed_category": "Network", "ticket_numbers": ["778215"]},
    "A05": {"error_codes": ["0x8007042B"]},
    "A10": {"confirmed_category": "Network"},
    "C01": {"error_codes": ["0x8007042B"]},
    "C02": {"ip_addresses": ["10.42.18.117"]},
    "C03": {"hostnames": ["FIN-LT-0423"]},
    "C05": {"file_paths": ["C:\\ProgramData\\CorpApp\\license.dat"]},
    "C06": {"ticket_numbers": ["611872"]},
    "C11": {"confirmed_category": "Hardware"},
    "D04": {"error_codes": ["0x8007042B"]},
    # F10 mentions "duplicate IP address" as a CONCEPT, not an actual IP
    # number -- must NOT false-positive an IP extraction.
    "F10": {},
    # G04 uses "turn_1: ... | turn_2: ..." -- pipe-separated, but none of
    # it is a real ticket/priority/category field. Must not false-positive.
    "G04": {},
    "G02": {},
    "G03": {},
    "G05": {},
    "G06": {},
}

FIELDS = ["confirmed_category", "error_codes", "ip_addresses", "hostnames",
          "file_paths", "ticket_numbers"]


def default_expected(field):
    return None if field == "confirmed_category" else []


def main():
    with open(DATA_PROCESSED / "eval_edge_case_queries.jsonl", encoding="utf-8") as f:
        rows = {}
        for line in f:
            r = json.loads(line)
            rows[r["query_id"]] = r

    passed, failed = 0, 0
    for qid, expected in EXPECTED.items():
        row = rows[qid]
        parsed = parse_query(row["query"])

        mismatches = []
        for field in FIELDS:
            exp = expected.get(field, default_expected(field))
            actual = getattr(parsed, field)
            if field == "confirmed_category":
                if exp != actual:
                    mismatches.append(f"{field}: expected {exp!r}, got {actual!r}")
            else:
                if sorted(exp) != sorted(actual):
                    mismatches.append(f"{field}: expected {sorted(exp)}, got {sorted(actual)}")

        if mismatches:
            failed += 1
            print(f"FAIL {qid}: {'; '.join(mismatches)}")
            print(f"      query: {row['query'][:100]!r}")
        else:
            passed += 1

    print(f"\n{passed} passed, {failed} failed, out of {len(EXPECTED)} hand-verified cases")

    # Also sweep every one of the other 47 queries in the file that have
    # NO hand-verified expectation above, and flag anything Step 1
    # extracted from them -- a human should eyeball these, since an
    # unexpected extraction on a query I didn't manually check could be
    # a real false positive I haven't caught yet.
    print("\n--- Extractions found on queries with no hand-verified expectation (review these) ---")
    unexpected_extractions = 0
    for qid, row in rows.items():
        if qid in EXPECTED:
            continue
        parsed = parse_query(row["query"])
        found = {f: getattr(parsed, f) for f in FIELDS if getattr(parsed, f)}
        if found:
            unexpected_extractions += 1
            print(f"  {qid}: {found}")
            print(f"      query: {row['query'][:100]!r}")

    if unexpected_extractions == 0:
        print("  (none)")

    return failed == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
