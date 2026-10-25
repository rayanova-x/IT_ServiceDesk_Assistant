"""
Step 1 of the pipeline: regex pre-parsing. No LLM, instant, free.

Per PROJECT_HANDOFF.md Section 3:
  -> strip ticket#/priority/employee-name (routing info, not search-relevant)
  -> if explicit "Category:" field present, trust it directly
  -> regex-extract fixed-format identifiers (error codes, IPs, hostnames,
     file paths) -> confirmed_entities (feeds the case-facts memory block,
     not yet built)

Patterns are grounded in real examples from
data/processed/eval_edge_case_queries.jsonl (categories A and C) rather
than guessed formats -- see that file for the exact phrasings each
pattern is meant to catch.
"""

import re
from dataclasses import dataclass, field


@dataclass
class ParsedQuery:
    cleaned_text: str  # original text with routing-info lines stripped
    confirmed_category: str | None = None  # from an explicit "Category:" field
    error_codes: list[str] = field(default_factory=list)
    ip_addresses: list[str] = field(default_factory=list)
    hostnames: list[str] = field(default_factory=list)
    file_paths: list[str] = field(default_factory=list)
    ticket_numbers: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Routing-info fields: ticket#, priority, employee-name-adjacent lines.
# These get STRIPPED from the text handed to search/understanding -- they're
# bookkeeping, not part of the actual problem description. Still captured
# separately (ticket_numbers, confirmed_category) since they're useful
# elsewhere (case-facts memory, direct category trust).
# ---------------------------------------------------------------------------
# Fields can share one line separated by "|" (e.g. "Priority: Medium |
# Category: Network"), not just one field per line -- strip these segments
# individually rather than requiring the whole line to be a single field.
TICKET_FIELD_SEGMENT = re.compile(
    r"(?:^|\|)\s*(ticket\s*#?\s*:?\s*\d+|priority\s*:\s*[^|\n]*|category\s*:\s*[^|\n]*|subcategory\s*:\s*[^|\n]*)",
    re.IGNORECASE | re.MULTILINE,
)

# "Ticket #778215" or "ticket 611872" appearing inline within prose, not just
# on its own structured-field line (see C06: "Following up on ticket 611872").
TICKET_NUMBER_INLINE = re.compile(r"\bticket\s*#?\s*(\d{4,})\b", re.IGNORECASE)

# Category values are short labels (e.g. "Network", "Hardware"), not full
# sentences -- stop at the first punctuation/pipe/newline/field boundary so
# a mid-sentence "Category: X" (see C11's "Ticket says Category: Hardware,
# but honestly...") doesn't swallow the rest of the sentence as the value.
CATEGORY_FIELD = re.compile(
    r"category\s*:\s*([^|\n,.;:]+?)(?=\s+(?:priority|category|subcategory|ticket)\s*:|[|\n,.;]|$)",
    re.IGNORECASE | re.MULTILINE,
)

# ---------------------------------------------------------------------------
# Fixed-format identifiers. These are open-vocabulary-adjacent but
# structurally recognizable -- exactly the kind of token BM25 should see
# verbatim (Section 5: "error codes are not metadata filters, they go into
# BM25 query text").
# ---------------------------------------------------------------------------
# Windows-style hex error codes (0x8007042B) -- the only error-code format
# seen in this project's real/synthetic data so far.
ERROR_CODE = re.compile(r"\b0x[0-9A-Fa-f]{6,8}\b")

IP_ADDRESS = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

# Asset-tag / hostname style identifiers: 2-5 letter dept/site code, hyphens,
# digits (e.g. FIN-LT-0423). Deliberately narrow -- generic all-caps words
# would false-positive constantly otherwise.
HOSTNAME = re.compile(r"\b[A-Z]{2,5}-[A-Z]{2,5}-\d{2,6}\b")

# Windows file paths (C:\...) and Unix-style absolute paths.
FILE_PATH = re.compile(r"\b[A-Za-z]:\\(?:[^\\\s]+\\)*[^\\\s]+|(?<!\w)/(?:[^/\s]+/)*[^/\s]+\.\w+")


def parse_query(raw_text: str) -> ParsedQuery:
    """
    Run all Step 1 regex extraction over raw_text. Extraction runs on the
    ORIGINAL text (before stripping) so identifiers embedded inside
    structured fields (e.g. a ticket number mentioned only in a "Ticket #"
    line) are still caught; only the routing-info lines are removed from
    cleaned_text afterward.
    """
    confirmed_category = None
    cat_match = CATEGORY_FIELD.search(raw_text)
    if cat_match:
        confirmed_category = cat_match.group(1).strip()

    ticket_numbers = sorted(set(TICKET_NUMBER_INLINE.findall(raw_text)))
    error_codes = sorted(set(ERROR_CODE.findall(raw_text)))
    ip_addresses = sorted(set(IP_ADDRESS.findall(raw_text)))
    hostnames = sorted(set(HOSTNAME.findall(raw_text)))
    file_paths = sorted(set(FILE_PATH.findall(raw_text)))

    cleaned_text = TICKET_FIELD_SEGMENT.sub("", raw_text)
    # "Description:" is a label, not a field to discard -- the text after
    # it IS the actual problem statement we want to keep, so only strip
    # the label prefix, not the content.
    cleaned_text = re.sub(r"(?im)^\s*description\s*:\s*", "", cleaned_text)
    # Lines that are now empty (or just leftover "|" separators) after
    # stripping every field segment should disappear entirely, not leave
    # blank/pipe-only lines behind.
    cleaned_text = "\n".join(
        line for line in cleaned_text.splitlines() if line.strip(" |")
    )
    cleaned_text = re.sub(r"\n\s*\n+", "\n\n", cleaned_text).strip()

    return ParsedQuery(
        cleaned_text=cleaned_text,
        confirmed_category=confirmed_category,
        error_codes=error_codes,
        ip_addresses=ip_addresses,
        hostnames=hostnames,
        file_paths=file_paths,
        ticket_numbers=ticket_numbers,
    )


if __name__ == "__main__":
    import json
    from pathlib import Path

    DATA_PROCESSED = Path(__file__).parent.parent / "data" / "processed"
    with open(DATA_PROCESSED / "eval_edge_case_queries.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    print("Step 1 regex parsing -- smoke test against real A/C category queries\n")
    for r in rows:
        if r["letter"] not in ("A", "C"):
            continue
        parsed = parse_query(r["query"])
        found = []
        if parsed.confirmed_category:
            found.append(f"category={parsed.confirmed_category!r}")
        if parsed.error_codes:
            found.append(f"error_codes={parsed.error_codes}")
        if parsed.ip_addresses:
            found.append(f"ips={parsed.ip_addresses}")
        if parsed.hostnames:
            found.append(f"hostnames={parsed.hostnames}")
        if parsed.file_paths:
            found.append(f"file_paths={parsed.file_paths}")
        if parsed.ticket_numbers:
            found.append(f"ticket_numbers={parsed.ticket_numbers}")
        print(f"{r['query_id']} ({r['subtype']}): {', '.join(found) if found else '(nothing extracted)'}")
