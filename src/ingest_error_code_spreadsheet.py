"""
Second ingestion CONNECTOR, for a genuinely different source shape than
Confluence: a spreadsheet of known error codes (data/raw/error_code_lookup.csv),
not prose SOP pages. Produces chunks_from_error_codes.jsonl in the SAME
canonical chunk schema chunk_confluence.py produces (chunk_id, page_id,
title, category, status, version_number, last_updated, labels,
restricted_to, header_path, chunk_index, secondary_split, token_count,
text) plus one addition -- source_type -- so a chunk's origin connector
stays traceable after both sources are combined for embedding.

WHY A SEPARATE CONNECTOR SCRIPT, NOT ONE MORE CASE INSIDE
chunk_confluence.py (researched, not assumed -- see PROJECT_HANDOFF.md):
real multi-source RAG ingestion architectures use one connector per
source (different auth, pagination, and native structure per source),
each normalizing its own source into the SAME canonical schema, with
convergence happening at the embedding/storage step, not the connector
step. So Confluence and this spreadsheet stay two separate scripts
producing two separate chunk files -- embed_and_store.py is the one
place they combine, mirroring how a real platform (e.g. Unstructured's
connector model) keeps per-source connectors independent.

WHY A ROW = ONE CHUNK (not further split, unlike Confluence's per-
section splitting): each row is already a complete, self-contained
answer ("this code means this, fix it this way") -- there's no
sub-structure within a row worth cutting on, unlike a multi-step
diagnostic SOP page. Splitting a row further would destroy the one
fact it represents.

WHY THIS CONTENT DOESN'T DUPLICATE THE EXISTING KB (verified, not
assumed -- checked confluence_export.jsonl directly before writing
this): grepped every page title for error/code/lookup/table/inventory/
asset -- only 6 matches, all prose diagnostic GUIDES (e.g. "Network SSL
VPN failure -- Full Diagnostic Guide"), none of which mention the
specific error codes below by their exact code string. 0x8007042B in
particular appears in real eval queries (A05/C01/D04,
eval_edge_case_queries.jsonl) and is answered today only by the general
VPN diagnostic prose page, which never states the exact code. This
spreadsheet adds a genuinely new, specific fact the existing KB doesn't
have -- and creates a real retrieval-quality question worth testing:
does a query naming the exact code now correctly prefer this terse,
precise row over the long general-purpose SOP page?

SCOPE DISCIPLINE (2026-09-21, real mistake found and corrected): the
first version of this spreadsheet had 15 rows, 7 of which were for
systems this project has ZERO other coverage for (printers, Windows
Update, Outlook/Exchange, badge readers, MDM) -- invented content that
LOOKED plausible in isolation but didn't fit this project's actual real
ticket taxonomy (data/raw/enriched_tickets.csv's real Subcategory
values). Caught only because the user explicitly asked to cross-check
scope before building eval coverage on top of it. Rebuilt to keep ONLY
rows whose `subcategory` value is a REAL subcategory from that dataset
(verified directly, not assumed) -- 4 original VPN rows (exact match:
"Network SSL VPN failure") plus 7 new rows for real subcategories that
had zero prior error-code coverage (License expired, Application
deadlock, Application crash on deploy, DHCP failure, Duplicate IP
detected, Latency spikes, Switch port failure) -- each cross-checked
against its real Confluence SOP page to confirm the page exists and
doesn't already state a code (same diligence as the original 0x8007042B
check). See PROJECT_HANDOFF.md for the full before/after. The
`subcategory` column exists specifically so this cross-reference is
explicit and checkable, not implicit in a free-text category guess.

Run: python src/ingest_error_code_spreadsheet.py
"""

import csv
import json
from datetime import date
from pathlib import Path

SRC_DIR = Path(__file__).parent
PROJECT_ROOT = SRC_DIR.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_CHUNKS = PROJECT_ROOT / "data" / "chunks"

SOURCE_CSV = DATA_RAW / "error_code_lookup.csv"
OUTPUT_CHUNKS = DATA_CHUNKS / "chunks_from_error_codes.jsonl"

# Page IDs in the 900xxx range so they never collide with the Confluence
# corpus's 200xxx page IDs (see confluence_export.jsonl) -- a real
# multi-source system needs this kind of namespacing precisely because
# two independent connectors will never coordinate their ID schemes.
PAGE_ID_PREFIX = "900"


def row_to_text(row: dict) -> str:
    """
    Turn one spreadsheet row into embeddable text. Written as a short,
    labeled paragraph rather than a raw comma-joined row -- an
    embedding model trained on natural language matches "Error code
    0x8007042B... caused by..." far better than a terse
    "0x8007042B,VPN Client,Tunnel..." string, even though the
    UNDERLYING data is tabular. This is the real ingestion decision a
    spreadsheet source forces that a prose source never has to make.
    """
    return (
        f"Error code {row['error_code']} ({row['system']}).\n"
        f"Cause: {row['short_cause']}\n"
        f"Fix: {row['fix_summary']}"
    )


def cross_check_subcategories(rows: list[dict], real_subcategories: set[str]) -> None:
    """
    Fail loudly rather than silently ingest a row whose `subcategory`
    doesn't match a REAL subcategory from enriched_tickets.csv -- see
    module docstring's "SCOPE DISCIPLINE" note. This is the check that
    was missing the first time this file was built; it now runs every
    time this connector runs, not just once by hand.
    """
    bad = [r for r in rows if r["subcategory"] not in real_subcategories]
    if bad:
        bad_list = ", ".join(f"{r['error_code']} ({r['subcategory']!r})" for r in bad)
        raise ValueError(
            f"{len(bad)} row(s) reference a subcategory that doesn't exist in "
            f"enriched_tickets.csv's real Subcategory values -- out of project "
            f"scope, fix or remove before ingesting: {bad_list}"
        )


def token_count(text: str) -> int:
    # Same word-based estimate as chunk_confluence.py, for consistency
    # across sources feeding the same downstream chunk-size assumptions.
    return round(len(text.split()) * 1.3)


def main():
    with open(SOURCE_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Loaded {len(rows)} rows from {SOURCE_CSV.name}")

    # Cross-check every row's subcategory against the REAL ticket
    # taxonomy before ingesting anything -- see cross_check_subcategories()
    # and the module docstring's "SCOPE DISCIPLINE" note.
    import pandas as pd
    tickets = pd.read_csv(DATA_RAW / "enriched_tickets.csv")
    real_subcategories = set(tickets["Subcategory"].unique())
    cross_check_subcategories(rows, real_subcategories)
    print(f"Cross-checked all {len(rows)} rows against real ticket subcategories -- OK")

    today = date.today().isoformat()
    chunks = []
    for i, row in enumerate(rows):
        page_id = f"{PAGE_ID_PREFIX}{i:03d}"
        text = row_to_text(row)
        chunks.append({
            "chunk_id": f"{page_id}-C00",
            "page_id": page_id,
            "title": f"Error code {row['error_code']} -- {row['system']}",
            "category": row["category"],
            "status": "current",
            # No real revision history for a spreadsheet row the way a
            # Confluence page has -- version 1, dated today's ingestion
            # run, is the honest value rather than a fabricated history.
            "version_number": 1,
            "last_updated": today,
            "labels": ["error-code-lookup"],
            "restricted_to": [],
            "header_path": row["system"],
            "subcategory": row["subcategory"],  # cross-referenced against real ticket taxonomy, see cross_check_subcategories()
            "chunk_index": 0,
            "secondary_split": False,
            "token_count": token_count(text),
            "text": text,
            "source_type": "spreadsheet",  # see module docstring -- traceable after merging with Confluence chunks
        })

    with open(OUTPUT_CHUNKS, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"Wrote {len(chunks)} chunks to {OUTPUT_CHUNKS}")
    by_category = {}
    for c in chunks:
        by_category[c["category"]] = by_category.get(c["category"], 0) + 1
    print("By category:", by_category)


if __name__ == "__main__":
    main()
