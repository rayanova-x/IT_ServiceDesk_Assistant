"""
One-off diagnostic (not part of the pipeline): scans data/raw/rag_corpus_full.jsonl
for the exact bug found in the VPN confuser-pair investigation (2026-09-14) --
a markdown "### Heading" line immediately followed by body text with NO blank
line between them. strip_cross_references() (src/build_confluence_export.py)
splits on blank lines only, so a heading+body block with no blank line gets
treated as one paragraph, joined, and re-emitted as a single line still
starting with "### " -- which to_storage_format() then wraps ENTIRELY in one
<h4> tag, including the body sentence(s) that should have stayed as
retrievable text. Confirmed on "Network SSL VPN failure" (SOP-0046): its
distinguishing "Overview" symptom sentence was swallowed into the <h4> label
and never made it into any embedded/searchable chunk.

Run: python src/scan_heading_bug.py
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
REPO_ROOT = PROJECT_ROOT.parent
DATA_RAW = REPO_ROOT / "data" / "raw"


def main():
    rows = [json.loads(l) for l in open(DATA_RAW / "rag_corpus_full.jsonl", encoding="utf-8")]

    affected = []
    for r in rows:
        body = r.get("body", "")
        lines = body.split("\n")
        doc_headings_affected = []
        for i, line in enumerate(lines):
            if line.strip().startswith("### "):
                if i + 1 < len(lines) and lines[i + 1].strip() != "":
                    doc_headings_affected.append((line.strip(), lines[i + 1].strip()[:60]))
        if doc_headings_affected:
            affected.append((r["doc_id"], r["title"], doc_headings_affected))

    print(f"Total docs scanned: {len(rows)}")
    print(f"Documents with >=1 affected heading: {len(affected)}")
    total_instances = sum(len(headings) for _, _, headings in affected)
    print(f"Total affected heading instances: {total_instances}")
    print()

    out_path = REPO_ROOT / "data" / "processed" / "heading_bug_scan.jsonl"
    with open(out_path, "w", encoding="utf-8") as out_f:
        for doc_id, title, headings in affected:
            record = {"doc_id": doc_id, "title": title, "affected_headings": [
                {"heading": h, "next_line_preview": nl} for h, nl in headings
            ]}
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    for doc_id, title, headings in affected:
        print(f"{doc_id} | {title}")
        for h, nl in headings:
            safe_nl = nl.encode("ascii", "replace").decode("ascii")
            print(f"    {h!r} -> {safe_nl!r}")
    print(f"\nSaved full scan to {out_path}")


if __name__ == "__main__":
    main()
