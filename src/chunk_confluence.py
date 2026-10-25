"""
Chunker for the simulated Confluence export (confluence_export.jsonl).

What changed from the markdown version, and why:
  1. Body text is now HTML (<h3>, <h4>, <p>, <ol><li>), not markdown (##, ###).
     So we parse with BeautifulSoup and cut on <h4> tags -- <h3> is just the
     page's own title repeated inside the body, not a real subsection, so it
     is never treated as a cut point.
  2. Trashed pages are excluded BEFORE chunking. This isn't a shortcut --
     Confluence's real search API defaults to 'current' status content only,
     so a real ingestion pipeline would never see trashed pages either.
  3. There's no flat 'category' field anymore. It's derived from the page's
     parent in the 'ancestors' page-tree, the same way you'd have to do it
     against a real Confluence space.
  4. doc_type / confuses_with / superseded_by do NOT exist on these pages
     (correctly) -- so there is nothing for the chunker to carry for them,
     even by accident. What DOES get carried: status, version (date +
     revision number -- your only signal for silently-stale pages), labels,
     and read restrictions (useful later for permission-aware retrieval).

Requires: pip install beautifulsoup4 langchain-text-splitters
"""

import json
import re
from pathlib import Path
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

SRC_DIR = Path(__file__).parent
DATA_PROCESSED = SRC_DIR.parent / "data" / "processed"
DATA_CHUNKS = SRC_DIR.parent / "data" / "chunks"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MAX_CHUNK_TOKENS = 220        # above this, a section gets force-split
SECONDARY_CHUNK_SIZE = 800    # characters, for the forced split
SECONDARY_OVERLAP = 120       # ~15% overlap, only on forced splits


def token_count(text: str) -> int:
    """Same word-based estimate as before -- see the earlier notebook for
    why, and how to swap in real tiktoken if you have network access."""
    return round(len(text.split()) * 1.3)


secondary_splitter = RecursiveCharacterTextSplitter(
    chunk_size=SECONDARY_CHUNK_SIZE,
    chunk_overlap=SECONDARY_OVERLAP,
    separators=["\n\n", "\n", ". ", " "],
)


# ---------------------------------------------------------------------------
# Step 1: turn a page's HTML body into a list of (header, text) sections
# ---------------------------------------------------------------------------
def split_into_sections(body_html: str, page_title: str):
    """
    Walk the top-level HTML tags in order. Start a new section every time
    we hit an <h4>. The <h3> at the very start is skipped as a cut point --
    it's just the page's own title repeated, not a real subsection (the
    same reason we didn't treat a document's own title as a chapter break
    in the markdown version).
    """
    soup = BeautifulSoup(body_html, "html.parser")

    sections = []
    current_header = page_title   # used if the page has no <h4> at all
    current_elements = []

    for el in soup.children:
        tag_name = getattr(el, "name", None)
        if tag_name is None:
            continue  # skip stray whitespace text nodes between tags
        if tag_name == "h3":
            continue  # the page's own title -- not a real section boundary
        if tag_name == "h4":
            if current_elements:
                sections.append((current_header, current_elements))
            current_header = el.get_text().strip()
            current_elements = []
        else:
            current_elements.append(el)

    if current_elements:
        sections.append((current_header, current_elements))

    return sections


def section_to_text(elements) -> str:
    """
    Convert a section's HTML elements back into clean, readable plain text
    for embedding -- paragraphs as lines, numbered lists kept numbered
    (get_text() alone would just run list items together with no
    separator, which is why we rebuild the numbering by hand here).
    """
    lines = []
    for el in elements:
        if el.name == "p":
            text = el.get_text().strip()
            if text:
                lines.append(text)
        elif el.name == "ol":
            for i, li in enumerate(el.find_all("li", recursive=False), start=1):
                lines.append(f"{i}. {li.get_text().strip()}")
        else:
            text = el.get_text().strip()
            if text:
                lines.append(text)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step 2: chunk one page, attaching realistic metadata
# ---------------------------------------------------------------------------
def chunk_page(page: dict) -> list[dict]:
    sections = split_into_sections(page["body"]["storage"]["value"], page["title"])

    # Category isn't a direct field -- derive it from the page tree, the
    # same way you'd walk a real Confluence space's parent pages.
    category = page["ancestors"][0]["title"] if page["ancestors"] else None

    label_names = [l["name"] for l in page["metadata"]["labels"]["results"]]
    restriction_groups = [
        g["name"] for g in page["restrictions"]["read"]["restrictions"]["group"]["results"]
    ]

    chunks = []
    chunk_idx = 0
    for header_path, elements in sections:
        text = section_to_text(elements)
        if not text:
            continue

        n_tokens = token_count(text)
        if n_tokens <= MAX_CHUNK_TOKENS:
            chunks.append(_make_chunk(page, text, header_path, chunk_idx, n_tokens,
                                       category, label_names, restriction_groups))
            chunk_idx += 1
        else:
            for sub in secondary_splitter.split_text(text):
                chunks.append(_make_chunk(page, sub, header_path, chunk_idx, token_count(sub),
                                           category, label_names, restriction_groups, sub_split=True))
                chunk_idx += 1

    return chunks


def _make_chunk(page, text, header_path, chunk_idx, n_tokens,
                 category, label_names, restriction_groups, sub_split=False):
    return {
        "chunk_id": f"{page['id']}-C{chunk_idx:02d}",
        "page_id": page["id"],
        "title": page["title"],
        "category": category,
        "status": page["status"],
        "version_number": page["version"]["number"],
        "last_updated": page["version"]["when"],
        "labels": label_names,
        "restricted_to": restriction_groups,
        "header_path": header_path,
        "chunk_index": chunk_idx,
        "secondary_split": sub_split,
        "token_count": n_tokens,
        "text": text,
        "source_type": "confluence",  # see ingest_error_code_spreadsheet.py's docstring --
        # every connector tags its own chunks so origin stays traceable after
        # embed_and_store.py combines chunks from multiple source connectors.
    }


# ---------------------------------------------------------------------------
# Run it
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    with open(DATA_PROCESSED / "confluence_export.jsonl", encoding="utf-8") as f:
        pages = [json.loads(line) for line in f]

    # Exclude trashed pages BEFORE chunking. Real Confluence search/content
    # APIs default to 'current' status only -- content in 'current' and
    # 'trashed' status will be fetched only if you explicitly ask for both.
    # So this mirrors real retrieval behavior, it isn't us taking a shortcut.
    retrievable_pages = [p for p in pages if p["status"] != "trashed"]
    print(f"{len(pages)} total pages -> {len(retrievable_pages)} retrievable "
          f"(excluded {len(pages) - len(retrievable_pages)} trashed)")

    all_chunks = []
    for page in retrievable_pages:
        all_chunks.extend(chunk_page(page))

    with open(DATA_CHUNKS / "chunks_from_confluence.jsonl", "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"Chunks produced: {len(all_chunks)}")
    print(f"Avg chunks/page: {len(all_chunks) / len(retrievable_pages):.2f}")
    secondary = sum(1 for c in all_chunks if c["secondary_split"])
    print(f"Chunks from a forced secondary split: {secondary}")

    # Sanity check: confirm no forbidden fields snuck onto any chunk
    forbidden = {"doc_type", "confuses_with", "supersedes", "superseded_by"}
    leaked = [c for c in all_chunks if forbidden & c.keys()]
    print(f"Chunks with a forbidden field: {len(leaked)} (should be 0)")
