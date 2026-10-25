"""
Looks up a KB article's full content for the citation side-panel (added
2026-10-24): when Step 8 flags low confidence and the staffer is shown
ranked candidates instead of a synthesized answer, they previously only
saw a title and a last_updated/status line -- no way to read the actual
article without leaving the tool. This loads the real page body
(confluence_export.jsonl's ADF/storage HTML, the same source
chunk_confluence.py originally chunked from) so the UI can render it
side-by-side with the chat, with the section matching whichever chunk
was actually retrieved highlighted -- not just the whole page dumped
with no orientation.

Chunk header_path values are real, verified to match literal <h1-6>
heading text already present in the stored HTML body (confirmed
2026-10-24 against page 200004's 9-section "Full Diagnostic Guide" --
every chunk's header_path has a literal matching heading). This match is
exact-text, not fuzzy, so this module does the heading-to-anchor
rewrite deterministically rather than guessing a scroll position.

PERMISSION FILTER: reuses the exact same restricted_to/allowed_groups
check as Step 4.5 (retrieval_pipeline.py's _passes_permission_filter) --
not a weaker, independently re-derived check. A client only ever sees a
citation that already survived that same filter during retrieval, but
this endpoint's own server-side check is NOT optional: a malicious or
buggy client could request any page_id directly, bypassing whatever the
UI happened to render.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from config import DATA_DIR

DATA_PROCESSED = DATA_DIR / "processed"

_HEADING_RE = re.compile(r"(<h[1-6][^>]*>)(.*?)(</h[1-6]>)", re.IGNORECASE | re.DOTALL)


@dataclass
class ArticleContent:
    page_id: str
    title: str
    html: str  # full body HTML, with heading anchors injected
    status: str
    last_updated: str
    restricted_to: list[str]
    matched_anchor: str | None  # the #id to scroll/highlight to, if a header_path was given and matched


def _load_pages() -> dict[str, dict]:
    pages = {}
    with open(DATA_PROCESSED / "confluence_export.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            pages[row["id"]] = row
    return pages


def _load_chunk_restrictions() -> dict[str, list[str]]:
    """page_id -> restricted_to, read from the same chunk file Step 4.5
    filters against (not re-derived from the raw Confluence restrictions
    shape) -- single source of truth for "who can see this page."""
    restrictions: dict[str, list[str]] = {}
    with open(DATA_DIR / "chunks" / "chunks_from_confluence.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            restrictions[row["page_id"]] = row.get("restricted_to") or []
    return restrictions


def _slugify(heading_text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", heading_text.strip().lower())
    return slug.strip("-")


def _inject_heading_anchors(html: str) -> tuple[str, dict[str, str]]:
    """Adds an id="<slug>" to every heading tag so the frontend can
    scroll to `#<slug>` directly, with no client-side text search needed.
    Returns (rewritten_html, {heading_text: slug}) so the caller can look
    up the right slug for a given header_path."""
    heading_to_slug: dict[str, str] = {}

    def _add_id(m: re.Match) -> str:
        open_tag, text, close_tag = m.group(1), m.group(2), m.group(3)
        slug = _slugify(re.sub(r"<[^>]+>", "", text))  # strip any inline tags before slugifying
        heading_to_slug[text.strip()] = slug
        # Insert id="slug" into the opening tag, e.g. <h3> -> <h3 id="slug">
        new_open = open_tag[:-1] + f' id="{slug}">'
        return f"{new_open}{text}{close_tag}"

    rewritten = _HEADING_RE.sub(_add_id, html)
    return rewritten, heading_to_slug


def get_article(page_id: str, header_path: str | None,
                 allowed_groups: set[str] | None) -> ArticleContent | None:
    """
    Returns the full article content for the side panel, or None if the
    page doesn't exist OR the caller isn't permitted to see it (same
    restricted_to check Step 4.5 already applies during retrieval --
    returning None here, not a partial/redacted body, on a denial).

    header_path: the specific chunk's header_path (e.g. "Resolution
    summary"), used to find the matching heading's anchor so the panel
    can scroll/highlight directly to it. None or no match -> the whole
    article is still returned, just with matched_anchor=None (panel
    opens at the top).
    """
    pages = _load_pages()
    page = pages.get(page_id)
    if page is None:
        return None

    restrictions = _load_chunk_restrictions()
    restricted_to = restrictions.get(page_id, [])
    # Same rule as retrieval_pipeline.py's _passes_permission_filter:
    # allowed_groups=None means "no caller identity known" (old
    # unfiltered behavior, e.g. an offline script) -- a real UI request
    # always has allowed_groups set, resolved server-side from the login
    # session, so this only denies when there IS a real identity that
    # doesn't have the required group.
    if restricted_to and allowed_groups is not None and not (set(restricted_to) & allowed_groups):
        return None

    html = page["body"]["storage"]["value"]
    rewritten_html, heading_to_slug = _inject_heading_anchors(html)

    matched_anchor = None
    if header_path:
        matched_anchor = heading_to_slug.get(header_path.strip())

    return ArticleContent(
        page_id=page_id,
        title=page["title"],
        html=rewritten_html,
        status=page.get("status", "current"),
        last_updated=page.get("version", {}).get("when", ""),
        restricted_to=restricted_to,
        matched_anchor=matched_anchor,
    )


def list_accessible_pages(allowed_groups: set[str]) -> list[dict]:
    """
    Every current (non-trashed) page the caller's allowed_groups can
    actually open, for the chat UI's own "Knowledge Base" sidebar tab --
    unlike search_pages() (the admin eval-case picker), this MUST apply
    the same restricted_to permission check as get_article()/Step 4.5,
    since staffers browse this list directly rather than only seeing
    citations that already survived retrieval-time filtering.
    """
    pages = _load_pages()
    restrictions = _load_chunk_restrictions()
    result = []
    for page in pages.values():
        if page.get("status") != "current":
            continue
        restricted_to = restrictions.get(page["id"], [])
        if restricted_to and not (set(restricted_to) & allowed_groups):
            continue
        result.append({
            "page_id": page["id"],
            "title": page["title"],
            "category": page["ancestors"][0]["title"] if page.get("ancestors") else "Uncategorized",
            "last_updated": page.get("version", {}).get("when", ""),
        })
    result.sort(key=lambda p: p["title"])
    return result


def search_pages(query: str, limit: int = 20) -> list[dict]:
    """
    Simple title-substring search over the KB -- added 2026-10-24 for the
    evals page's "write a test case" KB-page picker, so a new case's
    expected_page_id is chosen by searching real titles rather than typed
    as a raw page_id a user would have to already know. Deliberately
    plain substring matching, not fuzzy/semantic search: this is a small
    admin-facing picker over ~200 known titles, not a retrieval quality
    surface, so exact-enough substring matching is the right amount of
    mechanism -- no embedding model call needed for this.

    Returns EVERY page regardless of status (including trashed/
    superseded) -- deliberately, since a real eval case might legitimately
    want to test against a stale/trashed page (see the golden set's own
    "silently_stale_page"/"trashed_page_topic" subtypes), so filtering
    those out here would make certain case types impossible to author.
    """
    pages = _load_pages()
    q = query.strip().lower()
    matches = [
        {
            "page_id": p["id"], "title": p["title"], "status": p.get("status", "current"),
            "category": p["ancestors"][0]["title"] if p.get("ancestors") else None,
        }
        for p in pages.values()
        if not q or q in p["title"].lower()
    ]
    matches.sort(key=lambda m: m["title"])
    return matches[:limit]
