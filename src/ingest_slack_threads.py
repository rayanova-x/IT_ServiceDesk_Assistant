"""
Third ingestion CONNECTOR -- a structurally different source again, this
time conversational rather than prose (Confluence) or tabular (the
error-code spreadsheet). Reads data/raw/slack_threads.json (simulated,
shaped like a real IT-helpdesk triage channel export -- same honesty
convention as the Confluence corpus: authored to be realistic, not a
live Slack API pull, since there's no real workspace to connect to),
and produces chunks_from_slack_threads.jsonl in the same canonical chunk
schema the other two connectors produce.

WHY THIS SOURCE NEEDS A DIFFERENT INGESTION APPROACH THAN THE OTHER TWO
(researched, not assumed -- see PROJECT_HANDOFF.md's "Communication
archives" writeup): a raw Slack thread is mostly NOISE relative to its
signal. A single message like "yep that was it, updated it, loads fine
now" means nothing on its own -- no product, no symptom, no fix stated
in that sentence. Naively chunking per-message (like a prose page's
per-section chunking) would produce embeddable text with no retrievable
meaning. The real-world guidance found (Snyk's own RAG-over-Slack
writeup, Paragon's Slack-ingestion guide, both fetched directly
2026-09-22) converges on: treat the THREAD as the unit, not the message.

CONTEXTUALIZATION, PER ANTHROPIC'S OWN "CONTEXTUAL RETRIEVAL" TECHNIQUE
(fetched directly from anthropic.com/engineering/contextual-retrieval
and the Claude Cookbook's contextual-embeddings guide, 2026-09-22 -- not
paraphrased from memory): rather than asking an LLM to invent a clean
synthetic Q&A pair (which risks smoothing over or fabricating content
that wasn't actually said), Anthropic's approach SITUATES the real
original text -- generate a short (50-100 token) explanatory blurb
describing what a chunk covers within its source document, and prepend
that to the chunk BEFORE embedding. The original noisy exchange is
preserved verbatim; a description is added alongside it, not instead of
it. Their own measured numbers: this alone cut retrieval failures 35%,
and combined with contextual BM25 + reranking, 67% -- across multiple
domains (not conversational data specifically, that's not something
their public writeup tests, so applying it here is a reasoned
extension of the technique, not a repeated-verbatim claim).

WHAT'S DIFFERENT HERE FROM ANTHROPIC'S EXACT REFERENCE IMPLEMENTATION,
STATED HONESTLY: their cookbook example uses Claude with explicit
`cache_control: {"type": "ephemeral"}` prompt caching to keep the
per-chunk cost down when many chunks share one large parent document.
This project's LLM provider is Gemini (see PROJECT_HANDOFF.md's hard
constraints), which has a different caching mechanism, and at this
dataset's scale (4 threads, well under Gemini's per-request context
limit) the caching optimization isn't worth the added complexity -- the
CORE technique (situate a chunk within its full document via one LLM
call before embedding) is what's implemented; the specific cost
optimization Anthropic paired it with is not, and that tradeoff is
explicit rather than silently dropped.

SCOPE DISCIPLINE (same convention as ingest_error_code_spreadsheet.py's
cross_check_subcategories()): every thread's topic_subcategory is
checked against enriched_tickets.csv's real Subcategory values before
ingesting, and every thread's related_page_id is checked against the
real Confluence corpus to confirm it exists -- so this source only ever
adds content for real, in-scope subcategories, cross-referenced to a
real page, same discipline that caught the spreadsheet's scope drift.

WHY THIS DOESN'T DUPLICATE THE EXISTING KB: each thread's topic was
picked from the pool of real subcategories that have NO existing
error-code spreadsheet row AND were checked against their Confluence
page's actual diagnostic content -- the thread's informal "what fixed
it" exchange stays consistent with (not contradicting, not
re-explaining in full) the official SOP's documented resolution steps.
The genuinely new thing this source adds is the INFORMAL,
PEER-TO-PEER framing -- "someone else on the team already solved this
exact case, here's the real back-and-forth" -- which is a different
kind of evidence than "here's the official procedure," even when they
point to the same underlying fix.

GAP-FILLING THREADS (added 2026-09-22, real user question: "so when do
slack convos help here then"): the 4 original threads above all
corroborate an EXISTING Confluence SOP -- useful (peer-confirmed, real-
world evidence alongside the official doc, see step9_generation.py's
source-authority handling), but not the strongest case for why this
source matters, since the official doc already answers the question on
its own. Checked directly (not assumed): every real subcategory in
enriched_tickets.csv already has a matching Confluence page, so there's
no existing topic with genuinely ZERO official coverage to point at.
The realistic case where a Slack thread is the ONLY answer is narrower
and more specific than any general subcategory -- e.g. a known bug in
one specific app version, with a manual workaround the team found,
too narrow/temporary to ever get written up as its own formal SOP page
(this happens constantly in real IT teams: tribal knowledge for a
specific, dated bug that a general troubleshooting guide couldn't have
anticipated). A thread like this has `related_page_id: null` -- no
Confluence page to corroborate, by design, because none exists for this
specific case. cross_check_threads() only validates related_page_id
when a thread actually provides one; a null value is a deliberate,
valid case, not a data error. Category for a null-page-id thread is
derived by matching its topic_subcategory against the real Confluence
taxonomy anyway (see main()), since the BROADER subcategory still
exists even though this specific narrow case doesn't have its own page.

Run: python src/ingest_slack_threads.py
"""

import json
import os
import re
from datetime import date
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai

SRC_DIR = Path(__file__).parent
PROJECT_ROOT = SRC_DIR.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_CHUNKS = PROJECT_ROOT / "data" / "chunks"

SOURCE_JSON = DATA_RAW / "slack_threads.json"
OUTPUT_CHUNKS = DATA_CHUNKS / "chunks_from_slack_threads.jsonl"

load_dotenv(PROJECT_ROOT / ".env")

# Page IDs in the 910xxx range -- distinct from Confluence's 200xxx and
# the spreadsheet's 900xxx, so a third independent connector never
# collides with either existing one. Same reasoning as the spreadsheet
# connector's own PAGE_ID_PREFIX note.
PAGE_ID_PREFIX = "910"

SITUATE_PROMPT = """You are helping build a search index of resolved IT support \
conversations. Below is a full Slack thread from an internal IT helpdesk triage \
channel, and then one specific exchange from within it (the part where the \
reporter's problem actually gets diagnosed and resolved).

Write a short (1-2 sentence, under 60 words) description that situates this \
exchange for search: state the specific problem reported and the specific fix \
that resolved it, in plain language. Do NOT invent details not present in the \
thread. Do NOT include unrelated side-conversation from the thread. Answer with \
ONLY the description, nothing else.

<full_thread>
{full_thread}
</full_thread>

<exchange_to_situate>
{exchange}
</exchange_to_situate>"""


def _format_thread(messages: list[dict]) -> str:
    return "\n".join(f"[{m['user']}, {m['time']}] {m['text']}" for m in messages)


def _extract_resolving_exchange(messages: list[dict]) -> str:
    """
    Real Slack triage threads keep going after a problem is resolved
    (someone says thanks, someone else asks an unrelated question --
    see the module docstring). The RETRIEVABLE chunk should be the
    problem-report-through-resolution exchange, not the whole thread
    including that trailing noise. Implemented here as a simple
    heuristic (first message through the message containing a
    resolution-confirming word), not an LLM call -- keeping this step
    deterministic and free, since the LLM call this connector DOES make
    (situate_chunk below) is reserved for the part that actually needs
    judgment (describing what happened), not for something a keyword
    check can do reliably enough at this dataset's size.
    """
    RESOLUTION_MARKERS = ("thanks", "🙏", "fine now", "got it", "mounted right away", "corrected", "loads fine")
    cutoff = len(messages)
    for i, m in enumerate(messages):
        if any(marker in m["text"].lower() for marker in RESOLUTION_MARKERS):
            cutoff = i + 1
            break
    return _format_thread(messages[:cutoff])


def situate_chunk(client: genai.Client, model: str, full_thread: str, exchange: str) -> str:
    """
    Anthropic's Contextual Retrieval technique, applied via Gemini (see
    module docstring for the honest provider/caching-mechanism
    difference from Anthropic's own reference implementation): one LLM
    call per chunk, given the FULL parent thread plus the specific
    exchange to situate, returns a short description of what problem
    was reported and how it was resolved.
    """
    prompt = SITUATE_PROMPT.format(full_thread=full_thread, exchange=exchange)
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text.strip()


def cross_check_threads(threads: list[dict], real_subcategories: set[str], real_page_ids: set[str]) -> None:
    """
    Same discipline as ingest_error_code_spreadsheet.py's
    cross_check_subcategories() -- fail loudly on scope drift rather
    than silently ingesting an out-of-project-scope or unlinked thread.

    related_page_id is OPTIONAL (null allowed) as of 2026-09-22 -- see
    "GAP-FILLING THREADS" in the module docstring: a thread whose
    problem has no existing Confluence SOP at all is a real, deliberate
    case (that's the whole point of adding one), not a data error, so
    only threads that DO name a related_page_id get checked that it's
    real. topic_subcategory is still always required and checked --
    even a gap-filling thread must be grounded in a real subcategory
    from the project's actual ticket taxonomy, same scope discipline as
    every other source.
    """
    bad_subcats = [t for t in threads if t["topic_subcategory"] not in real_subcategories]
    bad_pages = [
        t for t in threads
        if t.get("related_page_id") is not None and t["related_page_id"] not in real_page_ids
    ]
    if bad_subcats or bad_pages:
        msgs = []
        if bad_subcats:
            msgs.append(f"{len(bad_subcats)} thread(s) reference a subcategory not in "
                        f"enriched_tickets.csv: {[t['thread_id'] for t in bad_subcats]}")
        if bad_pages:
            msgs.append(f"{len(bad_pages)} thread(s) reference a related_page_id not in "
                        f"confluence_export.jsonl: {[t['thread_id'] for t in bad_pages]}")
        raise ValueError("; ".join(msgs))


def token_count(text: str) -> int:
    return round(len(text.split()) * 1.3)


def main():
    with open(SOURCE_JSON, encoding="utf-8") as f:
        threads = json.load(f)
    print(f"Loaded {len(threads)} threads from {SOURCE_JSON.name}")

    tickets = pd.read_csv(DATA_RAW / "enriched_tickets.csv")
    real_subcategories = set(tickets["Subcategory"].unique())
    with open(DATA_PROCESSED / "confluence_export.jsonl", encoding="utf-8") as f:
        real_page_ids = {json.loads(line)["id"] for line in f}
    cross_check_threads(threads, real_subcategories, real_page_ids)
    print(f"Cross-checked all {len(threads)} threads against real subcategories + page IDs -- OK")

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    model = os.environ["GEMINI_MODEL"]

    today = date.today().isoformat()
    chunks = []
    for i, thread in enumerate(threads):
        full_thread = _format_thread(thread["messages"])
        exchange = _extract_resolving_exchange(thread["messages"])
        situated_context = situate_chunk(client, model, full_thread, exchange)
        text = f"{situated_context}\n\nFull exchange:\n{exchange}"

        page_id = f"{PAGE_ID_PREFIX}{i:03d}"
        chunks.append({
            "chunk_id": f"{page_id}-C00",
            "page_id": page_id,
            "title": f"Slack thread: {thread['topic_subcategory']} ({thread['channel']})",
            "category": None,  # derived below from the linked Confluence page's own category
            "status": "current",
            "version_number": 1,
            "last_updated": today,
            "labels": ["slack-thread", "contextual-retrieval"],
            "restricted_to": [],
            "header_path": thread["channel"],
            "chunk_index": 0,
            "secondary_split": False,
            "token_count": token_count(text),
            "text": text,
            "source_type": "slack_thread",
            "related_page_id": thread["related_page_id"],
        })
        print(f"  {thread['thread_id']}: {situated_context}")

    # category derivation, same convention as chunk_confluence.py
    # deriving category from the page tree:
    #   - if the thread names a related_page_id (corroborates an
    #     existing SOP), derive category from THAT specific page.
    #   - if not (a genuine gap-filling thread, no SOP exists yet --
    #     see "GAP-FILLING THREADS" in the module docstring), fall back
    #     to any OTHER real Confluence page whose title matches the
    #     thread's topic_subcategory, since that subcategory's real
    #     category placement in the taxonomy doesn't depend on whether
    #     THIS specific thread happens to link to a specific page.
    with open(DATA_PROCESSED / "confluence_export.jsonl", encoding="utf-8") as f:
        pages = [json.loads(line) for line in f]
    page_lookup_by_id = {p["id"]: p for p in pages}
    page_lookup_by_title = {p["title"]: p for p in pages}
    for c, thread in zip(chunks, threads):
        related = page_lookup_by_id.get(c["related_page_id"])
        if related is None:
            related = page_lookup_by_title.get(thread["topic_subcategory"])
        c["category"] = related["ancestors"][0]["title"] if related and related["ancestors"] else "none"

    with open(OUTPUT_CHUNKS, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(chunks)} chunks to {OUTPUT_CHUNKS}")


if __name__ == "__main__":
    main()
