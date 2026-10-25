import json, random, re
from datetime import datetime, timedelta
from pathlib import Path
from superseded_content import SUPERSEDED

random.seed(77)

def derive_created_date(when_date: str, version_number: int) -> str:
    """
    history.createdDate must diverge from version.when for actively-edited
    pages (version_number > 1) -- a page revised multiple times was created
    BEFORE its most recent edit, not at the same moment. Pages with
    version_number == 1 (created once, never touched -- the superseded/
    unmarked-stale tier) correctly keep createdDate == version.when, since
    that IS the realistic case for them.

    Offset scales with version_number: each revision beyond the first
    implies 30-120 days existed between edits, so a heavily-revised page
    looks like it's been around and maintained for longer.
    """
    days_per_revision = random.randint(30, 120)
    offset_days = (version_number - 1) * days_per_revision
    when_dt = datetime.strptime(when_date, "%Y-%m-%d")
    created_dt = when_dt - timedelta(days=offset_days)
    return created_dt.strftime("%Y-%m-%d")

SRC_DIR = Path(__file__).parent
DATA_RAW = SRC_DIR.parent / "data" / "raw"
DATA_PROCESSED = SRC_DIR.parent / "data" / "processed"

def strip_cross_references(text):
    """
    Remove any content that names another document via a quoted
    'see "Title"' reference -- case-insensitive, regardless of grammatical
    wrapping (fixes a real bug: an earlier version of this cleanup only
    matched specific phrasings like "-- see X instead" and missed a
    capitalized "See X" starting its own sentence).

    Works at two granularities:
      - Bulleted lines (e.g. inside a "Common misdiagnoses" list): the
        WHOLE bullet is dropped, since each such bullet exists only to
        make the cross-reference.
      - Plain prose paragraphs: only the OFFENDING SENTENCE is dropped, so
        legitimate neighboring sentences in the same paragraph survive.

    FIXED 2026-09-14 (found while investigating the VPN confuser-pair
    retrieval weakness, see PROJECT_HANDOFF.md): a "### Heading" line with
    NO blank line before its body text (a common style in this corpus --
    e.g. "### Overview" directly followed by the overview paragraph) was
    being treated as part of the SAME paragraph as that body text, since
    splitting below only looks for blank lines. The heading line and the
    body lines then got joined with spaces into one line still starting
    with "### ", which to_storage_format() wraps ENTIRELY in a single
    <h4> tag -- silently turning the body sentences into part of a
    section TITLE (chunked as a header, never embedded as searchable
    text) rather than real content. Scanned all 205 docs
    (src/scan_heading_bug.py): 15 documents affected, and every single
    one lost its "Overview" section this way -- exactly the sentence
    meant to state each document's distinguishing symptom, gone from the
    retrievable corpus for the very documents most likely to need it to
    avoid being confused with a lookalike. Fixed by treating a
    "### "/"## " heading line as its own paragraph boundary, so it can
    never be merged with the text below it, regardless of blank lines.
    """
    # A markdown heading line always starts its own paragraph and never
    # merges with what follows, even with no blank line separating them --
    # insert an artificial blank-line marker before every heading line so
    # the split below (which looks for blank lines) also splits there.
    text = re.sub(r'(?m)^(#{2,3}\s.*)$', r'\n\n\1\n\n', text)

    paragraphs = re.split(r'(\n\s*\n)', text)
    result = []
    for para in paragraphs:
        if para.strip() == "" or not para.strip('\n'):
            result.append(para)
            continue
        # A heading paragraph (possibly with a trailing colon/space quirk
        # from the substitution above) is never a cross-reference and
        # never needs sentence-splitting -- pass it through untouched.
        if re.match(r'^#{2,3}\s', para.strip()):
            result.append(para.strip())
            continue
        lines = para.split('\n')
        if any(re.match(r'^\s*-\s', l) for l in lines):
            bullets, current = [], []
            for line in lines:
                if re.match(r'^\s*-\s', line) and current:
                    bullets.append(' '.join(x.strip() for x in current))
                    current = [line]
                else:
                    current.append(line)
            if current:
                bullets.append(' '.join(x.strip() for x in current))
            surviving = [b for b in bullets if not re.search(r'\bsee\s+"', b, re.IGNORECASE)]
            result.append('\n'.join(surviving))
        else:
            joined = ' '.join(l.strip() for l in lines)
            sentences = re.split(r'(?<=[.:])\s+(?=[A-Z0-9])', joined)
            surviving = [s for s in sentences if not re.search(r'\bsee\s+"', s, re.IGNORECASE)]
            result.append(' '.join(surviving))
    return ''.join(result)

with open(DATA_RAW / 'rag_corpus_full.jsonl', encoding='utf-8') as f:
    old_docs = [json.loads(l) for l in f]

# ---------------------------------------------------------------------------
# Space + page-tree hierarchy
# ---------------------------------------------------------------------------
SPACE = {"id": 101, "key": "ITOPS", "name": "IT Operations Space", "type": "global",
         "_links": {"self": "https://company.atlassian.net/wiki/rest/api/space/ITOPS"}}

# Parent page IDs (fixed, so ancestors can reference them)
PARENTS = {
    "Access": {"id": "900001", "title": "Access & Identity Management"},
    "Security": {"id": "900002", "title": "Security Operations"},
    "Software": {"id": "900003", "title": "Software & Applications Support"},
    "Hardware": {"id": "900004", "title": "Hardware & Device Support"},
    "Network": {"id": "900005", "title": "Network Infrastructure"},
    "Policy": {"id": "900006", "title": "IT Policies & Procedures"},
    "Onboarding": {"id": "900007", "title": "Onboarding & Provisioning"},
    "Facilities": {"id": "900008", "title": "Facilities & Regional IT"},
}
# "Archived" sub-pages, one per IT category, for the minority of superseded
# docs that a diligent team member actually filed away properly.
ARCHIVE_CHILD = {
    cat: {"id": f"90{i}010", "title": f"Archived Procedures — {PARENTS[cat]['title']}"}
    for i, cat in enumerate(["Access", "Security", "Software", "Hardware", "Network"], 1)
}

AUTHORS = [
    {"accountId": "5b10ac8d82e05b22cc7d4ef5", "displayName": "Marcus Vance", "publicName": "Marcus V."},
    {"accountId": "6c21bd9e93f16c33dd8e5fa6", "displayName": "Priya Nadarajan", "publicName": "Priya N."},
    {"accountId": "7d32ce0fa4027d44ee9f6ab7", "displayName": "Tom Alsworth", "publicName": "Tom A."},
    {"accountId": "8e43df1b5138e55ff0a7bc08", "displayName": "Renee Okafor", "publicName": "Renee O."},
]

def author():
    return random.choice(AUTHORS)


# ---------------------------------------------------------------------------
# Lightweight markdown -> Confluence storage-format (XHTML-ish) converter
# ---------------------------------------------------------------------------
def to_storage_format(md_body: str) -> str:
    lines = md_body.strip().split("\n")
    html = []
    in_list = False
    for line in lines:
        line = line.strip()
        if not line:
            if in_list:
                html.append("</ol>")
                in_list = False
            continue
        line = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", line)
        m = re.match(r"^\d+\.\s+(.*)", line)
        if m:
            if not in_list:
                html.append("<ol>")
                in_list = True
            html.append(f"<li>{m.group(1)}</li>")
            continue
        if in_list:
            html.append("</ol>")
            in_list = False
        if line.startswith("### "):
            html.append(f"<h4>{line[4:]}</h4>")
        elif line.startswith("## "):
            html.append(f"<h3>{line[3:]}</h3>")
        else:
            html.append(f"<p>{line}</p>")
    if in_list:
        html.append("</ol>")
    return "".join(html)


# ---------------------------------------------------------------------------
# Rebuild body text for superseded docs -- NO self-aware "this is outdated"
# language. It reads exactly like a normal, confident, current procedure.
# ---------------------------------------------------------------------------
superseded_by_subcat = {s["subcategory"]: s for s in SUPERSEDED}

def clean_superseded_body(subcat):
    s = superseded_by_subcat[subcat]
    lines = [f"## {subcat}", "", f"**Symptoms:** {s['old_symptoms']}", "", "**Resolution steps:**"]
    for i, r in enumerate(s["old_resolution"], 1):
        lines.append(f"{i}. {r}")
    lines.append("")
    lines.append("**Escalation criteria:** If the above steps do not resolve the issue, "
                  "escalate to the relevant support team.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Assign the three staleness tiers for superseded docs.
# ---------------------------------------------------------------------------
superseded_docs = [d for d in old_docs if d["doc_type"] == "superseded"]
random.shuffle(superseded_docs)
n = len(superseded_docs)
TRASHED = set(d["doc_id"] for d in superseded_docs[:5])           # actually cleaned up
LABELED = set(d["doc_id"] for d in superseded_docs[5:15])         # labeled + archived, still readable
UNMARKED = set(d["doc_id"] for d in superseded_docs[15:])         # no signal but an old date/version

print(f"Superseded tiers -- trashed:{len(TRASHED)} labeled+archived:{len(LABELED)} unmarked:{len(UNMARKED)}")

# ---------------------------------------------------------------------------
# Category label/keyword tags per IT category, for realistic label sets
# ---------------------------------------------------------------------------
TOPIC_TAGS = {
    "Access": ["identity", "entra-id", "access-management"],
    "Security": ["security", "incident-response", "soc"],
    "Software": ["applications", "saas", "app-support"],
    "Hardware": ["hardware", "endpoint", "desktop-support"],
    "Network": ["network", "connectivity", "infrastructure"],
}

confluence_pages = []
ground_truth_relationships = []  # kept OUT of the corpus, eval-only
next_page_id = 200001

for d in old_docs:
    page_id = str(next_page_id)
    next_page_id += 1

    doc_type = d["doc_type"]
    category = d["category"]
    subcat = d.get("subcategory")

    # ---- Title & body: strip every self-aware "old/confusable/superseded" cue ----
    if doc_type == "superseded":
        title = subcat  # no "(Legacy)" suffix -- reads as a completely normal page
        body_md = clean_superseded_body(subcat)
    elif doc_type == "current_sop":
        title = subcat
        body_md = strip_cross_references(d["body"])
    elif doc_type == "confusable":
        title = subcat
        body_md = strip_cross_references(d["body"])
    else:  # noise
        title = d["title"]
        body_md = d["body"]

    body_html = to_storage_format(body_md)

    # ---- status / labels / properties / ancestors, by tier ----
    labels = []
    properties = {"source_system": "Confluence"}
    status = "current"

    if doc_type in ("current_sop", "confusable"):
        parent = PARENTS.get(category, PARENTS["Policy"])
        ancestors = [{"id": parent["id"], "type": "page", "status": "current", "title": parent["title"]}]
        labels = [{"prefix": "global", "name": "active-sop", "id": "33001"}]
        labels += [{"prefix": "global", "name": t, "id": f"33{i:03d}"} for i, t in enumerate(TOPIC_TAGS.get(category, []), 10)]
        properties["doc_status"] = "active"
        version_number = random.randint(2, 7)  # actively maintained pages get revised
        when_date = d["last_updated"]

    elif doc_type == "superseded":
        parent = PARENTS.get(category, PARENTS["Policy"])
        if d["doc_id"] in TRASHED:
            status = "trashed"
            ancestors = [{"id": parent["id"], "type": "page", "status": "current", "title": parent["title"]}]
            labels = [{"prefix": "global", "name": "archived", "id": "33099"}]
            properties["doc_status"] = "deprecated"
        elif d["doc_id"] in LABELED:
            archive_parent = ARCHIVE_CHILD[category]
            ancestors = [
                {"id": parent["id"], "type": "page", "status": "current", "title": parent["title"]},
                {"id": archive_parent["id"], "type": "page", "status": "current", "title": archive_parent["title"]},
            ]
            labels = [{"prefix": "global", "name": "deprecated", "id": "33098"}]
            properties["doc_status"] = "deprecated"
        else:  # UNMARKED -- the hard, realistic case
            ancestors = [{"id": parent["id"], "type": "page", "status": "current", "title": parent["title"]}]
            labels = [{"prefix": "global", "name": "active-sop", "id": "33001"}]  # never updated -- still says active!
            properties["doc_status"] = "active"  # incorrect, and that's the point
        version_number = random.choice([1, 1, 2])  # low -- created once, rarely touched
        when_date = d["last_updated"]  # this old date is the ONLY reliable signal for the unmarked tier

    else:  # noise
        noise_parent_key = random.choice(["Policy", "Onboarding", "Facilities"])
        parent = PARENTS[noise_parent_key]
        ancestors = [{"id": parent["id"], "type": "page", "status": "current", "title": parent["title"]}]
        labels = [{"prefix": "global", "name": "reference", "id": "33050"}]
        properties["doc_status"] = "active"
        version_number = random.randint(1, 3)
        when_date = d["last_updated"]

    # ---- read restrictions: Security procedures locked down, rest open to IT dept ----
    if category == "Security" and doc_type in ("current_sop", "confusable", "superseded"):
        restriction_group = "security-team-group"
    elif doc_type in ("current_sop", "confusable", "superseded"):
        restriction_group = "it-department-group"
    else:
        restriction_group = None  # general policy pages -- visible space-wide

    restrictions = {"read": {"restrictions": {"group": {"results": (
        [{"type": "group", "name": restriction_group}] if restriction_group else []
    )}}}}

    created_date = derive_created_date(when_date, version_number)

    page = {
        "id": page_id,
        "type": "page",
        "status": status,
        "title": title,
        "space": SPACE,
        "history": {
            "latest": True,
            "createdBy": {"type": "known", **author()},
            "createdDate": created_date,
        },
        "version": {
            "by": {"type": "known", **author()},
            "when": when_date,
            "number": version_number,
            "minorEdit": False,
        },
        "ancestors": ancestors,
        "metadata": {
            "labels": {"results": labels, "start": 0, "limit": 200, "size": len(labels)},
            "properties": properties,
        },
        "restrictions": restrictions,
        "body": {"storage": {"value": body_html, "representation": "storage"}},
        "_links": {
            "webui": f"/spaces/ITOPS/pages/{page_id}",
            "self": f"https://company.atlassian.net/wiki/rest/api/content/{page_id}",
        },
    }
    confluence_pages.append(page)

    # ---- ground truth (EVAL ONLY -- never shipped inside the retrievable corpus) ----
    if d.get("confuses_with") or d.get("supersedes") or d.get("superseded_by"):
        ground_truth_relationships.append({
            "page_id": page_id,
            "subcategory": subcat,
            "doc_type_true": doc_type,
            "confuses_with_subcat": d["confuses_with"]["subcategory"] if d.get("confuses_with") else None,
            "supersedes_page_true": None,   # filled in second pass below
            "superseded_by_page_true": None,
            "staleness_tier": (
                "trashed" if d["doc_id"] in (TRASHED if doc_type == "superseded" else set()) else
                "labeled_archived" if d["doc_id"] in (LABELED if doc_type == "superseded" else set()) else
                "unmarked_stale" if d["doc_id"] in (UNMARKED if doc_type == "superseded" else set()) else
                None
            ),
        })

# second pass: resolve supersedes/superseded_by page-id links using subcategory + doc_type
subcat_current_page = {p["title"]: p["id"] for p, d in zip(confluence_pages, old_docs) if d["doc_type"] == "current_sop"}
for gt, d in zip(ground_truth_relationships, [d for d in old_docs if d.get("confuses_with") or d.get("supersedes") or d.get("superseded_by")]):
    if d.get("superseded_by"):
        cur_title = next(dd["subcategory"] for dd in old_docs if dd["doc_id"] == d["superseded_by"])
        gt["superseded_by_page_true"] = subcat_current_page.get(cur_title)

print(f"\nBuilt {len(confluence_pages)} Confluence-style pages")
print(f"Ground truth relationships recorded (eval-only): {len(ground_truth_relationships)}")

with open(DATA_PROCESSED / 'confluence_export.jsonl', 'w', encoding='utf-8') as f:
    for p in confluence_pages:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

with open(DATA_PROCESSED / 'eval_only_ground_truth.jsonl', 'w', encoding='utf-8') as f:
    for g in ground_truth_relationships:
        f.write(json.dumps(g, ensure_ascii=False) + "\n")

print("Saved confluence_export.jsonl (the retrievable corpus) and eval_only_ground_truth.jsonl (eval-only, never fed to retrieval)")
