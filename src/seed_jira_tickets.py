"""
One-off script: seeds the Jira Service Management project (IH) with a
handful of realistic tickets, reusing real messy queries already built
and verified in data/processed/eval_edge_case_queries.jsonl -- so the
first real Jira-integration test has real, varied ticket content to
fetch (raw ticket dump, email forward, technical description, emotional
venting, and the F07 stale-duplicate case that's been a focus of
testing throughout this project) instead of synthetic placeholder text.

Uses Jira's REST API v3 issue-create endpoint directly (basic auth via
email + API token, same credentials as the rest of this integration).
Only `project` and `summary` are actually required fields for this
project's [System] Service request issue type (checked via createmeta
before writing this, not assumed) -- description is optional but used
here to carry the full messy query text.

Run: python src/seed_jira_tickets.py
"""

import base64
import json
import os
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent
REPO_ROOT = PROJECT_ROOT.parent
load_dotenv(REPO_ROOT / ".env")

SITE = os.environ["JIRA_SITE_URL"]
EMAIL = os.environ["JIRA_EMAIL"]
TOKEN = os.environ["JIRA_API_TOKEN"]
PROJECT_KEY = os.environ["JIRA_PROJECT_KEY"]

DATA_PROCESSED = REPO_ROOT / "data" / "processed"

# (query_id, short human summary) -- summary is what an end user would
# plausibly type as a ticket title; the full messy query text becomes
# the description, exactly as captured in the real eval data.
SEED_TICKETS = [
    ("A01", "VPN keeps dropping right after login"),
    ("A07", "FW: VPN issue -- branch office reports"),
    ("C02", "Machine can't reach internal network"),
    ("E05", "VPN disconnecting for 3 days straight"),
    ("F07", "Suspect a contractor account is compromised"),
]


def _auth_header() -> dict:
    token = base64.b64encode(f"{EMAIL}:{TOKEN}".encode()).decode()
    return {
        "Authorization": f"Basic {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def _adf_description(text: str) -> dict:
    """Jira's v3 API requires the description field in Atlassian
    Document Format (ADF), not plain text -- one paragraph per line."""
    return {
        "type": "doc",
        "version": 1,
        "content": [
            {"type": "paragraph", "content": [{"type": "text", "text": line}]}
            for line in text.split("\n") if line.strip()
        ],
    }


def create_ticket(summary: str, description: str) -> str:
    body = {
        "fields": {
            "project": {"key": PROJECT_KEY},
            "summary": summary,
            "description": _adf_description(description),
            "issuetype": {"id": "10001"},  # [System] Service request
        }
    }
    req = urllib.request.Request(
        f"{SITE}/rest/api/3/issue",
        data=json.dumps(body).encode("utf-8"),
        headers=_auth_header(),
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        result = json.loads(resp.read())
        return result["key"]


def main():
    with open(DATA_PROCESSED / "eval_edge_case_queries.jsonl", encoding="utf-8") as f:
        rows = {r["query_id"]: r for r in (json.loads(line) for line in f)}

    created = []
    for qid, summary in SEED_TICKETS:
        query_text = rows[qid]["query"]
        try:
            key = create_ticket(summary, query_text)
            print(f"Created {key}: {summary!r} (from eval query {qid})")
            created.append({"jira_key": key, "eval_query_id": qid, "summary": summary})
        except Exception as e:
            print(f"FAILED for {qid} ({summary!r}): {type(e).__name__}: {e}")

    out_path = DATA_PROCESSED / "seeded_jira_tickets.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for row in created:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nSaved mapping of {len(created)} seeded tickets to {out_path}")


if __name__ == "__main__":
    main()
