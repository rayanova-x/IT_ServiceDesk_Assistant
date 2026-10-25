"""
Real Jira Service Management client -- fetches an issue's description,
comments, and status via the Jira REST API v3, so a ticket's real
content can seed the first message of its chat session instead of
requiring an agent to re-type the problem.

Deliberately minimal: this is a read-only fetch client, no write
actions (posting comments, changing status) are implemented here --
per the design decision to keep any write-back to Jira a manual
copy-paste action by the human agent, not an automated API call. See
PROJECT_HANDOFF.md for the full reasoning (Atlassian's own real Rovo
"Request Router" agent, checked directly against its official docs,
treats priority/type/reply suggestions as human-approved, not
autonomous -- the one thing it does write automatically is a
context-summary comment, not a resolution presented as fact).

Auth: HTTP Basic with email + API token (Jira Cloud's standard REST
API v3 auth pattern, confirmed working against the real project
during setup -- see .env's JIRA_* variables).

Text extraction: Jira's v3 API returns description/comments in
Atlassian Document Format (ADF), a nested JSON structure, not plain
text. _adf_to_text() walks it and extracts plain text -- handles the
common node types (paragraph, text, bulletList/orderedList) actually
seen in real ticket content; falls back to skipping unrecognized node
types rather than crashing, since ADF has many node types this
helpdesk use case will likely never encounter (tables, media, etc.).
"""

import base64
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent
REPO_ROOT = PROJECT_ROOT.parent
load_dotenv(REPO_ROOT / ".env")


@dataclass
class JiraComment:
    author: str
    body_text: str
    created: str


@dataclass
class JiraIssueSummary:
    """Lightweight row for a ticket LIST (sidebar) -- no description/
    comments fetch, unlike JiraIssue, since a list view only needs
    enough to let the IT staffer pick a ticket to open."""
    key: str
    summary: str
    status: str


@dataclass
class JiraIssue:
    key: str
    summary: str
    description_text: str
    status: str
    comments: list[JiraComment] = field(default_factory=list)

    def to_seed_message(self) -> str:
        """Formats this ticket's real content into one message, ready
        to feed as the first turn of a chat session -- exactly what an
        IT staffer would otherwise have to re-type by hand. Comments
        are included so a ticket already worked partially (e.g. by a
        prior shift) carries that history into the conversation."""
        parts = [f"Ticket {self.key}: {self.summary}", "", self.description_text]
        if self.comments:
            parts.append("")
            parts.append("Comments so far:")
            for c in self.comments:
                parts.append(f"- [{c.author}, {c.created}] {c.body_text}")
        return "\n".join(p for p in parts if p is not None)


def _auth_header() -> dict:
    email = os.environ["JIRA_EMAIL"]
    token = os.environ["JIRA_API_TOKEN"]
    encoded = base64.b64encode(f"{email}:{token}".encode()).decode()
    return {"Authorization": f"Basic {encoded}", "Accept": "application/json"}


def _adf_to_text(node) -> str:
    """Extract plain text from an Atlassian Document Format node tree.
    Handles paragraph/text/bulletList/orderedList/listItem -- the node
    types actually expected in real ticket descriptions/comments.
    Unrecognized node types are skipped (their children, if any, are
    still walked) rather than raising, since ADF has many node types
    this use case doesn't need to render specially."""
    if node is None:
        return ""
    node_type = node.get("type")
    if node_type == "text":
        return node.get("text", "")

    children = node.get("content", [])
    if node_type in ("bulletList", "orderedList"):
        lines = []
        for i, item in enumerate(children, start=1):
            prefix = "- " if node_type == "bulletList" else f"{i}. "
            item_text = "".join(_adf_to_text(c) for c in item.get("content", []))
            lines.append(prefix + item_text)
        return "\n".join(lines)

    text = "".join(_adf_to_text(c) for c in children)
    if node_type == "paragraph":
        return text + "\n"
    return text


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers=_auth_header())
    with urllib.request.urlopen(req, timeout=15) as resp:
        import json
        return json.loads(resp.read())


class JiraFetchError(Exception):
    """Raised when a Jira issue can't be fetched -- distinct from a
    generic exception so callers (e.g. chat_service.py) can catch this
    specifically and show a clear message rather than a stack trace."""


def get_issue(issue_key: str) -> JiraIssue:
    """
    Fetch one issue's summary, description, status, and comments from
    the real Jira instance configured in .env. Raises JiraFetchError
    (not a bare exception) on any failure -- 404 (issue doesn't exist),
    401/403 (auth/permission), or network issues -- with the real
    underlying error message preserved, not swallowed.
    """
    site = os.environ["JIRA_SITE_URL"]
    url = f"{site}/rest/api/3/issue/{issue_key}"

    try:
        data = _get(url)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise JiraFetchError(f"Ticket {issue_key} not found in Jira.") from e
        raise JiraFetchError(f"Jira returned {e.code} fetching {issue_key}: {e.reason}") from e
    except urllib.error.URLError as e:
        raise JiraFetchError(f"Could not reach Jira ({site}) to fetch {issue_key}: {e}") from e

    fields = data["fields"]
    description_text = _adf_to_text(fields.get("description")).strip()

    comments = []
    for c in fields.get("comment", {}).get("comments", []):
        body_text = _adf_to_text(c.get("body")).strip()
        author = c.get("author", {}).get("displayName", "Unknown")
        created = c.get("created", "")[:10]  # date only, not full timestamp
        comments.append(JiraComment(author=author, body_text=body_text, created=created))

    return JiraIssue(
        key=data["key"],
        summary=fields.get("summary", ""),
        description_text=description_text,
        status=fields.get("status", {}).get("name", "Unknown"),
        comments=comments,
    )


def list_open_issues(limit: int = 50) -> list[JiraIssueSummary]:
    """
    List tickets in the configured project that aren't Done -- what an
    IT staffer's sidebar should show (the tickets still needing work),
    not the full historical list. Uses the JQL search endpoint (POST
    /rest/api/3/search/jql, the current v3 search API -- the older GET
    /rest/api/3/search is deprecated) with a project + status filter,
    ordered newest-first so a freshly created/updated ticket surfaces
    at the top same as a real Jira queue view.

    Raises JiraFetchError on failure, same convention as get_issue().
    """
    site = os.environ["JIRA_SITE_URL"]
    project_key = os.environ["JIRA_PROJECT_KEY"]
    jql = f'project = "{project_key}" AND statusCategory != Done ORDER BY updated DESC'
    url = f"{site}/rest/api/3/search/jql?jql={urllib.parse.quote(jql)}&maxResults={limit}&fields=summary,status"

    try:
        data = _get(url)
    except urllib.error.HTTPError as e:
        raise JiraFetchError(f"Jira returned {e.code} listing open issues: {e.reason}") from e
    except urllib.error.URLError as e:
        raise JiraFetchError(f"Could not reach Jira ({site}) to list open issues: {e}") from e

    return [
        JiraIssueSummary(
            key=issue["key"],
            summary=issue["fields"].get("summary", ""),
            status=issue["fields"].get("status", {}).get("name", "Unknown"),
        )
        for issue in data.get("issues", [])
    ]


if __name__ == "__main__":
    import sys

    key = sys.argv[1] if len(sys.argv) > 1 else "IH-5"
    print(f"Fetching {key}...")
    issue = get_issue(key)
    print(f"Status: {issue.status}")
    print(f"Comments: {len(issue.comments)}")
    print()
    print("--- seed message ---")
    print(issue.to_seed_message())
