"""
Real correctness test for step_route_or_search.py's new ask_clarifying_question
branch (added 2026-11-09 -- see that module's "CLARIFY ADDED" docstring
section for the real production evidence that motivated it:
data/processed/knowledge_gaps.jsonl showed genuinely topic-less messages
like "it's not working"/"the issue?" on a fresh session silently becoming
content-free search queries like "What is the issue?", guaranteed to find
nothing).

Uses the REAL Gemini API (same convention as test_permission_filter.py's
sibling live tests) -- this is specifically testing the model's own
judgment call between clarify/search/direct_reply, which can't be mocked
meaningfully. Costs real API calls; not free/instant.

Checks, built directly from the real logged cases that motivated this:
  1. A genuinely topic-less message with NO prior context (fresh session)
     -> clarify_question, NOT a content-free search_calls entry.
  2. The SAME vague phrasing ("whats the issue") WHEN a real topic already
     exists in conversation -> search_calls carrying that topic forward,
     NOT clarify -- the critical negative case: this branch must not
     regress the exact behavior the original router/contextualizer merge
     was built to fix (see step_route_or_search.py's main docstring).
  3. A real, specific IT question (not vague at all) -> search_calls,
     never clarify -- clarify must stay narrow, not become a new way to
     avoid searching.
  4. A greeting/off-topic message -> direct_reply, never clarify --
     clarify and direct_reply must stay mutually exclusive outcomes.
  5. clarify_question is always a real, non-empty question string when
     RouteResult.clarify_question is set.

Run: python tests/test_clarify_branch.py
"""

import sys
from pathlib import Path

TESTS_DIR = Path(__file__).parent
PROJECT_ROOT = TESTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from google import genai  # noqa: E402

from step_route_or_search import route_or_search  # noqa: E402

# Pulled directly from data/processed/knowledge_gaps.jsonl's real logged
# cases -- see this module's docstring. These are not invented examples.
TOPICLESS_NO_CONTEXT = [
    "it's not working",
    "the issue?",
    "what's the problem",
]

VPN_CONTEXT = (
    "user: My VPN keeps disconnecting right after I log in, error 0x8007042B.\n"
    "assistant: That matches a known tunnel-negotiation issue. Verify your VPN "
    "client version against the gateway's supported list, and check local "
    "firewall/antivirus isn't blocking the tunnel's port.\n"
)


def _client() -> genai.Client:
    import os
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def check_topicless_no_context_clarifies(errors: list) -> None:
    client = _client()
    for msg in TOPICLESS_NO_CONTEXT:
        result = route_or_search("", msg, client=client)
        if not result.clarify_question:
            errors.append(
                f"expected clarify_question for topic-less message {msg!r} with NO prior "
                f"context, got direct_reply={result.direct_reply!r} "
                f"search_calls={[c.query for c in result.search_calls]!r} -- this is exactly "
                f"the content-free-search regression this branch exists to prevent"
            )


def check_same_vague_phrasing_with_context_still_searches(errors: list) -> None:
    """
    THE critical negative case: "whats the issue" must still carry the
    real VPN topic forward via search_knowledge_base when context exists
    -- clarify must not fire just because phrasing is vague, only when
    there is genuinely nothing to resolve it against.
    """
    client = _client()
    for msg in ("whats the issue", "what's the issue", "what do you think"):
        result = route_or_search(VPN_CONTEXT, msg, client=client)
        if result.clarify_question:
            errors.append(
                f"{msg!r} WITH real VPN context in conversation incorrectly triggered "
                f"clarify_question ({result.clarify_question!r}) instead of carrying the "
                f"VPN topic forward via search_knowledge_base -- this would be a real "
                f"regression of the original router/contextualizer merge's whole purpose"
            )
            continue
        if not result.search_calls:
            errors.append(f"{msg!r} with VPN context produced neither clarify nor search_calls")
            continue
        query = result.search_calls[0].query.lower()
        if "vpn" not in query:
            errors.append(
                f"{msg!r} with VPN context searched for {query!r}, which doesn't carry the "
                f"real VPN topic forward"
            )


def check_specific_question_never_clarifies(errors: list) -> None:
    client = _client()
    specific_questions = [
        "My VPN keeps disconnecting right after I log in, error 0x8007042B.",
        "What DNS server should corporate laptops use?",
        "How do I reset a user's MFA device?",
    ]
    for msg in specific_questions:
        result = route_or_search("", msg, client=client)
        if result.clarify_question:
            errors.append(
                f"specific question {msg!r} incorrectly triggered clarify_question "
                f"({result.clarify_question!r}) instead of a real search -- clarify must "
                f"stay narrow to genuinely topic-less input"
            )
        if not result.search_calls:
            errors.append(f"specific question {msg!r} produced no search_calls at all (got {result!r})")


def check_greeting_never_clarifies(errors: list) -> None:
    client = _client()
    for msg in ("hi", "thanks", "who are you"):
        result = route_or_search("", msg, client=client)
        if result.clarify_question:
            errors.append(
                f"greeting/identity message {msg!r} incorrectly triggered clarify_question "
                f"({result.clarify_question!r}) instead of direct_reply -- clarify and "
                f"direct_reply must stay mutually exclusive"
            )
        if not result.direct_reply:
            errors.append(f"greeting/identity message {msg!r} produced no direct_reply (got {result!r})")


def check_clarify_question_is_substantive(errors: list) -> None:
    client = _client()
    result = route_or_search("", "it's not working", client=client)
    if not result.clarify_question:
        errors.append("setup problem: expected a clarify_question to check substantiveness of")
        return
    q = result.clarify_question.strip()
    if len(q) < 10 or q.lower() in ("can you clarify?", "could you clarify?"):
        errors.append(f"clarify_question is too generic/short to be useful: {q!r}")


def main() -> bool:
    checks = [
        ("topic-less message with no context -> clarify, not a content-free search",
         check_topicless_no_context_clarifies),
        ("same vague phrasing WITH real context still searches, never clarifies (critical negative case)",
         check_same_vague_phrasing_with_context_still_searches),
        ("a specific real question never clarifies", check_specific_question_never_clarifies),
        ("a greeting/identity question never clarifies", check_greeting_never_clarifies),
        ("clarify_question is a real, substantive question", check_clarify_question_is_substantive),
    ]

    errors: list = []
    for name, fn in checks:
        before = len(errors)
        print(f"Running: {name} ...")
        fn(errors)
        status = "FAIL" if len(errors) > before else "ok"
        print(f"  [{status}] {name}")

    print()
    if errors:
        print(f"{len(errors)} PROBLEMS FOUND:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("All clarify-branch checks passed.")
    return len(errors) == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
