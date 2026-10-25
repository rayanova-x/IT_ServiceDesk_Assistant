"""
Step 4 of the pipeline: ambiguity gate. Pure decision logic, no LLM call --
Step 3 already produced the `ambiguous` and `category` signals this step
acts on.

REDESIGNED 2026-09-13 (see PROJECT_HANDOFF.md Section 3, Step 4): the
original design asked the user ONE clarifying question and looped back to
Step 3. That assumed a chat interface talking to a confused employee.
This tool is used BY IT SUPPORT STAFF (see PROJECT_HANDOFF.md top of
file) -- an expert user, not someone who needs to be interviewed. Per
real agent-assist product research (Zendesk, Dynamics 365, Forethought,
Ada), the correct behavior for an expert-facing tool is: don't interrupt
with a question-and-answer loop. Instead, widen the search (drop the
category filter) and mark the result set as low-confidence/ambiguous so
the downstream answer step shows several ranked candidates instead of
confidently picking one -- the staffer's own expertise resolves it.

SIMPLIFIED 2026-10-24: the old caller (step3_understanding.py) produced
a separate `ambiguous` boolean alongside `category`, distinguishing
"genuinely ambiguous across categories" from "category=unclear, no exact
fit" -- even though, per this function's own prior docstring, BOTH cases
already produced identical downstream behavior
(search_category=None, low_confidence=True). The new caller
(step_route_or_search.py, replacing Steps 0/2/3 with one tool-calling
model reasoning over the full conversation) only has `category`, with
"unclear" as its single escape hatch for anything that doesn't clearly
fit one of the 8 real values -- so this function now checks only that,
collapsing a distinction that was already behaviorally identical, not
dropping real information.
"""

from dataclasses import dataclass

CATEGORY_VALUES = (
    "IT Policies & Procedures",
    "Facilities & Regional IT",
    "Onboarding & Provisioning",
    "Hardware & Device Support",
    "Security Operations",
    "Software & Applications Support",
    "Network Infrastructure",
    "Access & Identity Management",
    "unclear",
)


@dataclass
class GateDecision:
    search_category: str | None  # None means: don't filter by category
    low_confidence: bool  # True -> downstream should show ranked candidates,
                          # not confidently pick one result


def apply_ambiguity_gate(category: str) -> GateDecision:
    """
    Decide how Step 5 (retrieval) and Step 9 (answer presentation) should
    treat this query's category/confidence.

    Rules (see module docstring and PROJECT_HANDOFF.md Section 3):
      - category="unclear" -> no category filter (widen the search),
        low_confidence=True -- covers BOTH genuine ambiguity across
        categories and "no exact category fits," which already behaved
        identically before this simplification.
      - category is one of the 8 real values -> use it as the search
        filter, low_confidence=False.
    """
    if category == "unclear":
        return GateDecision(search_category=None, low_confidence=True)

    if category not in CATEGORY_VALUES:
        # Should be unreachable if the tool-call schema's enum caught
        # this, but never trust a field blindly across a module boundary.
        raise ValueError(
            f"category {category!r} is not one of the allowed values"
        )

    return GateDecision(search_category=category, low_confidence=False)


if __name__ == "__main__":
    from step3_understanding import UnderstandingResult

    test_cases = [
        UnderstandingResult(
            rewritten_query="VPN disconnects after login.",
            category="Network Infrastructure",
            error_codes=[],
            wants_latest=False,
            ambiguous=False,
        ),
        UnderstandingResult(
            rewritten_query="Screen went black.",
            category="unclear",
            error_codes=[],
            wants_latest=False,
            ambiguous=True,
        ),
        UnderstandingResult(
            rewritten_query="What does error code 0x8007042B mean?",
            category="unclear",
            error_codes=["0x8007042B"],
            wants_latest=False,
            ambiguous=False,
        ),
    ]

    print("Step 4 ambiguity gate -- smoke test\n")
    for u in test_cases:
        decision = apply_ambiguity_gate(u)
        print(f"category={u.category!r} ambiguous={u.ambiguous} "
              f"-> search_category={decision.search_category!r} "
              f"low_confidence={decision.low_confidence}")
