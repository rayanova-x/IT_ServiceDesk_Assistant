"""
Step 3 of the pipeline: structured query understanding. Own call, only
runs for messages Step 2 classified as genuine_question (see
PROJECT_HANDOFF.md Section 3 and Section 5 -- kept separate from Step 2
since routing may grow agentic later, extraction won't).

Design per the sourced LLM-call-design guidance (claudecertificationguide.com,
read 2026-09-13):
  - Explicit categorical criteria for each field in the system prompt.
  - 2-4 few-shot examples, each with a stated reason, pulled from real
    project data (data/processed/eval_edge_case_queries.jsonl), covering:
    a clean single-category case, a compound/bundled case needing
    list[str], a genuinely ambiguous case needing "unclear", an error-code
    case, and a wants_latest=True case.
  - Forced schema output. category has an explicit "unclear" fallback so
    the model isn't pressured to fabricate a plausible-but-wrong pick from
    the 8 real values (Section 3/6). error_codes is nullable/optional.
  - Lightweight self-check field (ambiguous) built into the schema itself,
    per the validation-retry guidance's "build discrepancy detection into
    the schema" pattern -- Step 4 reads this directly, no LLM needed to
    re-derive it.
  - Validation happens AFTER parsing, separately from schema enforcement:
    schema enforcement only guarantees valid JSON shape, not that
    error_codes/category content is actually correct.
"""

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

PROJECT_ROOT = Path(__file__).parent.parent
import sys  # noqa: E402
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from gemini_retry import call_with_retry  # noqa: E402
from prompt_delimiting import wrap_untrusted  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

# The 8 real categories, verified directly against confluence_export.jsonl's
# ancestors[0].title field (PROJECT_HANDOFF.md Section 3), plus the
# explicit "unclear" escape hatch -- never a closed 8-value set with no
# fallback.
REAL_CATEGORIES = (
    "IT Policies & Procedures",
    "Facilities & Regional IT",
    "Onboarding & Provisioning",
    "Hardware & Device Support",
    "Security Operations",
    "Software & Applications Support",
    "Network Infrastructure",
    "Access & Identity Management",
)
CATEGORY_VALUES = REAL_CATEGORIES + ("unclear",)

# Same hex-error-code format Step 1's regex already looks for -- used here
# to validate Gemini's error_codes output post-hoc (schema enforcement
# guarantees shape, not that the codes are real).
ERROR_CODE_PATTERN = re.compile(r"^0x[0-9A-Fa-f]{6,8}$")

SYSTEM_PROMPT = """You are the query-understanding step for an internal IT \
helpdesk assistant. The message has already been confirmed to be a \
genuine IT question (not off-topic, not an explicit request for a human). \
Extract the following, using these EXPLICIT rules:

- "rewritten_query": a clean, standalone restatement of the actual \
problem, stripped of greetings/filler/venting. If the message bundles \
TWO OR MORE genuinely unrelated problems (e.g. a VPN issue AND a separate \
email issue), return a list with one rewritten query per distinct \
problem. If it's one problem (even if described messily, with typos, or \
in another language), return a single string, not a list.

- "category": pick EXACTLY ONE of these 8 values if the problem clearly \
fits one of them: {categories}. If the message is genuinely ambiguous \
across categories, or you cannot tell which one it fits, return \
"unclear" -- do NOT guess a plausible-sounding category just to fill \
the field.

- "error_codes": a list of exact error codes/identifiers verbatim from \
the message (e.g. "0x8007042B"). If NO error code is mentioned, return \
an empty list -- do not invent one.

- "wants_latest": true ONLY if the message explicitly signals it wants \
the CURRENT/most up-to-date version of a procedure (words like "latest", \
"current", "up to date", "don't want an outdated version"). Otherwise \
false. A message that is merely about a recent event ("since yesterday") \
does NOT count -- that's a timing detail about the problem, not a \
recency request about the KB article.

- "ambiguous": true if category is "unclear" AND the message is too \
vague to search confidently even with a fallback category. False \
otherwise, including when category is "unclear" but the message still \
has enough detail to search on (e.g. specific symptoms) even without \
knowing the exact department.""".format(categories=", ".join(REAL_CATEGORIES))

# 5 few-shot examples with reasoning, pulled from real project data
# (data/processed/eval_edge_case_queries.jsonl), covering: clean single
# category, compound/bundled, genuinely ambiguous, error code, and
# wants_latest=True.
FEW_SHOT_EXAMPLES = [
    {
        "message": "My SSL VPN drops right after I authenticate, been happening since yesterday.",
        "output": {
            "rewritten_query": "SSL VPN connection drops immediately after authentication.",
            "category": "Network Infrastructure",
            "error_codes": [],
            "wants_latest": False,
            "ambiguous": False,
        },
        "reason": "A single, clear network problem. 'Since yesterday' is a timing detail, not a recency request about the article, so wants_latest stays False.",
    },
    {
        "message": "My VPN and my email are both broken today -- VPN disconnects right after I log in, and separately Outlook won't sync any new mail.",
        "output": {
            "rewritten_query": [
                "VPN disconnects right after login.",
                "Outlook is not syncing new mail.",
            ],
            "category": "Network Infrastructure",
            "error_codes": [],
            "wants_latest": False,
            "ambiguous": False,
        },
        "reason": "Two genuinely unrelated problems bundled in one message -- split into a list. category reflects the first/primary issue since the schema only has one category field; downstream retrieval runs once per rewritten_query.",
    },
    {
        "message": "My screen just went black.",
        "output": {
            "rewritten_query": "Screen went black.",
            "category": "unclear",
            "error_codes": [],
            "wants_latest": False,
            "ambiguous": True,
        },
        "reason": "Could plausibly be Hardware, Software, or Network Infrastructure (remote desktop) -- genuinely ambiguous, and too vague to search confidently even with a fallback. Do not guess a category.",
    },
    {
        "message": "Getting error 0x8007042B every time I try to connect to VPN.",
        "output": {
            "rewritten_query": "VPN connection fails with error 0x8007042B.",
            "category": "Network Infrastructure",
            "error_codes": ["0x8007042B"],
            "wants_latest": False,
            "ambiguous": False,
        },
        "reason": "A specific error code is present verbatim and should be captured exactly, not paraphrased or normalized.",
    },
    {
        "message": "What's the current, up to date procedure for handling a certificate access denial? I don't want an outdated version.",
        "output": {
            "rewritten_query": "What is the current procedure for handling a certificate access denial?",
            "category": "Access & Identity Management",
            "error_codes": [],
            "wants_latest": True,
            "ambiguous": False,
        },
        "reason": "Explicit recency language ('current, up to date', 'don't want an outdated version') triggers wants_latest=True -- this is a real signal about the KB article's freshness, not just a timing detail.",
    },
]


@dataclass
class UnderstandingResult:
    rewritten_query: str | list[str]
    category: str
    error_codes: list[str]
    wants_latest: bool
    ambiguous: bool
    validation_warnings: list[str] = field(default_factory=list)


def _build_schema() -> dict:
    return {
        "type": "OBJECT",
        "properties": {
            "rewritten_query": {
                "anyOf": [
                    {"type": "STRING"},
                    {"type": "ARRAY", "items": {"type": "STRING"}},
                ]
            },
            "category": {"type": "STRING", "enum": list(CATEGORY_VALUES)},
            "error_codes": {"type": "ARRAY", "items": {"type": "STRING"}},
            "wants_latest": {"type": "BOOLEAN"},
            "ambiguous": {"type": "BOOLEAN"},
        },
        "required": ["rewritten_query", "category", "error_codes", "wants_latest", "ambiguous"],
    }


def _build_prompt(message: str, message_source: str = "user_message") -> str:
    examples_text = "\n\n".join(
        f'Message: "{ex["message"]}"\n'
        f"Output: {json.dumps(ex['output'])}\n"
        f'Reason: {ex["reason"]}'
        for ex in FEW_SHOT_EXAMPLES
    )
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Examples:\n\n{examples_text}\n\n"
        f"Now process this message:\n{wrap_untrusted(message, message_source)}"
    )


def _validate(parsed: dict) -> list[str]:
    """
    Post-parse validation -- schema enforcement only guarantees valid
    JSON shape, not semantic correctness (see the validation-retry
    guidance). Returns a list of warning strings; does not raise, so
    callers can decide whether a warning is retry-worthy.
    """
    warnings = []
    if parsed["category"] not in CATEGORY_VALUES:
        warnings.append(f"category {parsed['category']!r} is not one of the 9 allowed values")
    for code in parsed["error_codes"]:
        if not ERROR_CODE_PATTERN.match(code):
            warnings.append(f"error_code {code!r} doesn't match the expected 0x... hex format")
    if parsed["category"] == "unclear" and not parsed["ambiguous"]:
        # Not necessarily wrong (see the "screen went black but has enough
        # symptom detail" case in the system prompt), but worth flagging
        # for review since it's an unusual combination.
        warnings.append("category=unclear but ambiguous=False -- verify this wasn't a forced-guess situation")
    return warnings


def understand_query(message: str, client: genai.Client | None = None,
                      message_source: str = "user_message") -> UnderstandingResult:
    """
    Run Step 3 on a message already confirmed genuine_question by Step 2.

    message_source: see step2_routing.py's classify_intent() docstring --
    same "ticket" vs "user_message" distinction, passed through unchanged.
    """
    if client is None:
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    model = os.environ["GEMINI_MODEL"]

    response = call_with_retry(
        "step3_understanding",
        client.models.generate_content,
        model=model,
        contents=_build_prompt(message, message_source),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=_build_schema(),
        ),
    )

    parsed = json.loads(response.text)
    warnings = _validate(parsed)

    return UnderstandingResult(
        rewritten_query=parsed["rewritten_query"],
        category=parsed["category"],
        error_codes=parsed["error_codes"],
        wants_latest=parsed["wants_latest"],
        ambiguous=parsed["ambiguous"],
        validation_warnings=warnings,
    )


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from query_parsing import parse_query

    test_messages = [
        "My SSL VPN connection keeps failing right after I authenticate. Worked fine yesterday.",
        "My VPN and my email are both broken today -- VPN disconnects right after I log in, and separately Outlook won't sync any new mail.",
        "My screen just went black.",
        "Getting error 0x8007042B every time I try to connect to VPN.",
        "What's the current, up to date procedure for handling a certificate access denial? I don't want an outdated version.",
    ]

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    print("Step 3 understanding -- smoke test\n")
    for msg in test_messages:
        parsed = parse_query(msg)
        result = understand_query(parsed.cleaned_text, client=client)
        print(f"MESSAGE: {msg}")
        print(f"  rewritten_query: {result.rewritten_query}")
        print(f"  category: {result.category}")
        print(f"  error_codes: {result.error_codes}")
        print(f"  wants_latest: {result.wants_latest}")
        print(f"  ambiguous: {result.ambiguous}")
        if result.validation_warnings:
            print(f"  WARNINGS: {result.validation_warnings}")
        print()
