"""
Replaces Steps 0 (contextualize) + 2 (route) + 3 (understand) with ONE
model call that reasons over the full conversation and decides, in a
single pass, whether to reply directly (conversational filler, an
off-topic decline, an identity question) or call a search_knowledge_base
tool -- one call per distinct issue, for a compound message.

WHY THIS REPLACED THE 3-STAGE GATE (2026-10-24): the old design ran
three SEPARATE, sequential LLM calls -- Step 0 rewrote the message using
conversation context, Step 2 classified the REWRITTEN text into
off_topic/conversational/genuine_question with NO visibility into the
original conversation, Step 3 then extracted structured fields from
whatever Step 2 passed through. Each stage reasoned from a narrower
slice of context than the one before it, and real, repeated live-UI
testing over several hours found them disagreeing with each other on
nearly every borderline message shape: "hi" with prior context got
rewritten into a fabricated full question by Step 0, which then
correctly-but-wrongly routed as genuine_question; "who are u" satisfied
BOTH off_topic's and conversational's rule wording simultaneously and
classified inconsistently run to run; "whats the issue" sometimes lost
the real topic and sometimes kept it, depending on which few-shot
example the model happened to pattern-match against. Every fix was a
new hand-written rule chasing the last bug, each risking the next
regression (one fix for "hi" broke "whats the issue" again).

This is a genuine, scoped move from this project's workflow design
toward an AGENTIC one -- correctly, per Anthropic's own definition
("LLMs autonomously using tools in a loop") -- for EXACTLY this one
decision: whether/how to search. It does NOT extend to the rest of the
pipeline. Steps 4.5 (permission filtering), 5/6.5 (retrieval+rerank), 7
(recency), 8 (confidence), and 9 (generation) are UNCHANGED, still a
fixed deterministic workflow -- in particular Step 8's calibrated
confidence threshold (measured against 112 real labeled queries) is
deliberately NOT replaced by this model's own judgment, since letting an
LLM self-report "my retrieval was good enough" is exactly the unreliable
self-assessment pattern this project has avoided everywhere else. The
scope of what changed is narrow and deliberate: only the question "is
this a real IT question, and if so, what should I search for" moved
from 3 blind sequential gates to one model reasoning over everything at
once with a tool -- matching how real conversational RAG products
(Claude, ChatGPT, Copilot) actually work when given a knowledge base,
confirmed via direct testing against the real google-genai SDK (see
this module's test cases) before committing to the design, not assumed.

Verified directly (2026-10-24) against the real Gemini API before
building this: (1) a compound message ("VPN AND Outlook broken")
correctly produced TWO separate function calls in one response,
replacing Step 3's special-cased rewritten_query list mechanism; (2)
every one of the conversational/off-topic/identity-question cases that
broke the old 3-stage gate today was handled correctly, in a single
call, with no separate routing step; (3) the hardest real case (a vague
"whats the issue" follow-up needing the actual prior VPN topic) produced
a correct, well-formed search query on the first attempt.
"""

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from gemini_retry import call_with_retry  # noqa: E402
from prompt_delimiting import wrap_untrusted  # noqa: E402
from step3_understanding import CATEGORY_VALUES, ERROR_CODE_PATTERN  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

SEARCH_TOOL_NAME = "search_knowledge_base"

SEARCH_TOOL = types.FunctionDeclaration(
    name=SEARCH_TOOL_NAME,
    description=(
        "Searches the internal IT knowledge base for a specific technical/IT topic. "
        "Call this ONCE PER DISTINCT ISSUE if the staffer's message bundles two or more "
        "genuinely unrelated problems (e.g. a VPN issue AND a separate email issue) -- "
        "do not combine unrelated issues into one call. If ONE question spans several "
        "topics (e.g. comparing two things), call it once per topic so each search stays "
        "focused, and set same_question=true on each of those calls."
    ),
    parameters={
        "type": "OBJECT",
        "properties": {
            "query": {
                "type": "STRING",
                "description": (
                    "A clean, standalone query for the knowledge-base article that would "
                    "hold the answer, written as a short natural-language question or "
                    "statement (not a list of keywords), the way that article would be "
                    "written. For a "
                    "vague follow-up, carry forward the real topic from the conversation, "
                    "never a generic restatement like 'what is the issue'. For a policy, "
                    "definition or general question, search for that topic itself and leave "
                    "the ticket's specifics out unless the answer depends on them."
                ),
            },
            "category": {
                "type": "STRING",
                "enum": list(CATEGORY_VALUES),
                "description": (
                    "Pick the ONE category that clearly fits, or 'unclear' if genuinely "
                    "ambiguous across categories or you cannot tell -- do NOT guess a "
                    "plausible-sounding category just to fill this field."
                ),
            },
            "error_codes": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
                "description": "Exact error codes/identifiers verbatim from the message (e.g. '0x8007042B'). Empty list if none.",
            },
            "wants_latest": {
                "type": "BOOLEAN",
                "description": (
                    "true ONLY if the staffer explicitly signals wanting the CURRENT/most "
                    "up-to-date version of a procedure (words like 'latest', 'current', "
                    "'up to date', 'don't want an outdated version'). A message merely about "
                    "a recent event ('since yesterday') does NOT count."
                ),
            },
            "same_question": {
                "type": "BOOLEAN",
                "description": (
                    "true ONLY when this call is one part of a single question that also "
                    "needs the other call(s) in this turn to be answered (e.g. the two "
                    "sides of a comparison), so the results get answered together. false "
                    "for a standalone call or for separate, unrelated issues."
                ),
            },
        },
        "required": ["query", "category", "error_codes", "wants_latest"],
    },
)

SYSTEM_PROMPT = """You are an internal IT helpdesk agent-assist tool, used BY IT SUPPORT \
STAFF to look up information for a ticket -- not by the end employee directly. Read the \
staffer's new message together with the conversation so far and decide how to respond.

You have ONE tool: search_knowledge_base. Use these rules:

- If the message is a bare greeting, acknowledgment, or filler with NO real question \
(e.g. "hi", "ok", "thanks", "got it"), reply directly in plain text with a short \
acknowledgment -- do NOT call the tool.

- If the message is a short question directed AT YOU (who/what you are, what you can \
do, whether you're a bot) rather than about the ticket or any IT topic, reply directly \
in plain text -- do NOT call the tool.

- If the message is genuinely unrelated to IT/the ticket (e.g. food, weather, \
non-IT facilities, HR/benefits questions), reply directly in plain text, briefly \
noting that it looks unrelated to the current ticket/IT scope and naming what the \
ticket IS actually about if you know it from the conversation -- do NOT call the tool.

- If the message is a genuine IT question -- including frustrated, vague, or \
ambiguous phrasing, and including a vague follow-up like "what's the issue" or "what \
do you think" that refers back to something already discussed or already answered \
earlier in this conversation -- call search_knowledge_base. Write the query for the \
knowledge-base article that would hold the answer: for a vague follow-up, carry \
forward the real topic from the conversation, never a generic restatement; for a \
policy, definition or general question, search for that topic itself and leave the \
ticket's specifics out unless the answer depends on them. If the message bundles two \
or more genuinely unrelated IT problems, call the tool ONCE PER DISTINCT ISSUE. If \
one question spans several topics (for example comparing two things), call the tool \
once per topic so each search stays focused, and set same_question=true on those \
calls so they are answered together.

- If the message asks something you can already answer directly from the conversation \
itself (e.g. "what's the ticket number", restating a fact already stated earlier), \
answer directly in plain text using ONLY what's actually in the conversation -- do NOT \
call the tool and do NOT guess at information not present.

Never call the tool for something you already answered in plain text in this same turn, \
and never answer in plain text AND call the tool in the same turn -- pick exactly one.

Examples of writing the search (they show the pattern; do not copy the wording):
- Ticket about a VPN failure. Staffer asks "who has to approve a software install \
request?" -> ONE call, query "who approves software installation requests" -- a policy \
question, so the VPN specifics stay out of the query.
- Ticket about a printer showing offline. Staffer asks "how is that different from a \
stuck print queue?" -> TWO calls, both with same_question=true: "printer shows \
offline" and "print queue stuck" -- each search stays on one topic, and the results \
are answered together."""


@dataclass
class SearchCall:
    """One call to search_knowledge_base -- the per-sub-question payload
    that used to come from Step 3's rewritten_query/category/error_codes/
    wants_latest fields."""
    query: str
    category: str  # one of CATEGORY_VALUES
    error_codes: list[str]
    wants_latest: bool
    validation_warnings: list[str] = field(default_factory=list)
    same_question: bool = False


@dataclass
class RouteResult:
    """Exactly one of (direct_reply, search_calls) is populated -- never
    both, per the system prompt's own explicit instruction."""
    direct_reply: str | None = None
    search_calls: list[SearchCall] = field(default_factory=list)


def _build_contents(conversation_text: str, new_message: str, message_source: str) -> str:
    parts = [SYSTEM_PROMPT]
    if conversation_text:
        parts.append(f"Conversation so far:\n{conversation_text}")
    parts.append(f"Now handle this new message:\n{wrap_untrusted(new_message, message_source)}")
    return "\n\n".join(parts)


def _validate_search_call(query: str, category: str, error_codes: list[str]) -> list[str]:
    """Same post-parse validation spirit as step3_understanding.py's
    _validate() -- tool-call argument schema enforcement guarantees shape
    (category is one of the enum values), not semantic correctness of
    error_codes."""
    warnings = []
    for code in error_codes:
        if not ERROR_CODE_PATTERN.match(code):
            warnings.append(f"error_code {code!r} doesn't match the expected 0x... hex format")
    return warnings


def route_or_search(conversation_text: str, new_message: str, client: genai.Client | None = None,
                     message_source: str = "user_message") -> RouteResult:
    """
    conversation_text: the prior conversation, already rendered as plain
    text (role: message per line) -- same shape step0_context.py's
    current_turns rendering used, built by the caller (pipeline.py) from
    ConversationState so this module stays free of that dataclass's
    details. Empty string on a genuine first message.

    new_message: the staffer's raw new message, NOT pre-rewritten by any
    earlier step -- this call does its own contextualization internally,
    replacing step0_context.py's separate rewrite pass.
    """
    if client is None:
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    model = os.environ["GEMINI_MODEL"]

    tools = [types.Tool(function_declarations=[SEARCH_TOOL])]
    contents = _build_contents(conversation_text, new_message, message_source)

    response = call_with_retry(
        "step_route_or_search",
        client.models.generate_content,
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(tools=tools),
    )

    parts = response.candidates[0].content.parts if response.candidates else []
    calls = [p.function_call for p in parts if getattr(p, "function_call", None)]
    texts = [p.text for p in parts if getattr(p, "text", None)]

    if calls:
        search_calls = []
        for fc in calls:
            args = dict(fc.args)
            query = args.get("query", "")
            category = args.get("category", "unclear")
            if category not in CATEGORY_VALUES:
                # Schema enforcement should make this impossible for a
                # declared-tool call, but never trust that blindly across
                # a module boundary (same principle as step3_understanding.py).
                category = "unclear"
            error_codes = list(args.get("error_codes") or [])
            wants_latest = bool(args.get("wants_latest", False))
            search_calls.append(SearchCall(
                query=query, category=category, error_codes=error_codes,
                wants_latest=wants_latest,
                validation_warnings=_validate_search_call(query, category, error_codes),
                same_question=bool(args.get("same_question", False)),
            ))
        return RouteResult(search_calls=search_calls)

    direct_reply = " ".join(t.strip() for t in texts if t.strip()) or None
    return RouteResult(direct_reply=direct_reply)
