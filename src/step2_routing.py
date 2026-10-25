"""
Step 2 of the pipeline: intent/routing classification. Own call, kept
separate from Step 3 (see PROJECT_HANDOFF.md Section 5 -- routing has
real potential to grow agentic later, extraction doesn't).

WHO THE USER IS (corrected 2026-09-13, see PROJECT_HANDOFF.md top of
file): this tool is used BY IT SUPPORT STAFF to look up KB articles/
procedures for a ticket, not by the confused employee directly. An IT
staffer is never going to ask this tool to "connect me to a human" --
they ARE the human. The "escalate" intent from the original design
assumed a chatbot talking to an employee and has been DROPPED -- see
PROJECT_HANDOFF.md Section 3 for the real-world agent-assist research
(Zendesk/Dynamics 365/Forethought/Ada) that informed this.

Design per PROJECT_HANDOFF.md Section 3 and the sourced LLM-call-design
guidance (claudecertificationguide.com, read 2026-09-13):
  - Explicit categorical criteria in the system prompt, not vague
    instructions -- each intent has a testable boundary.
  - 2-4 few-shot examples, each with a stated reason, not bare
    input->output pairs.
  - Forced schema output (response_schema + JSON mime type) -- the model
    always returns exactly this shape, never free text.
  - canned_answer intent dropped (no real canned Q&A list exists yet --
    see Section 3, don't build a dead code path).
"""

import json
import os
from dataclasses import dataclass
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

VALID_INTENTS = ("off_topic", "genuine_question", "conversational")

# "conversational" added 2026-10-23: found via live UI testing (a ticket
# session opened, then "hi" sent as the first chat message) that a bare
# greeting/acknowledgment with NO IT content was being forced into
# off_topic (correct per the OLD 2-intent rule -- "hi" really does
# mention zero systems/devices/accounts) and then chat_service.py's
# caller (ui/static/index.html) bounced it with "I'll route this to a
# human agent." That's wrong in a ticket-attached session: the staffer
# IS the human, there's nothing to route, and the open ticket's context
# shouldn't be abandoned over a greeting. The fix is a THIRD routing
# bucket, not a change to off_topic's own rule (which is correct in
# isolation) -- researched 2026-10-23: production query routers commonly
# separate "needs retrieval" / "off-topic" / "no action needed" rather
# than forcing every message into a binary. chat_service.py handles
# conversational with a short acknowledgment and skips the rest of the
# pipeline entirely (no Step 3/retrieval/generation cost).
SYSTEM_PROMPT = """You are the routing step for an internal IT helpdesk agent-assist \
tool, used BY IT SUPPORT STAFF to look up information for a ticket -- not by the \
end employee directly. Read the message (which may be a pasted ticket, or the \
staffer's own query) and classify it into exactly one of three intents, using these \
EXPLICIT rules -- do not use general judgment beyond them:

- "conversational": the message is EITHER (a) a bare greeting, acknowledgment, or \
filler with NO question and NO new information of its own (e.g. "hi", "ok", \
"thanks", "got it"), OR (b) a short question directed AT THE ASSISTANT ITSELF -- \
about who/what it is, what it can do, or whether it's a bot/human -- rather than \
about the ticket or any IT topic (e.g. "who are you", "are you a bot", "what can \
you help with"). Check this FIRST, before off_topic -- neither kind mentions an IT \
system, but neither is a real off-topic QUESTION ABOUT SOMETHING ELSE either.

- "off_topic": the message is an actual statement or question about a DIFFERENT, \
unrelated topic -- NOT about the assistant itself (see conversational above) -- \
and contains ZERO mention of a company system, device, account, application, \
network, or IT process. Purely personal, social, or unrelated-to-work topics \
(food, weather, facilities unrelated to IT, HR/benefits questions) are off_topic.

- "genuine_question": everything else that concerns an IT/company system, \
device, account, or process -- including frustrated, vague, ambiguous, \
or policy-related messages, as long as they aren't off_topic. This is \
the default for any real IT ticket content, however it's phrased.

When in doubt, check for ANY mention of a system/device/account/process --  \
if one exists, it is genuine_question, not off_topic."""

# 2 few-shot examples, each with a stated reason (not bare input->output),
# per the few-shot guidance: 2-4 examples, reasoning included so the model
# generalizes the rule rather than pattern-matching the exact wording.
FEW_SHOT_EXAMPLES = [
    {
        "message": "What time does the cafeteria on the 4th floor close today?",
        "intent": "off_topic",
        "reason": "No company system, device, account, or IT process is mentioned at all -- this is a facilities/food question unrelated to IT.",
    },
    {
        "message": "This is honestly ridiculous, THIRD day in a row my VPN just refuses to stay connected after login, I have work to do and this company's IT setup is a joke.",
        "intent": "genuine_question",
        "reason": "Frustrated venting, but it names a real system (VPN) and a real problem -- still a genuine ticket to resolve, regardless of tone.",
    },
    {
        "message": "hi",
        "intent": "conversational",
        "reason": "A bare greeting with no question and no IT content -- not the same as an off-topic question about something else, so it gets its own intent rather than being forced into off_topic.",
    },
    {
        "message": "who are u",
        "intent": "conversational",
        "reason": "A short question, but directed AT THE ASSISTANT ITSELF (who/what it is), not about the ticket or any IT topic -- added 2026-10-24 after this exact message was found to classify inconsistently (off_topic some runs, conversational others) because it satisfies BOTH rules' literal wording: it IS a question (excluding it from the old 'no question' conversational rule) AND it has zero IT content (satisfying off_topic's rule). A question about the assistant is its own case, not an off-topic question about a different real-world topic like the cafeteria example above.",
    },
]


@dataclass
class RoutingResult:
    intent: str
    raw_response: str


def _build_schema() -> dict:
    return {
        "type": "OBJECT",
        "properties": {
            "intent": {"type": "STRING", "enum": list(VALID_INTENTS)},
        },
        "required": ["intent"],
    }


def _build_prompt(message: str, message_source: str = "user_message") -> str:
    examples_text = "\n\n".join(
        f'Message: "{ex["message"]}"\n'
        f'Intent: {ex["intent"]}\n'
        f'Reason: {ex["reason"]}'
        for ex in FEW_SHOT_EXAMPLES
    )
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Examples:\n\n{examples_text}\n\n"
        f"Now classify this message:\n{wrap_untrusted(message, message_source)}"
    )


def classify_intent(message: str, client: genai.Client | None = None,
                     message_source: str = "user_message") -> RoutingResult:
    """
    Run Step 2 on a single (already Step-0/Step-1-processed) message.
    Returns a RoutingResult with intent guaranteed to be one of
    VALID_INTENTS (enforced by the schema, not just hoped for).

    message_source: passed straight to _build_prompt()/wrap_untrusted() --
    "ticket" for a Jira ticket's seed content (may originate from the end
    employee who filed it, not the IT staffer), "user_message" (default)
    otherwise. See pipeline.py's run_pipeline() docstring for why this
    distinction exists.
    """
    if client is None:
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    model = os.environ["GEMINI_MODEL"]

    response = call_with_retry(
        "step2_routing",
        client.models.generate_content,
        model=model,
        contents=_build_prompt(message, message_source),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=_build_schema(),
        ),
    )

    parsed = json.loads(response.text)
    intent = parsed["intent"]
    if intent not in VALID_INTENTS:
        # Schema enforcement should make this impossible, but never trust
        # that blindly -- schema enforcement only guarantees syntax, not
        # semantics (see the validation-retry guidance, Section 5).
        raise ValueError(f"Model returned an intent outside the enum: {intent!r}")

    return RoutingResult(intent=intent, raw_response=response.text)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from query_parsing import parse_query

    test_messages = [
        "My VPN keeps disconnecting right after I log in.",
        "What time does the cafeteria on the 4th floor close today?",
        "Who do I escalate this VPN outage to if it's affecting the whole Chennai office?",
        "This is honestly ridiculous, THIRD day in a row my VPN just refuses to stay connected after login, I have work to do and this company's IT setup is a joke.",
        "it's not working",
    ]

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    print("Step 2 routing -- smoke test\n")
    for msg in test_messages:
        parsed = parse_query(msg)
        result = classify_intent(parsed.cleaned_text, client=client)
        print(f"[{result.intent}] {msg}")
