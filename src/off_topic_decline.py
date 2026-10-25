"""
Generates a context-aware decline message when Step 2 classifies a
message as off_topic -- added 2026-10-24 after a direct review question
("what if Step 2 got it wrong?"): Step 2 only ever sees ONE rewritten
message, with no visibility into the ticket/conversation beyond whatever
Step 0 already folded into it. A flat "that's off-topic" bounce gives a
genuinely misclassified question nothing to work with, so this spends
ONE extra LLM call -- only on the off_topic path, which should be rare
(most real messages from IT staff working a ticket are conversational or
genuine_question) -- to generate a decline that names what the system
currently understands the ticket/conversation to be about, so a
misclassified message can be corrected in one turn instead of silently
failing.

Deliberately NOT a retry/re-route back into Step 2 or Step 3 -- that
would be trusting the model's own "actually, let me reconsider" judgment,
the same unreliable self-assessment pattern this project avoids
elsewhere (Step 8's calibrated threshold, citation verification). This
only changes what's SHOWN to the staffer, not the routing decision
itself; if Step 2 was wrong, the staffer rephrases and Step 2 gets
another, independent look at the rephrased message next turn.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from google import genai

PROJECT_ROOT = Path(__file__).parent
REPO_ROOT = PROJECT_ROOT.parent
load_dotenv(REPO_ROOT / ".env")

import sys  # noqa: E402
sys.path.insert(0, str(PROJECT_ROOT))
from gemini_retry import call_with_retry  # noqa: E402
from prompt_delimiting import wrap_untrusted  # noqa: E402

SYSTEM_PROMPT = """You are replying to an internal IT helpdesk agent-assist tool's \
staffer after their message was classified as off-topic -- it does not appear to \
concern an IT system, ticket, or company process. Write ONE short, polite reply \
(1-2 sentences) that:
- States plainly that the message seems unrelated to the ticket/IT scope.
- If a current ticket topic or recent conversation subject is given below, name it \
briefly so the staffer can immediately tell whether this was a misunderstanding and \
rephrase, rather than guessing why their message was declined.
- Does NOT answer the off-topic message itself, and does NOT apologize at length.
- Does NOT say anything about "routing to a human agent" -- the reader IS the \
support staffer; there is no one else to route to.

Output ONLY the reply text, nothing else."""


@dataclass
class OffTopicDecline:
    message: str


def _build_prompt(off_topic_message: str, ticket_or_topic_context: str | None) -> str:
    parts = [SYSTEM_PROMPT]
    if ticket_or_topic_context:
        parts.append(f"Current ticket/conversation topic:\n{ticket_or_topic_context}")
    parts.append(f"Now write the reply to this message:\n{wrap_untrusted(off_topic_message, 'user_message')}")
    return "\n\n".join(parts)


def generate_decline(off_topic_message: str, ticket_or_topic_context: str | None = None,
                      client: genai.Client | None = None) -> OffTopicDecline:
    """
    off_topic_message: the staffer's message Step 2 classified as
    off_topic (the standalone/Step-0-rewritten version, same text Step 2
    itself saw).
    ticket_or_topic_context: a short string describing what the
    session's ticket/conversation is actually about, if known -- e.g.
    "Ticket IH-4: VPN disconnecting for 3 days straight". None on a
    session with no prior context yet (shouldn't normally happen, since
    a ticket's own seed message is never off_topic, but handled
    defensively).
    """
    if client is None:
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    model = os.environ["GEMINI_MODEL"]

    response = call_with_retry(
        "off_topic_decline",
        client.models.generate_content,
        model=model,
        contents=_build_prompt(off_topic_message, ticket_or_topic_context),
    )
    return OffTopicDecline(message=response.text.strip())
