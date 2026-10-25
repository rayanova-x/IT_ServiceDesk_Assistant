"""
Shared helper for delimiting UNTRUSTED text before it's interpolated into
an LLM prompt -- the same structural-separation pattern already used for
retrieved chunk content (step9_generation.py's _sanitize_chunk_text(),
ADR-005), extended here to the other untrusted source this pipeline
handles every turn: the IT staffer's own raw message.

WHY THIS EXISTS (2026-10-23): ADR-005 closed the corpus-injection gap
(retrieved documents can't break out of their <document> tag) but left
user input going into Step 0/2/3's prompts as raw, undelimited text --
e.g. step2_routing.py's f'Message: "{message}"'. Per the layered-defense
guidance this project already follows (see ADR-005's own citation of
"delimiting alone is necessary, not sufficient"), every untrusted input
source should get the same treatment, not just the one that happened to
be tested first. Researched 2026-10-23: no production RAG pattern treats
this as a separate pass/fail "injection checker" step -- delimiters make
injection harder to hide, not impossible (no source claims a hard
structural boundary exists for LLM prompts the way it does for SQL).
The actual practice is tagging untrusted content distinctly at every call
site, which is what this module does.

Generic over the tag name (unlike _sanitize_chunk_text(), which is
narrowly scoped to <document>/<documents> since that's the one tag name
that prompt's structure depends on) because this module wraps user input
in a DIFFERENT tag (<user_message>) at three different call sites.
"""

import re


def wrap_untrusted(text: str, tag: str) -> str:
    """
    Wraps `text` in `<tag>...</tag>`, escaping any literal occurrence of
    that tag's own opening/closing sequence (case-insensitive, with or
    without attributes) INSIDE the text first -- so a message containing
    a literal "</user_message>" can't break out of its own boundary and
    have trailing text read as a new top-level instruction, the same
    attack class ADR-005 confirmed and fixed for retrieved chunk content.

    Does NOT touch any other angle-bracket content (code samples, shell
    redirects, comparison operators) -- only the specific tag name this
    call is using, matching _sanitize_chunk_text()'s "narrow, not a
    blanket escape" design.
    """
    safe = re.sub(
        rf"</?{re.escape(tag)}\b[^>]*>",
        lambda m: m.group(0).replace("<", "&lt;").replace(">", "&gt;"),
        text,
        flags=re.IGNORECASE,
    )
    return f"<{tag}>\n{safe}\n</{tag}>"
