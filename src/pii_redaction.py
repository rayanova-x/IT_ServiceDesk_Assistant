"""
Redacts PII from free-text fields BEFORE they're written to durable
storage (SQLite) or structured logs -- never before an LLM call.

WHY THE ORDERING MATTERS, EXPLICITLY (this was a real design question,
not an assumption): Step 1's regex extraction and Step 3's LLM extraction
both need the REAL, unredacted text to correctly pull error codes, IPs,
hostnames, and ticket numbers into case_facts -- redacting before
extraction would break the pipeline's actual job. The fix is not "redact
less," it's "redact at a different point": those structured identifiers
are NOT PII (they're operational data an IT staffer needs visible, and
they're already stored as their own clean fields, never as free text) and
are left untouched everywhere, including after redaction. What this
module targets is free-text PII that has no structured field of its own
-- a name, an email address, a phone number appearing in prose -- which
only ever needs to exist for the LLM call itself, not for permanent
storage.

Sourced pattern (verified via direct search, not assumed): the
recommended shape for PII in LLM-touching pipelines is "redact the
persisted copy, never the in-flight copy" -- send the real text to the
model (Anthropic's own commercial API terms: inputs/outputs aren't used
for training, default 30-day retention, 0-day available by agreement --
a much smaller exposure window than an UNREDACTED value stored
permanently in this project's own SQLite file with no retention limit at
all), and apply redaction only at the point data is about to become
durable. This module is called from chat_service.py immediately before
each session_store.log_turn() call -- never before the Gemini calls in
pipeline.py/step9_generation.py.

HONEST LIMITATION: these are regex patterns, not an NER model. They
catch structurally regular PII (emails, phone numbers in common formats)
well, and catch names poorly to not at all -- a general name like "John
Smith" has no reliable regex signature distinguishable from an ordinary
two-capitalized-words phrase. A real NLP-based redactor (e.g. Microsoft
Presidio, the commonly-cited practical default for this exact problem)
would be the correct next step if this project ever handled real
(not synthetic) employee data at volume. This module is deliberately
scoped to what regex can do reliably, stated as a limitation rather than
silently pretending name redaction works.
"""

import re

# Emails -- standard pattern, deliberately not trying to validate TLDs
# exhaustively (over-matching a non-email string is far less harmful here
# than under-matching a real one).
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")

# Phone numbers -- North American and generic international formats with
# separators (space/dash/dot/parens). Deliberately requires at least 7
# digits to avoid false-positiving on short numeric sequences (ticket
# numbers, port numbers) that already have their own extraction path and
# are not phone numbers.
PHONE = re.compile(
    r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
)

# Employee-ID-style tokens matching this project's own synthetic data
# convention (EMP-00001, see data/raw/employees.csv) -- redacted here
# because an employee ID is more directly re-identifying than an IP or
# hostname, even though structurally it looks similar to the HOSTNAME
# pattern Step 1 extracts. Kept as a SEPARATE pattern (not reusing Step
# 1's regex) since Step 1's hostname extraction is deliberately narrow
# (2-5 letter dept/site code) and an employee ID has a different, real
# shape worth matching on its own terms.
EMPLOYEE_ID = re.compile(r"\bEMP-\d{4,6}\b", re.IGNORECASE)


def redact_for_storage(text: str | None) -> str | None:
    """
    Returns a copy of `text` with free-text PII replaced by bracketed
    placeholders, safe to write to SQLite or a structured log line. None
    passes through as None (several call sites log an optional field).

    Deliberately does NOT touch error codes, IPs, hostnames, file paths,
    or ticket numbers -- those are operational identifiers with their own
    structured fields elsewhere in the schema, not free-text PII, and an
    IT staffer re-reading a logged turn needs them intact to make sense
    of what happened.
    """
    if text is None:
        return None
    redacted = EMAIL.sub("[EMAIL]", text)
    redacted = PHONE.sub("[PHONE]", redacted)
    redacted = EMPLOYEE_ID.sub("[EMPLOYEE_ID]", redacted)
    return redacted
