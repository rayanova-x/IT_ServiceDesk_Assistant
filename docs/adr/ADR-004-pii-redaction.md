# ADR-004: PII redaction happens at the storage boundary, never before extraction

**Status:** Accepted, 2026-10-15
**Decision owner:** Project owner, resolved after a direct architecture-review question about
whether redaction and structured-field extraction were in conflict

## Context

Real personal data flows through this system in several places that had
no protection until this decision: the raw message a staffer types or a
Jira ticket's real description/comments (`raw_text`), the rewritten query
(`standalone_text`/`sub_query`), and the generated answer
(`answer_text`/`caveat`) — all persisted to SQLite indefinitely, with no
redaction, no retention limit, and no data classification pass anywhere
in the pipeline. `session_store.py`'s own schema comment had
acknowledged this directly: the table design was "reasoned from first
principles, not sourced" with no real privacy framework behind it.

Fixing this raised a real architectural question, not just an
implementation detail: **Step 1's regex extraction and Step 3's LLM
extraction both need the real, unredacted text** to correctly pull error
codes, IPs, hostnames, and ticket numbers into `case_facts` — redacting
before those steps run would break the pipeline's actual job of
understanding the query. The question was whether "add PII redaction"
and "extraction needs real text" were fundamentally incompatible goals.

They aren't, once the two concerns are separated by *when* they apply,
not *whether* they apply. Checked against sourced guidance (verified
directly, not assumed) on PII handling in LLM-touching pipelines: the
recommended pattern is to redact the *persisted* copy of data, never the
*in-flight* copy sent to the model. Anthropic's own commercial API terms
reinforce why this ordering is safe: API inputs/outputs aren't used for
training and default to 30-day retention (0-day available by
agreement) — a smaller, time-bounded exposure than this project's own
prior behavior of storing the same unredacted text **permanently**, with
no retention limit at all, in its own SQLite file.

## Decision

Redaction happens in exactly one place: inside `chat_service.py`,
immediately before each `session_store.log_turn()` /
`log_access_denial()` call — after Steps 1–9 have already run on the
real text, and after the real (unredacted) answer has already been
returned to the UI for that request. Nothing upstream of that point
changes. `pii_redaction.py`'s `redact_for_storage()` replaces free-text
PII (emails, phone numbers, employee-ID-style tokens matching this
project's own synthetic data convention) with bracketed placeholders.

Critically, it does **not** touch error codes, IP addresses, hostnames,
file paths, or ticket numbers — those are operational identifiers an IT
staffer needs intact to make sense of a logged turn, they already have
their own clean structured fields elsewhere in the schema (never stored
as free text requiring redaction), and they are not PII in an IT-support
context the way a name or personal email is.

Verified directly, not assumed: a test message containing both a real
error code and a real email address in the same sentence correctly
produced a `synthesized` answer referencing the error code (confirming
extraction and generation both saw the real text), while the stored
`raw_text` had the email redacted and the error code intact — proving
the two concerns don't conflict when separated by pipeline stage rather
than treated as one all-or-nothing redaction pass.

## Alternatives considered

- **Redact before Step 1** (at raw message intake). Rejected: this is
  the design that actually would have broken extraction — Step 1's
  regex and Step 3's LLM call both need to see real error codes/IPs/
  hostnames, which a redaction pass applied this early could plausibly
  damage if its patterns ever overlapped with those structured formats.
- **Redact only in the structured log lines, leave SQLite untouched**.
  Rejected: SQLite is the more durable, more queryable copy (no
  retention limit, directly joinable across sessions) — it's the bigger
  exposure, not the smaller one, so it needs the stronger protection,
  not weaker.
- **A full NER-based redactor (e.g. Microsoft Presidio)** instead of
  regex. Not rejected, deferred — see Consequences below. Regex is a
  real, working first step; a proper NLP-based redactor is the correct
  next step if this project ever handles real (not synthetic) employee
  data at volume.

## Consequences

- Names are **not** reliably redacted. A regex pattern has no reliable
  signature for "this is a person's name" distinguishable from any other
  two-capitalized-words phrase — this is a stated, known limitation of
  this implementation, not a silent gap. Verified directly: "Contact
  John Smith" passes through completely unredacted in testing, while
  "john.smith@corp.com" two words later is correctly caught.
- There is still no retention policy — redacted text is stored
  indefinitely, same as before. This decision closes the "what's stored
  is unprotected" gap; it does not close the separate "nothing ever gets
  deleted" gap, which remains open.
- The in-flight Gemini call is unaffected by this decision in every way
  — same real text in, same real text out, as before. This decision
  only changes what happens to a copy of that text *after* the request
  completes.

## Revisit when

This project ever handles real (not synthetic) employee data at volume,
or a reviewer specifically asks about name redaction — at that point,
replace or augment the regex patterns with an actual NER-based redactor
(Presidio is the commonly-cited practical default for this exact
problem) rather than extending the regex approach further, since regex
genuinely cannot solve the name-detection problem no matter how many
more patterns are added to it.
