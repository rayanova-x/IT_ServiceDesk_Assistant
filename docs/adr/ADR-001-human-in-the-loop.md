# ADR-001: Human-in-the-loop by design — no escalation path, by choice

**Status:** Accepted, 2026-09-13
**Decision owner:** Project owner, confirmed with Anthropic's own agent-assist research pattern

## Context

The original architecture included an `escalate` intent — a fourth routing
outcome alongside off-topic/genuine-question handling, meant to hand off to
a human when the tool's own confidence was low. This was written on an
unexamined assumption: that the chatbot talks directly to a confused end
employee, the way a customer-support chatbot does.

That assumption was wrong for this tool. The confirmed user is **IT support
staff**, not the end employee — a staffer pastes in a ticket (or types
their own query) to find the right internal KB article to resolve *someone
else's* issue. An IT staffer is never going to ask this tool to "connect me
to a human," because they already are the human in the loop. Keeping
`escalate` would have built a real feature with no one on the other end of
it.

Four conditions have to hold for a human-in-the-loop design to be more than
theater: the human reviewing the output must be **competent** to judge it,
have **sufficient information** to judge it (not just a confidence number),
have **genuine authority to reject** it, and be **accountable** for the
outcome. A fake escalation path — routing a low-confidence case to a
"human" who is the same person who would have received the raw candidates
anyway — satisfies none of these better than just showing the candidates
directly.

## Decision

Drop `escalate` as an intent. Collapse routing to two outcomes:
`off_topic` (decline) and `genuine_question` (proceed). For low-confidence
or ambiguous results, do not interrupt with a clarifying-question loop —
show the IT staffer **ranked candidates with their confidence signal**
(reranker score, staleness, citations) and let their own domain expertise
pick. For a genuine zero-match, return an honest "no confident match
found" state and log the query to a knowledge-gap store
(`knowledge_gaps.jsonl`) for later KB curation.

This was cross-checked against how real agent-assist tools built for
internal support staff (not end customers) handle the same problem —
Zendesk's agent copilot, Dynamics 365 "Smart assist," Forethought Discover,
Ada — and the pattern is consistent: none of them implement "escalate the
AI's decision" for a staff-facing tool. They show ranked
candidates and log coverage gaps instead.

Checked against the four conditions above, this design satisfies all four
by construction, not by policy: the staffer is domain-competent by
definition of their job; they see the full ranked result set with
real signals (reranker score, last-updated date, status) rather than a
single collapsed verdict, which is sufficient information to judge, not a
black box; they have full authority to pick any candidate, ignore all of
them, or resolve the ticket from outside knowledge entirely; and they
remain exactly as accountable for the ticket's resolution as they were
before this tool existed — nothing about the design shifts that
accountability onto the tool.

## Alternatives considered

- **Clarifying-question loop** (ask the user a follow-up question when
  ambiguous). Rejected: this is the right default for a conversational
  end-user chatbot, not for an expert working a queue — interrupting an
  expert's workflow with a question has a real cost these researched
  products evidently avoid, which is why none of them default to it for a
  staff-facing surface.
- **Fake escalation to "a human"** (route low-confidence cases to a
  generic "escalated" state). Rejected: there's no one new on the other
  end of that handoff — it would just relabel the same ranked-candidates
  view as a dead-end state instead of an actionable one.

## Consequences

- The tool never silently commits to a wrong answer on a hard case — the
  low-confidence branch is code-enforced (Step 9 never calls the LLM for
  synthesis when `low_confidence=True`), not a prompt instruction the
  model could ignore.
- Correctly, this does not fix the underlying retrieval weakness that
  causes most of the low-confidence cases (the confuser-pair problem — see
  the known-limitations summary). It only guarantees the tool is honest
  about not knowing, rather than guessing confidently. That's a narrower
  but more defensible guarantee.
- The knowledge-gap log is a second, genuinely useful artifact this
  design produces almost for free: a standing "what does our KB not cover
  well" report for whoever owns KB content, not just a fallback UX state.

## Revisit when

Real usage data shows IT staff blindly accepting the top-ranked candidate
without reading the alternatives or checking the staleness disclosure —
that would mean the "sufficient information" condition has silently
stopped holding even though the UI still technically shows it, and the
human-in-the-loop guarantee would need a harder gate (e.g., requiring
explicit acknowledgment of a low-confidence caveat) rather than passive
display.
