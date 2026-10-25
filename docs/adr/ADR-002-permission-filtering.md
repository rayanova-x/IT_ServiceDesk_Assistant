# ADR-002: Permission filtering is pre-retrieval, never post-hoc

**Status:** Accepted, 2026-09-13 (retrieval filter); Hardened 2026-10-12 (server-resolved identity, audit trail)
**Decision owner:** Project owner, cross-checked against Glean / Microsoft 365 Copilot / Atlassian Rovo /
Pinecone's access-control engineering guide

## Context

Of the 283 chunks in the Confluence source alone, **183 (64.7%)** carry a
`restricted_to` group requirement — most to `it-department-group`, a
smaller set further restricted to `security-team-group` for
security-sensitive procedures. This is not a marginal case; a majority of
the KB is access-controlled. A retrieval system that doesn't account for
this will, by default, surface restricted content to anyone who asks a
relevant question — the ranking logic has no inherent awareness of who is
asking.

Every real enterprise retrieval system researched before building this
(Glean, Microsoft 365 Copilot's Graph-permission-scoped Retrieval API,
Atlassian Rovo, Pinecone's own access-control engineering guide) converges
on the same architecture: tag content with its allowed group/role at
ingest time, then enforce that tag as a metadata filter **at the moment of
retrieval** — never as a check applied to results after they've already
been generated into context, and never as a prompt instruction asking the
model to politely decline to repeat what it was just shown. Once
restricted content is in a model's context, the model will faithfully use
it; an instruction not to is advisory text competing with everything else
in the prompt, not an enforced boundary.

## Decision

`restricted_to` is written onto every chunk at ingestion time (already
true for all three sources: Confluence, the error-code spreadsheet, Slack
threads). At query time, the caller's group memberships
(`allowed_groups`) are resolved **server-side**, from a login session
token looked up fresh against the `staff_accounts` table on every
request — never accepted as a client-supplied field, and never trusted
from a cached claim embedded in the token itself (a JWT-style embedded
claim could drift from a real, since-revoked group membership until the
token expired; a server-side lookup on every request always reflects the
current row).

That resolved `allowed_groups` set is passed into the retrieval layer as a
genuine pre-retrieval filter: a Chroma `where` clause on the dense-search
query (`restricted_to IN [...]`) and an equivalent filter on the BM25
candidate scan — both applied *before* either retriever produces a
candidate list, not as a post-filter on what they return. A chunk the
caller isn't permitted to see is never retrieved, never ranked, and never
reaches the generation step at all.

Every time this filter actually excludes content from a result, that
exclusion is logged to an `access_denials` audit table — who asked, what
was withheld, how many chunks, when — independent of whether the user
ever notices. This turns "the filter works" from an implementation claim
into something a security reviewer can query and verify after the fact.

## Alternatives considered

- **Post-hoc filtering** (retrieve everything, strip restricted results
  before display). Rejected: the content still passes through the
  embedding/ranking/generation path, meaning a bug in the stripping step
  — or a future generation-prompt change that references chunk content
  directly — reopens the leak. The restricted content should never exist
  in the pipeline's working set in the first place.
- **Prompt-instruction withholding** ("if this chunk is restricted,
  don't mention it"). Rejected on the same grounds the sourced research
  calls out explicitly: this is an advisory instruction the model weighs
  against everything else in context, not a structural guarantee — and
  it still requires the restricted content to have been embedded into
  context for the model to "choose" not to use, which is itself the
  exposure.
- **Client-supplied group claim** (accept `allowed_groups` as a request
  parameter). Rejected: this lets any caller simply assert full access by
  editing the request body — a real vulnerability, not a hypothetical
  one, for any API consumer other than the one trusted UI.

## Consequences

- Retrieval is measurably more conservative for callers with narrow
  group membership — a staffer without `security-team-group` access
  genuinely cannot retrieve security-restricted procedures, even when
  those procedures would otherwise rank as the best match for their
  query. Verified directly: an IT-department-only account asking about a
  security-restricted incident type correctly receives "no matching KB
  articles found" rather than the restricted content or a degraded
  partial match.
- This adds one extra local BM25 scan per query when `allowed_groups` is
  supplied, solely to compute the audit-log diff (what got excluded).
  That cost is paid only by callers with a real identity attached — eval
  scripts and other callers that pass no identity are unaffected and
  retrieval remains unfiltered for them, matching prior behavior exactly.
- The filter is only as good as the `restricted_to` tagging at ingest
  time. A future ingestion source that fails to set this field correctly
  would silently default to unrestricted, since the filter has no way to
  distinguish "deliberately open" from "forgot to tag."

## Revisit when

A new ingestion source is added. The tagging discipline that makes this
filter meaningful (every chunk's `restricted_to` reflecting its real
access requirement) has to be re-verified for each new connector, not
assumed to carry over — this is exactly the kind of cross-source
consistency bug the project has already found once before, in a different
subsystem (BM25 silently missing a new source's chunks entirely until
caught by a targeted test).
