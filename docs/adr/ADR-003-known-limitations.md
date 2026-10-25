# ADR-003: Known limitations — documented as permanent, bounded risks, not defects awaiting a fix

**Status:** Living document, last updated 2026-10-15
**Decision owner:** Project owner

## Context

Hallucination, non-determinism, retrieval ranking errors, and
staleness-without-signal are not bugs that a bigger model, a stricter
prompt, or more tuning eventually eliminates — they are properties of how
retrieval-augmented generation works, and a credible system design treats
them as permanent constraints to build compensating controls around,
not problems to claim are solved. Several of this project's hardest
findings were explicitly investigated and left *unresolved*, with the
honest result written down rather than quietly re-framed as success. This
document collects those findings in one place, since they currently live
scattered across a 2000+ line build log.

## The limitations, as actually measured

**1. Confuser-pair retrieval — the single largest unresolved weakness.**
Two documents sharing vocabulary but describing different failure modes
(e.g., "Network SSL VPN failure" vs. "VPN client hangs on connect") can
still cause the wrong one to rank first. Reranking cut the confuser-trap
rate from 75% to 50% (2 of 4 known decoy cases still won at the time of
that measurement) — a real, measured improvement, explicitly *not*
claimed as a fix. The reranker matches on topic-level similarity without
reasoning about a stated distinguishing condition in the text (e.g., a
decoy document's own body explicitly says its symptom happens "rather
than at boot," which the reranker doesn't weight as disqualifying).
**Compensating control:** Step 8's confidence threshold, calibrated
from real data, correctly routes most of these cases to a
candidates-shown state rather than a confident wrong answer — verified
directly on a real case (query C11) where the correct answer won but with
a thin enough margin that the system correctly hedged instead of
asserting it.

**2. Confidence calibration has no reliable gradient below the threshold —
and, confirmed 2026-10-15, is not uniform across categories either.**
Calibration against 112 labeled queries found an honestly strange result:
scores below 0 land at roughly 45-50% accuracy, while scores in the 0–2
range score *worse* — around 38% — before accuracy becomes reliable
(75-80%) at 2.0 and above. This is not a smooth curve, and the project
does not pretend otherwise: the threshold is a single hard cut at 2.0
specifically because the data doesn't support a finer-grained scale below
it. The anomaly was traced to the confuser-pair weakness above landing
moderate (not low) scores while still being wrong, rather than treated as
noise to average over.

A follow-up slice of the same calibration run by category (added
2026-10-15, after a direct review question — "shouldn't calibration be
done per-document-type, not aggregate?" — correctly identified this as
untested) confirms the single global threshold is an averaging artifact,
not a universal truth. Real per-category numbers at the 2.0 cut:
**Access & Identity Management** (n=16) and **Network Infrastructure**
(n=35, the category containing the documented VPN confuser pair) both
have the relationship *backwards* — below-threshold accuracy (70% and
61% respectively) is higher than at-or-above-threshold accuracy (67% and
53%) — meaning a low score is not a reliable "this is wrong" signal in
either category. **Software & Applications Support** (n=22) is the
category where 2.0 works as strongly intended (17% below vs. 70% above,
a 53-point swing) — and is very likely the category the single global
threshold is implicitly tuned around, applied uniformly to every other
category regardless of fit. Security Operations and Hardware & Device
Support both behave reasonably close to the global curve. This is not
yet acted on (the threshold is still one global 2.0 in `step8_confidence.py`)
— it's reported here as a confirmed, measured finding, not a guess, per
this document's own stated rule.

**3. Recency reranking measured zero effect, and that result was kept,
not hidden or re-tuned away.** Step 7's recency-weighted rerank produced
no ranking changes on its tested sample at any weight from 1.0x to 8.0x.
Root-caused directly (not assumed): the sample lacked genuine
cross-document competition for recency to act on. This matches a
real, independently published finding (arXiv 2509.19376, "Freshness and
the Limits of Heuristic Trend Detection in Temporal RAG") that a fixed
recency-decay formula is corpus/query-composition-sensitive and "cannot
rescue" cases where relevance, not freshness, is the actual bottleneck —
their own published numbers show the same effect can collapse to 0.00 on
real data. The code was kept as-is rather than chased with further tuning
against a sample too small to actually exercise the mechanism either way.

**4. Ten knowledge-base documents are silently stale with zero retrievable
signal.** By design (for realism), 10 of the corpus's superseded documents
carry no marker distinguishing them from current ones — same `status`,
same label, only an old `version.when` and low revision count give them
away. Query understanding alone cannot catch these. The compensating
control is a second, independent layer: unconditional staleness
disclosure at generation time (every citation always carries its
`last_updated`/`status`, regardless of whether the query signaled a
recency intent) plus same-topic-pair detection that surfaces a stale
duplicate as a discrete structured field rather than leaving it to prose.
8 of these 10 pairs were confirmed, by direct testing, to already
co-occur in real top-5 results — this was checked before building the
detection feature, not assumed.

**5. Prompt-injection risk from the retrieved corpus — tested and fixed,
2026-10-15 (ADR-005).** This item originally read "low-probability but
not yet tested" — a review request to actually test it, not just assess
it, found a real, reproducible vulnerability: a chunk whose text
contained a literal `</document></documents>` sequence could break out
of its own tag boundary and have trailing text read as a new top-level
instruction, confirmed on two separate live Gemini calls (model output
was the literal word "COMPROMISED," exactly as the injected text
requested). Three other attack shapes tested in the same pass — a direct
"ignore previous instructions" override, a credential-exfiltration
request, a role-hijack attempt — were all already correctly resisted
without any fix, meaning this was a narrow, specific weakness in this
pipeline's own tag structure, not a general injection susceptibility.
Fixed with a targeted sanitization pass on chunk text (escaping only the
specific tag names this prompt depends on, verified not to mangle
legitimate angle-bracket content like code examples) plus an explicit
system-prompt rule as a second layer. Re-tested after the fix: the
previously-successful attack is now resisted on three more runs, and a
real non-adversarial query confirmed no regression. See ADR-005 for the
full before/after evidence.

**6. PII redaction is regex-only, and name redaction specifically does
not work.** Addressed 2026-10-15 (ADR-004): free-text PII in stored
conversation turns (emails, phone numbers, employee-ID-style tokens) is
now redacted before being written to SQLite or structured logs, applied
at the storage boundary rather than before extraction so Step 1/3's need
for real error codes/IPs/hostnames is unaffected — verified directly on
a message containing both a real error code and a real email in the same
sentence, confirming the two don't conflict. What remains genuinely
unsolved: regex has no reliable signature for a person's name, so
"Contact John Smith" passes through completely unredacted while an email
two words later is correctly caught — confirmed, not assumed, by direct
testing. There is also still no retention policy; redacted text is kept
indefinitely, same as before this fix. A real NER-based redactor
(Presidio is the commonly-cited default) is the correct next step if
this project ever handles real, non-synthetic employee data at volume.

## Decision

Keep this document current as new findings surface, rather than letting
them live only in dated conversation history or scattered handoff
sections. A finding is added here when it's measured and real — not
every open question, only the ones with an actual number or a reproduced
result behind them.

## Consequences

- This document will occasionally look like a list of unsolved problems
  rather than a feature list. That's the intent — a reviewer who finds
  this before finding a claim of "retrieval works well" gets the honest
  version first.
- Some of these limitations (confuser-pair retrieval specifically) are
  the actual highest-leverage remaining work in the project, more than
  any further generation-prompt tuning. Keeping them visible here is
  partly a prioritization tool, not just a disclosure artifact.

## Revisit when

Any of the above gets a real fix with measured before/after numbers —
move it out of this document into the relevant ADR or the eval results
section, with the same honesty about partial vs. complete resolution that
the original finding was written with. A limitation moves out of this
list only when there's a number proving it, never on the strength of a
design change alone.
