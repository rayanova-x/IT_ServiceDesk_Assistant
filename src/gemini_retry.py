"""
Shared retry/error-handling wrapper for every live Gemini call in the
serving path (step2_routing.py, step3_understanding.py,
step9_generation.py). Built because NONE of the three call sites had any
error handling at all before this -- a single 429/503/timeout during a
real chat turn would propagate all the way up through chat_service.py
and ui/server.py as a bare, unhandled exception, which FastAPI turns
into a raw 500 with a stack trace. For a pilot real IT staff actually
use, that's an outage, not a cosmetic issue -- Gemini's free tier is
already documented (PROJECT_HANDOFF.md Section 1) to rate-limit at 15
req/min, so this isn't a hypothetical failure mode.

Uses tenacity (already in requirements.txt, previously unused anywhere
in the codebase) rather than hand-rolled retry loops -- standard,
well-tested exponential backoff instead of reinventing it three times
across three call sites.

Deliberately narrow about WHAT gets retried: only transient conditions
(429 rate-limit, 503 service-unavailable/overloaded, network timeouts).
A 400 (bad request -- e.g. a schema the model genuinely can't satisfy)
or 401/403 (bad API key) retrying would just waste time repeating a
failure that will never succeed differently -- those re-raise
immediately, not after 3 wasted attempts.
"""

import logging

from google.genai import errors
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

logger = logging.getLogger("gemini_retry")

RETRYABLE_HTTP_CODES = {429, 503}


class GeminiUnavailableError(Exception):
    """
    Raised when a Gemini call exhausted all retry attempts on a
    transient error. Distinct from a bare exception so callers
    (chat_service.py) can catch THIS specifically and degrade the user-
    facing response gracefully, rather than letting an arbitrary
    exception type propagate and trusting every caller up the chain to
    handle it correctly.
    """
    def __init__(self, stage: str, original: Exception):
        self.stage = stage
        self.original = original
        super().__init__(f"Gemini unavailable during {stage} after retries: {original}")


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, errors.APIError):
        return exc.code in RETRYABLE_HTTP_CODES
    # google-genai wraps some transport-level failures (connection reset,
    # read timeout) in httpx's own exception types rather than APIError --
    # retry those too, since they're exactly as transient.
    return isinstance(exc, (TimeoutError, ConnectionError))


def _log_retry(retry_state) -> None:
    exc = retry_state.outcome.exception()
    stage = retry_state.kwargs.get("_stage", "?")
    logger.warning(
        "gemini_retry attempt=%d stage=%s error=%r -- retrying",
        retry_state.attempt_number, stage, exc,
    )


gemini_retry = retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(3),
    # wait_exponential_jitter, not plain wait_exponential -- added
    # 2026-10-15 after an architecture-review flag that fixed-interval
    # backoff (the original wait_exponential(multiplier=1, min=1, max=10))
    # has no jitter. For a single-process local pilot this had no
    # observable impact (no fleet of concurrent workers to synchronize),
    # but it's a real gap against the documented pattern, and the fix is
    # essentially free: tenacity ships this strategy natively, no
    # hand-rolled combination needed. initial=1/max=10 preserves this
    # pipeline's existing min=1/max=10 backoff envelope exactly;
    # jitter=1 adds up to 1 extra second of randomized delay on top of
    # each exponential step. The formula
    # (min(initial * 2**n + random.uniform(0, jitter), max)) is Google
    # Cloud's own documented retry strategy
    # (cloud.google.com/storage/docs/retry-strategy) -- a reasonable
    # precedent to follow given this project already calls a Google API.
    wait=wait_exponential_jitter(initial=1, max=10, jitter=1),
    reraise=True,
    before_sleep=_log_retry,
)


def call_with_retry(stage: str, fn, *args, **kwargs):
    """
    Runs fn(*args, **kwargs) with the shared retry policy, converting an
    exhausted retry (or a genuinely non-retryable error) into
    GeminiUnavailableError tagged with `stage` (e.g. "step2_routing",
    "step9_generation") so chat_service.py can log/display WHICH stage
    failed, not just that something failed.
    """
    @gemini_retry
    def _attempt(_stage):
        return fn(*args, **kwargs)

    try:
        return _attempt(_stage=stage)
    except Exception as e:
        logger.error("gemini_retry exhausted stage=%s error=%r", stage, e)
        raise GeminiUnavailableError(stage, e) from e
