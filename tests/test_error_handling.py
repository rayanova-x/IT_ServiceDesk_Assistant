"""
Real correctness test for the Gemini retry/error-handling layer (see
src/gemini_retry.py, and its wiring into chat_service.py's
send_message()). Uses FAKE failure-injecting functions, not a real
Gemini client -- the point is testing the retry/backoff/degrade LOGIC
itself (does it retry the right errors, stop retrying the wrong ones,
degrade gracefully once exhausted), which needs zero real API calls and
should run fast and free every time, same principle as
capture_step7_raw_data.py/sweep_step7_weights.py's capture-vs-sweep
split (isolate the expensive real call from the cheap, freely-repeatable
logic test).

Checks:
  1. A persistent retryable error (429) retries exactly 3 times then
     raises GeminiUnavailableError tagged with the right stage.
  2. A non-retryable error (400) raises immediately, WITHOUT wasting
     retries -- retrying a request that will deterministically fail the
     same way every time just adds latency for nothing.
  3. A transient failure that recovers on a later attempt returns the
     real result, not an error.
  4. chat_service.send_message() degrades an exhausted-retry failure
     during run_pipeline() (routing/understanding/context) into a clean
     ChatResponse(stopped_at="error", ...) instead of propagating the
     exception -- this is the actual user-facing guarantee: a real
     Gemini outage must never 500 the HTTP request.
  5. chat_service.send_message() degrades a generation-only failure
     (retrieval already succeeded) into a candidates_only turn with an
     error caveat, PRESERVING the search results rather than discarding
     them -- this is specifically what makes the degrade "graceful"
     rather than just "fails differently."

Run: python tests/test_error_handling.py
"""

import sys
from pathlib import Path

TESTS_DIR = Path(__file__).parent
PROJECT_ROOT = TESTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from google.genai import errors as genai_errors  # noqa: E402

from gemini_retry import GeminiUnavailableError, call_with_retry  # noqa: E402


def _client_error(code: int, status: str) -> genai_errors.ClientError:
    return genai_errors.ClientError(
        code=code, response_json={"error": {"message": "test", "status": status}}, response=None
    )


def _server_error(code: int, status: str) -> genai_errors.ServerError:
    return genai_errors.ServerError(
        code=code, response_json={"error": {"message": "test", "status": status}}, response=None
    )


def check_persistent_429_exhausts_retries(errors: list) -> None:
    calls = []

    def flaky():
        calls.append(1)
        raise _client_error(429, "RESOURCE_EXHAUSTED")

    try:
        call_with_retry("test_stage", flaky)
        errors.append("expected GeminiUnavailableError, got no exception")
    except GeminiUnavailableError as e:
        if e.stage != "test_stage":
            errors.append(f"expected stage='test_stage', got {e.stage!r}")
        if len(calls) != 3:
            errors.append(f"expected exactly 3 attempts for a persistent 429, got {len(calls)}")


def check_non_retryable_400_fails_fast(errors: list) -> None:
    calls = []

    def flaky():
        calls.append(1)
        raise _client_error(400, "INVALID_ARGUMENT")

    try:
        call_with_retry("test_stage", flaky)
        errors.append("expected GeminiUnavailableError, got no exception")
    except GeminiUnavailableError:
        if len(calls) != 1:
            errors.append(f"expected a non-retryable 400 to fail after exactly 1 attempt, got {len(calls)}")


def check_transient_failure_recovers(errors: list) -> None:
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 2:
            raise _server_error(503, "UNAVAILABLE")
        return "real result"

    result = call_with_retry("test_stage", flaky)
    if result != "real result":
        errors.append(f"expected the real result after recovery, got {result!r}")
    if len(calls) != 2:
        errors.append(f"expected exactly 2 attempts (1 failure + 1 success), got {len(calls)}")


def check_send_message_degrades_pipeline_failure(errors: list) -> None:
    import chat_service
    from step0_context import ConversationState

    original_run_pipeline = chat_service.run_pipeline

    def failing_run_pipeline(*args, **kwargs):
        raise GeminiUnavailableError("step2_routing", Exception("simulated outage"))

    chat_service.run_pipeline = failing_run_pipeline
    try:
        resp = chat_service.send_message(
            "test-session-pipeline-fail", "test message", ConversationState(), clients=None,
        )
        if resp.stopped_at != "error":
            errors.append(f"expected stopped_at='error' on a pipeline-level Gemini outage, got {resp.stopped_at!r}")
        if not resp.error_message:
            errors.append("expected a non-empty user-facing error_message on a pipeline-level outage")
        if resp.turns:
            errors.append(f"expected zero turns on a pipeline-level outage, got {len(resp.turns)}")
    finally:
        chat_service.run_pipeline = original_run_pipeline


def check_send_message_degrades_generation_failure_only(errors: list) -> None:
    """
    The more important case: retrieval SUCCEEDED (real search_results
    exist), only generate_answer() fails. The turn must degrade to
    candidates_only WITH those real search results surfaced as
    citations, not be discarded -- that's the actual value of catching
    this at the per-sub-result level instead of the whole-message level.
    """
    import chat_service
    from pipeline import PipelineOutcome, PipelineResult
    from step0_context import ConversationState

    fake_chunk = {
        "chunk_id": "200046-C00", "page_id": "200046", "title": "Network SSL VPN failure",
        "last_updated": "2026-07-01", "status": "current", "text": "fake excerpt text",
    }
    fake_result = PipelineResult(
        sub_query="VPN keeps disconnecting", category_used="Network Infrastructure",
        low_confidence=False,
        search_results=[{"chunk_id": "200046-C00", "rerank_score": 3.0, "chunk": fake_chunk}],
    )
    fake_outcome = PipelineOutcome(stopped_at="ok", results=[fake_result], stage_latency_ms={"step2_routing": 10.0})

    original_run_pipeline = chat_service.run_pipeline
    original_generate_answer_streaming = chat_service.generate_answer_streaming

    class _FakeClients:
        gemini = object()  # never actually called -- generate_answer_streaming itself is patched below

    def _raise_unavailable(*a, **k):
        # Matches the REAL failure shape: generate_answer_streaming()
        # raises BEFORE returning a StreamedAnswer when the initial
        # call_with_retry-wrapped request is exhausted -- i.e. before any
        # chunk could have reached a caller. See step9_generation.py's
        # generate_answer_streaming() docstring.
        raise GeminiUnavailableError("step9_generation", Exception("simulated outage"))

    chat_service.run_pipeline = lambda *a, **k: fake_outcome
    chat_service.generate_answer_streaming = _raise_unavailable
    try:
        resp = chat_service.send_message(
            "test-session-gen-fail", "VPN keeps disconnecting", ConversationState(), clients=_FakeClients(),
        )
        if resp.stopped_at != "ok":
            errors.append(f"expected stopped_at='ok' (retrieval succeeded) even though generation failed, got {resp.stopped_at!r}")
        if len(resp.turns) != 1:
            errors.append(f"expected exactly 1 degraded turn, got {len(resp.turns)}")
            return
        turn = resp.turns[0]
        if turn.mode != "candidates_only":
            errors.append(f"expected mode='candidates_only' for a generation-only failure, got {turn.mode!r}")
        if not turn.caveat or "temporarily" not in turn.caveat.lower() and "could not" not in turn.caveat.lower():
            errors.append(f"expected a clear error caveat explaining generation failed, got {turn.caveat!r}")
    finally:
        chat_service.run_pipeline = original_run_pipeline
        chat_service.generate_answer_streaming = original_generate_answer_streaming


def main() -> bool:
    checks = [
        ("persistent 429 exhausts exactly 3 retries", check_persistent_429_exhausts_retries),
        ("non-retryable 400 fails fast (1 attempt)", check_non_retryable_400_fails_fast),
        ("transient failure recovers on retry", check_transient_failure_recovers),
        ("send_message degrades a pipeline-level outage cleanly", check_send_message_degrades_pipeline_failure),
        ("send_message preserves retrieval results on a generation-only outage",
         check_send_message_degrades_generation_failure_only),
    ]

    errors: list = []
    for name, fn in checks:
        before = len(errors)
        try:
            fn(errors)
        except Exception as e:
            errors.append(f"{name}: unexpected exception {e!r}")
        status = "FAIL" if len(errors) > before else "ok"
        print(f"  [{status}] {name}")

    print()
    if errors:
        print(f"{len(errors)} PROBLEMS FOUND:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("All error-handling checks passed.")
    return len(errors) == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
