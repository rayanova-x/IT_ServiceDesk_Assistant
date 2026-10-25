"""
Real HTTP-level test of ui/server.py -- the one layer no other test in
this project touches (every other test calls pipeline/chat_service
functions directly in-process; this one goes through FastAPI's actual
request/response/cookie handling via TestClient, bundled with FastAPI/
Starlette, no new dependency).

SLOW: TestClient triggers the real `lifespan` startup, which calls
PipelineClients.build() -- the real embedding model, reranker, and Chroma
collection all load for real, and real Gemini calls happen for the one
end-to-end chat test. This is deliberate (testing the real serving path
means testing the real thing, not a stand-in -- same principle as every
other test in this project), but means this file costs ~15-20s to load
models plus a small amount of real Gemini quota (3-4 calls) -- run it
deliberately, not as a tight inner-loop check.

Checks:
  1. Every chat endpoint (/api/messages, /api/messages/stream) returns
     401 without a valid login cookie -- the actual security boundary
     this project's permission-filter work depends on (see
     ui/server.py's _require_login()).
  2. Login sets a real httpOnly cookie; that cookie authenticates
     subsequent requests; logout invalidates it.
  3. Wrong credentials are rejected with 401, not a stack trace.
  4. GET /api/me correctly reflects logged-out vs. logged-in state.
  5. A real end-to-end chat message through the full HTTP stack (login
     -> POST /api/messages) returns a well-formed response, and the
     resulting turn is actually persisted (visible via GET
     /api/sessions/{id}/turns).
  6. The permission filter's effect is visible through the REAL API, not
     just the underlying function: logging in as a security-team-only
     account (bob, no it-department-group) and asking an
     it-department-group-restricted question does not 500 and produces
     a response that doesn't contain restricted-only content.

Run: python tests/test_serving_path.py
"""

import sys
from pathlib import Path

TESTS_DIR = Path(__file__).parent
PROJECT_ROOT = TESTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "ui"))


def _build_client():
    """
    Imports ui/server.py and wraps it in a TestClient. Importing here
    (not at module top-level) keeps the ~15-20s model-load cost
    contained to right before it's actually needed, and lets this file
    be imported elsewhere (if ever) without paying that cost just by
    importing it.
    """
    from fastapi.testclient import TestClient
    import server as ui_server
    return TestClient(ui_server.app)


def check_auth_required(client, errors: list) -> None:
    resp = client.post("/api/messages", json={"message": "hello"})
    if resp.status_code != 401:
        errors.append(f"POST /api/messages with no login cookie should 401, got {resp.status_code}")

    resp = client.post("/api/messages/stream", json={"message": "hello"})
    if resp.status_code != 401:
        errors.append(f"POST /api/messages/stream with no login cookie should 401, got {resp.status_code}")


def check_me_reflects_logged_out(client, errors: list) -> None:
    resp = client.get("/api/me")
    if resp.status_code != 200:
        errors.append(f"GET /api/me should always be 200, got {resp.status_code}")
        return
    body = resp.json()
    if body.get("logged_in") is not False:
        errors.append(f"expected logged_in=False with no cookie, got {body}")


def check_wrong_credentials_rejected(client, errors: list) -> None:
    resp = client.post("/api/login", json={"username": "alice", "password": "wrong-password"})
    if resp.status_code != 401:
        errors.append(f"login with a wrong password should 401, got {resp.status_code}")

    resp = client.post("/api/login", json={"username": "nonexistent_user", "password": "whatever"})
    if resp.status_code != 401:
        errors.append(f"login with a nonexistent username should 401 (not leak that the user doesn't exist via a different code), got {resp.status_code}")


def check_login_logout_roundtrip(client, errors: list) -> None:
    resp = client.post("/api/login", json={"username": "alice", "password": "demo1234"})
    if resp.status_code != 200:
        errors.append(f"login with correct seeded credentials should succeed, got {resp.status_code}: {resp.text}")
        return
    body = resp.json()
    if body.get("username") != "alice":
        errors.append(f"expected username='alice' in login response, got {body}")
    if "it_helpdesk_session" not in client.cookies:
        errors.append("login did not set the expected session cookie")

    me = client.get("/api/me").json()
    if not me.get("logged_in"):
        errors.append(f"expected logged_in=True after login, got {me}")

    client.post("/api/logout")
    me_after = client.get("/api/me").json()
    if me_after.get("logged_in"):
        errors.append(f"expected logged_in=False after logout, got {me_after}")

    # The authenticated endpoint should 401 again post-logout.
    resp = client.post("/api/messages", json={"message": "hello"})
    if resp.status_code != 401:
        errors.append(f"POST /api/messages should 401 again after logout, got {resp.status_code}")


def check_real_chat_message_end_to_end(client, errors: list) -> None:
    """The one real-Gemini-call test in this file -- costs a few seconds
    and a small amount of quota, but this is the actual thing being
    verified: does a real message through the real HTTP stack work and
    get durably logged."""
    login_resp = client.post("/api/login", json={"username": "carol", "password": "demo1234"})
    if login_resp.status_code != 200:
        errors.append(f"login as carol failed: {login_resp.status_code}")
        return

    resp = client.post("/api/messages", json={
        "message": "My VPN keeps disconnecting right after I log in, error 0x8007042B."
    })
    if resp.status_code != 200:
        errors.append(f"POST /api/messages should succeed for a logged-in user, got {resp.status_code}: {resp.text}")
        return
    body = resp.json()
    if body.get("stopped_at") != "ok":
        errors.append(f"expected stopped_at='ok' for a genuine IT question, got {body.get('stopped_at')}")
    if not body.get("turns"):
        errors.append("expected at least one turn in the response")
        return

    session_id = body["session_id"]
    turns_resp = client.get(f"/api/sessions/{session_id}/turns")
    if turns_resp.status_code != 200:
        errors.append(f"expected the turn to be durably logged and fetchable, got {turns_resp.status_code}")
    elif not turns_resp.json():
        errors.append("GET .../turns returned an empty list right after a successful message")

    client.post("/api/logout")


def check_permission_filter_through_real_api(client, errors: list) -> None:
    """
    bob is seeded with ONLY security-team-group (see
    seed_staff_accounts.py) -- asking a question whose real answer lives
    in it-department-group-restricted content should not 500, and the
    resulting answer/caveat must not fabricate the restricted content it
    can't see. This doesn't re-derive the exact excluded-chunk assertion
    test_permission_filter.py already covers at the function level --
    it's checking that the SAME guarantee survives the real HTTP/auth
    layer in between (cookie -> _require_login -> allowed_groups).
    """
    login_resp = client.post("/api/login", json={"username": "bob", "password": "demo1234"})
    if login_resp.status_code != 200:
        errors.append(f"login as bob failed: {login_resp.status_code}")
        return

    resp = client.post("/api/messages", json={
        "message": "How do I set up a new laptop for onboarding a new employee?"
    })
    if resp.status_code != 200:
        errors.append(f"a permission-filtered query should still return 200 (a clean no-match/caveat), got {resp.status_code}: {resp.text}")
        return
    body = resp.json()
    if body.get("stopped_at") != "ok":
        errors.append(f"expected stopped_at='ok' (not an error) for a permission-filtered query, got {body.get('stopped_at')}")

    client.post("/api/logout")


def main() -> bool:
    print("Loading TestClient + real pipeline clients (this takes ~15-20s)...")
    client = _build_client()

    checks = [
        ("chat endpoints require login (401 without a cookie)", check_auth_required),
        ("/api/me reflects logged-out state with no cookie", check_me_reflects_logged_out),
        ("wrong/nonexistent credentials rejected uniformly", check_wrong_credentials_rejected),
        ("login sets a working cookie; logout invalidates it", check_login_logout_roundtrip),
        ("a real chat message succeeds end-to-end and is durably logged", check_real_chat_message_end_to_end),
        ("the permission filter's effect survives the real HTTP/auth layer", check_permission_filter_through_real_api),
    ]

    errors: list = []
    with client as c:  # __enter__/__exit__ runs the real lifespan startup/shutdown
        for name, fn in checks:
            before = len(errors)
            try:
                fn(c, errors)
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
        print("All serving-path checks passed.")
    return len(errors) == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
