"""
Real correctness test for Step 4.5 (permission filtering -- see
src/retrieval_pipeline.py's _passes_permission_filter()/
access_denial_info(), src/auth.py, src/session_store.py's staff_accounts/
access_denials tables). No LLM calls -- BM25/permission filtering are
both fully local, so this runs against the REAL 299-chunk corpus with
zero API cost, same convention as calibrate_confidence.py.

Checks:
  1. An unrestricted caller (allowed_groups=None) sees unchanged results
     -- the permission filter must be strictly opt-in, never a surprise
     behavior change for every pre-existing call site that doesn't pass
     an identity (eval scripts, offline tools).
  2. A caller without security-team-group genuinely cannot retrieve
     security-team-group-restricted content -- not just "ranked lower,"
     actually absent from the candidate list BM25/dense search return.
  3. access_denial_info() correctly reports the exact excluded groups and
     a nonzero count when something was actually withheld, and returns
     None when nothing was.
  4. A caller WITH the right group sees the restricted content normally
     -- the filter isn't accidentally blocking everyone.
  5. auth.py: password hashing/verification round-trips correctly, and a
     wrong password is rejected.
  6. session_store.py: a seeded staff account's allowed_groups survive a
     real DB round-trip (upsert -> get_staff_account), and an auth
     session created for it resolves to the SAME fresh group list (not a
     stale cached one) -- the core guarantee auth.py's module docstring
     is built on (never trust a client-held claim, always resolve fresh).

Run: python tests/test_permission_filter.py
"""

import sys
import tempfile
from pathlib import Path

TESTS_DIR = Path(__file__).parent
PROJECT_ROOT = TESTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import auth  # noqa: E402
from retrieval_pipeline import CHUNKS, access_denial_info, bm25_search  # noqa: E402

RESTRICTED_SECURITY_QUERY = "What is the procedure after a suspicious contractor compromise?"


def check_unfiltered_unchanged(errors: list) -> None:
    a = bm25_search(RESTRICTED_SECURITY_QUERY, top_k=10)
    b = bm25_search(RESTRICTED_SECURITY_QUERY, top_k=10, allowed_groups=None)
    ids_a = [r["chunk_id"] for r in a]
    ids_b = [r["chunk_id"] for r in b]
    if ids_a != ids_b:
        errors.append(f"allowed_groups=None changed results vs. omitting the param entirely: {ids_a} != {ids_b}")


def check_restricted_content_excluded(errors: list) -> None:
    results = bm25_search(RESTRICTED_SECURITY_QUERY, top_k=15, allowed_groups={"it-department-group"})
    leaked = [r for r in results if "security-team-group" in (r["chunk"].get("restricted_to") or [])]
    if leaked:
        errors.append(
            f"security-team-group-restricted chunks leaked to an it-department-group-only caller: "
            f"{[c['chunk_id'] for c in leaked]}"
        )
    if not results:
        errors.append("expected SOME results for an it-department-group caller (general-access chunks), got none")


def check_access_denial_info_accurate(errors: list) -> None:
    denial = access_denial_info(RESTRICTED_SECURITY_QUERY, [], None, {"it-department-group"})
    if denial is None:
        errors.append("expected a real access denial for a security-restricted query against an IT-only caller")
        return
    if denial["excluded_groups"] != ["security-team-group"]:
        errors.append(f"expected excluded_groups=['security-team-group'], got {denial['excluded_groups']}")
    if denial["excluded_count"] <= 0:
        errors.append(f"expected excluded_count > 0, got {denial['excluded_count']}")

    # A query whose top matches are all unrestricted chunks should report
    # no denial. (Verified directly against the real corpus: this query's
    # top-10 BM25 matches are all restricted_to=[] -- not an "off-topic/
    # no-match" query, since on this small corpus BM25 returns SOME
    # positive-score match for almost any real-word query, restricted or
    # not; "unrestricted topic", not "no results", is the real condition
    # being tested here.)
    no_denial = access_denial_info(
        "onboarding checklist for a new employee laptop setup", [], None, {"it-department-group"}
    )
    if no_denial is not None:
        errors.append(f"expected no denial for a query whose matches are all unrestricted content, got {no_denial}")


def check_full_access_caller_unaffected(errors: list) -> None:
    results = bm25_search(RESTRICTED_SECURITY_QUERY, top_k=10,
                           allowed_groups={"it-department-group", "security-team-group"})
    security_chunks = [c for c in CHUNKS if "security-team-group" in (c.get("restricted_to") or [])]
    if not security_chunks:
        errors.append("test setup problem: no security-team-group chunks exist in the real corpus at all")
        return
    found_any_security_chunk = any(
        "security-team-group" in (r["chunk"].get("restricted_to") or []) for r in results
    )
    if not found_any_security_chunk:
        errors.append(
            "a caller WITH security-team-group access unexpectedly got zero security-restricted "
            "results for a query specifically about restricted content -- filter may be over-blocking"
        )


def check_password_hashing(errors: list) -> None:
    h = auth.hash_password("correct-password")
    if not auth.verify_password("correct-password", h):
        errors.append("verify_password rejected the correct password against its own hash")
    if auth.verify_password("wrong-password", h):
        errors.append("verify_password ACCEPTED a wrong password -- critical auth bug")


def check_staff_account_roundtrip(errors: list) -> None:
    import session_store
    with tempfile.TemporaryDirectory() as tmpdir:
        session_store.DB_PATH = Path(tmpdir) / "test_sessions.db"
        session_store.init_db()

        session_store.upsert_staff_account(
            username="test_user", password_hash=auth.hash_password("pw123"),
            display_name="Test User", allowed_groups=["it-department-group"],
        )
        account = session_store.get_staff_account("test_user")
        if account is None:
            errors.append("get_staff_account returned None right after upsert_staff_account")
            return
        if account["allowed_groups"] != ["it-department-group"]:
            errors.append(f"allowed_groups didn't round-trip: got {account['allowed_groups']}")

        token = "test-token-123"
        session_store.create_auth_session(token, "test_user", "2099-01-01T00:00:00")
        resolved = session_store.get_auth_session(token)
        if resolved is None:
            errors.append("get_auth_session returned None for a freshly created session")
            return
        if resolved["allowed_groups"] != ["it-department-group"]:
            errors.append(f"auth session resolved stale/wrong groups: {resolved['allowed_groups']}")

        # Re-upsert with CHANGED groups, confirm the SAME existing token
        # now resolves the NEW groups -- the core "never trust a cached
        # claim" guarantee (see auth.py's module docstring).
        session_store.upsert_staff_account(
            username="test_user", password_hash=auth.hash_password("pw123"),
            display_name="Test User", allowed_groups=["it-department-group", "security-team-group"],
        )
        resolved_after_change = session_store.get_auth_session(token)
        if resolved_after_change["allowed_groups"] != ["it-department-group", "security-team-group"]:
            errors.append(
                "an existing login token did NOT pick up a changed group membership on its next "
                f"resolution -- got {resolved_after_change['allowed_groups']}, expected the updated list. "
                "This is exactly the stale-JWT-claim failure mode auth.py is designed to avoid."
            )

        session_store.delete_auth_session(token)
        if session_store.get_auth_session(token) is not None:
            errors.append("get_auth_session still returned a session after delete_auth_session")


def main() -> bool:
    checks = [
        ("unfiltered call unchanged", check_unfiltered_unchanged),
        ("restricted content excluded for an unauthorized caller", check_restricted_content_excluded),
        ("access_denial_info accuracy", check_access_denial_info_accurate),
        ("full-access caller unaffected", check_full_access_caller_unaffected),
        ("password hashing round-trip", check_password_hashing),
        ("staff account + auth session DB round-trip, including live group-change propagation",
         check_staff_account_roundtrip),
    ]

    errors: list = []
    for name, fn in checks:
        before = len(errors)
        fn(errors)
        status = "FAIL" if len(errors) > before else "ok"
        print(f"  [{status}] {name}")

    print()
    if errors:
        print(f"{len(errors)} PROBLEMS FOUND:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("All permission-filter/auth checks passed.")
    return len(errors) == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
