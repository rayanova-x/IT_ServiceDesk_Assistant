"""
Step 4.5 identity: IT-staff login + server-resolved permission groups.

Deliberately minimal, not OAuth/SSO -- this tool runs as a single local
pilot server (see PROJECT_HANDOFF.md), and a real enterprise SSO
integration is explicitly an AWS-migration-phase concern, not something
worth building against a toy user base now. What this DOES need to be
real about: group membership must never be something the CLIENT gets to
assert. An earlier version of this project's design let allowed_groups
flow in as a plain request parameter -- that's a real vulnerability (any
caller could claim any group), not just a simplification. Fixed by
resolving allowed_groups server-side from a login session token the
server itself issued, looked up fresh against staff_accounts on every
request (see session_store.get_auth_session()) rather than trusting a
signed-but-client-held claim (a JWT) that could drift from a real,
since-changed group membership until it expires.

Passwords: bcrypt (already in requirements.txt, no new dependency).
Session tokens: cryptographically random (secrets.token_urlsafe), stored
server-side in session_store.auth_sessions, set as an httpOnly cookie --
never readable or forgeable from JS, and a stolen token is useless after
its TTL or an explicit logout (session_store.delete_auth_session()).
"""

import secrets
from datetime import datetime, timedelta

import bcrypt

import session_store

SESSION_TTL_HOURS = 12
GUEST_SESSION_TTL_HOURS = 2  # shorter than a real staff session -- a demo
                              # visit is a few minutes, not a work shift

# Public-demo personas (2026-10-25): the login screen's "choose a persona"
# cards offer these four, mirroring seed_staff_accounts.py's real
# alice/bob/carol/dave roles/access exactly, so a demo visitor sees the
# SAME permission-filtering behavior a real staffer would. Kept here (not
# imported from seed_staff_accounts.py, which is a one-off CLI script)
# since this is the runtime source of truth the login endpoint reads.
GUEST_PERSONAS = {
    "alice": {"display_name": "Alice Chen", "role": "IT Support", "allowed_groups": ["it-department-group"]},
    "bob": {"display_name": "Bob Okafor", "role": "Security Team", "allowed_groups": ["security-team-group"]},
    "carol": {"display_name": "Carol Singh", "role": "IT Support", "allowed_groups": ["it-department-group"]},
    "dave": {"display_name": "Dave Martinez", "role": "Senior / Lead (full access)",
             "allowed_groups": ["it-department-group", "security-team-group"]},
}


def hash_password(plaintext: str) -> str:
    return bcrypt.hashpw(plaintext.encode(), bcrypt.gensalt()).decode()


def verify_password(plaintext: str, password_hash: str) -> bool:
    return bcrypt.checkpw(plaintext.encode(), password_hash.encode())


class LoginError(Exception):
    """Raised on bad credentials -- distinct from a generic exception so
    the login endpoint can return a clean 401, not a stack trace."""


def login(username: str, password: str) -> dict:
    """
    Verifies credentials against staff_accounts and issues a new session
    token. Returns {"token", "username", "display_name", "allowed_groups"}.
    Raises LoginError on an unknown username or wrong password -- the
    SAME error either way, so a login failure never discloses whether a
    given username exists (standard practice, not paranoia: this endpoint
    has no rate limiting yet, so username enumeration is a real exposure
    otherwise -- see the handoff doc's open-items note on this).
    """
    account = session_store.get_staff_account(username)
    if account is None or not verify_password(password, account["password_hash"]):
        raise LoginError("Invalid username or password.")

    token = secrets.token_urlsafe(32)
    expires_at = (datetime.now() + timedelta(hours=SESSION_TTL_HOURS)).isoformat()
    session_store.create_auth_session(token, username, expires_at)
    return {
        "token": token,
        "username": username,
        "display_name": account["display_name"],
        "allowed_groups": account["allowed_groups"],
    }


def resolve_session(token: str | None) -> dict | None:
    """
    Looks up a login session token and returns the CURRENT staff account
    row (fresh group membership, not a cached claim -- see module
    docstring), or None if the token is missing, unknown, or expired.
    Expiry is checked here (not just left to the DB row existing) since
    there's no background job sweeping expired rows in a project this
    size -- an expired-but-not-yet-cleaned-up row must still be rejected.
    """
    if not token:
        return None
    session = session_store.get_auth_session(token)
    if session is None:
        return None
    if datetime.fromisoformat(session["expires_at"]) < datetime.now():
        session_store.delete_auth_session(token)
        return None
    return session


def logout(token: str) -> None:
    session_store.delete_auth_session(token)


def guest_login(persona: str) -> dict:
    """
    Public-demo entry point (2026-10-25): no password, picks one of
    GUEST_PERSONAS and creates a brand-new, UNIQUE throwaway account for
    THIS visit (username = "<persona>-guest-<random>") rather than
    reusing a shared "alice" account across every visitor. This is what
    makes a demo visitor's session genuinely isolated: session_user_map,
    "My Conversations", the review queue, etc. already filter by exact
    username, so a unique-per-visit username gives each visitor their own
    empty slate with ZERO new filtering logic anywhere else in the app --
    they simply have never sent a message under this username before.

    is_guest=True on the created account is what server.py's admin-only
    endpoints (Finish Review, Run Golden Eval) check to refuse a demo
    visitor from mutating the real golden set / run log / review queue.

    Raises LoginError for an unknown persona key -- same error type as a
    bad password, so callers handle both the same way.
    """
    if persona not in GUEST_PERSONAS:
        raise LoginError(f"Unknown persona: {persona!r}")
    info = GUEST_PERSONAS[persona]
    guest_username = f"{persona}-guest-{secrets.token_hex(4)}"
    # A real (if never-used) bcrypt hash of a random password -- guest
    # accounts are never logged into by username/password, only via this
    # function, but staff_accounts.password_hash is NOT NULL and auth.py
    # treats "has no real password" as a distinct concept from "has an
    # empty/guessable one," so a genuine random hash is used, not a
    # placeholder string.
    throwaway_hash = hash_password(secrets.token_urlsafe(24))
    session_store.upsert_staff_account(
        username=guest_username, password_hash=throwaway_hash,
        display_name=info["display_name"], allowed_groups=info["allowed_groups"],
        is_guest=True,
    )
    token = secrets.token_urlsafe(32)
    expires_at = (datetime.now() + timedelta(hours=GUEST_SESSION_TTL_HOURS)).isoformat()
    session_store.create_auth_session(token, guest_username, expires_at)
    return {
        "token": token, "username": guest_username, "display_name": info["display_name"],
        "allowed_groups": info["allowed_groups"], "is_guest": True,
    }
