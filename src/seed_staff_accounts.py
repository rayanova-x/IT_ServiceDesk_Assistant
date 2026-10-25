"""
Seeds a small, fixed set of fictional IT-support staff accounts for
local demo/pilot use -- see auth.py for why this isn't OAuth/SSO, and
PROJECT_HANDOFF.md for the "who the real user is" note (IT support
staff, not end employees -- a genuinely different population from
employees.csv).

Four FICTIONAL accounts, matching the two real restricted_to groups that
already exist in the KB data (it-department-group, security-team-group):
  - alice / carol: general IT support staff, it-department-group only.
  - bob: security-team-group only (a narrower, security-focused role --
    included to prove the filter also withholds general IT content from
    someone WITHOUT it-department-group, not just the reverse).
  - dave: both groups (a senior/lead role with full access).
These four are also the public login screen's persona-picker source
(auth.GUEST_PERSONAS) -- a demo visitor who clicks a card gets a
brand-new throwaway account cloning that persona's access, never this
real row itself (see auth.guest_login()'s docstring).

Plus ONE real account for the project owner (2026-10-25): "admin," full
access, distinct from the four persona names specifically so there's no
ambiguity between "my own real session" and "a demo visitor who typed
dave's name into the sign-in form instead of clicking his card." Signing
in as admin (or alice/bob/carol/dave typed directly, not via a persona
card) is a REAL, persistent staff login -- not a guest account, so it's
the only kind of login allowed to use "Finish Review"/"Run Golden Eval"
(see server.py's _require_non_guest()).

Passwords are intentionally simple -- this is seed data for a local
pilot, not real production credentials. Re-running this script is safe
and idempotent: upsert_staff_account() REPLACEs each row, so updating
this file's group list and re-running it is how you'd change an
account's access, not a manual SQL edit.
"""

import auth
import session_store

STAFF_ACCOUNTS = [
    {"username": "alice", "password": "demo1234", "display_name": "Alice Chen",
     "allowed_groups": ["it-department-group"]},
    {"username": "bob", "password": "demo1234", "display_name": "Bob Okafor",
     "allowed_groups": ["security-team-group"]},
    {"username": "carol", "password": "demo1234", "display_name": "Carol Singh",
     "allowed_groups": ["it-department-group"]},
    {"username": "dave", "password": "demo1234", "display_name": "Dave Martinez",
     "allowed_groups": ["it-department-group", "security-team-group"]},
    {"username": "admin", "password": "demo4321", "display_name": "Admin",
     "allowed_groups": ["it-department-group", "security-team-group"]},
]


def main() -> None:
    session_store.init_db()
    for acct in STAFF_ACCOUNTS:
        session_store.upsert_staff_account(
            username=acct["username"],
            password_hash=auth.hash_password(acct["password"]),
            display_name=acct["display_name"],
            allowed_groups=acct["allowed_groups"],
        )
        print(f"Seeded {acct['username']} ({acct['display_name']}) -> {acct['allowed_groups']}")


if __name__ == "__main__":
    main()
