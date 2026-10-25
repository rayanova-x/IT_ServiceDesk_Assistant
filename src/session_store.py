"""
Production-style session/conversation persistence: SQLite (one file,
no server, built into Python -- no new dependency), storing every real
session/turn this tool ever handles, with the SAME deterministic
signals the eval scripts already compute logged automatically per turn.

WHY SQLITE OVER JSONL (2026-09-18, explicit user decision): every prior
eval script in this project used JSONL (data/processed/*.jsonl) --
appropriate for a fixed, offline eval set. Live production conversation
data needs to be queried by session, by date, by category, by whether a
knowledge gap fired, etc. -- a real embedded database, not another flat
file to grep through, matches how a real production system would store
this.

WHY THESE FIELDS, NOT MORE (sourced, not guessed): checked Anthropic's
own engineering docs first (anthropic.com/engineering/demystifying-evals-
for-ai-agents and .../multi-agent-research-system, both fetched directly
2026-09-18) for what a real production system logs. Two concrete,
transferable findings:
  1. Anthropic's own multi-agent production system explicitly monitors
     "agent decision patterns and interaction structures... without
     monitoring the contents of individual conversations, to maintain
     user privacy" -- this shaped the decision to log the SAME
     deterministic, already-computed signals every eval script in this
     project already tracks (category_used, low_confidence, mode,
     confidence score, whether a same-topic pair or knowledge gap
     fired) rather than adding a NEW per-turn LLM-judge call on every
     real conversation (that stays a deliberate, curated eval-sample
     activity -- see judge_generation_eval.py -- not a live-traffic cost).
  2. The standard unit Anthropic calls a "transcript/trace/trajectory"
     (outputs, tool calls, reasoning, intermediate results) is what
     `turns` + its FK'd `citations`/`superseded_alternatives` tables
     reconstruct for one turn -- the full pipeline decision trail, not
     just the final answer text.
HONEST GAP: neither Anthropic doc gives a concrete logging SCHEMA or a
PII/retention framework -- that part of this design (the actual table
shapes below, and treating `raw_text`/`rewritten_text` as the fields
most likely to contain PII) is reasoned from first principles, not
sourced. Flagged here rather than presented as if it were.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime

from config import SESSION_DB_PATH as DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id      TEXT PRIMARY KEY,
    started_at       TEXT NOT NULL,
    closed_at        TEXT,
    summary          TEXT,               -- set by close_session(), NULL while live
    case_facts_json  TEXT NOT NULL,       -- kept for backward compat with pre-existing rows;
                                          -- conversation_state_json (below) is now the source of
                                          -- truth and includes case_facts within it too
    conversation_state_json TEXT         -- full ConversationState.to_dict() snapshot (case_facts +
                                          -- past_sessions + current_turns), updated each turn -- see
                                          -- save_conversation_state()/load_conversation_state(). NULL
                                          -- for rows written before this existed (pre-2026-10-12);
                                          -- those sessions simply have nothing to rehydrate, same as
                                          -- a brand new session would.
);

CREATE TABLE IF NOT EXISTS turns (
    turn_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id        TEXT NOT NULL REFERENCES sessions(session_id),
    turn_index         INTEGER NOT NULL,   -- 0-based position within the session
    timestamp          TEXT NOT NULL,
    raw_text           TEXT NOT NULL,      -- what the user actually typed
    standalone_text    TEXT,               -- Step 0's rewrite, NULL if Step 0 was skipped
    used_step0         INTEGER NOT NULL,   -- 0/1
    stopped_at         TEXT NOT NULL,      -- "declined" | "ok" (was "off_topic"/"conversational" before 2026-10-24)
    sub_query          TEXT,               -- the search query for this sub-result (NULL if declined)
    category_used      TEXT,               -- Step 4's search_category (NULL if ambiguous/declined)
    wants_latest        INTEGER,           -- 0/1
    low_confidence      INTEGER,           -- 0/1
    top_rerank_score     REAL,              -- Step 8's confidence.top_score, NULL if no results
    mode                TEXT,               -- "synthesized" | "candidates_only" | NULL (declined)
    answer_text          TEXT,               -- NULL for candidates_only, set for synthesized/declined
    caveat               TEXT,
    knowledge_gap_logged  INTEGER NOT NULL DEFAULT 0,  -- 0/1, mirrors step8_confidence.py's own log
    same_topic_pair_detected INTEGER NOT NULL DEFAULT 0,  -- 0/1, len(superseded_alternatives) > 0
    citation_accuracy    REAL,  -- 0.0-1.0, NULL if nothing was cited or mode != synthesized;
                                -- see step9_generation.py's citation_accuracy_from() (deterministic,
                                -- no LLM call -- rag_evals.md 4.2's Citation Accuracy metric)
    latency_ms_json       TEXT,  -- {"step0_context": 12.3, "retrieval_rerank": 340.1, ...} per
                                 -- stage, from live_events.StageTimer -- NULL for pre-existing rows
    total_latency_ms      REAL   -- sum of latency_ms_json's values, denormalized for cheap AVG() queries
);

CREATE TABLE IF NOT EXISTS citations (
    citation_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    turn_id       INTEGER NOT NULL REFERENCES turns(turn_id),
    page_id       TEXT NOT NULL,
    title         TEXT NOT NULL,
    last_updated  TEXT NOT NULL,
    status        TEXT NOT NULL,
    header_path   TEXT  -- added 2026-10-24 for the citation side-panel (article_lookup.py):
                         -- which section of the article this citation's chunk came from
);

CREATE TABLE IF NOT EXISTS superseded_alternatives (
    alt_id                INTEGER PRIMARY KEY AUTOINCREMENT,
    turn_id               INTEGER NOT NULL REFERENCES turns(turn_id),
    current_page_id       TEXT NOT NULL,
    older_page_id         TEXT NOT NULL,
    older_title           TEXT NOT NULL,
    older_last_updated    TEXT NOT NULL,
    current_last_updated  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);
CREATE INDEX IF NOT EXISTS idx_turns_category ON turns(category_used);
CREATE INDEX IF NOT EXISTS idx_turns_low_confidence ON turns(low_confidence);
CREATE INDEX IF NOT EXISTS idx_citations_turn ON citations(turn_id);
CREATE INDEX IF NOT EXISTS idx_alts_turn ON superseded_alternatives(turn_id);

-- Step 4.5 identity: real IT support agent accounts, seeded by
-- seed_staff_accounts.py (bcrypt-hashed passwords, never plaintext).
-- allowed_groups_json is the source of truth check_confidence's
-- permission filter resolves server-side from the login session cookie
-- -- never trusted from client-supplied request data (see
-- auth.py/ui/server.py's login flow). This is a genuinely different
-- population from employees.csv (end users tickets are filed about,
-- not the IT staff using this tool) -- see PROJECT_HANDOFF.md's
-- "who the user is" note.
CREATE TABLE IF NOT EXISTS staff_accounts (
    username            TEXT PRIMARY KEY,
    password_hash       TEXT NOT NULL,       -- bcrypt, see auth.py
    display_name        TEXT NOT NULL,
    allowed_groups_json TEXT NOT NULL,       -- JSON list, e.g. ["it-department-group"]
    is_guest            INTEGER NOT NULL DEFAULT 0  -- 0/1, see auth.create_guest_account();
                                                     -- guests are denied admin-only eval actions
                                                     -- (Finish Review / Run Golden Eval) and are
                                                     -- real throwaway rows, one per demo visitor
);

-- Server-side login sessions -- a random opaque token in an httpOnly
-- cookie maps to a username here. Deliberately NOT a JWT: a JWT with
-- embedded group claims would let a client-visible (even if signed)
-- token drift from staff_accounts if a group membership changes before
-- the token expires; a server-side lookup on every request always
-- reflects the current, real row.
CREATE TABLE IF NOT EXISTS auth_sessions (
    token        TEXT PRIMARY KEY,
    username     TEXT NOT NULL REFERENCES staff_accounts(username),
    created_at   TEXT NOT NULL,
    expires_at   TEXT NOT NULL
);

-- Step 4.5 audit trail: every time the permission filter actually
-- excluded restricted content from a result, not just that a search
-- ran. This is the artifact a security reviewer would want before
-- signing off on a pilot touching restricted KB content -- who asked,
-- what was withheld, when.
CREATE TABLE IF NOT EXISTS access_denials (
    denial_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        TEXT NOT NULL,
    session_id       TEXT NOT NULL,
    username         TEXT,
    sub_query        TEXT NOT NULL,
    excluded_groups  TEXT NOT NULL,   -- comma-joined groups that caused an exclusion
    excluded_count   INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_auth_sessions_username ON auth_sessions(username);
CREATE INDEX IF NOT EXISTS idx_access_denials_session ON access_denials(session_id);

-- Deliberately SEPARATE from `sessions`/`turns` (2026-10-25 user decision):
-- in a real production deployment those tables should stay anonymous
-- (session_id only), so this mapping is its own table -- updatable
-- independently (e.g. to correct a misattributed session) without ever
-- requiring a username column on the transcript tables themselves.
CREATE TABLE IF NOT EXISTS session_user_map (
    session_id   TEXT PRIMARY KEY,
    username     TEXT NOT NULL,
    mapped_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_session_user_map_username ON session_user_map(username);
"""


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(SCHEMA)
        # Migration for DBs created before citation_accuracy existed (added
        # 2026-09-18) -- CREATE TABLE IF NOT EXISTS above is a no-op on an
        # existing turns table, so a real ALTER is needed for anyone
        # reopening a pre-existing sessions.db rather than a fresh one.
        existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(turns)")}
        if "citation_accuracy" not in existing_cols:
            conn.execute("ALTER TABLE turns ADD COLUMN citation_accuracy REAL")
        if "latency_ms_json" not in existing_cols:
            conn.execute("ALTER TABLE turns ADD COLUMN latency_ms_json TEXT")
        if "total_latency_ms" not in existing_cols:
            conn.execute("ALTER TABLE turns ADD COLUMN total_latency_ms REAL")
        existing_citation_cols = {row["name"] for row in conn.execute("PRAGMA table_info(citations)")}
        if "header_path" not in existing_citation_cols:
            conn.execute("ALTER TABLE citations ADD COLUMN header_path TEXT")
        existing_session_cols = {row["name"] for row in conn.execute("PRAGMA table_info(sessions)")}
        if "conversation_state_json" not in existing_session_cols:
            conn.execute("ALTER TABLE sessions ADD COLUMN conversation_state_json TEXT")
        existing_staff_cols = {row["name"] for row in conn.execute("PRAGMA table_info(staff_accounts)")}
        if "is_guest" not in existing_staff_cols:
            conn.execute("ALTER TABLE staff_accounts ADD COLUMN is_guest INTEGER NOT NULL DEFAULT 0")


def create_session(session_id: str, case_facts: dict) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO sessions (session_id, started_at, case_facts_json) VALUES (?, ?, ?)",
            (session_id, datetime.now().isoformat(), json.dumps(case_facts)),
        )


def update_session_facts(session_id: str, case_facts: dict) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE sessions SET case_facts_json = ? WHERE session_id = ?",
            (json.dumps(case_facts), session_id),
        )


def save_conversation_state(session_id: str, state_dict: dict) -> None:
    """
    Persists the FULL ConversationState snapshot (case_facts +
    past_sessions + current_turns -- see ConversationState.to_dict()),
    not just case_facts. Called after every turn (chat_service.py), same
    as update_session_facts() already was -- this is what makes a
    mid-ticket conversation survive a server restart: without this, only
    case_facts was ever durable, and current_turns (which Step 0's
    rewrite prompt actually reads) lived ONLY in ui/server.py's in-memory
    _live_sessions dict, gone the instant the process restarted.
    """
    with _connect() as conn:
        conn.execute(
            "UPDATE sessions SET conversation_state_json = ? WHERE session_id = ?",
            (json.dumps(state_dict), session_id),
        )


def load_conversation_state(session_id: str) -> dict | None:
    """
    Returns the last-saved full ConversationState snapshot for a session,
    or None if the session doesn't exist yet or was written before this
    column existed (pre-2026-10-12 rows have conversation_state_json =
    NULL) -- callers (ui/server.py) treat None the same as "no prior
    state," i.e. start a fresh ConversationState, exactly the pre-existing
    behavior for a session this couldn't rehydrate.
    """
    with _connect() as conn:
        row = conn.execute(
            "SELECT conversation_state_json FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None or row["conversation_state_json"] is None:
            return None
        return json.loads(row["conversation_state_json"])


def close_session(session_id: str, summary: str) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE sessions SET closed_at = ?, summary = ? WHERE session_id = ?",
            (datetime.now().isoformat(), summary, session_id),
        )


def next_turn_index(session_id: str) -> int:
    with _connect() as conn:
        row = conn.execute(
            "SELECT COALESCE(MAX(turn_index), -1) + 1 AS next_idx FROM turns WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return row["next_idx"]


def log_turn(
    session_id: str,
    raw_text: str,
    standalone_text: str | None,
    used_step0: bool,
    stopped_at: str,
    sub_query: str | None = None,
    category_used: str | None = None,
    wants_latest: bool | None = None,
    low_confidence: bool | None = None,
    top_rerank_score: float | None = None,
    mode: str | None = None,
    answer_text: str | None = None,
    caveat: str | None = None,
    knowledge_gap_logged: bool = False,
    citations: list[dict] | None = None,
    superseded_alternatives: list[dict] | None = None,
    citation_accuracy: float | None = None,
    latency_ms: dict | None = None,
) -> int:
    """Logs one turn (one sub-result -- a compound query with N
    sub-questions produces N calls to this function, same convention as
    PipelineResult). Returns the new turn_id."""
    turn_index = next_turn_index(session_id)
    same_topic_pair_detected = bool(superseded_alternatives)
    total_latency_ms = sum(latency_ms.values()) if latency_ms else None
    with _connect() as conn:
        cur = conn.execute(
            """INSERT INTO turns (
                session_id, turn_index, timestamp, raw_text, standalone_text, used_step0,
                stopped_at, sub_query, category_used, wants_latest, low_confidence,
                top_rerank_score, mode, answer_text, caveat, knowledge_gap_logged,
                same_topic_pair_detected, citation_accuracy, latency_ms_json, total_latency_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id, turn_index, datetime.now().isoformat(), raw_text, standalone_text,
                int(used_step0), stopped_at, sub_query, category_used,
                None if wants_latest is None else int(wants_latest),
                None if low_confidence is None else int(low_confidence),
                top_rerank_score, mode, answer_text, caveat, int(knowledge_gap_logged),
                int(same_topic_pair_detected), citation_accuracy,
                json.dumps(latency_ms) if latency_ms else None, total_latency_ms,
            ),
        )
        turn_id = cur.lastrowid
        for c in (citations or []):
            conn.execute(
                "INSERT INTO citations (turn_id, page_id, title, last_updated, status, header_path) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (turn_id, c["page_id"], c["title"], c["last_updated"], c["status"], c.get("header_path", "")),
            )
        for alt in (superseded_alternatives or []):
            conn.execute(
                """INSERT INTO superseded_alternatives
                   (turn_id, current_page_id, older_page_id, older_title, older_last_updated, current_last_updated)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (turn_id, alt["current_page_id"], alt["older_page_id"], alt["older_title"],
                 alt["older_last_updated"], alt["current_last_updated"]),
            )
        return turn_id


def get_session_turns(session_id: str) -> list[dict]:
    """
    Returns every turn for this session, each with its citations attached
    as a `citations` list -- added 2026-10-24 for the citation side-panel
    (previously this returned the raw `turns` row only, with no citation
    data at all, so a resumed/reopened ticket session had no way to
    render clickable citation chips even though a fresh live turn did).
    """
    with _connect() as conn:
        turn_rows = conn.execute(
            "SELECT * FROM turns WHERE session_id = ? ORDER BY turn_index ASC",
            (session_id,),
        ).fetchall()
        turns = [dict(r) for r in turn_rows]
        if not turns:
            return turns

        turn_ids = [t["turn_id"] for t in turns]
        placeholders = ",".join("?" * len(turn_ids))
        citation_rows = conn.execute(
            f"SELECT turn_id, page_id, title, last_updated, status, header_path "
            f"FROM citations WHERE turn_id IN ({placeholders})",
            turn_ids,
        ).fetchall()
        citations_by_turn: dict[int, list[dict]] = {}
        for r in citation_rows:
            citations_by_turn.setdefault(r["turn_id"], []).append({
                "page_id": r["page_id"], "title": r["title"], "last_updated": r["last_updated"],
                "status": r["status"], "header_path": r["header_path"] or "",
            })
        for t in turns:
            t["citations"] = citations_by_turn.get(t["turn_id"], [])
        return turns


def list_sessions(limit: int = 50) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """SELECT s.session_id, s.started_at, s.closed_at, s.summary,
                      COUNT(t.turn_id) AS n_turns
               FROM sessions s LEFT JOIN turns t ON t.session_id = s.session_id
               GROUP BY s.session_id ORDER BY s.started_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def list_ticket_sessions(issue_key: str) -> list[dict]:
    """
    All sessions for ONE ticket -- added 2026-10-25 for the "start a new
    session on this ticket" feature (chat_service.ticket_session_id()'s
    session_number parameter). Matches both the unsuffixed original
    ("jira:<KEY>") and every numbered one ("jira:<KEY>::2", "::3", ...)
    via a LIKE prefix match, then filters out any OTHER ticket whose key
    happens to be a prefix of this one (e.g. "jira:IH-1" must not also
    match a real "jira:IH-10" session) by checking the suffix shape
    explicitly rather than trusting the SQL LIKE alone.
    """
    base = f"jira:{issue_key}"
    with _connect() as conn:
        rows = conn.execute(
            """SELECT s.session_id, s.started_at, s.closed_at, s.summary,
                      COUNT(t.turn_id) AS n_turns
               FROM sessions s LEFT JOIN turns t ON t.session_id = s.session_id
               WHERE s.session_id = ? OR s.session_id LIKE ?
               GROUP BY s.session_id ORDER BY s.started_at ASC""",
            (base, f"{base}::%"),
        ).fetchall()
    results = []
    for r in rows:
        sid = r["session_id"]
        if sid == base:
            results.append({**dict(r), "session_number": 1})
        else:
            suffix = sid[len(base) + 2:]  # strip "jira:<KEY>::"
            if suffix.isdigit():
                results.append({**dict(r), "session_number": int(suffix)})
    results.sort(key=lambda r: r["session_number"])
    return results


def map_session_to_user(session_id: str, username: str) -> None:
    """Records (or corrects) which staffer a session belongs to, in the
    separate session_user_map table -- see its schema comment for why
    this isn't a column on `sessions` itself. REPLACE so re-mapping an
    existing session_id (e.g. fixing a mistake) is a normal upsert."""
    with _connect() as conn:
        conn.execute(
            """INSERT INTO session_user_map (session_id, username, mapped_at)
               VALUES (?, ?, ?)
               ON CONFLICT(session_id) DO UPDATE SET
                 username = excluded.username, mapped_at = excluded.mapped_at""",
            (session_id, username, datetime.now().isoformat()),
        )


def get_session_user(session_id: str) -> str | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT username FROM session_user_map WHERE session_id = ?", (session_id,)
        ).fetchone()
        return row["username"] if row else None


def list_sessions_for_user(username: str, limit: int = 50) -> list[dict]:
    """Same shape as list_sessions(), filtered to one staffer's own
    sessions via a join against session_user_map -- backs "My
    Conversations" actually meaning "mine," not every session in the DB."""
    with _connect() as conn:
        rows = conn.execute(
            """SELECT s.session_id, s.started_at, s.closed_at, s.summary,
                      COUNT(t.turn_id) AS n_turns
               FROM sessions s
               JOIN session_user_map m ON m.session_id = s.session_id
               LEFT JOIN turns t ON t.session_id = s.session_id
               WHERE m.username = ?
               GROUP BY s.session_id ORDER BY s.started_at DESC LIMIT ?""",
            (username, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def get_session_case_facts(session_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT case_facts_json FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        return json.loads(row["case_facts_json"]) if row else None


# ---------------------------------------------------------------------------
# Step 4.5 identity: staff accounts, login sessions, access-denial audit.
# ---------------------------------------------------------------------------

def upsert_staff_account(username: str, password_hash: str, display_name: str,
                          allowed_groups: list[str], is_guest: bool = False) -> None:
    """Used by seed_staff_accounts.py (real accounts) and
    auth.create_guest_account() (throwaway demo-visitor accounts, one per
    guest login -- see its own docstring). REPLACE, not INSERT OR IGNORE --
    re-running the seed script should refresh an existing account (e.g. a
    changed group list) rather than silently no-op."""
    with _connect() as conn:
        conn.execute(
            """INSERT INTO staff_accounts (username, password_hash, display_name, allowed_groups_json, is_guest)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(username) DO UPDATE SET
                 password_hash = excluded.password_hash,
                 display_name = excluded.display_name,
                 allowed_groups_json = excluded.allowed_groups_json,
                 is_guest = excluded.is_guest""",
            (username, password_hash, display_name, json.dumps(allowed_groups), int(is_guest)),
        )


def get_staff_account(username: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM staff_accounts WHERE username = ?", (username,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["allowed_groups"] = json.loads(d.pop("allowed_groups_json"))
        d["is_guest"] = bool(d["is_guest"])
        return d


def create_auth_session(token: str, username: str, expires_at: str) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO auth_sessions (token, username, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (token, username, datetime.now().isoformat(), expires_at),
        )


def get_auth_session(token: str) -> dict | None:
    """Returns the logged-in staff account for a login token, joined
    live against staff_accounts -- so a group-membership change takes
    effect on the account's NEXT request, not just its next login (see
    schema docstring: this is why login sessions aren't a self-contained
    JWT with embedded claims)."""
    with _connect() as conn:
        row = conn.execute(
            """SELECT a.token, a.username, a.expires_at, s.display_name, s.allowed_groups_json, s.is_guest
               FROM auth_sessions a JOIN staff_accounts s ON s.username = a.username
               WHERE a.token = ?""",
            (token,),
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["allowed_groups"] = json.loads(d.pop("allowed_groups_json"))
        d["is_guest"] = bool(d["is_guest"])
        return d


def delete_auth_session(token: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM auth_sessions WHERE token = ?", (token,))


def log_access_denial(session_id: str, username: str | None, sub_query: str,
                       excluded_groups: list[str], excluded_count: int) -> None:
    with _connect() as conn:
        conn.execute(
            """INSERT INTO access_denials
               (timestamp, session_id, username, sub_query, excluded_groups, excluded_count)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (datetime.now().isoformat(), session_id, username, sub_query,
             ",".join(excluded_groups), excluded_count),
        )


def recent_access_denials(limit: int = 20) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM access_denials ORDER BY denial_id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Production-monitoring-style aggregate queries -- the "decision patterns
# and structure" signals Anthropic's own engineering docs describe
# tracking, computed from what's already logged, no new LLM calls.
# ---------------------------------------------------------------------------

def monitoring_summary(since: str | None = None) -> dict:
    """Aggregate signals across all logged turns (optionally since an
    ISO timestamp) -- category distribution, low-confidence rate,
    knowledge-gap rate, same-topic-pair detection rate. Mirrors the same
    metrics eval_pipeline_e2e.py/calibrate_confidence.py already compute
    on the curated eval set, but over REAL logged traffic instead."""
    with _connect() as conn:
        where = "WHERE timestamp >= ?" if since else ""
        params = (since,) if since else ()
        total = conn.execute(f"SELECT COUNT(*) AS n FROM turns {where}", params).fetchone()["n"]
        if total == 0:
            return {"total_turns": 0}

        low_conf = conn.execute(
            f"SELECT COUNT(*) AS n FROM turns {where}{' AND' if where else 'WHERE'} low_confidence = 1",
            params,
        ).fetchone()["n"]
        gaps = conn.execute(
            f"SELECT COUNT(*) AS n FROM turns {where}{' AND' if where else 'WHERE'} knowledge_gap_logged = 1",
            params,
        ).fetchone()["n"]
        pairs = conn.execute(
            f"SELECT COUNT(*) AS n FROM turns {where}{' AND' if where else 'WHERE'} same_topic_pair_detected = 1",
            params,
        ).fetchone()["n"]
        # "declined" replaced "off_topic"/"conversational" 2026-10-24 --
        # see pipeline.py's PipelineOutcome.stopped_at comment. Dict key
        # kept as off_topic_rate for API/dashboard continuity, but now
        # means "direct reply, no search" broadly, not specifically
        # off-topic content.
        declined = conn.execute(
            f"SELECT COUNT(*) AS n FROM turns {where}{' AND' if where else 'WHERE'} stopped_at = 'declined'",
            params,
        ).fetchone()["n"]
        errors = conn.execute(
            f"SELECT COUNT(*) AS n FROM turns {where}{' AND' if where else 'WHERE'} stopped_at = 'error'",
            params,
        ).fetchone()["n"]
        denials = conn.execute("SELECT COUNT(*) AS n FROM access_denials").fetchone()["n"]
        latency_row = conn.execute(
            f"""SELECT AVG(total_latency_ms) AS avg_ms, MAX(total_latency_ms) AS max_ms,
                       COUNT(total_latency_ms) AS n_timed
                FROM turns {where}{' AND' if where else 'WHERE'} total_latency_ms IS NOT NULL""",
            params,
        ).fetchone()
        by_category = conn.execute(
            f"""SELECT category_used, COUNT(*) AS n FROM turns {where}
                {'AND' if where else 'WHERE'} category_used IS NOT NULL
                GROUP BY category_used ORDER BY n DESC""",
            params,
        ).fetchall()
        cit_row = conn.execute(
            f"""SELECT AVG(citation_accuracy) AS avg_acc, COUNT(citation_accuracy) AS n_cited
                FROM turns {where}{' AND' if where else 'WHERE'} citation_accuracy IS NOT NULL""",
            params,
        ).fetchone()
        by_mode = conn.execute(
            f"""SELECT mode, COUNT(*) AS n FROM turns {where}
                {'AND' if where else 'WHERE'} mode IS NOT NULL
                GROUP BY mode ORDER BY n DESC""",
            params,
        ).fetchall()

        return {
            "total_turns": total,
            "low_confidence_rate": round(low_conf / total, 3),
            "knowledge_gap_rate": round(gaps / total, 3),
            "same_topic_pair_rate": round(pairs / total, 3),
            "off_topic_rate": round(declined / total, 3),
            "error_rate": round(errors / total, 3),
            "access_denial_count": denials,
            "by_category": {r["category_used"]: r["n"] for r in by_category},
            "by_mode": {r["mode"]: r["n"] for r in by_mode},
            "mean_citation_accuracy": round(cit_row["avg_acc"], 3) if cit_row["avg_acc"] is not None else None,
            "n_turns_with_citations": cit_row["n_cited"],
            "mean_latency_ms": round(latency_row["avg_ms"], 1) if latency_row["avg_ms"] is not None else None,
            "max_latency_ms": round(latency_row["max_ms"], 1) if latency_row["max_ms"] is not None else None,
            "n_turns_timed": latency_row["n_timed"],
        }


def latency_breakdown(limit: int = 200) -> dict:
    """
    Average per-STAGE latency across the most recent `limit` timed turns
    -- answers "where is time going" (embedding/rerank vs. the Gemini
    calls), the dashboard panel most directly demonstrating this was
    engineered with real instrumentation, not just a response-time number.
    latency_ms_json's keys vary turn to turn (off_topic-stopped turns
    never reach retrieval, for instance), so this aggregates in Python
    rather than trying to pivot varying JSON keys in SQL.
    """
    with _connect() as conn:
        rows = conn.execute(
            """SELECT latency_ms_json FROM turns
               WHERE latency_ms_json IS NOT NULL
               ORDER BY turn_id DESC LIMIT ?""",
            (limit,),
        ).fetchall()

    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in rows:
        stages = json.loads(row["latency_ms_json"])
        for stage, ms in stages.items():
            sums[stage] = sums.get(stage, 0.0) + ms
            counts[stage] = counts.get(stage, 0) + 1

    return {
        "n_turns": len(rows),
        "stages": {
            stage: round(sums[stage] / counts[stage], 1) for stage in sums
        },
    }


def monitoring_timeseries(bucket: str = "day", limit: int = 30) -> list[dict]:
    """
    Per-bucket (default: per-day) breakdown of the same rates
    monitoring_summary() computes all-time -- what a real observability
    dashboard actually needs per rag_evals2.md's "Online evals" section
    ("track moving averages... over 7-day windows... catch silent
    regressions offline evals miss"). Buckets by the turn's calendar
    date (from its ISO timestamp's first 10 chars -- SQLite has no
    native DATE type, this is the standard substr trick), ordered
    oldest-first so a line chart renders left-to-right chronologically.
    Returns the most recent `limit` buckets that have at least one turn
    (no zero-filled gaps for days with no traffic -- a dashboard showing
    a flat 0% "low-confidence rate" on a day with zero turns would be
    misleading, so that day is simply absent rather than shown as 0).
    """
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT
                substr(timestamp, 1, 10) AS day,
                COUNT(*) AS total_turns,
                AVG(low_confidence) AS low_confidence_rate,
                AVG(knowledge_gap_logged) AS knowledge_gap_rate,
                AVG(same_topic_pair_detected) AS same_topic_pair_rate,
                AVG(CASE WHEN stopped_at = 'declined' THEN 1.0 ELSE 0.0 END) AS off_topic_rate,
                AVG(citation_accuracy) AS mean_citation_accuracy
            FROM turns
            GROUP BY day
            ORDER BY day DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    result = [
        {
            "day": r["day"],
            "total_turns": r["total_turns"],
            "low_confidence_rate": round(r["low_confidence_rate"], 3),
            "knowledge_gap_rate": round(r["knowledge_gap_rate"], 3),
            "same_topic_pair_rate": round(r["same_topic_pair_rate"], 3),
            "off_topic_rate": round(r["off_topic_rate"], 3),
            "mean_citation_accuracy": (
                round(r["mean_citation_accuracy"], 3) if r["mean_citation_accuracy"] is not None else None
            ),
        }
        for r in rows
    ]
    result.reverse()  # oldest first, see docstring
    return result


if __name__ == "__main__":
    init_db()
    print(f"Initialized {DB_PATH}")
    print(monitoring_summary())
