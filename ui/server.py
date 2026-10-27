"""
Local web UI backend for the IT Helpdesk RAG Assistant -- wraps
src/chat_service.py (which wraps the real pipeline) behind a small
FastAPI server, so the chatbot can be used interactively through a
browser instead of only via test scripts.

No new pipeline/business logic here -- pure API wiring, same convention
as pipeline.py and chat_service.py ("this module is ONLY the wiring").
PipelineClients (embedding model, reranker, Chroma, Gemini client) are
built ONCE at server startup and reused across every request -- these
are expensive to construct (model loads) and were always meant to be
built once per process, per PipelineClients' own docstring.

ConversationState objects live in an in-memory dict keyed by session_id
for the lifetime of the server process (NOT persisted across a server
restart -- session_store.py's SQLite database is the durable record;
this in-memory dict is only the live working state needed to keep
Step 0's case_facts/current_turns available between requests within one
run of the server). A real production deployment would need a proper
session store (Redis, etc.) for this -- out of scope for a local,
single-user tool.

Run: python ui/server.py
Open: http://localhost:8010 (8000 was already in use by another process
on this machine when this was built -- picked a free port instead of
touching whatever else was running there)
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

UI_DIR = Path(__file__).parent
PROJECT_ROOT = UI_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import asyncio  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import uuid  # noqa: E402
from datetime import datetime  # noqa: E402

from fastapi import FastAPI, HTTPException, Request, Response  # noqa: E402
from fastapi.responses import FileResponse, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from logging_setup import configure_logging  # noqa: E402
configure_logging()
logger = logging.getLogger("ui.server")

import auth  # noqa: E402
import jira_client  # noqa: E402
import live_events  # noqa: E402
import session_store  # noqa: E402
from chat_service import (  # noqa: E402
    ChatResponse,
    new_session_id,
    send_message,
    load_ticket_context,
    stream_message,
    ticket_session_id,
)
from pipeline import PipelineClients  # noqa: E402
from step0_context import ConversationState  # noqa: E402
from article_lookup import get_article, list_accessible_pages, search_pages  # noqa: E402
import golden_set_manager  # noqa: E402

AUTH_COOKIE_NAME = "it_helpdesk_session"


def _get_or_restore_conversation_state(session_id: str) -> ConversationState:
    """
    Returns this session's live ConversationState, rehydrating it from
    session_store.py's durable snapshot if this server process doesn't
    already have it in memory -- the real fix for the documented
    limitation in this module's own docstring ("NOT persisted across a
    server restart"): a server bounce used to mean every in-progress
    ticket conversation silently lost its case_facts/current_turns and
    Step 0 started rewriting as if it were a brand first message, even
    though session_store.db still had the real history the whole time.
    A session that was never logged before (a genuinely new session_id)
    has nothing to restore -- load_conversation_state() returns None,
    and this falls back to a fresh ConversationState(), identical to the
    pre-existing behavior for a brand new session.
    """
    if session_id in _live_sessions:
        return _live_sessions[session_id]
    saved = session_store.load_conversation_state(session_id)
    state = ConversationState.from_dict(saved) if saved is not None else ConversationState()
    _live_sessions[session_id] = state
    return state


def _require_login(request: Request) -> dict:
    """
    Resolves the CURRENT logged-in staff account from the httpOnly
    session cookie -- never from any client-supplied request field.
    This is the ONLY place allowed_groups is allowed to originate from
    for a real chat request; every endpoint below that touches the
    pipeline calls this rather than accepting an allowed_groups field on
    SendMessageRequest (which would let any caller simply claim full
    access). Raises 401 if there's no valid session -- callers must log
    in via POST /api/login first.
    """
    token = request.cookies.get(AUTH_COOKIE_NAME)
    session = auth.resolve_session(token)
    if session is None:
        raise HTTPException(status_code=401, detail="Not logged in.")
    return session


def _require_non_guest(staff: dict) -> None:
    """
    Blocks a demo-persona guest login (auth.guest_login()) from actions
    that mutate REAL shared state -- the golden set, its version history,
    and the run log. A public deploy's demo visitors must never be able
    to promote junk cases into the actual golden set or burn real
    embedding/reranker calls via "Run Golden Eval." Call AFTER
    _require_login(staff) on any such endpoint.
    """
    if staff.get("is_guest"):
        raise HTTPException(status_code=403, detail="Demo accounts can't modify the golden set or run evals.")

# Live ConversationState per session, for this server process's lifetime.
# See module docstring -- session_store.db is the durable record, this is
# only the live Step 0 working memory.
_live_sessions: dict[str, ConversationState] = {}
_clients: PipelineClients | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _clients
    session_store.init_db()
    print("Loading pipeline clients (embedding model, reranker, Chroma, Gemini)...")
    _clients = PipelineClients.build()
    print("Ready.")
    yield


app = FastAPI(title="IT Helpdesk RAG Assistant", lifespan=lifespan)


class SendMessageRequest(BaseModel):
    session_id: str | None = None  # None -> start a new session
    message: str


class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/api/login")
def post_login(req: LoginRequest, response: Response):
    try:
        session = auth.login(req.username, req.password)
    except auth.LoginError as e:
        logger.warning("login failed for username=%s", req.username, extra={"username": req.username})
        raise HTTPException(status_code=401, detail=str(e))
    response.set_cookie(
        AUTH_COOKIE_NAME, session["token"],
        httponly=True, samesite="lax", max_age=auth.SESSION_TTL_HOURS * 3600,
    )
    logger.info("login succeeded", extra={"username": session["username"]})
    live_events.broadcast("login", {"username": session["username"], "display_name": session["display_name"]})
    return {
        "username": session["username"],
        "display_name": session["display_name"],
        "allowed_groups": session["allowed_groups"],
    }


class GuestLoginRequest(BaseModel):
    persona: str  # one of auth.GUEST_PERSONAS' keys


@app.get("/api/personas")
def list_personas():
    """
    Backs the login screen's persona-picker cards -- public, no auth
    (a visitor hasn't logged in yet, that's the whole point). Returns
    display_name/role/allowed_groups for each of the four demo personas
    so the UI never hardcodes this list separately from auth.py's own
    source of truth.
    """
    return [
        {"persona": key, "display_name": info["display_name"], "role": info["role"],
         "allowed_groups": info["allowed_groups"]}
        for key, info in auth.GUEST_PERSONAS.items()
    ]


@app.post("/api/login/guest")
def post_guest_login(req: GuestLoginRequest, response: Response):
    """
    Public-demo entry point: no password, creates a brand-new isolated
    account for THIS visit -- see auth.guest_login()'s docstring for why
    this gives genuine per-visitor isolation with no new filtering logic
    anywhere else. Shorter cookie TTL than a real staff login
    (GUEST_SESSION_TTL_HOURS) since a demo visit is brief.
    """
    try:
        session = auth.guest_login(req.persona)
    except auth.LoginError as e:
        raise HTTPException(status_code=400, detail=str(e))
    response.set_cookie(
        AUTH_COOKIE_NAME, session["token"],
        httponly=True, samesite="lax", max_age=auth.GUEST_SESSION_TTL_HOURS * 3600,
    )
    logger.info("guest login succeeded", extra={"username": session["username"]})
    live_events.broadcast("login", {"username": session["username"], "display_name": session["display_name"]})
    return {
        "username": session["username"],
        "display_name": session["display_name"],
        "allowed_groups": session["allowed_groups"],
        "is_guest": True,
    }


@app.post("/api/logout")
def post_logout(request: Request, response: Response):
    token = request.cookies.get(AUTH_COOKIE_NAME)
    if token:
        auth.logout(token)
    response.delete_cookie(AUTH_COOKIE_NAME)
    return {"ok": True}


@app.get("/api/me")
def get_me(request: Request):
    """Lets the frontend check login state on page load without
    triggering a 401-driven redirect loop -- returns {"logged_in": False}
    instead of raising when there's no valid session."""
    token = request.cookies.get(AUTH_COOKIE_NAME)
    session = auth.resolve_session(token)
    if session is None:
        return {"logged_in": False}
    return {
        "logged_in": True,
        "username": session["username"],
        "display_name": session["display_name"],
        "allowed_groups": session["allowed_groups"],
        "is_guest": session["is_guest"],
    }


class TurnPayload(BaseModel):
    sub_query: str
    mode: str | None
    answer_text: str | None
    caveat: str | None
    staleness_disclosure: str
    superseded_alternatives: list[dict]
    low_confidence: bool | None
    citations: list[dict] = []  # {"page_id", "title", "header_path", "last_updated", "status"}


class SendMessageResponse(BaseModel):
    session_id: str
    stopped_at: str
    turns: list[TurnPayload]
    error_message: str | None = None
    message: str | None = None  # set for stopped_at "declined"


def _serialize(resp: ChatResponse, session_id: str) -> SendMessageResponse:
    return SendMessageResponse(
        session_id=session_id,
        stopped_at=resp.stopped_at,
        error_message=resp.error_message,
        message=resp.message,
        turns=[
            TurnPayload(
                sub_query=t.sub_query,
                mode=t.mode,
                answer_text=t.answer_text,
                caveat=t.caveat,
                staleness_disclosure=t.staleness_disclosure,
                # Already plain dicts as of 2026-10-24's streaming
                # refactor -- ChatTurnDisplay.superseded_alternatives now
                # comes from stream_message()'s own event dicts (built
                # once there, not re-derived from SupersededAlternative
                # objects at this layer), so no attribute-to-dict
                # conversion is needed here anymore.
                superseded_alternatives=t.superseded_alternatives,
                low_confidence=t.low_confidence,
                citations=t.citations,
            )
            for t in resp.turns
        ],
    )


@app.post("/api/messages", response_model=SendMessageResponse)
def post_message(req: SendMessageRequest, request: Request):
    staff = _require_login(request)
    session_id = req.session_id or new_session_id()
    session_store.map_session_to_user(session_id, staff["username"])
    state = _get_or_restore_conversation_state(session_id)

    resp = send_message(session_id, req.message, state, _clients,
                         allowed_groups=set(staff["allowed_groups"]), username=staff["username"])
    return _serialize(resp, session_id)


class ArticlePayload(BaseModel):
    page_id: str
    title: str
    html: str
    status: str
    last_updated: str
    matched_anchor: str | None


@app.get("/api/articles/{page_id}", response_model=ArticlePayload)
def get_article_content(page_id: str, request: Request, header_path: str | None = None):
    """
    Backs the citation side-panel: full article HTML (heading-anchored)
    for a clicked citation chip. header_path (query param, from the
    citation the staffer clicked) picks which section to scroll/
    highlight to -- see article_lookup.py for the anchor-matching logic.

    Server-side permission check is NOT optional here, even though a
    real UI client only ever shows citations that already survived Step
    4.5's filter during retrieval -- this endpoint accepts an arbitrary
    page_id directly, so a malicious or buggy client requesting a
    page_id it was never shown must still be denied, same as retrieval
    itself would deny it.
    """
    staff = _require_login(request)
    article = get_article(page_id, header_path, allowed_groups=set(staff["allowed_groups"]))
    if article is None:
        raise HTTPException(status_code=404, detail="Article not found or not accessible")
    return ArticlePayload(
        page_id=article.page_id, title=article.title, html=article.html,
        status=article.status, last_updated=article.last_updated,
        matched_anchor=article.matched_anchor,
    )


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _forward_as_sse(events):
    """
    Converts chat_service.py's own event-dict generator (stream_message())
    directly into SSE wire format, one-to-one -- used by
    post_message_stream() below.

    REAL STREAMING, added 2026-10-24 -- this used to chunk an
    ALREADY-COMPLETE answer into fixed-size pieces purely for visual
    effect, because generate_answer() ran synthesis to completion before
    anything could be shown. Now chat_service.py's stream_message()
    yields turn_delta events AS Gemini actually produces them (see
    step9_generation.py's generate_answer_streaming() and its own
    docstring for why this is safe: the verify-then-retract faithfulness
    check is a single prompt instruction within the one generation call,
    not a second round-trip, and direct testing confirmed streamed
    chunks contain only the final, already-verified text -- see
    PROJECT_HANDOFF.md for the full investigation). This function is now
    pure wire-format translation, no chunking logic of its own.
    """
    for event in events:
        event_type = event.pop("type")
        yield _sse_event(event_type, event)


@app.post("/api/messages/stream")
def post_message_stream(req: SendMessageRequest, request: Request):
    """Same request/response contract as POST /api/messages, but
    delivered as Server-Sent Events with REAL token-level streaming --
    see _forward_as_sse() and chat_service.stream_message()."""
    staff = _require_login(request)
    session_id = req.session_id or new_session_id()
    session_store.map_session_to_user(session_id, staff["username"])
    state = _get_or_restore_conversation_state(session_id)

    events = stream_message(session_id, req.message, state, _clients,
                             allowed_groups=set(staff["allowed_groups"]), username=staff["username"])
    return StreamingResponse(_forward_as_sse(events), media_type="text/event-stream")


class TicketSummary(BaseModel):
    key: str
    summary: str
    status: str
    session_id: str
    has_session: bool  # True if this ticket has been opened before (resumable)


@app.get("/api/tickets", response_model=list[TicketSummary])
def get_tickets(request: Request):
    """
    Lists open tickets from the real Jira project (see jira_client.py --
    JQL-filtered to statusCategory != Done, newest-updated first, same
    as a real agent queue view), each flagged with whether it already
    has a chat session (i.e. has been ACCEPTED before -- see
    TicketSummary.has_session) so the sidebar can distinguish
    not-yet-accepted vs. already-accepted/in-progress tickets.
    """
    _require_login(request)
    issues = jira_client.list_open_issues()
    tickets = []
    for issue in issues:
        session_id = ticket_session_id(issue.key)
        has_session = session_store.session_exists(session_id)
        tickets.append(TicketSummary(
            key=issue.key, summary=issue.summary, status=issue.status,
            session_id=session_id, has_session=has_session,
        ))
    return tickets


class TicketCommentPayload(BaseModel):
    author: str
    body_text: str
    created: str


class TicketDetail(BaseModel):
    key: str
    summary: str
    status: str
    description_text: str
    comments: list[TicketCommentPayload]
    has_session: bool


@app.get("/api/tickets/{issue_key}", response_model=TicketDetail)
def get_ticket_detail(issue_key: str, request: Request):
    """
    Read-only ticket detail (summary/description/comments/status) for the
    ticket-browser's main panel -- the "read it before deciding" step.
    Deliberately does NOT create a session or touch the pipeline; that
    only happens via POST /api/tickets/{key}/context, the explicit
    "Accept" action. Login is still required (same as every other ticket
    endpoint) since ticket content is real work data, not public.
    """
    _require_login(request)
    try:
        issue = jira_client.get_issue(issue_key)
    except jira_client.JiraFetchError as e:
        raise HTTPException(status_code=502, detail=str(e))
    session_id = ticket_session_id(issue_key)
    has_session = session_store.session_exists(session_id)
    return TicketDetail(
        key=issue.key, summary=issue.summary, status=issue.status,
        description_text=issue.description_text,
        comments=[TicketCommentPayload(author=c.author, body_text=c.body_text, created=c.created)
                   for c in issue.comments],
        has_session=has_session,
    )


@app.get("/api/tickets/{issue_key}/session")
def get_ticket_session(issue_key: str, request: Request, session_number: int = 1):
    """
    Resume path: returns this ticket's FULL turn history (every raw
    message + answer logged so far under its jira:<KEY>[::N] session) as
    a plain JSON list, same shape/convention as GET
    /api/sessions/{id}/turns -- a reopen replays instantly from
    session_store.py rather than being streamed turn-by-turn, since
    this is old, already-seen content, not new generation. Returns []
    (not 404) for a ticket that's never been opened, so the frontend
    can use this single call to decide seed-vs-resume without a
    separate existence check.

    session_number: added 2026-10-25 -- see ticket_session_id()'s own
    docstring. Defaults to 1 (the original session), matching every
    pre-existing caller of this endpoint.
    """
    _require_login(request)
    return session_store.get_session_turns(ticket_session_id(issue_key, session_number))


@app.get("/api/tickets/{issue_key}/sessions")
def get_ticket_sessions(issue_key: str, request: Request):
    """
    Lists EVERY session for this ticket (the original plus any "start
    new session" ones) -- backs the ticket chat window's session
    switcher. Always returns at least one logical entry even if the
    ticket was never opened (session_number=1, n_turns=0), so the
    frontend can offer "start session #2" without a separate existence
    check first.
    """
    _require_login(request)
    sessions = session_store.list_ticket_sessions(issue_key)
    if not sessions:
        sessions = [{"session_id": ticket_session_id(issue_key), "session_number": 1,
                     "started_at": None, "closed_at": None, "summary": None, "n_turns": 0}]
    return sessions


@app.post("/api/tickets/{issue_key}/context")
def open_ticket_context(issue_key: str, request: Request, session_number: int = 1):
    """
    Accept path: loads the real Jira ticket into this ticket's session as
    background context ONLY -- no search, no generated answer. The staffer's
    first message (typed, or the UI's "Suggest SOP" button, which just sends
    an ordinary message) runs through POST /api/messages/stream with the
    ticket already in the conversation. Safe to call again on an already-
    loaded session (see chat_service.load_ticket_context()).
    """
    staff = _require_login(request)
    session_id = ticket_session_id(issue_key, session_number)
    session_store.map_session_to_user(session_id, staff["username"])
    state = _get_or_restore_conversation_state(session_id)
    try:
        return load_ticket_context(issue_key, state, session_number=session_number)
    except jira_client.JiraFetchError as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/api/sessions")
def get_sessions(request: Request):
    # Auth check added 2026-10-25: found missing while wiring the evals
    # page's "My Conversations" tab (Stage 3), which makes this endpoint
    # more prominently used -- every other real data endpoint in this
    # file requires login, this one and the turns endpoint below didn't.
    #
    # Scoped to the logged-in staffer's OWN sessions (2026-10-25) via the
    # separate session_user_map table -- "My Conversations" previously
    # returned every session in the database regardless of who started
    # it, which didn't match the tab's name or intent.
    staff = _require_login(request)
    return session_store.list_sessions_for_user(staff["username"])


@app.get("/api/sessions/{session_id}/turns")
def get_session_turns(session_id: str, request: Request):
    _require_login(request)
    turns = session_store.get_session_turns(session_id)
    if not turns:
        raise HTTPException(status_code=404, detail="Session not found or has no turns")
    return turns


@app.post("/api/sessions/{session_id}/close")
def close_session(session_id: str, summary: str = ""):
    # Compress the in-memory ConversationState (if this process has it
    # live) BEFORE discarding it -- ConversationState.close_session()
    # moves current_turns into a past_sessions summary string, which
    # must be persisted too, or a later rehydration of this session
    # would silently skip straight from "never closed" to "gone,"
    # missing the compressed summary close_session() is specifically
    # for. A session this process never touched (restored on a
    # different process, or never opened this run) has nothing live to
    # compress -- only the DB-side closed_at/summary fields get set,
    # same as before this existed.
    state = _live_sessions.pop(session_id, None)
    if state is not None:
        state.close_session(summary)
        session_store.save_conversation_state(session_id, state.to_dict())
    session_store.close_session(session_id, summary)
    return {"ok": True}


@app.get("/api/monitoring")
def get_monitoring():
    return session_store.monitoring_summary()


@app.get("/api/monitoring/timeseries")
def get_monitoring_timeseries(limit: int = 30):
    return session_store.monitoring_timeseries(limit=limit)


@app.get("/api/monitoring/latency")
def get_monitoring_latency(limit: int = 200):
    """Per-pipeline-stage average latency over the most recent `limit`
    timed turns -- see session_store.latency_breakdown()."""
    return session_store.latency_breakdown(limit=limit)


@app.get("/api/monitoring/access-denials")
def get_recent_access_denials(limit: int = 20):
    return session_store.recent_access_denials(limit=limit)


@app.get("/api/monitoring/stream")
async def monitoring_stream():
    """
    Server-Sent Events feed of live activity -- logins, completed turns,
    and access denials -- pushed the instant they happen rather than
    polled. See live_events.py's module docstring for why push (not a
    fixed-interval poll) matches Anthropic's own "production monitoring...
    catches issues before you know about them" framing. The dashboard
    (ui/static/dashboard.html) opens ONE persistent connection to this
    and renders each incoming event into its live feed + re-ticks its
    rate tiles, instead of re-fetching /api/monitoring on a timer.
    """
    queue = live_events.subscribe()

    async def event_stream():
        try:
            yield _sse_event("connected", {})
            while True:
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=15)
                    yield _sse_event(payload["event"], payload["data"])
                except asyncio.TimeoutError:
                    yield _sse_event("heartbeat", {})  # keeps the connection alive through proxies/idle timeouts
        finally:
            live_events.unsubscribe(queue)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/monitoring/retrieval-eval")
def get_retrieval_eval():
    """
    Surfaces the last OFFLINE retrieval-eval run (src/eval_retrieval.py --
    Recall@5, MRR, Hit Rate@5, Precision@5, NDCG@5, confuser-trap rate)
    on the dashboard alongside live production-traffic metrics, per
    rag_evals2.md's point that offline eval scores are the baseline
    "moving averages over time windows" gets compared against. Reads the
    already-written results file rather than re-running the eval (that's
    a deliberate, on-demand activity -- `python src/eval_retrieval.py` --
    not something a page load should trigger; loading BAAI/bge-reranker-base
    and re-scoring ~110 queries on every dashboard refresh would defeat
    the purpose of a fast-loading monitoring page).
    """
    path = PROJECT_ROOT / "data" / "processed" / "retrieval_eval_results_reranked.jsonl"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No retrieval eval results found -- run src/eval_retrieval.py first.")

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    scorable = [r for r in rows if not r["unanswerable"]]
    n = len(scorable)
    if n == 0:
        return {"n": 0}

    def avg(key):
        return round(sum(r[key] for r in scorable) / n, 3)

    trap_queries = [r for r in scorable if r.get("confuser_trap_page_id")]
    trap_rate = round(sum(r["trap_beat_real"] for r in trap_queries) / len(trap_queries), 3) if trap_queries else None

    return {
        "n": n,
        "recall_at_5": avg("recall_at_5"),
        "top_1_accuracy": avg("top_1_correct"),
        "mrr": avg("mrr"),
        "hit_rate_at_5": avg("hit_rate_at_5"),
        "precision_at_5": avg("precision_at_5"),
        "ndcg_at_5": avg("ndcg_at_5"),
        "confuser_trap_rate": trap_rate,
        "n_trap_queries": len(trap_queries),
        "generated_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
    }


# Human-readable description of what each letter-category in
# eval_edge_case_queries.jsonl actually checks -- the golden set itself
# only has a `letter` + `subtype` field (e.g. "A"/"raw_ticket_dump");
# this mapping was derived by reading PROJECT_HANDOFF.md's own A-G map
# and the real subtypes present under each letter (verified directly
# against the file, not guessed) -- see ui/static/evals.html's Golden Set
# tab, added 2026-10-24.
GOLDEN_SET_CATEGORY_LABELS = {
    "A": "Input format variance (raw ticket dumps, pasted logs, transcriptions)",
    "B": "Message structure (rambling, fragmented, bundled, well-formed)",
    "C": "Entity extraction (error codes, IPs, hostnames, file paths, dates)",
    "D": "Query intent types (how-to, policy Q&A, escalation, meta questions)",
    "E": "Ambiguity & off-topic (typos, code-mixing, genuinely ambiguous, off-topic)",
    "F": "Staleness & answerability (stale/trashed pages, confuser pairs, no answer exists)",
    "G": "Multi-turn conversation (follow-ups, topic changes, resuming after a gap)",
}


@app.get("/api/monitoring/golden-set")
def get_golden_set():
    """
    Browsable view of the real hand-written golden set
    (eval_edge_case_queries.jsonl), grouped by letter-category (what
    behavior each group checks -- see GOLDEN_SET_CATEGORY_LABELS above)
    and joined against the real retrieval eval results
    (retrieval_eval_results_reranked.jsonl, matched by query_id) so each
    case shows its actual last-measured outcome, not just its definition.
    A case with no matching result row (e.g. the G02-G06 multi-turn cases,
    historically blocked pending Step 0) is shown with result=null rather
    than silently omitted.
    """
    golden_path = PROJECT_ROOT / "data" / "processed" / "eval_edge_case_queries.jsonl"
    if not golden_path.exists():
        raise HTTPException(status_code=404, detail="No golden set found.")

    cases = [json.loads(line) for line in golden_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    results_path = PROJECT_ROOT / "data" / "processed" / "retrieval_eval_results_reranked.jsonl"
    results_by_qid = {}
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                results_by_qid[row["query_id"]] = row

    groups: dict[str, list[dict]] = {}
    for c in cases:
        letter = c["letter"]
        result = results_by_qid.get(c["query_id"])
        groups.setdefault(letter, []).append({
            "query_id": c["query_id"],
            "subtype": c["subtype"],
            "query": c["query"],
            "expected_page_id": c["expected_page_id"],
            "unanswerable": c["unanswerable"],
            "confuser_trap_page_id": c.get("confuser_trap_page_id"),
            "notes": c.get("notes"),
            "result": (
                {
                    "recall_at_5": result["recall_at_5"],
                    "top_1_correct": result["top_1_correct"],
                    "hit_rank": result.get("hit_rank"),
                    "trap_beat_real": result.get("trap_beat_real"),
                }
                if result else None
            ),
        })

    return {
        "groups": [
            {
                "letter": letter,
                "label": GOLDEN_SET_CATEGORY_LABELS.get(letter, letter),
                "cases": cases_list,
            }
            for letter, cases_list in sorted(groups.items())
        ],
        "total_cases": len(cases),
        "n_unanswerable": sum(1 for c in cases if c["unanswerable"]),
    }


@app.get("/api/kb-pages/search")
def search_kb_pages(request: Request, q: str = ""):
    """
    Backs the evals page's "write a test case" KB-page picker -- search
    real article titles to pick a case's expected_page_id, instead of
    requiring a raw page_id typed from memory. See article_lookup.py's
    search_pages() for why this is plain substring matching, not
    semantic search (a small admin picker over ~200 known titles, not a
    retrieval-quality surface).
    """
    _require_login(request)
    return search_pages(q)


@app.get("/api/kb-pages/mine")
def list_my_kb_pages(request: Request):
    """
    Backs the chat UI's "Knowledge Base" sidebar tab: every page the
    LOGGED-IN staffer can actually open, grouped by category client-side.
    Permission-filtered server-side via list_accessible_pages() -- same
    restricted_to check as get_article()/Step 4.5, not the unfiltered
    admin picker search_pages() uses.
    """
    staff = _require_login(request)
    return list_accessible_pages(set(staff["allowed_groups"]))


class CustomEvalCaseRequest(BaseModel):
    query: str
    expected_page_id: str | None = None  # None = no ground truth, not scored, observation-only
    unanswerable: bool = False
    notes: str = ""
    source: str = "written"  # "written" (Run a Test Case tab) | "conversation" (marked from a real turn)
    source_session_id: str | None = None  # set when source == "conversation"
    source_turn_index: int | None = None


def _run_and_score_custom_case(req: CustomEvalCaseRequest) -> dict:
    """
    Shared by both ways a custom eval case can be created -- written from
    scratch (Run a Test Case tab) or marked from a real logged turn (My
    Conversations tab). Runs the query through the real retrieval
    pipeline (hybrid_search_reranked -- the exact same function
    eval_retrieval.py itself uses) and, if expected_page_id was given,
    scores it with the SAME score_query() function eval_retrieval.py
    uses for the real golden set -- not a reimplementation that could
    silently drift from the real metric definitions.

    Appends the scored result to data/processed/eval_custom_cases.jsonl
    -- DELIBERATELY NEVER the original eval_edge_case_queries.jsonl. That
    file has been through multiple rounds of real human correction this
    project (see PROJECT_HANDOFF.md's labeling-lesson notes on C09/F05/
    F09) -- mixing UI-authored or conversation-derived, unreviewed cases
    into it would make a carefully-verified golden set indistinguishable
    from untrusted on-the-fly entries. The Eval Numbers tab's "combined"
    view reads both files separately and reports them as distinct
    sources, never silently merges them into one file on disk.

    unanswerable=True queries are never scored for retrieval (same rule
    the real golden set follows) -- they're recorded for the record, but
    contribute nothing to recall/MRR/etc.
    """
    if _clients is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready yet.")

    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from eval_retrieval import score_query
    from retrieval_pipeline import hybrid_search_reranked

    case_id = f"CUSTOM-{uuid.uuid4().hex[:8]}"
    query_row = {
        "query_id": case_id,
        "source": req.source,
        "query": req.query,
        "expected_page_id": req.expected_page_id,
        "unanswerable": req.unanswerable,
        "confuser_trap_page_id": None,
    }

    if req.unanswerable or req.expected_page_id is None:
        # Observation-only: run retrieval, show what came back, but don't
        # claim a score against an answer key that doesn't exist.
        results = hybrid_search_reranked(
            req.query, _clients.chroma_collection, _clients.embedding_model, _clients.reranker, top_n=5,
        )
        scored = {
            "query_id": case_id, "source": req.source, "unanswerable": req.unanswerable,
            "expected_page_id": None, "confuser_trap_page_id": None,
            "retrieved_page_ids": [r["chunk"]["page_id"] for r in results],
            "hit_rank": None, "recall_at_5": None, "top_1_correct": None,
            "mrr": None, "hit_rate_at_5": None, "precision_at_5": None, "ndcg_at_5": None,
            "trap_beat_real": None,
        }
    else:
        search_fn = lambda q: hybrid_search_reranked(  # noqa: E731
            q, _clients.chroma_collection, _clients.embedding_model, _clients.reranker, top_n=5,
        )
        scored = score_query(query_row, search_fn)

    scored["query"] = req.query
    scored["notes"] = req.notes
    scored["created_at"] = datetime.now().isoformat()
    scored["source_session_id"] = req.source_session_id
    scored["source_turn_index"] = req.source_turn_index

    custom_path = PROJECT_ROOT / "data" / "processed" / "eval_custom_cases.jsonl"
    with open(custom_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(scored, ensure_ascii=False) + "\n")

    return scored


@app.post("/api/evals/custom-cases/run")
def run_custom_eval_case(req: CustomEvalCaseRequest, request: Request):
    """Write-a-test-case entry point (Run a Test Case tab) -- see
    _run_and_score_custom_case() for the real logic."""
    _require_login(request)
    req.source = "written"
    req.source_session_id = None
    req.source_turn_index = None
    return _run_and_score_custom_case(req)


class MarkTurnAsEvalCaseRequest(BaseModel):
    session_id: str
    turn_index: int
    expected_page_id: str | None = None
    unanswerable: bool = False
    notes: str = ""


@app.post("/api/evals/custom-cases/from-conversation")
def mark_turn_as_eval_case(req: MarkTurnAsEvalCaseRequest, request: Request):
    """
    My Conversations tab entry point: turns ONE real logged turn (from
    sessions.db, a conversation the staffer actually had through the
    live UI) into a scored eval case. This is specifically the
    "refine the eval set with real production questions" step from
    rag_evals.md's own bootstrap method (section 4.4, step 3) -- the one
    step this project's otherwise-thorough golden set never actually
    did, since every existing case was hand-written/synthetic, not
    pulled from real usage. See _run_and_score_custom_case()'s docstring
    for why this still goes to the separate custom-cases file, not the
    original golden set.

    Uses the turn's STORED text (raw_text, already PII-redacted at the
    point it was logged -- see pii_redaction.py's "redact the persisted
    copy" rule) as the query, not a live re-fetch of the in-flight
    unredacted version -- consistent with treating sessions.db as the
    durable, already-sanitized record.
    """
    _require_login(request)
    turns = session_store.get_session_turns(req.session_id)
    turn = next((t for t in turns if t["turn_index"] == req.turn_index), None)
    if turn is None:
        raise HTTPException(status_code=404, detail="Turn not found in that session.")
    if not turn["raw_text"]:
        raise HTTPException(status_code=400, detail="This turn has no raw_text to use as a query.")

    case_req = CustomEvalCaseRequest(
        query=turn["raw_text"],
        expected_page_id=req.expected_page_id,
        unanswerable=req.unanswerable,
        notes=req.notes,
        source="conversation",
        source_session_id=req.session_id,
        source_turn_index=req.turn_index,
    )
    return _run_and_score_custom_case(case_req)


@app.get("/api/evals/custom-cases")
def list_custom_eval_cases(request: Request):
    """All custom cases run so far, newest first -- read from their own
    separate file (never the original golden set, see
    run_custom_eval_case()'s docstring for why)."""
    _require_login(request)
    custom_path = PROJECT_ROOT / "data" / "processed" / "eval_custom_cases.jsonl"
    if not custom_path.exists():
        return {"cases": []}
    rows = [json.loads(line) for line in custom_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows.reverse()
    return {"cases": rows}


@app.get("/api/evals/review-queue")
def get_review_queue(request: Request):
    """Every production-marked custom case (from "My Conversations")
    awaiting a human approve/reject decision before it can be promoted
    into the golden set -- see golden_set_manager.list_review_queue()."""
    _require_login(request)
    return {"queue": golden_set_manager.list_review_queue()}


class ReviewDecisionRequest(BaseModel):
    query_id: str
    status: str  # "approved" | "rejected" | "pending"


@app.post("/api/evals/review-queue/decide")
def decide_review_case(req: ReviewDecisionRequest, request: Request):
    _require_login(request)
    try:
        return golden_set_manager.set_review_status(req.query_id, req.status)
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/evals/review-queue/finish")
def finish_review(request: Request):
    """Promotes every currently-approved review case into a NEW golden
    set version (golden_set_vN.jsonl) -- see
    golden_set_manager.promote_approved_to_golden_set()."""
    staff = _require_login(request)
    _require_non_guest(staff)
    try:
        return golden_set_manager.promote_approved_to_golden_set()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/evals/golden-set/versions")
def get_golden_set_versions(request: Request):
    _require_login(request)
    return {
        "latest_version": golden_set_manager.latest_golden_set_version(),
        "versions": golden_set_manager.list_versions(),
    }


@app.get("/api/evals/golden-run/status")
def get_golden_run_status(request: Request):
    """Backs the "Run Golden Eval" button's change-detection: tells the
    UI whether anything has actually changed since the last real run, so
    a click that would do nothing can show a notification instead of
    re-running the full pipeline for no reason."""
    _require_login(request)
    status = golden_set_manager.needs_rerun()
    status["last_run"] = golden_set_manager.last_run()
    return status


@app.get("/api/evals/golden-run/history")
def get_golden_run_history(request: Request):
    _require_login(request)
    return {"runs": golden_set_manager.list_runs()}


@app.post("/api/evals/golden-run/run")
def run_golden_eval(request: Request, force: bool = False):
    """
    Triggers a REAL golden-set retrieval eval run (loads the reranker,
    re-scores every case) -- expensive, so this only runs when
    needs_rerun() says something actually changed, unless force=true is
    explicitly passed (a staffer deliberately overriding the skip, e.g.
    after a change needs_rerun() can't detect like a score_query() bug
    fix). On skip, returns the reason without touching the pipeline or
    the results file -- no wasted model load, no wasted LLM/embedding
    calls, matching the user's explicit ask not to waste calls when
    nothing changed.
    """
    staff = _require_login(request)
    _require_non_guest(staff)
    if _clients is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready yet.")

    status = golden_set_manager.needs_rerun()
    if not status["changed"] and not force:
        return {"ran": False, "reason": status["reason"], "last_run": golden_set_manager.last_run()}

    metrics = golden_set_manager.run_golden_set_retrieval_eval(_clients)
    return {"ran": True, "metrics": metrics}


@app.get("/api/monitoring/retrieval-eval-combined")
def get_retrieval_eval_combined():
    """
    Same aggregate metrics as /api/monitoring/retrieval-eval, but
    computed over the ORIGINAL golden set PLUS every scored custom case
    -- lets the Eval Numbers tab show "what if we count the cases I just
    wrote/marked too" without ever writing them into the original file.
    Only custom cases that were actually scored (had a real
    expected_page_id, not observation-only) are included, same
    scorable/unanswerable rule the original eval uses.
    """
    original_path = PROJECT_ROOT / "data" / "processed" / "retrieval_eval_results_reranked.jsonl"
    custom_path = PROJECT_ROOT / "data" / "processed" / "eval_custom_cases.jsonl"

    rows = []
    if original_path.exists():
        rows += [json.loads(line) for line in original_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    n_custom = 0
    if custom_path.exists():
        custom_rows = [json.loads(line) for line in custom_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        scorable_custom = [r for r in custom_rows if not r["unanswerable"] and r.get("recall_at_5") is not None]
        rows += scorable_custom
        n_custom = len(scorable_custom)

    scorable = [r for r in rows if not r["unanswerable"]]
    n = len(scorable)
    if n == 0:
        return {"n": 0, "n_custom_included": 0}

    def avg(key):
        return round(sum(r[key] for r in scorable) / n, 3)

    trap_queries = [r for r in scorable if r.get("confuser_trap_page_id")]
    trap_rate = round(sum(r["trap_beat_real"] for r in trap_queries) / len(trap_queries), 3) if trap_queries else None

    return {
        "n": n,
        "n_custom_included": n_custom,
        "recall_at_5": avg("recall_at_5"),
        "top_1_accuracy": avg("top_1_correct"),
        "mrr": avg("mrr"),
        "hit_rate_at_5": avg("hit_rate_at_5"),
        "precision_at_5": avg("precision_at_5"),
        "ndcg_at_5": avg("ndcg_at_5"),
        "confuser_trap_rate": trap_rate,
        "n_trap_queries": len(trap_queries),
    }


@app.get("/api/monitoring/generation-eval")
def get_generation_eval():
    """
    Surfaces the last offline generation-quality judge run
    (src/judge_generation_eval.py) -- faithfulness/relevance LLM-judge
    scores plus the deterministic contract-violation count. Computes
    aggregates LIVE from the per-row judged file (real field names:
    faithfulness_score/relevance_score, pending bool, judge_model,
    complex bool, contract_violations list) rather than trusting
    generation_eval_summary.json, which was found to be stale against
    this exact file (summary claimed n_synthesized=1 from an old
    snapshot; PROJECT_HANDOFF.md's own narrative separately claims n=7
    judged -- neither matches this file's real current state, which is
    1 judged, 5 genuinely pending, 16 correctly never attempted because
    they never reached synthesized mode). Reading raw rows every time
    means this number can't drift out of sync with disk again -- see
    ADR-003's "a limitation moves out of the list only when there's a
    number proving it" rule, applied here to a number instead.
    """
    path = PROJECT_ROOT / "data" / "processed" / "generation_eval_judged.jsonl"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No generation eval results found -- run src/judge_generation_eval.py first.")

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    judged = [r for r in rows if r.get("faithfulness_score") is not None]
    pending = [r for r in rows if r.get("pending")]
    candidates_only = [r for r in rows if r.get("mode") == "candidates_only"]

    def avg(key, items):
        vals = [r[key] for r in items if r.get(key) is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    self_graded = [r for r in judged if r.get("judge_model") == "gemini-3.5-flash-lite"]
    independently_graded = [r for r in judged if r.get("judge_model") == "gemini-3.6-flash"]
    total_violations = sum(len(r.get("contract_violations") or []) for r in rows)

    return {
        "n_total_rows": len(rows),
        "n_judged": len(judged),
        "n_pending": len(pending),
        "n_candidates_only": len(candidates_only),
        "mean_faithfulness": avg("faithfulness_score", judged),
        "mean_relevance": avg("relevance_score", judged),
        "n_self_graded": len(self_graded),
        "n_independently_graded": len(independently_graded),
        "mean_faithfulness_self_graded": avg("faithfulness_score", self_graded),
        "mean_faithfulness_independently_graded": avg("faithfulness_score", independently_graded),
        "contract_violations": total_violations,
        "generated_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
    }


@app.get("/api/monitoring/calibration")
def get_calibration():
    """
    Surfaces the Step 8 confidence-threshold calibration curve
    (src/calibrate_confidence.py) -- both the aggregate score-bucket view
    and, as of 2026-10-15, a per-category breakdown at the current
    threshold. Added specifically because the aggregate-only view
    couldn't answer whether the single global 2.0 cutoff is actually
    valid across every category, or dominated by whichever category
    happens to contain this sample's worst confuser pairs -- see
    ADR-003. Reads the raw per-query records and buckets them here
    rather than trusting any pre-computed summary.
    """
    path = PROJECT_ROOT / "data" / "processed" / "confidence_calibration.jsonl"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No calibration data found -- run src/calibrate_confidence.py first.")

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    scored = [r for r in rows if r.get("top_score") is not None and not r.get("unanswerable")]

    buckets = [(-100, -5), (-5, -2), (-2, 0), (0, 2), (2, 5), (5, 100)]
    bucket_results = []
    for lo, hi in buckets:
        in_bucket = [r for r in scored if lo <= r["top_score"] < hi]
        if not in_bucket:
            continue
        correct = sum(1 for r in in_bucket if r["top_correct"])
        bucket_results.append({
            "range_low": lo, "range_high": hi,
            "n": len(in_bucket), "correct": correct,
            "accuracy": round(correct / len(in_bucket), 3),
        })

    THRESHOLD = 2.0
    by_category: dict = {}
    for r in scored:
        cat = r.get("category") or "(no category -- empty result set)"
        by_category.setdefault(cat, []).append(r)

    category_results = []
    for cat, cat_rows in sorted(by_category.items()):
        below = [r for r in cat_rows if r["top_score"] < THRESHOLD]
        above = [r for r in cat_rows if r["top_score"] >= THRESHOLD]
        category_results.append({
            "category": cat,
            "n": len(cat_rows),
            "below_threshold_n": len(below),
            "below_threshold_accuracy": round(sum(r["top_correct"] for r in below) / len(below), 3) if below else None,
            "at_or_above_threshold_n": len(above),
            "at_or_above_threshold_accuracy": round(sum(r["top_correct"] for r in above) / len(above), 3) if above else None,
        })

    return {
        "n_scored": len(scored),
        "threshold": THRESHOLD,
        "buckets": bucket_results,
        "by_category": category_results,
        "has_category_data": any(r.get("category") for r in scored),
        "generated_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
    }


app.mount("/static", StaticFiles(directory=str(UI_DIR / "static")), name="static")


@app.get("/")
def index():
    return FileResponse(str(UI_DIR / "static" / "index.html"))


@app.get("/dashboard")
def dashboard():
    return FileResponse(str(UI_DIR / "static" / "dashboard.html"))


@app.get("/evals")
def evals_page():
    return FileResponse(str(UI_DIR / "static" / "evals.html"))


if __name__ == "__main__":
    import os
    import uvicorn
    # HOST/PORT overridable via env (same convention as config.py's path
    # vars) -- 127.0.0.1-only is correct for local dev, but a container
    # (Hugging Face Spaces expects 0.0.0.0:7860) needs to bind every
    # interface, not just loopback, or external traffic never reaches it.
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8010"))
    uvicorn.run(app, host=host, port=port)
