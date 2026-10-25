"""
In-memory pub/sub for pushing live events to the observability dashboard
over Server-Sent Events, instead of the dashboard polling /api/monitoring
on a timer. Matches the framing in Anthropic's own "demystifying evals"
engineering post (fetched directly, verified 2026-10-12): offline evals
run on a fixed test set pre-launch, while "production monitoring kicks in
post-launch" to "reveal real user behavior... [and] catch issues before
you know about them" -- that's a REACTIVE, as-it-happens signal, which a
fixed-interval poll approximates but doesn't actually deliver (there's
always up-to-the-poll-interval lag, and most polls return nothing new).

Deliberately NOT a message queue/Redis pub-sub -- this is a single local
server process (see ui/server.py's own docstring on scope), so an
in-memory list of asyncio.Queues per connected dashboard client is the
right-sized solution. A real multi-instance deployment (the AWS-migration
phase) would need a shared broker; noted here, not built, since nothing
in this project runs more than one server process today.

Events carry only the SAME deterministic signals session_store.py
already logs per turn (category, mode, confidence, latency, whether an
access denial fired) -- never raw conversation text -- matching the
project's existing privacy stance (Anthropic's multi-agent-research-
system post: "monitor agent decision patterns... without monitoring the
contents of individual conversations").
"""

import asyncio
import json
import time
from dataclasses import dataclass, field

_subscribers: list[asyncio.Queue] = []


def subscribe() -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue(maxsize=100)
    _subscribers.append(q)
    return q


def unsubscribe(q: asyncio.Queue) -> None:
    if q in _subscribers:
        _subscribers.remove(q)


def broadcast(event_type: str, data: dict) -> None:
    """
    Fire-and-forget push to every currently-connected dashboard client.
    Never raises/blocks the caller (chat_service.py's request path) on a
    slow or full subscriber queue -- a dashboard that's fallen behind
    just misses an event rather than stalling a real user's chat request,
    which matters far more than the dashboard staying perfectly current.
    """
    payload = {"event": event_type, "data": data, "server_time": time.time()}
    for q in list(_subscribers):
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            pass  # see docstring -- drop, don't block or raise


@dataclass
class StageTimer:
    """
    Tiny per-turn stopwatch -- pipeline.py/chat_service.py call
    .mark(stage_name) after each stage completes, producing a
    {stage: latency_ms} dict for both the structured log line and the
    dashboard's latency breakdown. Not a profiler -- just wall-clock
    deltas between stage boundaries that already exist in the code.
    """
    _last: float = field(default_factory=time.perf_counter)
    marks: dict = field(default_factory=dict)

    def mark(self, stage: str) -> None:
        now = time.perf_counter()
        self.marks[stage] = round((now - self._last) * 1000, 1)
        self._last = now
