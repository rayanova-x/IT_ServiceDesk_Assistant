"""
Golden set versioning + review-queue promotion (2026-10-25 user feature).

Design, per explicit user decisions in this session:
  - Promoted cases never get appended into the original hand-verified
    eval_edge_case_queries.jsonl in place. Instead each promotion writes
    a NEW FULL SNAPSHOT file, golden_set_v{N}.jsonl -- v1 is a frozen
    copy of the original 67 cases, v2 = v1 + newly-approved cases, etc.
    Every past version stays openable/re-runnable for comparison; this
    module never deletes or rewrites an existing version file.
  - A separate golden_set_meta.json tracks version history (who/when/
    how many cases/which query_ids were added in each version) so a
    staffer can always see what changed between versions without diffing
    JSONL files by hand.
  - Promotion candidates come from eval_custom_cases.jsonl rows with
    source="conversation" (marked from a real logged turn in the chat
    UI's "My Conversations" tab) that have a real expected_page_id (not
    observation-only/unanswerable-only) and review_status == "pending".
    A staffer must explicitly approve or reject each one; only approved
    rows get promoted on "Finish Review". review_status lives as an
    extra field directly on the custom-case row in eval_custom_cases.jsonl
    (updated in place), not a second queue file -- that file is already
    the durable record of every custom case ever run, this just adds
    state to it rather than duplicating it.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path

from config import DATA_DIR

DATA_PROCESSED = DATA_DIR / "processed"
ORIGINAL_GOLDEN_SET = DATA_PROCESSED / "eval_edge_case_queries.jsonl"
CUSTOM_CASES_PATH = DATA_PROCESSED / "eval_custom_cases.jsonl"
META_PATH = DATA_PROCESSED / "golden_set_meta.json"
RUN_LOG_PATH = DATA_PROCESSED / "eval_run_log.jsonl"

# Same model identifiers eval_retrieval.py actually constructs (kept here
# as the single source of truth for change-detection, not re-derived from
# a live model object -- see get_model_version()'s docstring).
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"


def get_model_version() -> str:
    """
    A short string identifying the retrieval models a golden-eval run
    used -- embedding model + reranker, the two real pipeline components
    eval_retrieval.py.score_query() exercises. Hardcoded to match the
    literal strings eval_retrieval.py passes to TextEmbedding/
    TextCrossEncoder (see its own imports) rather than introspecting a
    live model object, since this needs to be comparable WITHOUT loading
    the models (the whole point of change-detection is skipping that
    costly load when nothing changed).
    """
    return f"{EMBEDDING_MODEL_NAME}|{RERANKER_MODEL_NAME}"


def _golden_set_path(version: int) -> Path:
    return DATA_PROCESSED / f"golden_set_v{version}.jsonl"


def _load_meta() -> dict:
    if not META_PATH.exists():
        return {"latest_version": 0, "versions": []}
    return json.loads(META_PATH.read_text(encoding="utf-8"))


def _save_meta(meta: dict) -> None:
    META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def _ensure_v1_bootstrapped() -> dict:
    """
    First-ever call: freezes the current 67-case eval_edge_case_queries.jsonl
    as golden_set_v1.jsonl, so there's always a versioned file to build on
    -- the original file itself is never read directly by anything built
    after this point, only v1 onward.
    """
    meta = _load_meta()
    if meta["latest_version"] >= 1:
        return meta

    rows = [json.loads(line) for line in ORIGINAL_GOLDEN_SET.read_text(encoding="utf-8").splitlines() if line.strip()]
    _golden_set_path(1).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8",
    )
    meta = {
        "latest_version": 1,
        "versions": [{
            "version": 1,
            "created_at": datetime.now().isoformat(),
            "n_cases": len(rows),
            "n_added": len(rows),
            "added_query_ids": [r["query_id"] for r in rows],
            "note": "Initial version -- frozen snapshot of the original hand-verified golden set.",
        }],
    }
    _save_meta(meta)
    return meta


def latest_golden_set_version() -> int:
    return _ensure_v1_bootstrapped()["latest_version"]


def load_golden_set(version: int | None = None) -> list[dict]:
    """Every case in the given version (default: latest), in the same
    shape eval_retrieval.py.load_edge_case_queries() already expects."""
    v = version or latest_golden_set_version()
    path = _golden_set_path(v)
    if not path.exists():
        raise FileNotFoundError(f"No golden set version {v} on disk ({path}).")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def list_versions() -> list[dict]:
    return _ensure_v1_bootstrapped()["versions"]


def golden_set_hash(version: int | None = None) -> str:
    """Content hash of a golden set version's exact file bytes -- used by
    change-detection to answer "has the golden set changed since the last
    real eval run" without diffing case-by-case."""
    v = version or latest_golden_set_version()
    path = _golden_set_path(v)
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Review queue -- pending custom cases marked from real conversations.
# ---------------------------------------------------------------------------

def _load_custom_cases() -> list[dict]:
    if not CUSTOM_CASES_PATH.exists():
        return []
    return [json.loads(line) for line in CUSTOM_CASES_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def _save_custom_cases(rows: list[dict]) -> None:
    CUSTOM_CASES_PATH.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n" if rows else "",
        encoding="utf-8",
    )


def list_review_queue() -> list[dict]:
    """
    Every production-marked case (source == "conversation") that's a real
    scored case (not observation-only/unanswerable) and hasn't been
    reviewed yet. review_status defaults to "pending" for any row written
    before this field existed, via .get() -- an old row is never silently
    excluded just because it predates this feature.
    """
    rows = _load_custom_cases()
    return [
        r for r in rows
        if r.get("source") == "conversation"
        and r.get("expected_page_id") is not None
        and not r.get("unanswerable")
        and r.get("review_status", "pending") == "pending"
    ]


def set_review_status(query_id: str, status: str) -> dict:
    """status: 'approved' | 'rejected' | 'pending' (to undo). Updates the
    row IN PLACE in eval_custom_cases.jsonl -- see module docstring for
    why this doesn't live in a second queue file."""
    if status not in ("approved", "rejected", "pending"):
        raise ValueError(f"Invalid review status: {status!r}")
    rows = _load_custom_cases()
    target = None
    for r in rows:
        if r.get("query_id") == query_id:
            r["review_status"] = status
            target = r
            break
    if target is None:
        raise KeyError(f"No custom case with query_id={query_id!r}")
    _save_custom_cases(rows)
    return target


def promote_approved_to_golden_set() -> dict:
    """
    "Finish Review": every custom case currently marked review_status ==
    "approved" gets folded into a NEW golden set version (see module
    docstring for why this is a new snapshot file, not an in-place
    append). Each promoted case is tagged letter="H" (a new group,
    "real production cases") and review_status is flipped to "promoted"
    on its eval_custom_cases.jsonl row so it can't be promoted twice and
    drops out of list_review_queue() (which only shows "pending").

    Returns the new version's metadata, or raises if there's nothing
    approved to promote (callers should check list_review_queue()'s
    approved count before calling, so this is a hard error, not a no-op --
    calling it with nothing approved is a caller bug, not a normal path).
    """
    rows = _load_custom_cases()
    approved = [r for r in rows if r.get("review_status") == "approved"]
    if not approved:
        raise ValueError("No approved cases to promote.")

    meta = _ensure_v1_bootstrapped()
    current_version = meta["latest_version"]
    current_cases = load_golden_set(current_version)

    new_cases = []
    added_ids = []
    for r in approved:
        new_cases.append({
            "query_id": r["query_id"],
            "letter": "H",
            "subtype": "production_conversation",
            "query": r["query"],
            "expected_page_id": r["expected_page_id"],
            "unanswerable": False,
            "notes": r.get("notes") or "Promoted from a real staffer conversation via the review queue.",
        })
        added_ids.append(r["query_id"])

    new_version = current_version + 1
    all_cases = current_cases + new_cases
    _golden_set_path(new_version).write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in all_cases) + "\n", encoding="utf-8",
    )

    version_entry = {
        "version": new_version,
        "created_at": datetime.now().isoformat(),
        "n_cases": len(all_cases),
        "n_added": len(new_cases),
        "added_query_ids": added_ids,
        "note": f"Promoted {len(new_cases)} case(s) reviewed from real production conversations.",
    }
    meta["latest_version"] = new_version
    meta["versions"].append(version_entry)
    _save_meta(meta)

    # Mark promoted rows so they drop out of the pending queue and can't
    # be promoted again on a future "Finish Review" click.
    for r in rows:
        if r.get("query_id") in added_ids:
            r["review_status"] = "promoted"
    _save_custom_cases(rows)

    return version_entry


# ---------------------------------------------------------------------------
# Run log -- every real golden-eval run, so past numbers are never
# silently overwritten and a staffer can always see what a given number
# was actually computed from.
# ---------------------------------------------------------------------------

def _load_run_log() -> list[dict]:
    if not RUN_LOG_PATH.exists():
        return []
    return [json.loads(line) for line in RUN_LOG_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def last_run() -> dict | None:
    runs = _load_run_log()
    return runs[-1] if runs else None


def list_runs(limit: int = 20) -> list[dict]:
    runs = _load_run_log()
    return list(reversed(runs))[:limit]


def needs_rerun() -> dict:
    """
    Compares the CURRENT latest golden set's content hash + model version
    against the last logged real run. Returns {"changed": bool, "reason":
    str, "current_hash":..., "current_model_version":...} -- the "Run
    Golden Eval" button calls this first so an unchanged golden set/model
    never triggers a wasted real pipeline run (every query re-embedded +
    reranked + a wall-clock cost) just because a staffer clicked the
    button out of habit.
    """
    current_hash = golden_set_hash()
    current_model_version = get_model_version()
    current_version = latest_golden_set_version()
    prev = last_run()
    if prev is None:
        return {
            "changed": True, "reason": "No prior run on record.",
            "current_hash": current_hash, "current_model_version": current_model_version,
            "current_golden_set_version": current_version,
        }
    if prev["golden_set_hash"] != current_hash:
        return {
            "changed": True, "reason": "Golden set has changed since the last run.",
            "current_hash": current_hash, "current_model_version": current_model_version,
            "current_golden_set_version": current_version,
        }
    if prev["model_version"] != current_model_version:
        return {
            "changed": True, "reason": "Retrieval model/reranker has changed since the last run.",
            "current_hash": current_hash, "current_model_version": current_model_version,
            "current_golden_set_version": current_version,
        }
    return {
        "changed": False,
        "reason": "No changes since the last run -- golden set and model version are identical.",
        "current_hash": current_hash, "current_model_version": current_model_version,
        "current_golden_set_version": current_version,
    }


def run_golden_set_retrieval_eval(clients) -> dict:
    """
    The real, costly re-run the "Run Golden Eval" button triggers:
    every case in the LATEST golden set version through the real
    hybrid_search_reranked() pipeline (same function production code and
    eval_retrieval.py itself use), scored with eval_retrieval.score_query()
    -- not a reimplementation. Overwrites retrieval_eval_results_reranked.jsonl
    (the exact file /api/monitoring/retrieval-eval and /golden-set already
    read), so every existing dashboard/evals-page endpoint picks up the
    new numbers with no changes of their own. Also appends one entry to
    the run log via log_run() so the PREVIOUS numbers are never lost --
    see log_run()'s docstring.

    Callers (the server endpoint) MUST check needs_rerun() first and
    refuse to call this when nothing changed -- this function itself does
    not skip, by design, so a caller that deliberately wants to force a
    re-run (e.g. after fixing a bug in score_query itself, which
    needs_rerun() has no way to detect) always can.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from eval_retrieval import score_query
    from retrieval_pipeline import hybrid_search_reranked
    import live_events

    cases = load_golden_set()
    # Same scorable/blocked split eval_retrieval.load_edge_case_queries()
    # applies: only G01 among any "G" letter cases is scorable (Step 0
    # conversation contextualization doesn't support multi-turn yet).
    query_rows = []
    for c in cases:
        if c.get("letter") == "G" and c["query_id"] != "G01":
            continue
        query_rows.append({
            "query_id": c["query_id"],
            "source": "edge_case",
            "query": c["query"],
            "expected_page_id": c["expected_page_id"],
            "confuser_trap_page_id": c.get("confuser_trap_page_id"),
            "unanswerable": c["unanswerable"],
        })

    search_fn = lambda q: hybrid_search_reranked(  # noqa: E731
        q, clients.chroma_collection, clients.embedding_model, clients.reranker, top_n=5,
    )
    total = len(query_rows)
    live_events.broadcast("golden_run_progress", {"done": 0, "total": total, "query_id": None})
    results = []
    for i, q in enumerate(query_rows, start=1):
        results.append(score_query(q, search_fn))
        live_events.broadcast("golden_run_progress", {"done": i, "total": total, "query_id": q["query_id"]})

    out_path = DATA_PROCESSED / "retrieval_eval_results_reranked.jsonl"
    out_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n", encoding="utf-8",
    )

    scorable = [r for r in results if not r["unanswerable"]]
    n = len(scorable)
    metrics = {"n": n}
    if n > 0:
        def avg(key):
            return round(sum(r[key] for r in scorable) / n, 3)
        trap_queries = [r for r in scorable if r.get("confuser_trap_page_id")]
        metrics.update({
            "recall_at_5": avg("recall_at_5"), "top_1_accuracy": avg("top_1_correct"),
            "mrr": avg("mrr"), "hit_rate_at_5": avg("hit_rate_at_5"),
            "precision_at_5": avg("precision_at_5"), "ndcg_at_5": avg("ndcg_at_5"),
            "confuser_trap_rate": (
                round(sum(r["trap_beat_real"] for r in trap_queries) / len(trap_queries), 3)
                if trap_queries else None
            ),
            "n_trap_queries": len(trap_queries),
        })

    log_run(metrics)
    return metrics


def log_run(metrics: dict) -> dict:
    """Appends one real golden-eval run to the run log -- called after an
    actual pipeline run completes, never on a skipped/no-op click."""
    entry = {
        "run_at": datetime.now().isoformat(),
        "golden_set_version": latest_golden_set_version(),
        "golden_set_hash": golden_set_hash(),
        "model_version": get_model_version(),
        "n_cases": metrics.get("n"),
        "metrics": metrics,
    }
    with open(RUN_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry
