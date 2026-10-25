"""
Real correctness test for Step 0 (src/step0_context.py) against the 5
real multi-turn eval queries (G02-G06) that were previously BLOCKED --
these need Step 0 specifically to be testable at all (see
PROJECT_HANDOFF.md Section 8). Each query's "prior_turn: ... |
current_turn: ..." style encoding is parsed into a real ConversationState
and run through the actual run_pipeline() with Step 0 wired in.

This is the first time these 5 queries can be scored at all.

Run: python tests/test_step0_context.py
"""

import json
import sys
import time
from pathlib import Path

TESTS_DIR = Path(__file__).parent
PROJECT_ROOT = TESTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pipeline import PipelineClients, run_pipeline  # noqa: E402
from step0_context import ConversationState  # noqa: E402

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def build_g02_state():
    state = ConversationState()
    state.add_turn("user", "My VPN keeps disconnecting right after I log in.")
    state.add_turn("assistant", "Try restarting the VPN client and checking your network adapter settings.")
    return state, "okay that didn't help, what if I'm connecting over public wifi at a cafe?"


def build_g03_state():
    state = ConversationState()
    state.close_session("user reported VPN disconnect issue 3 days ago, ticket left open, no resolution confirmed")
    return state, "hey, still having that same VPN problem from earlier this week"


def build_g04_state():
    state = ConversationState()
    state.add_turn("user", "Quick question about VPN")
    state.add_turn("user", "it's been disconnecting right after I log in")
    return state, "happening since yesterday, is there a fix?"


def build_g05_state():
    state = ConversationState()
    state.add_turn("user", "My VPN keeps disconnecting right after I log in.")
    state.add_turn("assistant", "Try restarting the VPN client.")
    return state, "actually never mind that got fixed, separate question -- why does my account keep getting locked after failed MFA attempts?"


def build_g06_state():
    state = ConversationState()
    state.add_turn("user", "How do I fix a certificate access denial?")
    state.add_turn("assistant", "Check the cert expiry, verify device enrollment, then reissue if needed.")
    return state, "wait, is that the most current version of the steps? I don't want to follow something outdated."


BUILDERS = {
    "G02": build_g02_state,
    "G03": build_g03_state,
    "G04": build_g04_state,
    "G05": build_g05_state,
    "G06": build_g06_state,
}


def main():
    with open(DATA_PROCESSED / "eval_edge_case_queries.jsonl", encoding="utf-8") as f:
        rows = {r["query_id"]: r for r in (json.loads(line) for line in f)}

    print("Building pipeline clients (loading models)...")
    clients = PipelineClients.build()

    print("\nStep 0 multi-turn eval -- the 5 previously-blocked G02-G06 queries\n")
    results = []
    for qid, builder in BUILDERS.items():
        row = rows[qid]
        state, new_message = builder()
        outcome = run_pipeline(new_message, clients, conversation_state=state)

        expected = row["expected_page_id"]
        expected_list = expected if isinstance(expected, list) else [expected]
        expected_list = [e for e in expected_list if e is not None]

        hit_rank = None
        wants_latest_correct = None
        if outcome.stopped_at == "ok" and outcome.results:
            retrieved = [sr["chunk"]["page_id"] for sr in outcome.results[0].search_results]
            for rank, pid in enumerate(retrieved, start=1):
                if pid in expected_list and hit_rank is None:
                    hit_rank = rank

        # G06 specifically tests whether recency intent survives the
        # rewrite -- check wants_latest came through correctly.
        if qid == "G06":
            wants_latest_correct = outcome.wants_latest is True

        print(f"{qid}: stopped_at={outcome.stopped_at}")
        if outcome.results:
            print(f"  rewritten: {outcome.results[0].sub_query!r}")
        print(f"  expected_page_id={expected} unanswerable={row['unanswerable']} "
              f"hit_rank={hit_rank}" + (f" wants_latest={outcome.wants_latest}" if qid == "G06" else ""))
        results.append({
            "query_id": qid, "hit_rank": hit_rank, "unanswerable": row["unanswerable"],
            "stopped_at": outcome.stopped_at, "wants_latest_correct": wants_latest_correct,
        })
        print()
        time.sleep(15)

    print("=== SUMMARY ===")
    for r in results:
        status = "OK" if (r["hit_rank"] is not None or r["unanswerable"]) else "MISS"
        print(f"  {r['query_id']}: {status} (hit_rank={r['hit_rank']})")
    if any(r["wants_latest_correct"] is not None for r in results):
        g06 = next(r for r in results if r["query_id"] == "G06")
        print(f"  G06 wants_latest signal preserved through rewrite: {g06['wants_latest_correct']}")


if __name__ == "__main__":
    main()
