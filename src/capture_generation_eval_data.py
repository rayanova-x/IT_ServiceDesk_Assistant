"""
Captures real Step 9 generation output (query, retrieved context, and the
actual synthesized answer or candidates_only caveat) for a stratified
20-query sample, ONE TIME -- costs Gemini quota (routing + understanding +
generation calls). See data/processed/eval_generation_sample.jsonl for the
sample and why each query was picked.

WHY THIS EXISTS (2026-09-14): same reasoning as capture_step7_raw_data.py
-- separate the EXPENSIVE part (real pipeline + Gemini generation, run
once) from judging (an LLM-as-judge pass that we'll want to re-run/tune
without re-spending quota on generation itself). See
judge_generation_eval.py, which reads this script's output.

This is the "generation" half of the RAG evaluation triad
(reference_docs/rag_evals.md, rag_evals2.md): retrieval accuracy is
already measured (eval_retrieval.py, eval_pipeline_e2e.py); this captures
what's needed to measure faithfulness/groundedness and answer relevance
-- the half that has never been formally evaled (PROJECT_HANDOFF.md
Section 8, item 9).

4 multi-turn queries (G01/G03/G05/G06) need a ConversationState built the
same way tests/test_step0_context.py does it -- reused here rather than
re-invented.

Run: python src/capture_generation_eval_data.py
"""

import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
import sys  # noqa: E402
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import PipelineClients, run_pipeline  # noqa: E402
from step0_context import ConversationState  # noqa: E402
from step9_generation import generate_answer  # noqa: E402

REPO_ROOT = PROJECT_ROOT.parent
DATA_PROCESSED = REPO_ROOT / "data" / "processed"


def build_g01_state():
    # Control case: no prior context, Step 0 should be a no-op.
    return ConversationState(), "My VPN keeps disconnecting right after I log in."


def build_g03_state():
    state = ConversationState()
    state.close_session("user reported VPN disconnect issue 3 days ago, ticket left open, no resolution confirmed")
    return state, "hey, still having that same VPN problem from earlier this week"


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


MULTI_TURN_BUILDERS = {
    "G01": build_g01_state,
    "G03": build_g03_state,
    "G05": build_g05_state,
    "G06": build_g06_state,
}


def main():
    with open(DATA_PROCESSED / "eval_edge_case_queries.jsonl", encoding="utf-8") as f:
        query_rows = {r["query_id"]: r for r in (json.loads(line) for line in f)}
    with open(DATA_PROCESSED / "eval_generation_sample.jsonl", encoding="utf-8") as f:
        sample_rows = [json.loads(line) for line in f]

    print(f"Capturing generation output for {len(sample_rows)} queries (Gemini calls happen ONCE here)")
    print("Building pipeline clients (loading models)...")
    clients = PipelineClients.build()

    out_path = DATA_PROCESSED / "generation_eval_capture.jsonl"
    with open(out_path, "w", encoding="utf-8") as out_f:
        for i, sample in enumerate(sample_rows, start=1):
            qid = sample["query_id"]
            row = query_rows[qid]

            if qid in MULTI_TURN_BUILDERS:
                state, message = MULTI_TURN_BUILDERS[qid]()
            else:
                state, message = None, row["query"]

            for attempt in range(4):
                try:
                    outcome = run_pipeline(message, clients, conversation_state=state)
                    break
                except Exception as e:
                    if attempt == 3:
                        outcome = None
                        break
                    wait = 10 * (attempt + 1)
                    print(f"  ({qid}: {type(e).__name__}, retrying in {wait}s...)")
                    time.sleep(wait)

            if outcome is None:
                print(f"[{i}/{len(sample_rows)}] {qid:6s} FAILED (pipeline)")
                time.sleep(15)
                continue

            if outcome.stopped_at != "ok" or not outcome.results:
                print(f"[{i}/{len(sample_rows)}] {qid:6s} stopped_at={outcome.stopped_at}, no results -- recorded, not generated")
                out_f.write(json.dumps({
                    "query_id": qid, "reason": sample["reason"], "message": message,
                    "stopped_at": outcome.stopped_at, "sub_results": [],
                }, ensure_ascii=False) + "\n")
                out_f.flush()
                time.sleep(15)
                continue

            sub_results = []
            for r in outcome.results:
                for attempt in range(4):
                    try:
                        gen = generate_answer(r.sub_query, r.search_results, r.low_confidence, client=clients.gemini)
                        break
                    except Exception as e:
                        if attempt == 3:
                            gen = None
                            break
                        wait = 10 * (attempt + 1)
                        print(f"  ({qid} generation: {type(e).__name__}, retrying in {wait}s...)")
                        time.sleep(wait)

                if gen is None:
                    print(f"    sub_query generation FAILED for {qid}")
                    continue

                sub_results.append({
                    "sub_query": r.sub_query,
                    "category_used": r.category_used,
                    "low_confidence": r.low_confidence,
                    "context_chunks": [
                        {
                            "page_id": sr["chunk"]["page_id"],
                            "title": sr["chunk"]["title"],
                            "text": sr["chunk"]["text"],
                            "last_updated": sr["chunk"]["last_updated"],
                            "status": sr["chunk"]["status"],
                        }
                        for sr in r.search_results
                    ],
                    "mode": gen.mode,
                    "answer_text": gen.answer_text,
                    "caveat": gen.caveat,
                    "citations": [c.page_id for c in gen.citations],
                })
                time.sleep(15)

            expected = row["expected_page_id"]
            record = {
                "query_id": qid,
                "reason": sample["reason"],
                "message": message,
                "unanswerable": row["unanswerable"],
                "expected_page_id": expected,
                "stopped_at": outcome.stopped_at,
                "sub_results": sub_results,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            print(f"[{i}/{len(sample_rows)}] {qid:6s} captured, "
                  f"{len(sub_results)} sub-result(s), mode(s)={[s['mode'] for s in sub_results]}")

    print(f"\nSaved raw capture to {out_path}")
    print("Now run judge_generation_eval.py to score faithfulness/relevance -- no further generation calls needed.")


if __name__ == "__main__":
    main()
