"""
Frozen dev / held-out split for every retrieval and pipeline eval.

WHY: the eval, calibration and e2e scripts used to share one query set, so
anything tuned on it (the 2.0 confidence threshold, few-shot examples,
prompt wording) was also graded on it. This module gives every script the
same two buckets:

  dev      -- the ONLY queries a prompt, few-shot example, threshold or weight
              may be derived from.
  heldout  -- report-only. Never tune on it. Run it once per finished change.

The assignment is computed once (`python src/eval_split.py --build`), written
to data/processed/eval_split.json and committed. `--build` refuses to
overwrite an existing file without --force, so the held-out set can't be
quietly re-carved after seeing results.

How the split is built:
  - ~30% of GROUPS go to heldout, picked by a hash of the group key (so it is
    reproducible), stratified by source (+ subtype for edge-case queries).
  - Queries with identical text stay together (EC01 repeats C01), and the two
    phrasings per error code / Slack thread stay together, so near-duplicates
    can't straddle the split.
  - Any query whose text appears in a prompt's source file (a few-shot
    example) is pinned to dev -- e.g. C01/E01/F01 in step3_understanding.py.

Known limit: queries about the same KB page can still land on both sides (many
edge cases target the same VPN page). The held-out set tests new phrasings,
not new pages, and at ~50 queries its error bars are wide (+-10-15 points).
Treat it as a sanity check, not a precise estimate.
"""

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))

from config import DATA_DIR  # noqa: E402

DATA_PROCESSED = DATA_DIR / "processed"
SPLIT_PATH = DATA_PROCESSED / "eval_split.json"

HELDOUT_FRACTION = 0.30
SEED = "eval-split-v1"

# Source files whose text may contain a verbatim eval query (few-shot examples).
PROMPT_SOURCES = [
    "step0_context.py", "step2_routing.py", "step3_understanding.py",
    "step_route_or_search.py", "step9_generation.py",
]

HELDOUT_WARNING = (
    "!! Running on the HELD-OUT split: report only. Do not change any prompt, "
    "few-shot example, threshold or weight based on these results."
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _hash(key: str) -> int:
    return int(hashlib.sha256(f"{SEED}|{key}".encode()).hexdigest(), 16)


def load_all_queries() -> list[dict]:
    """Every scorable query from the four eval sources, via eval_retrieval's
    own loaders (imported lazily: eval_retrieval imports this module)."""
    import eval_retrieval as er
    subcat_to_page = er.load_confluence_lookup()
    edge, _blocked = er.load_edge_case_queries()
    return (er.build_ticket_queries(subcat_to_page) + edge
            + er.load_error_code_queries() + er.load_slack_thread_queries())


def _group_key(q: dict) -> str:
    # Pairs of phrasings for one code / thread share a group; everything else
    # groups by identical query text.
    if q["source"] in ("error_code", "slack_thread"):
        return f"{q['source']}|{q['expected_page_id']}"
    return f"text|{_norm(q['query'])}"


def _stratum(q: dict) -> str:
    if q["source"] == "edge_case":
        return f"edge_case|{q['query_id'][0]}"  # letter A-F
    return q["source"]


def _prompt_text() -> str:
    parts = []
    for name in PROMPT_SOURCES:
        path = SRC_DIR / name
        if path.exists():
            parts.append(_norm(path.read_text(encoding="utf-8")))
    return "\n".join(parts)


def build_split(queries: list[dict]) -> dict:
    prompt_text = _prompt_text()

    groups: dict[str, list[dict]] = defaultdict(list)
    for q in queries:
        groups[_group_key(q)].append(q)

    ids = [q["query_id"] for q in queries]
    assert len(ids) == len(set(ids)), "duplicate query_id across eval sources"

    pinned: dict[str, str] = {}
    for key, members in groups.items():
        for q in members:
            if len(_norm(q["query"])) >= 20 and _norm(q["query"]) in prompt_text:
                for m in members:
                    pinned[m["query_id"]] = "query text appears in a prompt source file (few-shot example)"

    by_stratum: dict[str, list[str]] = defaultdict(list)
    for key, members in groups.items():
        by_stratum[_stratum(members[0])].append(key)

    heldout_keys: set[str] = set()
    for _stratum_name, keys in sorted(by_stratum.items()):
        keys = sorted(keys, key=_hash)
        n_held = round(len(keys) * HELDOUT_FRACTION) if len(keys) >= 3 else 0
        taken = 0
        for key in keys:
            if taken >= n_held:
                break
            if any(m["query_id"] in pinned for m in groups[key]):
                continue  # pinned groups stay in dev
            heldout_keys.add(key)
            taken += 1

    splits = {
        q["query_id"]: ("heldout" if _group_key(q) in heldout_keys else "dev")
        for q in queries
    }
    return {
        "version": 1,
        "seed": SEED,
        "heldout_fraction": HELDOUT_FRACTION,
        "pinned_dev": pinned,
        "splits": splits,
    }


_cache: dict | None = None


def _load() -> dict:
    global _cache
    if _cache is None:
        if not SPLIT_PATH.exists():
            raise FileNotFoundError(
                f"{SPLIT_PATH} not found -- run `python src/eval_split.py --build` once."
            )
        _cache = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    return _cache


def get_split(query_id: str) -> str:
    """'dev' or 'heldout'. A query not in the frozen file (e.g. a newly added
    custom case) counts as dev, so it can never silently leak into heldout."""
    return _load()["splits"].get(query_id, "dev")


def select(queries: list[dict], split: str) -> list[dict]:
    """Filter query dicts (each with a 'query_id') to 'dev', 'heldout' or 'all'."""
    if split not in ("dev", "heldout", "all"):
        raise ValueError(f"split must be dev, heldout or all, not {split!r}")
    if split == "all":
        return queries
    if split == "heldout":
        print(HELDOUT_WARNING)
    return [q for q in queries if get_split(q["query_id"]) == split]


def add_split_argument(parser: argparse.ArgumentParser, default: str = "dev") -> None:
    parser.add_argument("--split", choices=["dev", "heldout", "all"], default=default,
                        help="which frozen eval split to run (default: %(default)s)")


def out_name(filename: str, split: str) -> str:
    """Split-specific output filename, so a dev or heldout run never overwrites
    the files the dashboard reads (those are the 'all' outputs)."""
    if split == "all":
        return filename
    stem, dot, ext = filename.rpartition(".")
    return f"{stem}.{split}.{ext}" if dot else f"{filename}.{split}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build", action="store_true", help="carve and write eval_split.json")
    ap.add_argument("--force", action="store_true", help="overwrite an existing split (don't, after seeing results)")
    ap.add_argument("--show", action="store_true", help="print counts of the existing split")
    args = ap.parse_args()

    if args.build:
        if SPLIT_PATH.exists() and not args.force:
            sys.exit(f"{SPLIT_PATH} already exists and is frozen. Refusing to overwrite (use --force only if you have not tuned on it).")
        queries = load_all_queries()
        data = build_split(queries)
        SPLIT_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {SPLIT_PATH}")
    elif not args.show:
        ap.print_help()
        return

    data = _load() if not args.build else json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    splits = data["splits"]
    n_dev = sum(1 for s in splits.values() if s == "dev")
    n_held = len(splits) - n_dev
    print(f"{len(splits)} queries: {n_dev} dev, {n_held} heldout, {len(data['pinned_dev'])} pinned to dev")
    for qid, why in sorted(data["pinned_dev"].items()):
        print(f"  pinned: {qid} ({why})")


if __name__ == "__main__":
    main()
