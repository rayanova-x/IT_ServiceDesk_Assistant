"""
Small-batch test runner for Step 2 routing, writing results incrementally
to a file so progress is actually visible (the earlier one-liner piped
through `grep` at the end, which buffers everything until the whole
pipeline exits -- looked stuck even if it wasn't).
"""
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from step2_routing import classify_intent  # noqa: E402
from query_parsing import parse_query  # noqa: E402
from google import genai  # noqa: E402
import os  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

BATCH_SIZE = 62

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

with open(PROJECT_ROOT / "data/processed/eval_edge_case_queries.jsonl", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

scorable = [r for r in rows if not (r["letter"] == "G" and r["query_id"] != "G01")]
batch = scorable[:BATCH_SIZE]

out_path = PROJECT_ROOT / "data" / "processed" / "step2_batch_test_results.txt"
with open(out_path, "w", encoding="utf-8") as out:
    for i, r in enumerate(batch, start=1):
        parsed = parse_query(r["query"])
        t0 = time.time()
        result = classify_intent(parsed.cleaned_text, client=client)
        elapsed = time.time() - t0
        line = f"[{i}/{len(batch)}] {r['query_id']:6s} {r['subtype']:35s} -> {result.intent}  ({elapsed:.1f}s)"
        print(line, flush=True)
        out.write(line + "\n")
        out.flush()
        time.sleep(4.5)

print("DONE")
