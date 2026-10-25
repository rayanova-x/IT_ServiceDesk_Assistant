"""
Quick, readable look at what's actually inside your Chroma collection --
no embeddings math, just: what chunks exist, what metadata do they carry,
and a category breakdown.

Run this first, before the embedding-space plot -- it's the "read the
table" step before the "see the picture" step.
"""

import chromadb
from collections import Counter
import csv

from config import CHROMA_DB_PATH, DATA_DIR

DATA_PROCESSED = DATA_DIR / "processed"

client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
collection = client.get_collection("it_helpdesk_kb")

print(f"Total chunks stored: {collection.count()}\n")

# include=["documents", "metadatas"] -- we don't need the raw embedding
# vectors for this readable view, just the text and metadata.
data = collection.get(include=["documents", "metadatas"])


# ---------------------------------------------------------------------------
# 1. Print a handful of chunks in full, so you can see exactly what one
#    stored record actually looks like.
# ---------------------------------------------------------------------------
print("=== Sample chunks (first 5) ===\n")
for i in range(min(5, len(data["ids"]))):
    m = data["metadatas"][i]
    print(f"[{data['ids'][i]}]")
    print(f"  Title:      {m['title']}")
    print(f"  Category:   {m['category']}")
    print(f"  Status:     {m['status']}  |  Version: {m['version_number']}  |  Updated: {m['last_updated']}")
    print(f"  Labels:     {m['labels']}")
    print(f"  Restricted: {m['restricted_to']}")
    print(f"  Text:       {data['documents'][i][:150]}...")
    print()


# ---------------------------------------------------------------------------
# 2. Category and status breakdown -- a quick sanity check on the shape
#    of what's actually in there.
# ---------------------------------------------------------------------------
categories = Counter(m["category"] for m in data["metadatas"])
statuses = Counter(m["status"] for m in data["metadatas"])

print("=== Category breakdown ===")
for cat, count in categories.most_common():
    print(f"  {cat}: {count}")

print("\n=== Status breakdown ===")
for status, count in statuses.items():
    print(f"  {status}: {count}")


# ---------------------------------------------------------------------------
# 3. Export EVERYTHING to a CSV -- easiest way to actually browse all 247
#    chunks yourself (open it in Excel, sort/filter by column, etc.)
# ---------------------------------------------------------------------------
with open(DATA_PROCESSED / "chroma_contents.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["chunk_id", "title", "category", "status", "version_number",
                      "last_updated", "labels", "restricted_to", "header_path", "text_preview"])
    for i in range(len(data["ids"])):
        m = data["metadatas"][i]
        writer.writerow([
            data["ids"][i], m["title"], m["category"], m["status"], m["version_number"],
            m["last_updated"], m["labels"], m["restricted_to"], m["header_path"],
            data["documents"][i][:200].replace("\n", " "),
        ])

print(f"\nExported all {len(data['ids'])} chunks to data/processed/chroma_contents.csv -- open it in Excel to browse everything.")
