"""
Visualize the embedding space -- since we can't look at 384 numbers per
chunk directly, we reduce each vector down to just 2 numbers (using PCA)
and plot every chunk as a point. Chunks with similar MEANING should land
near each other, even though PCA never looks at the text itself, only the
numbers.

This is genuinely useful for sanity-checking your RAG pipeline, not just a
pretty picture -- e.g. it's a fast way to visually check whether your
confusable-pair chunks and their "real" counterparts sit close together in
space (which is exactly what would make them hard for retrieval to tell
apart), and whether your 5 IT categories form distinct visual clusters at
all, or overlap heavily.

Install first: pip install scikit-learn matplotlib
"""

import chromadb
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from config import CHROMA_DB_PATH, REPO_ROOT

OUTPUTS_DIR = REPO_ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
collection = client.get_collection("it_helpdesk_kb")

# This time we DO need the actual embedding vectors, not just text/metadata.
data = collection.get(include=["embeddings", "metadatas", "documents"])
embeddings = np.array(data["embeddings"])
categories = [m["category"] for m in data["metadatas"]]

print(f"Loaded {len(embeddings)} embeddings, each with {embeddings.shape[1]} dimensions")

# Reduce 384 dimensions down to just 2, keeping as much of the original
# "spread" between points as possible. This is a real, lossy compression --
# think of it like a shadow of a 3D object on a wall: you lose some
# information, but the overall shape is still recognizable.
pca = PCA(n_components=2)
coords = pca.fit_transform(embeddings)
print(f"PCA explained variance (how much of the real structure survived "
      f"the squeeze to 2D): {pca.explained_variance_ratio_.sum():.1%}")


# ---------------------------------------------------------------------------
# Plot 1: colored by category -- do your 5 IT categories form visually
# distinct clusters, or do they overlap?
# ---------------------------------------------------------------------------
unique_cats = sorted(set(categories))
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_cats)))
cat_to_color = dict(zip(unique_cats, colors))

plt.figure(figsize=(12, 8))
for cat in unique_cats:
    idxs = [i for i, c in enumerate(categories) if c == cat]
    plt.scatter(coords[idxs, 0], coords[idxs, 1],
                label=cat, alpha=0.6, s=30, color=cat_to_color[cat])

plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
plt.title("Chunk embeddings, reduced to 2D (PCA), colored by category")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "embedding_space_by_category.png", dpi=150)
print("\nSaved outputs/embedding_space_by_category.png")


# ---------------------------------------------------------------------------
# Plot 2: colored by status -- do "current" and stale/superseded-style
# chunks sit apart, or are they scattered evenly (meaning status carries
# no semantic signal on its own -- which is expected, since staleness
# isn't a meaning difference, it's a metadata difference)
# ---------------------------------------------------------------------------
statuses = [m["status"] for m in data["metadatas"]]
unique_statuses = sorted(set(statuses))
status_colors = {"current": "steelblue", "trashed": "lightgray"}

plt.figure(figsize=(12, 8))
for status in unique_statuses:
    idxs = [i for i, s in enumerate(statuses) if s == status]
    plt.scatter(coords[idxs, 0], coords[idxs, 1],
                label=status, alpha=0.6, s=30,
                color=status_colors.get(status, "orange"))

plt.legend()
plt.title("Same embedding space, colored by page status")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "embedding_space_by_status.png", dpi=150)
print("Saved outputs/embedding_space_by_status.png")

plt.show()
