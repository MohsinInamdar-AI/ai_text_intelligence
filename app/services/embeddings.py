"""
Embedding and vector search service.

Responsible for:
- Encoding text into vectors
- Managing FAISS index
- Performing semantic search
"""

from typing import List
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# -------------------------------------------------
# CPU-friendly embedding model
# -------------------------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------------------------
# In-memory FAISS storage
# -------------------------------------------------
texts: List[str] = []
index = None

def add_texts(new_texts: List[str]):
    """
    Add multiple texts into FAISS index safely.
    """
    global texts, index

    if not new_texts:
        return

    embeddings = model.encode(new_texts)
    embeddings = np.array(embeddings).astype("float32")

    if index is None:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)

    index.add(embeddings)
    texts.extend(new_texts)

def add_text(text: str):
    """
    Backward-compatible helper for adding a single text.
    """
    add_texts([text])

def search_texts(query: str, k: int = 3) -> List[str]:
    """
    Perform safe semantic search.

    Includes guards against:
    - Empty index
    - Invalid FAISS indices
    """
    global index, texts

    if index is None or index.ntotal == 0 or not texts:
        return []

    k = min(k, len(texts), index.ntotal)

    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    _, indices = index.search(query_embedding, k)

    results = []
    for idx in indices[0]:
        if idx >= 0 and idx < len(texts):
            results.append(texts[idx])

    return results
