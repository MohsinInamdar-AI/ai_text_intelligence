"""
Simple RAG implementation.

Uses:
- FAISS for retrieval
- Static knowledge base file
- No external LLM required (CPU-friendly)
"""

import os
from app.services.embeddings import add_texts, search_texts

# -------------------------------------------------
# Knowledge base file path
# -------------------------------------------------
DATA_PATH = "app/data/rag_health_data.txt"

def load_rag_file_once():
    """
    Load and index RAG documents at application startup.
    """
    if not os.path.exists(DATA_PATH):
        print(f"[RAG] File not found: {DATA_PATH}")
        return

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # Split into paragraph chunks
    chunks = [
        chunk.strip()
        for chunk in content.split("\n\n")
        if chunk.strip()
    ]

    if not chunks:
        print("[RAG] No content found in RAG file")
        return

    add_texts(chunks)
    print(f"[RAG] Loaded {len(chunks)} knowledge chunks")

# Load knowledge base once
load_rag_file_once()

def langchain_rag_answer(query: str):
    """
    Retrieve relevant context and return grounded answer.
    """
    contexts = search_texts(query)

    if not contexts:
        return {
            "answer": "No relevant context found in the knowledge base.",
            "contexts": []
        }

    return {
        "answer": f"Based on retrieved information:\n\n{contexts[0]}",
        "contexts": contexts,
        "disclaimer": (
            "This response is for informational purposes only "
            "and not medical advice."
        )
    }
