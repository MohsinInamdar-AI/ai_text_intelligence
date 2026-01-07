"""
Semantic Search API Route

This endpoint demonstrates:
- Text embedding using Sentence Transformers
- Vector similarity search using FAISS
- How semantic search differs from keyword-based search

The endpoint accepts JSON input for better Swagger UX
and consistency with other APIs.
"""

from fastapi import APIRouter
from pydantic import BaseModel

# Import embedding services
from app.services.embeddings import add_text, search_texts

router = APIRouter()

# -------------------------------------------------
# Request schema (JSON body)
# -------------------------------------------------
class SemanticSearchRequest(BaseModel):
    """
    Request body for semantic search.

    Example:
    {
        "text": "Preventive healthcare improves long-term health."
    }
    """
    text: str

# -------------------------------------------------
# Semantic Search Endpoint
# -------------------------------------------------
@router.post("/semantic-search")
def semantic_search(request: SemanticSearchRequest):
    """
    Perform semantic search using vector embeddings.

    Workflow:
    1. Convert input text into embeddings
    2. Store embeddings in FAISS index
    3. Search for semantically similar texts
    4. Return closest matches

    This endpoint is CPU-friendly and runs locally.
    """

    # Store text embedding in vector database
    add_text(request.text)

    # Perform similarity search using the same text as query
    results = search_texts(request.text)

    return {
        "query": request.text,
        "results": results
    }
