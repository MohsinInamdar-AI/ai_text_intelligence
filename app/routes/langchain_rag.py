"""
RAG (Retrieval-Augmented Generation) endpoint.

Accepts a JSON request body for clean Swagger UI experience.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from app.services.langchain_rag import langchain_rag_answer

router = APIRouter()

class RAGRequest(BaseModel):
    """
    Request schema for RAG queries.
    """
    query: str

@router.post("/rag-query")
def rag_query(request: RAGRequest):
    """
    Answer user questions using retrieved knowledge base context.
    """
    return langchain_rag_answer(request.query)
