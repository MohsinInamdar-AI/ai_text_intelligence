"""
Main application entry point.

This file:
- Creates the FastAPI app
- Registers all API routes
- Exposes Swagger UI at /docs
"""


from fastapi import FastAPI

# Import route modules
from app.routes import analyze
from app.routes import summarize
from app.routes import semantic_search
from app.routes import langchain_rag

# -------------------------------------------------
# FastAPI application configuration
# -------------------------------------------------

app = FastAPI(
    title="AI Text Intelligence",
    description="NLP-powered API with Sentiment Analysis, Summarization, Semantic Search, and RAG",
    version="1.0.0",
)

# -------------------------------------------------
# Register API routers
# -------------------------------------------------

app.include_router(analyze.router, tags=["Analysis"])
app.include_router(summarize.router, tags=["Summarization"])
app.include_router(semantic_search.router, tags=["Semantic Search"])
app.include_router(langchain_rag.router, tags=["RAG"])

# -------------------------------------------------
# Health check / root endpoint
# -------------------------------------------------

@app.get("/")
def root():
    """
    Simple health endpoint to verify API is running.
    """
    return {
        "message": "AI Text Intelligence API is running",
        "docs": "/docs"
    }