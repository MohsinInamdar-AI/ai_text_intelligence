"""
Text summarization endpoint.

Uses a Transformer-based summarization model.
"""

from fastapi import APIRouter
from app.schemas.summarize import SummarizeRequest, SummarizeResponse
from app.services.summarizer import summarize_text

router = APIRouter()

@router.post("/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest):
    """
    Summarize input text using a pre-trained model (T5).

    Designed to be CPU-friendly.
    """
    return SummarizeResponse(
        summary=summarize_text(req.text)
    )
