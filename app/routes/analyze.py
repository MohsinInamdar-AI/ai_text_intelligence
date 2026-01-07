"""
Text analysis endpoint.

This route:
- Accepts raw text
- Returns sentiment + top keywords
"""

from fastapi import APIRouter
from app.schemas.analyze import AnalyzeRequest, AnalyzeResponse
from app.services.sentiment import analyze_sentiment
from app.services.keywords import extract_keywords

router = APIRouter()

@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_text(req: AnalyzeRequest):
    """
    Analyze text sentiment and extract keywords.

    Flow:
    1. Run sentiment classification (DistilBERT)
    2. Extract keywords using spaCy
    """
    sentiment = analyze_sentiment(req.text)
    keywords = extract_keywords(req.text)

    return AnalyzeResponse(
        sentiment=sentiment,
        keywords=keywords
    )
