"""
Sentiment analysis service using Hugging Face Transformers.
"""

from transformers import pipeline

# Pre-trained sentiment model (CPU-friendly)
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

def analyze_sentiment(text: str) -> str:
    """
    Classify sentiment as positive or negative.
    """
    result = sentiment_pipeline(text)[0]
    return result["label"].lower()
