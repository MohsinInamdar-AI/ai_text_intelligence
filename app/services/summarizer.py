"""
Text summarization service using Transformer model.
"""

from transformers import pipeline

summarizer = pipeline(
    "summarization",
    model="t5-small"
)

def summarize_text(text: str) -> str:
    """
    Generate a concise summary of input text.
    """
    summary = summarizer(
        text,
        max_length=100,
        min_length=30,
        do_sample=False
    )
    return summary[0]["summary_text"]
