"""
Keyword Extraction Service using spaCy.

WHY THIS FILE EXISTS:
- This service extracts meaningful keywords from input text.
- It is used by the /analyze endpoint to return top keywords.

IMPORTANT DESIGN DECISION:
- spaCy models are NOT loaded at import time.
- Instead, we use lazy loading to avoid CI/CD and Docker failures.

This pattern is commonly used in production ML systems
where models are large and environment-dependent.
"""

import spacy
import subprocess
import sys

# Cached spaCy model instance (loaded only once per process)
_nlp = None


def get_nlp():
    """
    Lazily load the spaCy NLP model.

    WHY LAZY LOADING?
    - Prevents application crashes during CI/CD pipelines
    - Prevents Docker container startup failures
    - Ensures model is downloaded only when actually needed

    BEHAVIOR:
    - Tries to load 'en_core_web_sm'
    - If missing, downloads it automatically
    - Caches the model for reuse
    """
    global _nlp

    if _nlp is None:
        try:
            # Attempt to load the spaCy model
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            # If model is not found, download it dynamically
            subprocess.check_call(
                [sys.executable, "-m", "spacy", "download", "en_core_web_sm"]
            )
            _nlp = spacy.load("en_core_web_sm")

    return _nlp


def extract_keywords(text: str, top_k: int = 5):
    """
    Extract top keywords from input text.

    METHOD:
    - Uses spaCy noun chunks (lightweight and fast)
    - Filters out overly long phrases
    - Removes duplicates while preserving order

    PARAMETERS:
    - text (str): Input text from user
    - top_k (int): Number of keywords to return (default = 5)

    RETURNS:
    - List[str]: Clean list of extracted keywords
    """

    # Load NLP model safely
    nlp = get_nlp()

    # Process the input text
    doc = nlp(text)

    # Extract noun-based keywords (max 3 words each)
    keywords = [
        chunk.text.lower()
        for chunk in doc.noun_chunks
        if len(chunk.text.split()) <= 3
    ]

    # Remove duplicates and limit to top_k
    return list(dict.fromkeys(keywords))[:top_k]
