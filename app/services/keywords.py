"""
Keyword Extraction Service using spaCy.

This service uses lazy loading to avoid import-time failures
in CI/CD and Docker environments.
"""

import spacy
import subprocess
import sys

_nlp = None  # Cached spaCy model


def get_nlp():
    """
    Lazily load the spaCy NLP model.
    Downloads it automatically if missing.
    """
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            subprocess.check_call(
                [sys.executable, "-m", "spacy", "download", "en_core_web_sm"]
            )
            _nlp = spacy.load("en_core_web_sm")
    return _nlp


def extract_keywords(text: str, top_k: int = 5):
    """
    Extract top keywords using noun chunks.
    """
    nlp = get_nlp()
    doc = nlp(text)

    keywords = [
        chunk.text.lower()
        for chunk in doc.noun_chunks
        if len(chunk.text.split()) <= 3
    ]

    return list(dict.fromkeys(keywords))[:top_k]
