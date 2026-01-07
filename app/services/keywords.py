
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_keywords(text: str):
    doc = nlp(text)
    return list(set([token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]))[:5]
