"""
templates.py

Templates and light paraphrase utilities to produce LLM-like replies without external LLM.
Uses deterministic strategies: fixed templates + synonym substitution using NLTK WordNet (conservative).
"""

from typing import List
import random
import nltk
from nltk.corpus import wordnet as wn

# Conservative templates per intent fallback (generic)
GENERIC_TEMPLATES = [
    "I'm sorry you're experiencing {symptoms}. {description} You might try {precaution}. If it worsens, seek medical attention. Not a medical professional — for emergencies, call local emergency services.",
    "{description} Based on what you said ({symptoms}), consider: {precaution}. If symptoms are severe, get urgent care. Not a medical professional — for emergencies, call local emergency services.",
    "Thanks for sharing. {description} For these symptoms ({symptoms}), {precaution}. If you're worried or symptoms are severe, please see a clinician. Not a medical professional — for emergencies, call local emergency services."
]

# small set of high-precision synonym substitution map for safety (avoid changing medical terms)
SAFE_SYNONYM_MAP = {
    "itching": ["itch", "irritation"],
    "fever": ["high temperature"],
    "pain": ["ache", "discomfort"],
    "rash": ["skin rash", "red patches"],
    "bleeding": ["hemorrhage (seek care)"]
}

def safe_synonym_substitute(text: str, max_replacements: int = 2) -> str:
    """
    Replace a small number of tokens using SAFE_SYNONYM_MAP deterministically.
    This avoids WordNet over-generalization.
    """
    out = text
    replacements = 0
    for key, syns in SAFE_SYNONYM_MAP.items():
        if replacements >= max_replacements:
            break
        if key in out:
            # deterministic: pick first synonym
            out = out.replace(key, syns[0], 1)
            replacements += 1
    return out

def choose_template(intent: str = None) -> str:
    """Return a template string; for now use generic templates. Could be extended per-intent."""
    # deterministic selection: choose based on hash of intent
    idx = 0
    if intent:
        idx = abs(hash(intent)) % len(GENERIC_TEMPLATES)
    return GENERIC_TEMPLATES[idx]

def render_reply(intent: str, symptoms: List[str], description: str, precautions: List[str]) -> str:
    """Render a human-like reply combining pieces."""
    # create symptom phrase
    if not symptoms:
        sympt_phrase = "your symptoms"
    elif len(symptoms) == 1:
        sympt_phrase = symptoms[0]
    elif len(symptoms) == 2:
        sympt_phrase = f"{symptoms[0]} and {symptoms[1]}"
    else:
        sympt_phrase = ", ".join(symptoms[:-1]) + f", and {symptoms[-1]}"

    # select a single concise precaution phrase
    prec = ""
    if precautions:
        # deterministic choose first non-empty precaution and join up to two
        prec_list = [p for p in precautions if p and str(p).strip().lower() not in ("nan", "")]
        prec = "; ".join(prec_list[:2]) if prec_list else "follow routine care"
    else:
        prec = "follow routine care and consult a clinician if it worsens"

    template = choose_template(intent)
    reply = template.format(symptoms=sympt_phrase, description=description or "", precaution=prec)
    # apply safe synonym substitution for mild variety
    reply = safe_synonym_substitute(reply)
    return reply
