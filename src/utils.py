"""
utils.py

Small utilities: save/load pickles, ensure directories, emergency keyword detection,
and basic morphological normalization for user queries.
"""

import os
import joblib
import re
from typing import List

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_pickle(obj, path: str):
    ensure_dir(os.path.dirname(path) or ".")
    joblib.dump(obj, path)

def load_pickle(path: str):
    return joblib.load(path)

_EMERGENCY_KEYWORDS = [
    "chest pain", "severe chest pain", "shortness of breath", "difficulty breathing",
    "severe bleeding", "unconscious", "loss of consciousness", "blackout", "no pulse",
    "not breathing", "severe allergic reaction", "anaphylaxis", "severe burn", "suicidal",
    "sudden weakness", "sudden numbness", "slurred speech"
]

def is_emergency_text(text: str) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    t = text.lower()
    # match any emergency keyword as substring
    for kw in _EMERGENCY_KEYWORDS:
        if kw in t:
            return True
    # also detect phrases with numbers like "bleeding heavily"
    if re.search(r"\b(bleeding heavily|bleeding profusely|cannot breathe|can't breathe)\b", t):
        return True
    return False

def normalize_query_text(text: str) -> str:
    """Light normalization: collapse whitespace, remove repeated punctuation, strip."""
    if not isinstance(text, str):
        text = str(text)
    txt = re.sub(r"\s+", " ", text).strip()
    txt = re.sub(r"([?.!]){2,}", r"\1", txt)
    return txt
