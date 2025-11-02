"""
preprocessing.py

Helpers to:
- auto-detect dataset columns
- load dataset and system CSVs robustly
- normalize symptom tokens
- synthesize natural-language utterances from symptom lists
- build TF-IDF vectorizer
- build symptom -> intent frequency map (used for rule-based overrides)
"""

from typing import List, Tuple, Dict
import re
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# Attempt to ensure WordNet availability if needed downstream
try:
    nltk.data.find("corpora/wordnet")
except Exception:
    try:
        nltk.download("wordnet", quiet=True)
    except Exception:
        pass

COMMON_INTENT_COLS = {"intent", "condition", "label", "disease", "diagnosis", "disease_name"}
COMMON_MESSAGE_COLS = {"text", "pattern", "message", "symptoms", "symptom_list"}
COMMON_RESPONSE_COLS = {"response", "responses", "reply"}

# Robust CSV reader
def read_csv_fuzzy(path: str) -> pd.DataFrame:
    """Read CSV or tab-separated file robustly; returns a DataFrame with string cells."""
    for sep in [",", "\t", ";"]:
        try:
            df = pd.read_csv(path, sep=sep, header=0, encoding="utf-8", dtype=str)
            return df.fillna("")
        except Exception:
            continue
    # fallback: read raw lines -> one-column df
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [l.strip() for l in f if l.strip()]
    return pd.DataFrame({"raw": lines})

def detect_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    """
    Return (intent_col, message_col, response_col_or_none)
    If message_col not available, synthesizes '__synth_message' joining columns 1..end.
    """
    cols = [c.lower().strip() for c in df.columns]
    intent_col = None
    message_col = None
    response_col = None

    for c, orig in zip(cols, df.columns):
        if c in COMMON_INTENT_COLS and intent_col is None:
            intent_col = orig
        if c in COMMON_MESSAGE_COLS and message_col is None:
            message_col = orig
        if c in COMMON_RESPONSE_COLS and response_col is None:
            response_col = orig

    if intent_col is None:
        intent_col = df.columns[0]

    if message_col is None:
        if df.shape[1] > 1:
            message_col = "__synth_message"
            df[message_col] = df.apply(lambda r: ", ".join([str(x).strip() for x in r[1:].values if str(x).strip() and str(x).strip().lower() not in ("nan", "na")]), axis=1)
        else:
            message_col = intent_col

    return intent_col, message_col, response_col

# Symptom normalization
SYM_TOKEN_CLEAN = re.compile(r"[_\-\s]+")

def normalize_symptom_token(tok: str) -> str:
    tok = str(tok).strip()
    if tok == "" or tok.lower() in ("nan", "na"):
        return ""
    tok = tok.replace("_", " ").replace("-", " ")
    tok = SYM_TOKEN_CLEAN.sub(" ", tok)
    tok = tok.lower()
    tok = tok.strip()
    return tok

def split_symptoms_from_message(message: str) -> List[str]:
    """Given a message that may be comma-separated tokens, split and normalize."""
    if not isinstance(message, str) or message.strip() == "":
        return []
    tokens = re.split(r"[,\;/\|]+", message)
    tokens = [normalize_symptom_token(t) for t in tokens]
    tokens = [t for t in tokens if t]
    return tokens

# Utterance synthesis templates
_SYNTH_TEMPLATES = [
    "I have been experiencing {symptoms}.",
    "I noticed {symptoms} recently.",
    "I'm getting {symptoms} and not sure why.",
    "I have {symptoms} — what could this be?",
    "For a few days I've had {symptoms}.",
]

def synthesize_utterances_from_symptoms(symptoms: List[str], n_variations: int = 4) -> List[str]:
    if not symptoms:
        return []
    if len(symptoms) == 1:
        sympt_str = symptoms[0]
    elif len(symptoms) == 2:
        sympt_str = f"{symptoms[0]} and {symptoms[1]}"
    else:
        sympt_str = ", ".join(symptoms[:-1]) + f", and {symptoms[-1]}"
    variations = []
    templates = _SYNTH_TEMPLATES.copy()
    random.shuffle(templates)
    for i in range(min(n_variations, len(templates))):
        variations.append(templates[i].format(symptoms=sympt_str))
    if len(symptoms) > 2 and len(variations) < n_variations:
        perm = symptoms[-1:] + symptoms[:-1]
        if len(perm) == 1:
            pstr = perm[0]
        elif len(perm) == 2:
            pstr = f"{perm[0]} and {perm[1]}"
        else:
            pstr = ", ".join(perm[:-1]) + f", and {perm[-1]}"
        variations.append(f"I am noticing {pstr}.")
    return list(dict.fromkeys(variations))

def build_tfidf(corpus: List[str], max_features: int = 20000, ngram_range=(1,2)) -> TfidfVectorizer:
    """Build and return a TF-IDF vectorizer fitted on corpus."""
    v = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    v.fit(corpus)
    return v

# ---------------- Symptom -> Intent map ----------------

def build_symptom_intent_map(df: pd.DataFrame, intent_col: str, message_col: str) -> Dict[str, Dict[str, int]]:
    """
    Build a mapping: symptom_token -> {intent: count}
    Uses the raw dataframe (assumes message_col may be '__synth_message' produced earlier).
    Only tokens of length >= 2 characters are kept to reduce noise.
    """
    symptom_map: Dict[str, Dict[str, int]] = {}
    if message_col not in df.columns:
        return symptom_map
    for _, row in df.iterrows():
        intent = str(row[intent_col]).strip()
        message = str(row[message_col]).strip() if message_col in df.columns else ""
        # If the message looks like an utterance (contains spaces) attempt to parse symptom tokens heuristically:
        # - prefer comma/semicolon-separated tokens; otherwise, try to extract tokens like 'fever', 'rash' using word tokens
        tokens = []
        if "," in message or ";" in message or "/" in message or "|" in message:
            tokens = split_symptoms_from_message(message)
        else:
            # fallback: split into words and keep short tokens that likely represent symptoms (heuristic)
            toks = re.findall(r"[A-Za-z0-9_\-]+", message)
            tokens = [normalize_symptom_token(t) for t in toks if len(t) >= 2]
        for tok in tokens:
            if not tok:
                continue
            inner = symptom_map.setdefault(tok, {})
            inner[intent] = inner.get(intent, 0) + 1
    return symptom_map
