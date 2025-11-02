"""
src/inference.py

Updated InferenceEngine implementing refined response logic:
 - Adaptive prediction: 3+ symptoms -> single disease, 1-2 symptoms -> up to 3 diseases
 - Clean structured reply format (disease-wise explanations and precautions)
 - No echoing of user input
 - Medication filtering for safety
"""

import os
import re
import joblib
import difflib
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter

# Try to reuse helpers from preprocessing/utils if available
try:
    from preprocessing import normalize_symptom_token, split_symptoms_from_message, read_csv_fuzzy
except Exception:
    def normalize_symptom_token(tok: str) -> str:
        tok = str(tok).strip()
        if tok == "" or tok.lower() in ("nan", "na"):
            return ""
        tok = tok.replace("_", " ").replace("-", " ")
        tok = re.sub(r"[_\-\s]+", " ", tok)
        tok = tok.lower().strip()
        return tok

    def split_symptoms_from_message(message: str) -> List[str]:
        if not isinstance(message, str) or message.strip() == "":
            return []
        tokens = re.split(r"[,\;/\|]+", message)
        tokens = [normalize_symptom_token(t) for t in tokens]
        tokens = [t for t in tokens if t]
        return tokens

    def read_csv_fuzzy(path: str):
        import pandas as pd
        for sep in [",", "\t", ";"]:
            try:
                df = pd.read_csv(path, sep=sep, header=0, dtype=str, encoding="utf-8").fillna("")
                return df
            except Exception:
                continue
        raise FileNotFoundError(path)

try:
    from utils import is_emergency_text
except Exception:
    def is_emergency_text(text: str) -> bool:
        t = str(text).lower()
        em = ["chest pain", "not breathing", "can't breathe", "cannot breathe", "severe bleeding", "loss of consciousness", "unconscious"]
        return any(kw in t for kw in em)

# ---------- constants ----------
DEFAULT_TOP_K = 3
FUZZY_SYM_CUTOFF = 0.75  # fuzzy match threshold for symptom token variants
MAX_SUGGESTIONS = 3
_TOKEN_SPLIT_RE = re.compile(r"[,\;/\|\.\?\n]")

# medication-like keywords to filter from precautions (safety)
_MEDICATION_KEYWORDS = {
    "antibiotic", "antibiotics", "antifungal", "antivirals", "aspirin", "ibuprofen", "paracetamol",
    "acetaminophen", "tablet", "capsule", "mg", "ml", "injection", "dose", "take", "apply", "ointment",
    "ointment", "cream", "spray", "prescription", "medication", "medications", "pills"
}

# ---------- helpers ----------
def _norm_tokens(text: str) -> List[str]:
    if not text:
        return []
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return toks

def _first_sentence(text: str) -> str:
    if not text:
        return ""
    parts = re.split(r"[\.!?]\s*", text.strip())
    if parts:
        s = parts[0].strip()
        if s and not s.endswith("."):
            s = s + "."
        return s
    return text

def _sanitize_precautions(items: List[str]) -> List[str]:
    """
    Remove medication-specific items or instructions. Keep generic, safety-first advice.
    If all items are medication-like, return a safe fallback list.
    """
    out = []
    for it in items:
        if not it:
            continue
        low = it.lower()
        # if any medication word appears as whole word or substring, treat as medication-like
        med_like = False
        for kw in _MEDICATION_KEYWORDS:
            if re.search(r"\b" + re.escape(kw) + r"\b", low):
                med_like = True
                break
        if med_like:
            continue
        # Avoid short non-informative items
        if len(it.strip()) < 3:
            continue
        out.append(it.strip())
    # If none remain, provide safe generic precautions
    if not out:
        return [
            "Rest and monitor symptoms",
            "Stay hydrated",
            "Seek medical advice from a clinician if symptoms persist or worsen"
        ]
    return out

def _pretty_bullets(items: List[str], max_items: int = 6) -> List[str]:
    out = []
    for i, it in enumerate(items):
        if i >= max_items:
            break
        out.append(f"• {it}")
    return out

# ---------- main engine ----------
class InferenceEngine:
    def __init__(self, models_dir: str = "models", threshold: float = 0.5):
        self.models_dir = models_dir or "."
        self.threshold = float(threshold)
        self.system_metadata: Dict[str, Any] = {}
        self.symptom_map: Dict[str, Dict[str, int]] = {}
        self.disease_symptoms: Dict[str, Counter] = {}
        self.disease_total_symptoms: Dict[str, int] = {}
        self.descriptions: Dict[str, str] = {}
        self.precautions: Dict[str, List[str]] = {}
        self.symptom_vocab: List[str] = []
        self._load_resources()

    # -------- Loading resources (system_metadata.pkl OR CSVs) --------
    def _load_resources(self):
        meta_path = os.path.join(self.models_dir, "system_metadata.pkl")
        if os.path.exists(meta_path):
            try:
                self.system_metadata = joblib.load(meta_path)
                self.descriptions = self.system_metadata.get("descriptions", {}) or {}
                self.precautions = self.system_metadata.get("precautions", {}) or {}
                symptom_map = self.system_metadata.get("symptom_map", {}) or {}
                self.symptom_map = symptom_map
                ds = defaultdict(Counter)
                for sym, intents in symptom_map.items():
                    for d, cnt in intents.items():
                        ds[d][sym] += int(cnt)
                self.disease_symptoms = dict(ds)
                for d, c in self.disease_symptoms.items():
                    self.disease_total_symptoms[d] = sum(c.values()) or len(c) or 1
                self.symptom_vocab = sorted(list({s for s in symptom_map.keys() if s}))
                return
            except Exception:
                # fallback to CSV loading
                pass

        # Attempt CSVs
        dataset_candidates = [
            os.path.join(self.models_dir, "dataset.csv"),
            os.path.join(self.models_dir, "data.csv"),
            os.path.join(".", "dataset.csv"),
            os.path.join(".", "data.csv"),
        ]
        desc_candidates = [
            os.path.join(self.models_dir, "symptom_description.csv"),
            os.path.join(".", "symptom_description.csv"),
        ]
        prec_candidates = [
            os.path.join(self.models_dir, "symptom_precaution.csv"),
            os.path.join(".", "symptom_precaution.csv"),
        ]

        dataset_path = next((p for p in dataset_candidates if os.path.exists(p)), None)
        if dataset_path:
            try:
                df = read_csv_fuzzy(dataset_path)
                lower_cols = [c.lower().strip() for c in df.columns]
                intent_col = None
                message_col = None
                for c, orig in zip(lower_cols, df.columns):
                    if c in {"intent", "condition", "disease", "label", "diagnosis", "disease_name"} and intent_col is None:
                        intent_col = orig
                    if c in {"text", "pattern", "message", "symptoms", "symptom_list"} and message_col is None:
                        message_col = orig
                if intent_col is None:
                    intent_col = df.columns[0]
                if message_col is None:
                    if df.shape[1] > 1:
                        message_col = "__synth_message"
                        df[message_col] = df.apply(lambda r: ", ".join([str(x).strip() for x in r[1:].values if str(x).strip() and str(x).strip().lower() not in ("nan", "na")]), axis=1)
                    else:
                        message_col = intent_col

                ds = defaultdict(Counter)
                for _, row in df.iterrows():
                    disease = str(row[intent_col]).strip()
                    message = str(row.get(message_col, "")).strip()
                    toks = []
                    if any(sep in message for sep in [",", ";", "/", "|"]):
                        toks = split_symptoms_from_message(message)
                    else:
                        toks_raw = re.findall(r"[A-Za-z0-9_\-]+", message)
                        toks = [normalize_symptom_token(t) for t in toks_raw if len(t) >= 2]
                    for t in toks:
                        if not t:
                            continue
                        ds[disease][t] += 1
                self.disease_symptoms = dict(ds)
                for d, c in self.disease_symptoms.items():
                    self.disease_total_symptoms[d] = sum(c.values()) or len(c) or 1
                sym_map = {}
                for d, counter in self.disease_symptoms.items():
                    for s, cnt in counter.items():
                        sym_map.setdefault(s, {})[d] = sym_map.setdefault(s, {}).get(d, 0) + cnt
                self.symptom_map = sym_map
                self.symptom_vocab = sorted(list(set(sym_map.keys())))
            except Exception:
                self.symptom_map = {}
                self.disease_symptoms = {}
                self.symptom_vocab = []
        else:
            self.symptom_map = {}
            self.disease_symptoms = {}
            self.symptom_vocab = []

        # descriptions
        desc_path = next((p for p in desc_candidates if os.path.exists(p)), None)
        if desc_path:
            try:
                df_desc = read_csv_fuzzy(desc_path)
                if df_desc.shape[1] >= 2:
                    kcol = df_desc.columns[0]
                    vcol = df_desc.columns[1]
                    for _, r in df_desc.iterrows():
                        k = str(r[kcol]).strip()
                        v = str(r[vcol]).strip()
                        if k:
                            self.descriptions[k] = v
                else:
                    for _, r in df_desc.iterrows():
                        raw = str(r[df_desc.columns[0]])
                        if "\t" in raw:
                            k, v = raw.split("\t", 1)
                            self.descriptions[k.strip()] = v.strip()
            except Exception:
                self.descriptions = {}

        # precautions
        prec_path = next((p for p in prec_candidates if os.path.exists(p)), None)
        if prec_path:
            try:
                df_prec = read_csv_fuzzy(prec_path)
                if df_prec.shape[1] >= 2:
                    for _, r in df_prec.iterrows():
                        key = str(r[df_prec.columns[0]]).strip()
                        vals = []
                        for c in df_prec.columns[1:]:
                            v = str(r[c]).strip()
                            if v and v.lower() not in ("nan", "na"):
                                vals.append(v)
                        if key:
                            self.precautions[key] = vals
                else:
                    for _, r in df_prec.iterrows():
                        raw = str(r[df_prec.columns[0]])
                        if "\t" in raw:
                            k, v = raw.split("\t", 1)
                            self.precautions[k.strip()] = [v.strip()]
            except Exception:
                self.precautions = {}

        # ensure totals present
        for d in self.disease_symptoms:
            self.disease_total_symptoms.setdefault(d, sum(self.disease_symptoms[d].values()) or len(self.disease_symptoms[d]) or 1)

    # -------- symptom extraction & mapping --------
    def _extract_candidate_tokens(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        text = text.strip().lower()
        parts = [p.strip() for p in _TOKEN_SPLIT_RE.split(text) if p.strip()]
        candidates = []
        for p in parts:
            subparts = re.split(r"\band\b|\bwith\b|\bplus\b", p)
            for sp in subparts:
                sp = sp.strip()
                if not sp:
                    continue
                words = re.findall(r"[a-z0-9]+", sp)
                if not words:
                    continue
                for n in range(1, min(4, len(words) + 1)):
                    for i in range(len(words) - n + 1):
                        gram = " ".join(words[i : i + n])
                        candidates.append(gram)
        seen = set()
        out = []
        for c in candidates:
            n = normalize_symptom_token(c)
            if n and n not in seen:
                seen.add(n)
                out.append(n)
        return out

    def _map_to_known_symptoms(self, candidate_tokens: List[str]) -> List[str]:
        if not candidate_tokens:
            return []
        matched = []
        vocab = self.symptom_vocab or []
        low_vocab = {v.lower(): v for v in vocab}
        for tok in candidate_tokens:
            if not tok:
                continue
            if tok in low_vocab:
                matched.append(low_vocab[tok])
                continue
            tok_tokens = set(_norm_tokens(tok))
            best = None
            best_overlap = 0
            for v_lower, v_orig in low_vocab.items():
                v_tokens = set(_norm_tokens(v_lower))
                overlap = len(tok_tokens & v_tokens)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best = v_orig
            if best and best_overlap >= 1:
                matched.append(best)
                continue
            if vocab:
                matches = difflib.get_close_matches(tok, list(low_vocab.keys()), n=1, cutoff=FUZZY_SYM_CUTOFF)
                if matches:
                    matched.append(low_vocab[matches[0]])
                    continue
            matched.append(tok)
        # dedupe preserve order
        seen = set()
        out = []
        for m in matched:
            if m not in seen:
                seen.add(m)
                out.append(m)
        return out

    # -------- scoring --------
    def _score_diseases(self, matched_symptoms: List[str]) -> List[Tuple[str, float, List[str]]]:
        scores = []
        if not matched_symptoms:
            return []
        mset = set(matched_symptoms)
        for d, counter in self.disease_symptoms.items():
            if not counter:
                continue
            total = float(self.disease_total_symptoms.get(d, sum(counter.values()) or len(counter) or 1))
            matched_items = []
            ssum = 0.0
            for s, cnt in counter.items():
                if s in mset:
                    matched_items.append(s)
                    ssum += min(cnt, 3)
            raw_frac = (ssum / total) if total > 0 else 0.0
            uniq_boost = min(len(matched_items) / (len(counter) + 1), 0.5)
            score = raw_frac * 0.9 + uniq_boost * 0.1
            score = max(0.0, min(1.0, float(score)))
            if matched_items:
                scores.append((d, score, matched_items))
        scores.sort(key=lambda x: (x[1], len(x[2])), reverse=True)
        return scores

    # -------- public infer --------
    def infer(self, query: str) -> Dict[str, Any]:
        q = (query or "").strip()
        if not q:
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "reply": "I didn't get any symptoms. Could you state them (e.g., 'fever, cough')?",
                "description": "",
                "precaution": [],
                "emergency": False,
            }

        # Emergency detection
        try:
            if is_emergency_text(q):
                reply = (
                    "⚠️ EMERGENCY ALERT: This message contains signs of a possible medical emergency "
                    "such as severe chest pain or breathing difficulty. Please call local emergency services or go to the nearest emergency department immediately."
                )
                return {
                    "intent": "emergency",
                    "confidence": 1.0,
                    "reply": reply,
                    "description": "",
                    "precaution": [],
                    "emergency": True,
                }
        except Exception:
            pass

        # Extract and map symptoms (do not echo input)
        candidates = self._extract_candidate_tokens(q)
        mapped = self._map_to_known_symptoms(candidates)
        # Filter common stop-words left-over
        mapped = [m for m in mapped if len(m) > 1 and m not in {"have", "i", "since", "suffering", "experiencing", "symptoms", "symptom"}]

        # If none matched, attempt word-level mapping
        if not mapped:
            words = re.findall(r"[a-z0-9]+", q.lower())
            cand_words = [normalize_symptom_token(w) for w in words if len(w) > 2]
            mapped = self._map_to_known_symptoms(cand_words)

        # Count number of *distinct* user-provided symptoms (after normalization)
        user_sym_count = len(mapped)

        # Score diseases
        scored = self._score_diseases(mapped)

        # Looser substring match if scored empty
        if not scored and self.symptom_vocab:
            substr_matches = []
            q_low = q.lower()
            for s in self.symptom_vocab:
                if s and s in q_low:
                    substr_matches.append(s)
            if substr_matches:
                scored = self._score_diseases(substr_matches)

        if not scored:
            clarification = "I couldn't confidently match those symptoms to any condition in my knowledge base. Please list symptoms separated by commas and mention how long you've had them."
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "reply": clarification,
                "description": "",
                "precaution": [],
                "emergency": False,
            }

        # Adaptive prediction:
        # - >=3 user symptoms -> pick only the single disease with max matched symptoms (top scorer)
        # - 1-2 symptoms -> return up to 3 diseases ranked by score
        if user_sym_count >= 3:
            top_k = [scored[0]]
        else:
            top_k = scored[:MAX_SUGGESTIONS]

        # Build structured reply per requested format
        disease_names = [t[0] for t in top_k]
        # Possible Condition(s): list just names (comma separated)
        possible_line = ", ".join(disease_names)

        # Explanations: short description (first sentence) for each disease
        explanation_blocks = []
        best_desc = ""
        for d, score, matched_items in top_k:
            # Prefer exact key lookup; case-insensitive fallback
            desc = self.descriptions.get(d) or self.descriptions.get(d.lower()) or ""
            if desc:
                short_desc = _first_sentence(desc)
            else:
                # fallback: short heuristic using matched symptoms
                if matched_items:
                    short_desc = f"Matched symptoms: {', '.join(matched_items)}."
                else:
                    short_desc = "Matches symptom pattern in dataset."
            explanation_blocks.append((d, short_desc))
            if not best_desc:
                best_desc = short_desc

        # Precautions: per-disease, sanitized, up to 3 bullets each
        precaution_blocks = []
        for d, _, _ in top_k:
            raw_prec = self.precautions.get(d) or self.precautions.get(d.lower()) or []
            sanitized = _sanitize_precautions(raw_prec)
            # ensure unique preserve order
            seen = set()
            filtered = []
            for p in sanitized:
                if p not in seen:
                    seen.add(p)
                    filtered.append(p)
            precaution_blocks.append((d, filtered[:3]))

        # Compose final reply string with spacing and bullets as required
        lines: List[str] = []
        lines.append("👩‍⚕️ Healthcare Assistant:")
        lines.append(f"Possible Condition(s): {possible_line}")
        lines.append("")  # blank line

        lines.append("Explanation:")
        lines.append("")  # blank line
        for d, desc in explanation_blocks:
            lines.append(f"{d}: {desc}")
            lines.append("")  # blank line between disease explanations

        lines.append("Precaution / Advice:")
        lines.append("")  # blank line

        for d, precs in precaution_blocks:
            lines.append(f"{d}:")
            for b in _pretty_bullets(precs, max_items=3):
                lines.append(b)
            lines.append("")  # blank line between diseases

        reply_text = "\n".join(lines).strip()

        # Determine top confidence
        top_score = float(top_k[0][1]) if top_k else 0.0
        intent_field = top_k[0][0] if (user_sym_count >= 3 or top_score >= 0.45) else "multiple"

        return {
            "intent": intent_field,
            "confidence": float(top_score),
            "reply": reply_text,
            "description": best_desc or "",
            "precaution": {d: p for d, p in precaution_blocks},
            "emergency": False,
        }

    # convenience properties
    @property
    def descriptions_map(self) -> Dict[str, str]:
        return self.descriptions

    @property
    def precautions_map(self) -> Dict[str, List[str]]:
        return self.precautions

    @property
    def symptom_vocabulary(self) -> List[str]:
        return self.symptom_vocab


