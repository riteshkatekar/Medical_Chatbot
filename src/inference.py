

####################################################################

# src/inference.py
"""
Inference engine for the medical chatbot.

Key behavior:
 - By default returns a single best prediction (top-1) and a concise friendly reply.
 - Uses symptom->disease maps from system_metadata.pkl when available, otherwise
   attempts to build them from dataset CSV(s).
 - Optional Keras classifier fallback (if TensorFlow is installed and artifacts exist).
 - Emergency detection using utils.is_emergency_text when available.
 - Sanitizes precaution text to avoid medication-prescribing language.
"""

from typing import Dict, Any, List, Optional, Tuple
import os
import re
import joblib
import difflib
import warnings
from collections import defaultdict, Counter

try:
    import numpy as np
except Exception:
    np = None  # numpy required for classifier; handle gracefully

# Try to import Keras/TensorFlow (optional)
try:
    import tensorflow as tf  # type: ignore
    from tensorflow import keras  # type: ignore
    _KERAS_AVAILABLE = True
except Exception:
    _KERAS_AVAILABLE = False

# Attempt to reuse helpers from your project
try:
    from preprocessing import normalize_symptom_token, split_symptoms_from_message, read_csv_fuzzy  # type: ignore
except Exception:
    # minimal fallbacks
    def normalize_symptom_token(tok: str) -> str:
        tok = str(tok or "").strip()
        if tok == "" or tok.lower() in ("nan", "na"):
            return ""
        tok = tok.replace("_", " ").replace("-", " ")
        tok = re.sub(r"[_\-\s]+", " ", tok)
        return tok.lower().strip()

    def split_symptoms_from_message(message: str) -> List[str]:
        if not isinstance(message, str) or not message.strip():
            return []
        tokens = re.split(r"[,\;/\|]+", message)
        tokens = [normalize_symptom_token(t) for t in tokens]
        return [t for t in tokens if t]

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
    from utils import is_emergency_text  # type: ignore
except Exception:
    def is_emergency_text(text: str) -> bool:
        if not isinstance(text, str):
            return False
        t = text.lower()
        for kw in ("chest pain", "not breathing", "can't breathe", "cannot breathe", "severe bleeding", "loss of consciousness", "unconscious", "suicidal"):
            if kw in t:
                return True
        return False

# ---------- constants ----------
FUZZY_SYM_CUTOFF = 0.75
MAX_PRECAUTIONS_TO_SHOW = 4
MEDICATION_KEYWORDS = {
    "antibiotic", "antibiotics", "antifungal", "antivirals", "aspirin", "ibuprofen", "paracetamol",
    "acetaminophen", "tablet", "capsule", "mg", "ml", "injection", "dose", "take", "apply", "ointment",
    "medication", "medications", "pills", "antacid", "antihistamine", "steroid"
}
_TOKEN_SPLIT_RE = re.compile(r"[,\;/\|\.\?\n]")

# ---------- small helpers ----------
def _norm_tokens(text: str) -> List[str]:
    if not text:
        return []
    return re.findall(r"[a-z0-9]+", text.lower())

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
    out = []
    for it in items:
        if not it:
            continue
        low = it.lower()
        med_like = False
        for kw in MEDICATION_KEYWORDS:
            if re.search(r"\b" + re.escape(kw) + r"\b", low):
                med_like = True
                break
        if med_like:
            continue
        s = it.strip()
        if len(s) < 3:
            continue
        out.append(s)
    if not out:
        return [
            "Rest and monitor symptoms.",
            "Stay hydrated and avoid strenuous activity.",
            "Seek medical advice if symptoms worsen or persist."
        ]
    # dedupe preserve order
    seen = set()
    res = []
    for r in out:
        if r not in seen:
            seen.add(r)
            res.append(r)
    return res

def _pretty_bullets(items: List[str], max_items: int = MAX_PRECAUTIONS_TO_SHOW) -> List[str]:
    out = []
    for i, it in enumerate(items):
        if i >= max_items:
            break
        out.append(f"• {it}")
    return out

def _compose_single_reply(disease: str, description: str, precautions: List[str]) -> str:
    parts = []
    parts.append("👩‍⚕️ Healthcare Assistant:")
    parts.append(f"Possible Condition: {disease}")
    parts.append("")  # blank line
    if description:
        parts.append("Explanation:")
        parts.append(f"{disease}: {_first_sentence(description)}")
        parts.append("")
    if precautions:
        parts.append("Precaution / Advice:")
        for b in _pretty_bullets(precautions, max_items=MAX_PRECAUTIONS_TO_SHOW):
            parts.append(b)
    return "\n".join(parts).strip()

# ---------- InferenceEngine ----------
class InferenceEngine:
    def __init__(self, models_dir: str = "models", threshold: float = 0.5):
        self.models_dir = models_dir or "."
        self.threshold = float(threshold)
        self.system_metadata: Dict[str, Any] = {}
        self.symptom_map: Dict[str, Dict[str, int]] = {}  # symptom -> {disease: count}
        self.disease_symptoms: Dict[str, Counter] = {}
        self.disease_total_symptoms: Dict[str, int] = {}
        self.descriptions: Dict[str, str] = {}
        self.precautions: Dict[str, List[str]] = {}
        self.symptom_vocab: List[str] = []
        # classifier artifacts (optional)
        self._keras_model = None
        self._tfidf = None
        self._label_encoder = None

        self._load_resources()

    def _load_resources(self):
        # 1) load system_metadata.pkl if present
        meta_path = os.path.join(self.models_dir, "system_metadata.pkl")
        if os.path.exists(meta_path):
            try:
                meta = joblib.load(meta_path)
                if isinstance(meta, dict):
                    self.system_metadata = meta
                    self.descriptions = meta.get("descriptions", {}) or {}
                    self.precautions = meta.get("precautions", {}) or {}
                    smap = meta.get("symptom_map", {}) or {}
                    if smap:
                        self.symptom_map = smap
                        ds = defaultdict(Counter)
                        for s, intents in smap.items():
                            for d, cnt in intents.items():
                                ds[d][s] += int(cnt)
                        self.disease_symptoms = dict(ds)
                        for d, cnt in self.disease_symptoms.items():
                            self.disease_total_symptoms[d] = sum(cnt.values()) or len(cnt) or 1
                        self.symptom_vocab = sorted({s for s in self.symptom_map.keys() if s})
            except Exception:
                # best-effort; continue
                self.system_metadata = {}

        # 2) attempt to load tfidf/label_encoder/keras model
        tfidf_path = os.path.join(self.models_dir, "tfidf.pkl")
        le_path = os.path.join(self.models_dir, "label_encoder.pkl")
        keras_candidates = [
            os.path.join(self.models_dir, "keras_model.keras"),
            os.path.join(self.models_dir, "keras_model.h5"),
            os.path.join(self.models_dir, "keras_model.hdf5"),
        ]
        keras_model_path = next((p for p in keras_candidates if os.path.exists(p)), None)

        if os.path.exists(tfidf_path):
            try:
                self._tfidf = joblib.load(tfidf_path)
            except Exception:
                self._tfidf = None

        if os.path.exists(le_path):
            try:
                self._label_encoder = joblib.load(le_path)
            except Exception:
                self._label_encoder = None

        if _KERAS_AVAILABLE and keras_model_path:
            try:
                self._keras_model = keras.models.load_model(keras_model_path)
            except Exception:
                self._keras_model = None

        # 3) fallback: if symptom_map empty, try to build from dataset.csv in repo root or data/ or models_dir
        if not self.symptom_map:
            candidate_paths = [
                os.path.join("data", "dataset.csv"),
                os.path.join(self.models_dir, "dataset.csv"),
                os.path.join(".", "dataset.csv"),
            ]
            dataset_path = next((p for p in candidate_paths if os.path.exists(p)), None)
            if dataset_path:
                try:
                    df = read_csv_fuzzy(dataset_path)
                    # detect columns similar to your preprocessing.detect_columns
                    cols_lower = [c.lower().strip() for c in df.columns]
                    intent_col = None
                    message_col = None
                    for c, orig in zip(cols_lower, df.columns):
                        if c in {"intent", "condition", "label", "disease", "diagnosis", "disease_name"} and intent_col is None:
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

        # 4) try to load descriptions/precautions CSVs if still empty
        if not self.descriptions:
            for candidate in (os.path.join(self.models_dir, "symptom_description.csv"), os.path.join("data", "symptom_description.csv"), os.path.join(".", "symptom_description.csv")):
                if os.path.exists(candidate):
                    try:
                        df_desc = read_csv_fuzzy(candidate)
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
                        break
                    except Exception:
                        continue

        if not self.precautions:
            for candidate in (os.path.join(self.models_dir, "symptom_precaution.csv"), os.path.join("data", "symptom_precaution.csv"), os.path.join(".", "symptom_precaution.csv")):
                if os.path.exists(candidate):
                    try:
                        df_prec = read_csv_fuzzy(candidate)
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
                        break
                    except Exception:
                        continue

    # ---------- classifier helper ----------
    def classify_text(self, text: str) -> Optional[Dict[str, Any]]:
        if not _KERAS_AVAILABLE or self._keras_model is None or self._tfidf is None or self._label_encoder is None:
            return None
        try:
            X = self._tfidf.transform([text])
            X_arr = X.toarray()
            probs = self._keras_model.predict(X_arr, verbose=0)[0]
            probs = np.asarray(probs, dtype=float)
            top_idx = int(probs.argmax())
            top_conf = float(probs[top_idx])
            try:
                intent_label = str(self._label_encoder.inverse_transform([top_idx])[0])
            except Exception:
                classes = getattr(self._label_encoder, "classes_", None)
                if classes is not None and top_idx < len(classes):
                    intent_label = str(classes[top_idx])
                else:
                    intent_label = "unknown"
            prob_dict = {}
            classes = getattr(self._label_encoder, "classes_", None)
            if classes is not None and len(classes) == len(probs):
                for c_idx, c in enumerate(classes):
                    prob_dict[str(c)] = float(probs[c_idx])
            return {"intent": intent_label, "confidence": top_conf, "probs": prob_dict}
        except Exception:
            return None

    # ---------- symptom mapping ----------
    def _extract_candidate_tokens(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
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
                        gram = " ".join(words[i: i + n])
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
        # dedupe
        seen = set()
        out = []
        for m in matched:
            if m not in seen:
                seen.add(m)
                out.append(m)
        return out

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

    # ---------- public infer (top-1 by default) ----------
    def infer(self, query: str, top_k: int = 1) -> Dict[str, Any]:
        q = (query or "").strip()
        if not q:
            return {
                "query": q,
                "intent": "unknown",
                "confidence": 0.0,
                "reply": "I didn't get any symptoms. Could you state them (e.g., 'fever, cough')?",
                "description": "",
                "precaution": [],
                "emergency": False,
            }

        # emergency detection
        try:
            if is_emergency_text(q):
                return {
                    "query": q,
                    "intent": "emergency",
                    "confidence": 1.0,
                    "reply": "⚠️ EMERGENCY: This sounds like a medical emergency. Call local emergency services immediately.",
                    "description": "",
                    "precaution": [],
                    "emergency": True,
                }
        except Exception:
            pass

        # 1) symptom extraction and mapping
        candidates = self._extract_candidate_tokens(q)
        mapped = self._map_to_known_symptoms(candidates)
        mapped = [m for m in mapped if len(m) > 1 and m not in {"have", "i", "since", "suffering", "experiencing", "symptoms", "symptom"}]

        if not mapped:
            words = re.findall(r"[a-z0-9]+", q.lower())
            cand_words = [normalize_symptom_token(w) for w in words if len(w) > 2]
            mapped = self._map_to_known_symptoms(cand_words)

        user_sym_count = len(mapped)

        # 2) symptom-based scoring
        scored = self._score_diseases(mapped)

        # 3) looser substring fallback
        if not scored and self.symptom_vocab:
            substr_matches = []
            q_low = q.lower()
            for s in self.symptom_vocab:
                if s and s in q_low:
                    substr_matches.append(s)
            if substr_matches:
                scored = self._score_diseases(substr_matches)

        # 4) classifier fallback (optional)
        classifier_result = self.classify_text(q) if _KERAS_AVAILABLE else None

        # If no scored and classifier missing, ask clarification
        if not scored and not classifier_result:
            return {
                "query": q,
                "intent": "unknown",
                "confidence": 0.0,
                "reply": "I couldn't confidently match those symptoms to any known condition. Please list symptoms separated by commas and mention duration.",
                "description": "",
                "precaution": [],
                "emergency": False,
            }

        # If classifier strongly confident, prefer it
        if classifier_result:
            clf_label = classifier_result.get("intent")
            clf_conf = float(classifier_result.get("confidence", 0.0))
            if clf_conf >= max(0.55, (scored[0][1] if scored else 0.0) + 0.15):
                desc = self.descriptions.get(clf_label, "") or self.descriptions.get(clf_label.lower(), "") or ""
                raw_prec = self.precautions.get(clf_label, []) or self.precautions.get(clf_label.lower(), []) or []
                sanitized = _sanitize_precautions(raw_prec)
                reply_text = _compose_single_reply(clf_label, desc, sanitized)
                return {
                    "query": q,
                    "intent": clf_label,
                    "confidence": clf_conf,
                    "reply": reply_text,
                    "description": _first_sentence(desc) or "",
                    "precaution": sanitized,
                    "emergency": False,
                }

        # Otherwise use symptom-based top result(s)
        if scored:
            top = scored[0]
            disease_name = top[0]
            top_score = float(top[1])
            desc = self.descriptions.get(disease_name, "") or self.descriptions.get(disease_name.lower(), "") or ""
            raw_prec = self.precautions.get(disease_name, []) or self.precautions.get(disease_name.lower(), []) or []
            sanitized = _sanitize_precautions(raw_prec)
            reply_text = _compose_single_reply(disease_name, desc or f"Matched symptoms: {', '.join(top[2])}.", sanitized)
            return {
                "query": q,
                "intent": disease_name,
                "confidence": top_score,
                "reply": reply_text,
                "description": _first_sentence(desc) or (f"Matched symptoms: {', '.join(top[2])}."),
                "precaution": sanitized,
                "emergency": False,
            }

        # If we reach here, fall back to classifier primary label
        if classifier_result:
            cl = classifier_result.get("intent", "unknown")
            conf = float(classifier_result.get("confidence", 0.0))
            desc = self.descriptions.get(cl, "") or ""
            raw_prec = self.precautions.get(cl, []) or []
            sanitized = _sanitize_precautions(raw_prec)
            reply_text = _compose_single_reply(cl, desc, sanitized)
            return {
                "query": q,
                "intent": cl,
                "confidence": conf,
                "reply": reply_text,
                "description": _first_sentence(desc) or "",
                "precaution": sanitized,
                "emergency": False,
            }

        # last resort
        return {
            "query": q,
            "intent": "unknown",
            "confidence": 0.0,
            "reply": "Sorry — I couldn't match that. Could you provide a short list of symptoms separated by commas?",
            "description": "",
            "precaution": [],
            "emergency": False,
        }

    # convenience accessors
    @property
    def descriptions_map(self) -> Dict[str, str]:
        return self.descriptions

    @property
    def precautions_map(self) -> Dict[str, List[str]]:
        return self.precautions

    @property
    def symptom_vocabulary(self) -> List[str]:
        return self.symptom_vocab
