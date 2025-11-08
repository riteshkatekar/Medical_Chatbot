<<<<<<< HEAD
# """
# src/inference.py 

# Goals:
# - Case-insensitive & normalized direct disease-name lookup (so "Drug Reaction" / "drug reaction" / "Drug reaction" work).
# - Robust symptom extraction using exact, token, and fuzzy matching against symptom_map keys.
# - Conservative symptom-vote override when classifier is unconfident or clearly contradicted.
# - Keep emergency handling and safe clarifying prompts.
# """

# from typing import Dict, Any, List, Optional
# import os
# import re
# import difflib
# import numpy as np
# import joblib
# from utils import is_emergency_text, normalize_query_text

# # Tunable thresholds
# CONFIDENCE_THRESHOLD = 0.5
# SYMPTOM_OVERRIDE_CONFIDENCE = 0.6   # classifier confidence below this may be overridden by symptom votes
# SYMPTOM_NORMALIZED_SCORE_OVERRIDE = 0.25  # proportion of votes required to override
# SYMPTOM_RAW_VOTE_MIN = 1  # minimum raw votes for best intent to be considered

# DISEASE_FUZZY_CUTOFF = 0.6
# SYMPTOM_FUZZY_CUTOFF = 0.7

# _non_alnum_re = re.compile(r'[^a-z0-9]')

# def _normalize_key(s: str) -> str:
#     """Normalize a string to a compact lowercase alphanumeric-only key."""
#     if not s:
#         return ""
#     s = s.lower().strip()
#     s = s.replace("_", " ")
#     # collapse whitespace
#     s = re.sub(r"\s+", " ", s)
#     # remove non-alphanumeric
#     s = _non_alnum_re.sub("", s)
#     return s

# def _fuzzy_lookup_normalized(topic: str, normalized_map: Dict[str, str], cutoff: float = DISEASE_FUZZY_CUTOFF) -> Optional[str]:
#     """
#     topic: raw text (will be normalized)
#     normalized_map: dict(normalized_key -> original_key)
#     returns original_key or None
#     """
#     if not topic or not normalized_map:
#         return None
#     nk = _normalize_key(topic)
#     if nk in normalized_map:
#         return normalized_map[nk]
#     # use difflib on normalized keys
#     keys = list(normalized_map.keys())
#     matches = difflib.get_close_matches(nk, keys, n=1, cutoff=cutoff)
#     if matches:
#         return normalized_map[matches[0]]
#     return None

# class InferenceEngine:
#     def __init__(self, models_dir: str, threshold: float = CONFIDENCE_THRESHOLD):
#         models_dir = os.path.abspath(models_dir)
#         self.tfidf = joblib.load(os.path.join(models_dir, "tfidf.pkl"))
#         self.clf = joblib.load(os.path.join(models_dir, "sklearn_baseline.pkl"))
#         self.le = joblib.load(os.path.join(models_dir, "label_encoder.pkl"))
#         meta = joblib.load(os.path.join(models_dir, "system_metadata.pkl"))
#         self.descriptions: Dict[str, str] = meta.get("descriptions", {}) if meta else {}
#         self.precautions: Dict[str, List[str]] = meta.get("precautions", {}) if meta else {}
#         self.symptom_map: Dict[str, Dict[str, int]] = meta.get("symptom_map", {}) if meta else {}
#         self.threshold = threshold

#         # Precompute normalized lookup maps for diseases and symptoms
#         self._disease_norm_map: Dict[str, str] = {}
#         for k in (self.descriptions.keys() if self.descriptions else []):
#             nk = _normalize_key(k)
#             if nk:
#                 self._disease_norm_map[nk] = k

#         # symptom normalized map (normalized_symptom -> original_symptom_key)
#         self._symptom_norm_map: Dict[str, str] = {}
#         for s in (self.symptom_map.keys() if self.symptom_map else []):
#             ns = _normalize_key(s)
#             if ns:
#                 self._symptom_norm_map[ns] = s

#         # flatten list of disease keys for fuzzy match fallback
#         self._disease_keys = list(self.descriptions.keys()) if self.descriptions else []

#     # ---------------- Symptom extraction & voting ----------------
#     def _extract_symptom_candidates(self, query: str) -> List[str]:
#         """
#         Return a list of symptom keys (as present in symptom_map) that match the query.
#         Strategy:
#           1) Exact multi-word substring match for symptom keys with spaces.
#           2) Word-boundary exact match for single-token symptoms.
#           3) Fuzzy match tokens against normalized symptom keys.
#         """
#         q = query.lower()
#         found = set()
#         # exact substring for multi-word symptom keys
#         for s in self.symptom_map.keys():
#             if not s:
#                 continue
#             s_norm = s.lower().strip()
#             if " " in s_norm:
#                 if s_norm in q:
#                     found.add(s)
#         # word-level exact match
#         words = re.findall(r"[a-z0-9]+", q)
#         wordset = set(words)
#         for s in self.symptom_map.keys():
#             s_norm = s.lower().strip()
#             if " " in s_norm:
#                 continue
#             if s_norm in wordset:
#                 found.add(s)
#         # fuzzy token matching: try each query token against normalized symptom keys
#         if words:
#             norm_keys = list(self._symptom_norm_map.keys())
#             for w in words:
#                 matches = difflib.get_close_matches(w, norm_keys, n=1, cutoff=SYMPTOM_FUZZY_CUTOFF)
#                 if matches:
#                     orig = self._symptom_norm_map[matches[0]]
#                     found.add(orig)
#         return list(found)

#     def _symptom_vote_intents(self, symptoms: List[str]) -> Dict[str, int]:
#         votes: Dict[str, int] = {}
#         for s in symptoms:
#             mapping = self.symptom_map.get(s, {})
#             for intent, cnt in mapping.items():
#                 votes[intent] = votes.get(intent, 0) + cnt
#         return votes

#     def _best_symptom_intent(self, symptoms: List[str]) -> Optional[Dict[str, Any]]:
#         if not symptoms:
#             return None
#         votes = self._symptom_vote_intents(symptoms)
#         if not votes:
#             return None
#         total = sum(votes.values())
#         best_intent, best_count = max(votes.items(), key=lambda x: x[1])
#         normalized = best_count / total if total > 0 else 0.0
#         return {"intent": best_intent, "score": normalized, "raw_votes": votes, "best_count": best_count}

#     # ---------------- Disease lookup helpers ----------------
#     def _direct_disease_lookup(self, query: str) -> Optional[str]:
#         """
#         Try to interpret the entire query as a disease name (case-insensitive, normalized, fuzzy).
#         Returns canonical disease key (as in descriptions) or None.
#         """
#         q = query.strip()
#         if not q:
#             return None
#         # Strip common leading question words like "what is", "tell me about", etc.
#         q_clean = re.sub(r'^(what is|what\'s|whats|tell me about|tell me|who is|who\'s)\s+', '', q, flags=re.I).strip()
#         # normalize and lookup
#         cand = _fuzzy_lookup_normalized(q_clean, self._disease_norm_map, cutoff=DISEASE_FUZZY_CUTOFF)
#         return cand

#     # ---------------- Core inference ----------------
#     def infer(self, query: str) -> Dict[str, Any]:
#         q_raw = query or ""
#         q = normalize_query_text(q_raw)
#         if not q:
#             return {
#                 "query": query,
#                 "intent": "unknown",
#                 "confidence": 0.0,
#                 "reply": "Sorry — I didn't catch that. Could you rephrase?",
#                 "description": "",
#                 "precaution": "",
#                 "emergency": False
#             }

#         # Emergency override
#         if is_emergency_text(q):
#             return {
#                 "query": query,
#                 "intent": "emergency",
#                 "confidence": 1.0,
#                 "reply": "This sounds like an emergency. Call your local emergency services or go to the nearest emergency department immediately. Not a medical professional — for emergencies, call local emergency services.",
#                 "description": "",
#                 "precaution": "",
#                 "emergency": True
#             }

#         # 1) Direct disease-name detection (stateless): if query is or strongly matches a disease name, return metadata
#         direct = self._direct_disease_lookup(q)
#         if direct:
#             desc = self.descriptions.get(direct, "")
#             prec = self.precautions.get(direct, [])
#             reply = f"{direct}: {desc} Not a medical professional — for emergencies, call local emergency services."
#             return {
#                 "query": query,
#                 "intent": direct,
#                 "confidence": 0.95,
#                 "reply": reply,
#                 "description": desc,
#                 "precaution": "; ".join(prec) if prec else "",
#                 "emergency": False
#             }

#         # 2) Classifier baseline
#         X = self.tfidf.transform([q])
#         probs = self.clf.predict_proba(X)[0]
#         idx = int(np.argmax(probs))
#         confidence = float(probs[idx])
#         predicted_label = str(self.le.inverse_transform([idx])[0])

#         # 3) Symptom evidence from query
#         symptom_keys = self._extract_symptom_candidates(q)
#         symptom_best = self._best_symptom_intent(symptom_keys) if symptom_keys else None

#         # 4) Symptom-based override (more permissive, but conservative)
#         if symptom_best:
#             sym_intent = symptom_best["intent"]
#             sym_score = float(symptom_best["score"])
#             best_count = int(symptom_best.get("best_count", 0))
#             # Allow override if classifier is weak or symptom evidence strong:
#             if (confidence < SYMPTOM_OVERRIDE_CONFIDENCE) and ((sym_score >= SYMPTOM_NORMALIZED_SCORE_OVERRIDE) or (best_count >= SYMPTOM_RAW_VOTE_MIN)):
#                 description = self.descriptions.get(sym_intent, "")
#                 precaution_list = self.precautions.get(sym_intent, [])
#                 reply = f"I think this matches {sym_intent}. {description} For this, consider: " + (", ".join(precaution_list[:3]) if precaution_list else "follow routine care")
#                 return {
#                     "query": query,
#                     "intent": sym_intent,
#                     "confidence": max(confidence, sym_score),
#                     "reply": reply + ". Not a medical professional — for emergencies, call local emergency services.",
#                     "description": description,
#                     "precaution": "; ".join(precaution_list) if precaution_list else "",
#                     "emergency": False
#                 }

#         # 5) If classifier low-confidence, ask clarifying question (stateless)
#         description = self._lookup_description_for_label(predicted_label)
#         precaution_list = self._lookup_precautions_for_label(predicted_label)

#         if confidence < self.threshold:
#             clar = "Can you tell me how long you've had these symptoms and whether you have a fever or other major changes?"
#             return {
#                 "query": query,
#                 "intent": predicted_label,
#                 "confidence": confidence,
#                 "reply": clar + " Not a medical professional — for emergencies, call local emergency services.",
#                 "description": description,
#                 "precaution": "; ".join(precaution_list) if precaution_list else "",
#                 "emergency": False
#             }

#         # 6) High-confidence classifier output
#         reply = f"{description} Based on what you said, consider: " + (", ".join(precaution_list[:3]) if precaution_list else "follow routine care")
#         return {
#             "query": query,
#             "intent": predicted_label,
#             "confidence": confidence,
#             "reply": reply + ". Not a medical professional — for emergencies, call local emergency services.",
#             "description": description,
#             "precaution": "; ".join(precaution_list) if precaution_list else "",
#             "emergency": False
#         }

#     # ---------------- small helpers ----------------
#     def _lookup_description_for_label(self, label: str) -> str:
#         # label may have different casing/spacing than keys in descriptions -> use normalized lookup
#         if not label:
#             return ""
#         # direct
#         if label in self.descriptions:
#             return self.descriptions[label]
#         # normalized fuzzy
#         cand = _fuzzy_lookup_normalized(label, self._disease_norm_map, cutoff=DISEASE_FUZZY_CUTOFF)
#         if cand:
#             return self.descriptions.get(cand, "")
#         return ""

#     def _lookup_precautions_for_label(self, label: str) -> List[str]:
#         if not label:
#             return []
#         if label in self.precautions:
#             return self.precautions[label]
#         cand = _fuzzy_lookup_normalized(label, self._disease_norm_map, cutoff=DISEASE_FUZZY_CUTOFF)
#         if cand:
#             return self.precautions.get(cand, [])
#         return []




# """
# src/inference.py
=======
"""
src/inference.py
>>>>>>> 61ea1752c12febf1ea3c97dca0ca50d61d299833

# Updated InferenceEngine implementing refined response logic:
#  - Adaptive prediction: 3+ symptoms -> single disease, 1-2 symptoms -> up to 3 diseases
#  - Clean structured reply format (disease-wise explanations and precautions)
#  - No echoing of user input
#  - Medication filtering for safety
# """

# import os
# import re
# import joblib
# import difflib
# from typing import Dict, List, Tuple, Optional, Any
# from collections import defaultdict, Counter

# # Try to reuse helpers from preprocessing/utils if available
# try:
#     from preprocessing import normalize_symptom_token, split_symptoms_from_message, read_csv_fuzzy
# except Exception:
#     def normalize_symptom_token(tok: str) -> str:
#         tok = str(tok).strip()
#         if tok == "" or tok.lower() in ("nan", "na"):
#             return ""
#         tok = tok.replace("_", " ").replace("-", " ")
#         tok = re.sub(r"[_\-\s]+", " ", tok)
#         tok = tok.lower().strip()
#         return tok

#     def split_symptoms_from_message(message: str) -> List[str]:
#         if not isinstance(message, str) or message.strip() == "":
#             return []
#         tokens = re.split(r"[,\;/\|]+", message)
#         tokens = [normalize_symptom_token(t) for t in tokens]
#         tokens = [t for t in tokens if t]
#         return tokens

#     def read_csv_fuzzy(path: str):
#         import pandas as pd
#         for sep in [",", "\t", ";"]:
#             try:
#                 df = pd.read_csv(path, sep=sep, header=0, dtype=str, encoding="utf-8").fillna("")
#                 return df
#             except Exception:
#                 continue
#         raise FileNotFoundError(path)

# try:
#     from utils import is_emergency_text
# except Exception:
#     def is_emergency_text(text: str) -> bool:
#         t = str(text).lower()
#         em = ["chest pain", "not breathing", "can't breathe", "cannot breathe", "severe bleeding", "loss of consciousness", "unconscious"]
#         return any(kw in t for kw in em)

# # ---------- constants ----------
# DEFAULT_TOP_K = 3
# FUZZY_SYM_CUTOFF = 0.75  # fuzzy match threshold for symptom token variants
# MAX_SUGGESTIONS = 3
# _TOKEN_SPLIT_RE = re.compile(r"[,\;/\|\.\?\n]")

# # medication-like keywords to filter from precautions (safety)
# _MEDICATION_KEYWORDS = {
#     "antibiotic", "antibiotics", "antifungal", "antivirals", "aspirin", "ibuprofen", "paracetamol",
#     "acetaminophen", "tablet", "capsule", "mg", "ml", "injection", "dose", "take", "apply", "ointment",
#     "ointment", "cream", "spray", "prescription", "medication", "medications", "pills"
# }

# # ---------- helpers ----------
# def _norm_tokens(text: str) -> List[str]:
#     if not text:
#         return []
#     toks = re.findall(r"[a-z0-9]+", text.lower())
#     return toks

# def _first_sentence(text: str) -> str:
#     if not text:
#         return ""
#     parts = re.split(r"[\.!?]\s*", text.strip())
#     if parts:
#         s = parts[0].strip()
#         if s and not s.endswith("."):
#             s = s + "."
#         return s
#     return text

# def _sanitize_precautions(items: List[str]) -> List[str]:
#     """
#     Remove medication-specific items or instructions. Keep generic, safety-first advice.
#     If all items are medication-like, return a safe fallback list.
#     """
#     out = []
#     for it in items:
#         if not it:
#             continue
#         low = it.lower()
#         # if any medication word appears as whole word or substring, treat as medication-like
#         med_like = False
#         for kw in _MEDICATION_KEYWORDS:
#             if re.search(r"\b" + re.escape(kw) + r"\b", low):
#                 med_like = True
#                 break
#         if med_like:
#             continue
#         # Avoid short non-informative items
#         if len(it.strip()) < 3:
#             continue
#         out.append(it.strip())
#     # If none remain, provide safe generic precautions
#     if not out:
#         return [
#             "Rest and monitor symptoms",
#             "Stay hydrated",
#             "Seek medical advice from a clinician if symptoms persist or worsen"
#         ]
#     return out

# def _pretty_bullets(items: List[str], max_items: int = 6) -> List[str]:
#     out = []
#     for i, it in enumerate(items):
#         if i >= max_items:
#             break
#         out.append(f"• {it}")
#     return out

# # ---------- main engine ----------
# class InferenceEngine:
#     def __init__(self, models_dir: str = "models", threshold: float = 0.5):
#         self.models_dir = models_dir or "."
#         self.threshold = float(threshold)
#         self.system_metadata: Dict[str, Any] = {}
#         self.symptom_map: Dict[str, Dict[str, int]] = {}
#         self.disease_symptoms: Dict[str, Counter] = {}
#         self.disease_total_symptoms: Dict[str, int] = {}
#         self.descriptions: Dict[str, str] = {}
#         self.precautions: Dict[str, List[str]] = {}
#         self.symptom_vocab: List[str] = []
#         self._load_resources()

#     # -------- Loading resources (system_metadata.pkl OR CSVs) --------
#     def _load_resources(self):
#         meta_path = os.path.join(self.models_dir, "system_metadata.pkl")
#         if os.path.exists(meta_path):
#             try:
#                 self.system_metadata = joblib.load(meta_path)
#                 self.descriptions = self.system_metadata.get("descriptions", {}) or {}
#                 self.precautions = self.system_metadata.get("precautions", {}) or {}
#                 symptom_map = self.system_metadata.get("symptom_map", {}) or {}
#                 self.symptom_map = symptom_map
#                 ds = defaultdict(Counter)
#                 for sym, intents in symptom_map.items():
#                     for d, cnt in intents.items():
#                         ds[d][sym] += int(cnt)
#                 self.disease_symptoms = dict(ds)
#                 for d, c in self.disease_symptoms.items():
#                     self.disease_total_symptoms[d] = sum(c.values()) or len(c) or 1
#                 self.symptom_vocab = sorted(list({s for s in symptom_map.keys() if s}))
#                 return
#             except Exception:
#                 # fallback to CSV loading
#                 pass

#         # Attempt CSVs
#         dataset_candidates = [
#             os.path.join(self.models_dir, "dataset.csv"),
#             os.path.join(self.models_dir, "data.csv"),
#             os.path.join(".", "dataset.csv"),
#             os.path.join(".", "data.csv"),
#         ]
#         desc_candidates = [
#             os.path.join(self.models_dir, "symptom_description.csv"),
#             os.path.join(".", "symptom_description.csv"),
#         ]
#         prec_candidates = [
#             os.path.join(self.models_dir, "symptom_precaution.csv"),
#             os.path.join(".", "symptom_precaution.csv"),
#         ]

#         dataset_path = next((p for p in dataset_candidates if os.path.exists(p)), None)
#         if dataset_path:
#             try:
#                 df = read_csv_fuzzy(dataset_path)
#                 lower_cols = [c.lower().strip() for c in df.columns]
#                 intent_col = None
#                 message_col = None
#                 for c, orig in zip(lower_cols, df.columns):
#                     if c in {"intent", "condition", "disease", "label", "diagnosis", "disease_name"} and intent_col is None:
#                         intent_col = orig
#                     if c in {"text", "pattern", "message", "symptoms", "symptom_list"} and message_col is None:
#                         message_col = orig
#                 if intent_col is None:
#                     intent_col = df.columns[0]
#                 if message_col is None:
#                     if df.shape[1] > 1:
#                         message_col = "__synth_message"
#                         df[message_col] = df.apply(lambda r: ", ".join([str(x).strip() for x in r[1:].values if str(x).strip() and str(x).strip().lower() not in ("nan", "na")]), axis=1)
#                     else:
#                         message_col = intent_col

#                 ds = defaultdict(Counter)
#                 for _, row in df.iterrows():
#                     disease = str(row[intent_col]).strip()
#                     message = str(row.get(message_col, "")).strip()
#                     toks = []
#                     if any(sep in message for sep in [",", ";", "/", "|"]):
#                         toks = split_symptoms_from_message(message)
#                     else:
#                         toks_raw = re.findall(r"[A-Za-z0-9_\-]+", message)
#                         toks = [normalize_symptom_token(t) for t in toks_raw if len(t) >= 2]
#                     for t in toks:
#                         if not t:
#                             continue
#                         ds[disease][t] += 1
#                 self.disease_symptoms = dict(ds)
#                 for d, c in self.disease_symptoms.items():
#                     self.disease_total_symptoms[d] = sum(c.values()) or len(c) or 1
#                 sym_map = {}
#                 for d, counter in self.disease_symptoms.items():
#                     for s, cnt in counter.items():
#                         sym_map.setdefault(s, {})[d] = sym_map.setdefault(s, {}).get(d, 0) + cnt
#                 self.symptom_map = sym_map
#                 self.symptom_vocab = sorted(list(set(sym_map.keys())))
#             except Exception:
#                 self.symptom_map = {}
#                 self.disease_symptoms = {}
#                 self.symptom_vocab = []
#         else:
#             self.symptom_map = {}
#             self.disease_symptoms = {}
#             self.symptom_vocab = []

#         # descriptions
#         desc_path = next((p for p in desc_candidates if os.path.exists(p)), None)
#         if desc_path:
#             try:
#                 df_desc = read_csv_fuzzy(desc_path)
#                 if df_desc.shape[1] >= 2:
#                     kcol = df_desc.columns[0]
#                     vcol = df_desc.columns[1]
#                     for _, r in df_desc.iterrows():
#                         k = str(r[kcol]).strip()
#                         v = str(r[vcol]).strip()
#                         if k:
#                             self.descriptions[k] = v
#                 else:
#                     for _, r in df_desc.iterrows():
#                         raw = str(r[df_desc.columns[0]])
#                         if "\t" in raw:
#                             k, v = raw.split("\t", 1)
#                             self.descriptions[k.strip()] = v.strip()
#             except Exception:
#                 self.descriptions = {}

#         # precautions
#         prec_path = next((p for p in prec_candidates if os.path.exists(p)), None)
#         if prec_path:
#             try:
#                 df_prec = read_csv_fuzzy(prec_path)
#                 if df_prec.shape[1] >= 2:
#                     for _, r in df_prec.iterrows():
#                         key = str(r[df_prec.columns[0]]).strip()
#                         vals = []
#                         for c in df_prec.columns[1:]:
#                             v = str(r[c]).strip()
#                             if v and v.lower() not in ("nan", "na"):
#                                 vals.append(v)
#                         if key:
#                             self.precautions[key] = vals
#                 else:
#                     for _, r in df_prec.iterrows():
#                         raw = str(r[df_prec.columns[0]])
#                         if "\t" in raw:
#                             k, v = raw.split("\t", 1)
#                             self.precautions[k.strip()] = [v.strip()]
#             except Exception:
#                 self.precautions = {}

#         # ensure totals present
#         for d in self.disease_symptoms:
#             self.disease_total_symptoms.setdefault(d, sum(self.disease_symptoms[d].values()) or len(self.disease_symptoms[d]) or 1)

#     # -------- symptom extraction & mapping --------
#     def _extract_candidate_tokens(self, text: str) -> List[str]:
#         if not text or not text.strip():
#             return []
#         text = text.strip().lower()
#         parts = [p.strip() for p in _TOKEN_SPLIT_RE.split(text) if p.strip()]
#         candidates = []
#         for p in parts:
#             subparts = re.split(r"\band\b|\bwith\b|\bplus\b", p)
#             for sp in subparts:
#                 sp = sp.strip()
#                 if not sp:
#                     continue
#                 words = re.findall(r"[a-z0-9]+", sp)
#                 if not words:
#                     continue
#                 for n in range(1, min(4, len(words) + 1)):
#                     for i in range(len(words) - n + 1):
#                         gram = " ".join(words[i : i + n])
#                         candidates.append(gram)
#         seen = set()
#         out = []
#         for c in candidates:
#             n = normalize_symptom_token(c)
#             if n and n not in seen:
#                 seen.add(n)
#                 out.append(n)
#         return out

#     def _map_to_known_symptoms(self, candidate_tokens: List[str]) -> List[str]:
#         if not candidate_tokens:
#             return []
#         matched = []
#         vocab = self.symptom_vocab or []
#         low_vocab = {v.lower(): v for v in vocab}
#         for tok in candidate_tokens:
#             if not tok:
#                 continue
#             if tok in low_vocab:
#                 matched.append(low_vocab[tok])
#                 continue
#             tok_tokens = set(_norm_tokens(tok))
#             best = None
#             best_overlap = 0
#             for v_lower, v_orig in low_vocab.items():
#                 v_tokens = set(_norm_tokens(v_lower))
#                 overlap = len(tok_tokens & v_tokens)
#                 if overlap > best_overlap:
#                     best_overlap = overlap
#                     best = v_orig
#             if best and best_overlap >= 1:
#                 matched.append(best)
#                 continue
#             if vocab:
#                 matches = difflib.get_close_matches(tok, list(low_vocab.keys()), n=1, cutoff=FUZZY_SYM_CUTOFF)
#                 if matches:
#                     matched.append(low_vocab[matches[0]])
#                     continue
#             matched.append(tok)
#         # dedupe preserve order
#         seen = set()
#         out = []
#         for m in matched:
#             if m not in seen:
#                 seen.add(m)
#                 out.append(m)
#         return out

#     # -------- scoring --------
#     def _score_diseases(self, matched_symptoms: List[str]) -> List[Tuple[str, float, List[str]]]:
#         scores = []
#         if not matched_symptoms:
#             return []
#         mset = set(matched_symptoms)
#         for d, counter in self.disease_symptoms.items():
#             if not counter:
#                 continue
#             total = float(self.disease_total_symptoms.get(d, sum(counter.values()) or len(counter) or 1))
#             matched_items = []
#             ssum = 0.0
#             for s, cnt in counter.items():
#                 if s in mset:
#                     matched_items.append(s)
#                     ssum += min(cnt, 3)
#             raw_frac = (ssum / total) if total > 0 else 0.0
#             uniq_boost = min(len(matched_items) / (len(counter) + 1), 0.5)
#             score = raw_frac * 0.9 + uniq_boost * 0.1
#             score = max(0.0, min(1.0, float(score)))
#             if matched_items:
#                 scores.append((d, score, matched_items))
#         scores.sort(key=lambda x: (x[1], len(x[2])), reverse=True)
#         return scores

#     # -------- public infer --------
#     def infer(self, query: str) -> Dict[str, Any]:
#         q = (query or "").strip()
#         if not q:
#             return {
#                 "intent": "unknown",
#                 "confidence": 0.0,
#                 "reply": "I didn't get any symptoms. Could you state them (e.g., 'fever, cough')?",
#                 "description": "",
#                 "precaution": [],
#                 "emergency": False,
#             }

#         # Emergency detection
#         try:
#             if is_emergency_text(q):
#                 reply = (
#                     "⚠️ EMERGENCY ALERT: This message contains signs of a possible medical emergency "
#                     "such as severe chest pain or breathing difficulty. Please call local emergency services or go to the nearest emergency department immediately."
#                 )
#                 return {
#                     "intent": "emergency",
#                     "confidence": 1.0,
#                     "reply": reply,
#                     "description": "",
#                     "precaution": [],
#                     "emergency": True,
#                 }
#         except Exception:
#             pass

#         # Extract and map symptoms (do not echo input)
#         candidates = self._extract_candidate_tokens(q)
#         mapped = self._map_to_known_symptoms(candidates)
#         # Filter common stop-words left-over
#         mapped = [m for m in mapped if len(m) > 1 and m not in {"have", "i", "since", "suffering", "experiencing", "symptoms", "symptom"}]

#         # If none matched, attempt word-level mapping
#         if not mapped:
#             words = re.findall(r"[a-z0-9]+", q.lower())
#             cand_words = [normalize_symptom_token(w) for w in words if len(w) > 2]
#             mapped = self._map_to_known_symptoms(cand_words)

#         # Count number of *distinct* user-provided symptoms (after normalization)
#         user_sym_count = len(mapped)

#         # Score diseases
#         scored = self._score_diseases(mapped)

#         # Looser substring match if scored empty
#         if not scored and self.symptom_vocab:
#             substr_matches = []
#             q_low = q.lower()
#             for s in self.symptom_vocab:
#                 if s and s in q_low:
#                     substr_matches.append(s)
#             if substr_matches:
#                 scored = self._score_diseases(substr_matches)

#         if not scored:
#             clarification = "I couldn't confidently match those symptoms to any condition in my knowledge base. Please list symptoms separated by commas and mention how long you've had them."
#             return {
#                 "intent": "unknown",
#                 "confidence": 0.0,
#                 "reply": clarification,
#                 "description": "",
#                 "precaution": [],
#                 "emergency": False,
#             }

#         # Adaptive prediction:
#         # - >=3 user symptoms -> pick only the single disease with max matched symptoms (top scorer)
#         # - 1-2 symptoms -> return up to 3 diseases ranked by score
#         if user_sym_count >= 3:
#             top_k = [scored[0]]
#         else:
#             top_k = scored[:MAX_SUGGESTIONS]

#         # Build structured reply per requested format
#         disease_names = [t[0] for t in top_k]
#         # Possible Condition(s): list just names (comma separated)
#         possible_line = ", ".join(disease_names)

#         # Explanations: short description (first sentence) for each disease
#         explanation_blocks = []
#         best_desc = ""
#         for d, score, matched_items in top_k:
#             # Prefer exact key lookup; case-insensitive fallback
#             desc = self.descriptions.get(d) or self.descriptions.get(d.lower()) or ""
#             if desc:
#                 short_desc = _first_sentence(desc)
#             else:
#                 # fallback: short heuristic using matched symptoms
#                 if matched_items:
#                     short_desc = f"Matched symptoms: {', '.join(matched_items)}."
#                 else:
#                     short_desc = "Matches symptom pattern in dataset."
#             explanation_blocks.append((d, short_desc))
#             if not best_desc:
#                 best_desc = short_desc

#         # Precautions: per-disease, sanitized, up to 3 bullets each
#         precaution_blocks = []
#         for d, _, _ in top_k:
#             raw_prec = self.precautions.get(d) or self.precautions.get(d.lower()) or []
#             sanitized = _sanitize_precautions(raw_prec)
#             # ensure unique preserve order
#             seen = set()
#             filtered = []
#             for p in sanitized:
#                 if p not in seen:
#                     seen.add(p)
#                     filtered.append(p)
#             precaution_blocks.append((d, filtered[:3]))

#         # Compose final reply string with spacing and bullets as required
#         lines: List[str] = []
#         lines.append("👩‍⚕️ Healthcare Assistant:")
#         lines.append(f"Possible Condition(s): {possible_line}")
#         lines.append("")  # blank line

#         lines.append("Explanation:")
#         lines.append("")  # blank line
#         for d, desc in explanation_blocks:
#             lines.append(f"{d}: {desc}")
#             lines.append("")  # blank line between disease explanations

#         lines.append("Precaution / Advice:")
#         lines.append("")  # blank line

#         for d, precs in precaution_blocks:
#             lines.append(f"{d}:")
#             for b in _pretty_bullets(precs, max_items=3):
#                 lines.append(b)
#             lines.append("")  # blank line between diseases

#         reply_text = "\n".join(lines).strip()

#         # Determine top confidence
#         top_score = float(top_k[0][1]) if top_k else 0.0
#         intent_field = top_k[0][0] if (user_sym_count >= 3 or top_score >= 0.45) else "multiple"

#         return {
#             "intent": intent_field,
#             "confidence": float(top_score),
#             "reply": reply_text,
#             "description": best_desc or "",
#             "precaution": {d: p for d, p in precaution_blocks},
#             "emergency": False,
#         }

#     # convenience properties
#     @property
#     def descriptions_map(self) -> Dict[str, str]:
#         return self.descriptions

#     @property
#     def precautions_map(self) -> Dict[str, List[str]]:
#         return self.precautions

#     @property
#     def symptom_vocabulary(self) -> List[str]:
#         return self.symptom_vocab














####################################################### above code is the best....Following is based on keras

# """
# src/inference.py

# Updated InferenceEngine implementing refined response logic:
#  - Adaptive prediction: 3+ symptoms -> single disease, 1-2 symptoms -> up to 3 diseases
#  - Clean structured reply format (disease-wise explanations and precautions)
#  - No echoing of user input
#  - Medication filtering for safety
#  - Uses Keras classifier (if available) as a fallback/comparator to symptom-map scoring
# """

# import os
# import re
# import joblib
# import difflib
# from typing import Dict, List, Tuple, Optional, Any
# from collections import defaultdict, Counter

# # Try to reuse helpers from preprocessing/utils if available
# try:
#     from preprocessing import normalize_symptom_token, split_symptoms_from_message, read_csv_fuzzy
# except Exception:
#     def normalize_symptom_token(tok: str) -> str:
#         tok = str(tok).strip()
#         if tok == "" or tok.lower() in ("nan", "na"):
#             return ""
#         tok = tok.replace("_", " ").replace("-", " ")
#         tok = re.sub(r"[_\-\s]+", " ", tok)
#         tok = tok.lower().strip()
#         return tok

#     def split_symptoms_from_message(message: str) -> List[str]:
#         if not isinstance(message, str) or message.strip() == "":
#             return []
#         tokens = re.split(r"[,\;/\|]+", message)
#         tokens = [normalize_symptom_token(t) for t in tokens]
#         tokens = [t for t in tokens if t]
#         return tokens

#     def read_csv_fuzzy(path: str):
#         import pandas as pd
#         for sep in [",", "\t", ";"]:
#             try:
#                 df = pd.read_csv(path, sep=sep, header=0, dtype=str, encoding="utf-8").fillna("")
#                 return df
#             except Exception:
#                 continue
#         raise FileNotFoundError(path)

# try:
#     from utils import is_emergency_text
# except Exception:
#     def is_emergency_text(text: str) -> bool:
#         t = str(text).lower()
#         em = ["chest pain", "not breathing", "can't breathe", "cannot breathe", "severe bleeding", "loss of consciousness", "unconscious"]
#         return any(kw in t for kw in em)

# # Try to import Keras/TensorFlow for classifier usage (optional fallback)
# try:
#     import tensorflow as tf
#     from tensorflow import keras
#     _KERAS_AVAILABLE = True
# except Exception:
#     _KERAS_AVAILABLE = False

# # ---------- constants ----------
# DEFAULT_TOP_K = 3
# FUZZY_SYM_CUTOFF = 0.75  # fuzzy match threshold for symptom token variants
# MAX_SUGGESTIONS = 3
# _TOKEN_SPLIT_RE = re.compile(r"[,\;/\|\.\?\n]")

# # medication-like keywords to filter from precautions (safety)
# _MEDICATION_KEYWORDS = {
#     "antibiotic", "antibiotics", "antifungal", "antivirals", "aspirin", "ibuprofen", "paracetamol",
#     "acetaminophen", "tablet", "capsule", "mg", "ml", "injection", "dose", "take", "apply", "ointment",
#     "ointment", "cream", "spray", "prescription", "medication", "medications", "pills"
# }

# # ---------- helpers ----------
# def _norm_tokens(text: str) -> List[str]:
#     if not text:
#         return []
#     toks = re.findall(r"[a-z0-9]+", text.lower())
#     return toks

# def _first_sentence(text: str) -> str:
#     if not text:
#         return ""
#     parts = re.split(r"[\.!?]\s*", text.strip())
#     if parts:
#         s = parts[0].strip()
#         if s and not s.endswith("."):
#             s = s + "."
#         return s
#     return text

# def _sanitize_precautions(items: List[str]) -> List[str]:
#     """
#     Remove medication-specific items or instructions. Keep generic, safety-first advice.
#     If all items are medication-like, return a safe fallback list.
#     """
#     out = []
#     for it in items:
#         if not it:
#             continue
#         low = it.lower()
#         # if any medication word appears as whole word or substring, treat as medication-like
#         med_like = False
#         for kw in _MEDICATION_KEYWORDS:
#             if re.search(r"\b" + re.escape(kw) + r"\b", low):
#                 med_like = True
#                 break
#         if med_like:
#             continue
#         # Avoid short non-informative items
#         if len(it.strip()) < 3:
#             continue
#         out.append(it.strip())
#     # If none remain, provide safe generic precautions
#     if not out:
#         return [
#             "Rest and monitor symptoms",
#             "Stay hydrated",
#             "Seek medical advice from a clinician if symptoms persist or worsen"
#         ]
#     return out

# def _pretty_bullets(items: List[str], max_items: int = 6) -> List[str]:
#     out = []
#     for i, it in enumerate(items):
#         if i >= max_items:
#             break
#         out.append(f"• {it}")
#     return out

# # ---------- main engine ----------
# class InferenceEngine:
#     def __init__(self, models_dir: str = "models", threshold: float = 0.5):
#         self.models_dir = models_dir or "."
#         self.threshold = float(threshold)
#         self.system_metadata: Dict[str, Any] = {}
#         self.symptom_map: Dict[str, Dict[str, int]] = {}
#         self.disease_symptoms: Dict[str, Counter] = {}
#         self.disease_total_symptoms: Dict[str, int] = {}
#         self.descriptions: Dict[str, str] = {}
#         self.precautions: Dict[str, List[str]] = {}
#         self.symptom_vocab: List[str] = []
#         # classifier artifacts
#         self._keras_model = None
#         self._tfidf = None
#         self._label_encoder = None
#         self._load_resources()

#     # -------- Loading resources (system_metadata.pkl OR CSVs) --------
#     def _load_resources(self):
#         # 1) Try to load metadata first
#         meta_path = os.path.join(self.models_dir, "system_metadata.pkl")
#         if os.path.exists(meta_path):
#             try:
#                 self.system_metadata = joblib.load(meta_path)
#                 self.descriptions = self.system_metadata.get("descriptions", {}) or {}
#                 self.precautions = self.system_metadata.get("precautions", {}) or {}
#                 symptom_map = self.system_metadata.get("symptom_map", {}) or {}
#                 self.symptom_map = symptom_map
#                 ds = defaultdict(Counter)
#                 for sym, intents in symptom_map.items():
#                     for d, cnt in intents.items():
#                         ds[d][sym] += int(cnt)
#                 self.disease_symptoms = dict(ds)
#                 for d, c in self.disease_symptoms.items():
#                     self.disease_total_symptoms[d] = sum(c.values()) or len(c) or 1
#                 self.symptom_vocab = sorted(list({s for s in symptom_map.keys() if s}))
#             except Exception:
#                 # continue to CSV fallback
#                 pass

#         # 2) Try to load tfidf + label encoder + keras model if available
#         tfidf_path = os.path.join(self.models_dir, "tfidf.pkl")
#         le_path = os.path.join(self.models_dir, "label_encoder.pkl")
#         keras_model_path = os.path.join(self.models_dir, "keras_model.h5")

#         if os.path.exists(tfidf_path):
#             try:
#                 self._tfidf = joblib.load(tfidf_path)
#             except Exception:
#                 self._tfidf = None

#         if os.path.exists(le_path):
#             try:
#                 self._label_encoder = joblib.load(le_path)
#             except Exception:
#                 self._label_encoder = None

#         # load keras model if available and tfidf + label encoder present
#         if _KERAS_AVAILABLE and os.path.exists(keras_model_path):
#             try:
#                 self._keras_model = keras.models.load_model(keras_model_path)
#             except Exception:
#                 self._keras_model = None

#         # if symptom_map empty, try to build from CSV dataset candidates (fallback)
#         if not self.symptom_map:
#             dataset_candidates = [
#                 os.path.join(self.models_dir, "dataset.csv"),
#                 os.path.join(self.models_dir, "data.csv"),
#                 os.path.join(".", "dataset.csv"),
#                 os.path.join(".", "data.csv"),
#             ]
#             dataset_path = next((p for p in dataset_candidates if os.path.exists(p)), None)
#             if dataset_path:
#                 try:
#                     df = read_csv_fuzzy(dataset_path)
#                     lower_cols = [c.lower().strip() for c in df.columns]
#                     intent_col = None
#                     message_col = None
#                     for c, orig in zip(lower_cols, df.columns):
#                         if c in {"intent", "condition", "disease", "label", "diagnosis", "disease_name"} and intent_col is None:
#                             intent_col = orig
#                         if c in {"text", "pattern", "message", "symptoms", "symptom_list"} and message_col is None:
#                             message_col = orig
#                     if intent_col is None:
#                         intent_col = df.columns[0]
#                     if message_col is None:
#                         if df.shape[1] > 1:
#                             message_col = "__synth_message"
#                             df[message_col] = df.apply(lambda r: ", ".join([str(x).strip() for x in r[1:].values if str(x).strip() and str(x).strip().lower() not in ("nan", "na")]), axis=1)
#                         else:
#                             message_col = intent_col

#                     ds = defaultdict(Counter)
#                     for _, row in df.iterrows():
#                         disease = str(row[intent_col]).strip()
#                         message = str(row.get(message_col, "")).strip()
#                         toks = []
#                         if any(sep in message for sep in [",", ";", "/", "|"]):
#                             toks = split_symptoms_from_message(message)
#                         else:
#                             toks_raw = re.findall(r"[A-Za-z0-9_\-]+", message)
#                             toks = [normalize_symptom_token(t) for t in toks_raw if len(t) >= 2]
#                         for t in toks:
#                             if not t:
#                                 continue
#                             ds[disease][t] += 1
#                     self.disease_symptoms = dict(ds)
#                     for d, c in self.disease_symptoms.items():
#                         self.disease_total_symptoms[d] = sum(c.values()) or len(c) or 1
#                     sym_map = {}
#                     for d, counter in self.disease_symptoms.items():
#                         for s, cnt in counter.items():
#                             sym_map.setdefault(s, {})[d] = sym_map.setdefault(s, {}).get(d, 0) + cnt
#                     self.symptom_map = sym_map
#                     self.symptom_vocab = sorted(list(set(sym_map.keys())))
#                 except Exception:
#                     self.symptom_map = {}
#                     self.disease_symptoms = {}
#                     self.symptom_vocab = []

#         # try to load descriptions and precautions from local CSVs if still empty
#         desc_candidates = [
#             os.path.join(self.models_dir, "symptom_description.csv"),
#             os.path.join(".", "symptom_description.csv"),
#         ]
#         prec_candidates = [
#             os.path.join(self.models_dir, "symptom_precaution.csv"),
#             os.path.join(".", "symptom_precaution.csv"),
#         ]

#         if not self.descriptions:
#             desc_path = next((p for p in desc_candidates if os.path.exists(p)), None)
#             if desc_path:
#                 try:
#                     df_desc = read_csv_fuzzy(desc_path)
#                     if df_desc.shape[1] >= 2:
#                         kcol = df_desc.columns[0]
#                         vcol = df_desc.columns[1]
#                         for _, r in df_desc.iterrows():
#                             k = str(r[kcol]).strip()
#                             v = str(r[vcol]).strip()
#                             if k:
#                                 self.descriptions[k] = v
#                     else:
#                         for _, r in df_desc.iterrows():
#                             raw = str(r[df_desc.columns[0]])
#                             if "\t" in raw:
#                                 k, v = raw.split("\t", 1)
#                                 self.descriptions[k.strip()] = v.strip()
#                 except Exception:
#                     self.descriptions = {}

#         if not self.precautions:
#             prec_path = next((p for p in prec_candidates if os.path.exists(p)), None)
#             if prec_path:
#                 try:
#                     df_prec = read_csv_fuzzy(prec_path)
#                     if df_prec.shape[1] >= 2:
#                         for _, r in df_prec.iterrows():
#                             key = str(r[df_prec.columns[0]]).strip()
#                             vals = []
#                             for c in df_prec.columns[1:]:
#                                 v = str(r[c]).strip()
#                                 if v and v.lower() not in ("nan", "na"):
#                                     vals.append(v)
#                             if key:
#                                 self.precautions[key] = vals
#                     else:
#                         for _, r in df_prec.iterrows():
#                             raw = str(r[df_prec.columns[0]])
#                             if "\t" in raw:
#                                 k, v = raw.split("\t", 1)
#                                 self.precautions[k.strip()] = [v.strip()]
#                 except Exception:
#                     self.precautions = {}

#         # ensure totals present
#         for d in self.disease_symptoms:
#             self.disease_total_symptoms.setdefault(d, sum(self.disease_symptoms[d].values()) or len(self.disease_symptoms[d]) or 1)

#     # -------- classifier helper (Keras) --------
#     def classify_text(self, text: str) -> Optional[Dict[str, Any]]:
#         """
#         Use TF-IDF + Keras classifier to predict intent and confidence.
#         Returns {"intent": <str>, "confidence": <float>, "probs": {label:prob, ...}} or None when classifier not available.
#         """
#         if not self._keras_model or not self._tfidf or not self._label_encoder:
#             return None
#         try:
#             X = self._tfidf.transform([text])
#             X_arr = X.toarray()
#             probs = self._keras_model.predict(X_arr, verbose=0)[0]
#             top_idx = int(probs.argmax())
#             top_conf = float(probs[top_idx])
#             try:
#                 intent_label = str(self._label_encoder.inverse_transform([top_idx])[0])
#             except Exception:
#                 # label encoder might map classes differently; attempt safe mapping
#                 classes = getattr(self._label_encoder, "classes_", None)
#                 if classes is not None and top_idx < len(classes):
#                     intent_label = str(classes[top_idx])
#                 else:
#                     intent_label = "unknown"
#             prob_dict = {}
#             classes = getattr(self._label_encoder, "classes_", None)
#             if classes is not None and len(classes) == len(probs):
#                 for c_idx, c in enumerate(classes):
#                     prob_dict[str(c)] = float(probs[c_idx])
#             return {"intent": intent_label, "confidence": top_conf, "probs": prob_dict}
#         except Exception:
#             return None

#     # -------- symptom extraction & mapping --------
#     def _extract_candidate_tokens(self, text: str) -> List[str]:
#         if not text or not text.strip():
#             return []
#         text = text.strip().lower()
#         parts = [p.strip() for p in _TOKEN_SPLIT_RE.split(text) if p.strip()]
#         candidates = []
#         for p in parts:
#             subparts = re.split(r"\band\b|\bwith\b|\bplus\b", p)
#             for sp in subparts:
#                 sp = sp.strip()
#                 if not sp:
#                     continue
#                 words = re.findall(r"[a-z0-9]+", sp)
#                 if not words:
#                     continue
#                 for n in range(1, min(4, len(words) + 1)):
#                     for i in range(len(words) - n + 1):
#                         gram = " ".join(words[i : i + n])
#                         candidates.append(gram)
#         seen = set()
#         out = []
#         for c in candidates:
#             n = normalize_symptom_token(c)
#             if n and n not in seen:
#                 seen.add(n)
#                 out.append(n)
#         return out

#     def _map_to_known_symptoms(self, candidate_tokens: List[str]) -> List[str]:
#         if not candidate_tokens:
#             return []
#         matched = []
#         vocab = self.symptom_vocab or []
#         low_vocab = {v.lower(): v for v in vocab}
#         for tok in candidate_tokens:
#             if not tok:
#                 continue
#             if tok in low_vocab:
#                 matched.append(low_vocab[tok])
#                 continue
#             tok_tokens = set(_norm_tokens(tok))
#             best = None
#             best_overlap = 0
#             for v_lower, v_orig in low_vocab.items():
#                 v_tokens = set(_norm_tokens(v_lower))
#                 overlap = len(tok_tokens & v_tokens)
#                 if overlap > best_overlap:
#                     best_overlap = overlap
#                     best = v_orig
#             if best and best_overlap >= 1:
#                 matched.append(best)
#                 continue
#             if vocab:
#                 matches = difflib.get_close_matches(tok, list(low_vocab.keys()), n=1, cutoff=FUZZY_SYM_CUTOFF)
#                 if matches:
#                     matched.append(low_vocab[matches[0]])
#                     continue
#             matched.append(tok)
#         # dedupe preserve order
#         seen = set()
#         out = []
#         for m in matched:
#             if m not in seen:
#                 seen.add(m)
#                 out.append(m)
#         return out

#     # -------- scoring --------
#     def _score_diseases(self, matched_symptoms: List[str]) -> List[Tuple[str, float, List[str]]]:
#         scores = []
#         if not matched_symptoms:
#             return []
#         mset = set(matched_symptoms)
#         for d, counter in self.disease_symptoms.items():
#             if not counter:
#                 continue
#             total = float(self.disease_total_symptoms.get(d, sum(counter.values()) or len(counter) or 1))
#             matched_items = []
#             ssum = 0.0
#             for s, cnt in counter.items():
#                 if s in mset:
#                     matched_items.append(s)
#                     ssum += min(cnt, 3)
#             raw_frac = (ssum / total) if total > 0 else 0.0
#             uniq_boost = min(len(matched_items) / (len(counter) + 1), 0.5)
#             score = raw_frac * 0.9 + uniq_boost * 0.1
#             score = max(0.0, min(1.0, float(score)))
#             if matched_items:
#                 scores.append((d, score, matched_items))
#         scores.sort(key=lambda x: (x[1], len(x[2])), reverse=True)
#         return scores

#     # -------- public infer --------
#     def infer(self, query: str) -> Dict[str, Any]:
#         q = (query or "").strip()
#         if not q:
#             return {
#                 "intent": "unknown",
#                 "confidence": 0.0,
#                 "reply": "I didn't get any symptoms. Could you state them (e.g., 'fever, cough')?",
#                 "description": "",
#                 "precaution": [],
#                 "emergency": False,
#             }

#         # Emergency detection
#         try:
#             if is_emergency_text(q):
#                 reply = (
#                     "⚠️ EMERGENCY ALERT: This message contains signs of a possible medical emergency "
#                     "such as severe chest pain or breathing difficulty. Please call local emergency services or go to the nearest emergency department immediately."
#                 )
#                 return {
#                     "intent": "emergency",
#                     "confidence": 1.0,
#                     "reply": reply,
#                     "description": "",
#                     "precaution": [],
#                     "emergency": True,
#                 }
#         except Exception:
#             pass

#         # Extract and map symptoms (do not echo input)
#         candidates = self._extract_candidate_tokens(q)
#         mapped = self._map_to_known_symptoms(candidates)
#         # Filter common stop-words left-over
#         mapped = [m for m in mapped if len(m) > 1 and m not in {"have", "i", "since", "suffering", "experiencing", "symptoms", "symptom"}]

#         # If none matched, attempt word-level mapping
#         if not mapped:
#             words = re.findall(r"[a-z0-9]+", q.lower())
#             cand_words = [normalize_symptom_token(w) for w in words if len(w) > 2]
#             mapped = self._map_to_known_symptoms(cand_words)

#         # Count number of *distinct* user-provided symptoms (after normalization)
#         user_sym_count = len(mapped)

#         # Score diseases
#         scored = self._score_diseases(mapped)

#         # Looser substring match if scored empty
#         if not scored and self.symptom_vocab:
#             substr_matches = []
#             q_low = q.lower()
#             for s in self.symptom_vocab:
#                 if s and s in q_low:
#                     substr_matches.append(s)
#             if substr_matches:
#                 scored = self._score_diseases(substr_matches)

#         # Try classifier fallback if no scored results or scored confidence is low
#         classifier_result = self.classify_text(q) if _KERAS_AVAILABLE else None

#         if not scored:
#             if classifier_result is None:
#                 clarification = "I couldn't confidently match those symptoms to any condition in my knowledge base. Please list symptoms separated by commas and mention how long you've had them."
#                 return {
#                     "intent": "unknown",
#                     "confidence": 0.0,
#                     "reply": clarification,
#                     "description": "",
#                     "precaution": [],
#                     "emergency": False,
#                 }
#             else:
#                 # use classifier output to build a lightweight reply
#                 intent_label = classifier_result.get("intent", "unknown")
#                 conf = float(classifier_result.get("confidence", 0.0))
#                 desc = self.descriptions.get(intent_label, "") or self.descriptions.get(intent_label.lower(), "")
#                 raw_prec = self.precautions.get(intent_label, []) or self.precautions.get(intent_label.lower(), []) or []
#                 sanitized = _sanitize_precautions(raw_prec)
#                 prec_map = {intent_label: sanitized}
#                 reply_text = f"👩‍⚕️ Healthcare Assistant:\nPossible Condition(s): {intent_label}\n\nExplanation:\n{intent_label}: {(_first_sentence(desc) or 'Matches symptom pattern in dataset.')}\n\nPrecaution / Advice:\n{intent_label}:\n"
#                 for b in _pretty_bullets(sanitized[:3]):
#                     reply_text += b + "\n"
#                 return {
#                     "intent": intent_label,
#                     "confidence": conf,
#                     "reply": reply_text,
#                     "description": _first_sentence(desc) or "",
#                     "precaution": prec_map,
#                     "emergency": False,
#                 }

#         # At this point we have symptom-based scored results
#         # Determine top score from symptom scoring
#         top_score = float(scored[0][1]) if scored else 0.0

#         # If classifier exists and is more confident, prefer classifier where appropriate
#         if classifier_result:
#             clf_intent = classifier_result.get("intent")
#             clf_conf = float(classifier_result.get("confidence", 0.0))
#             # If classifier highly confident and top_score is low, use classifier
#             if clf_conf >= max(0.5, top_score + 0.15):
#                 # use classifier prediction
#                 intent_label = clf_intent
#                 desc = self.descriptions.get(intent_label, "") or self.descriptions.get(intent_label.lower(), "")
#                 raw_prec = self.precautions.get(intent_label, []) or self.precautions.get(intent_label.lower(), []) or []
#                 sanitized = _sanitize_precautions(raw_prec)
#                 prec_map = {intent_label: sanitized}
#                 reply_text = f"👩‍⚕️ Healthcare Assistant:\nPossible Condition(s): {intent_label}\n\nExplanation:\n{intent_label}: {(_first_sentence(desc) or 'Matches symptom pattern in dataset.')}\n\nPrecaution / Advice:\n{intent_label}:\n"
#                 for b in _pretty_bullets(sanitized[:3]):
#                     reply_text += b + "\n"
#                 return {
#                     "intent": intent_label,
#                     "confidence": clf_conf,
#                     "reply": reply_text,
#                     "description": _first_sentence(desc) or "",
#                     "precaution": prec_map,
#                     "emergency": False,
#                 }

#         # Otherwise keep symptom-based results (apply adaptive prediction rules)
#         if user_sym_count >= 3:
#             top_k = [scored[0]]
#         else:
#             top_k = scored[:MAX_SUGGESTIONS]

#         disease_names = [t[0] for t in top_k]
#         possible_line = ", ".join(disease_names)

#         explanation_blocks = []
#         best_desc = ""
#         for d, score, matched_items in top_k:
#             desc = self.descriptions.get(d) or self.descriptions.get(d.lower()) or ""
#             if desc:
#                 short_desc = _first_sentence(desc)
#             else:
#                 if matched_items:
#                     short_desc = f"Matched symptoms: {', '.join(matched_items)}."
#                 else:
#                     short_desc = "Matches symptom pattern in dataset."
#             explanation_blocks.append((d, short_desc))
#             if not best_desc:
#                 best_desc = short_desc

#         precaution_blocks = []
#         for d, _, _ in top_k:
#             raw_prec = self.precautions.get(d) or self.precautions.get(d.lower()) or []
#             sanitized = _sanitize_precautions(raw_prec)
#             seen = set()
#             filtered = []
#             for p in sanitized:
#                 if p not in seen:
#                     seen.add(p)
#                     filtered.append(p)
#             precaution_blocks.append((d, filtered[:3]))

#         lines: List[str] = []
#         lines.append("👩‍⚕️ Healthcare Assistant:")
#         lines.append(f"Possible Condition(s): {possible_line}")
#         lines.append("")
#         lines.append("Explanation:")
#         lines.append("")
#         for d, desc in explanation_blocks:
#             lines.append(f"{d}: {desc}")
#             lines.append("")
#         lines.append("Precaution / Advice:")
#         lines.append("")
#         for d, precs in precaution_blocks:
#             lines.append(f"{d}:")
#             for b in _pretty_bullets(precs, max_items=3):
#                 lines.append(b)
#             lines.append("")

#         reply_text = "\n".join(lines).strip()
#         top_score = float(top_k[0][1]) if top_k else 0.0
#         intent_field = top_k[0][0] if (user_sym_count >= 3 or top_score >= 0.45) else "multiple"

#         return {
#             "intent": intent_field,
#             "confidence": float(top_score),
#             "reply": reply_text,
#             "description": best_desc or "",
#             "precaution": {d: p for d, p in precaution_blocks},
#             "emergency": False,
#         }

#     # convenience properties
#     @property
#     def descriptions_map(self) -> Dict[str, str]:
#         return self.descriptions

#     @property
#     def precautions_map(self) -> Dict[str, List[str]]:
#         return self.precautions

#     @property
#     def symptom_vocabulary(self) -> List[str]:
#         return self.symptom_vocab

























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
