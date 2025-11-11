
# src/chatbot.py 
"""
src/chatbot.py (stateless)

Revised, robust, importable chatbot helper.

Features:
- Lazy import / cached InferenceEngine loader.
- Improved topic extraction (many phrasings + fuzzy/token match).
- Stateless handling: each query is independent.
- Public API:
    - get_chatbot_response(query, models_dir, threshold) -> structured dict
    - chatbot_reply(query, models_dir, threshold) -> textual WhatsApp-style reply
- CLI interactive and single-query modes.
"""

import argparse
import os
import re
import sys
import random
import difflib
from typing import Any, Dict, List, Iterable, Optional, Tuple

# Module-level placeholder for lazy import
InferenceEngine = None  # type: ignore

# Try early import (best-effort). If it fails, _get_engine will import lazily.
try:
    from inference import InferenceEngine  # type: ignore
except Exception:
    try:
        # when run as package
        from .inference import InferenceEngine  # type: ignore
    except Exception:
        InferenceEngine = None  # will be loaded by _get_engine

# ---------------- Config / constants ----------------
GREETINGS = [
    "Hello! How can I help you today?",
    "Hi there! I'm your healthcare assistant.",
    "Hey! How are you feeling today?",
    "Good day! Tell me your symptoms and I'll assist."
]

FALLBACKS = [
    "I'm not sure I understood that. Could you rephrase?",
    "Sorry, I didn’t get that. Can you try again?",
    "I’m learning every day, but I missed that. Please rephrase."
]

# ---------------- Small-talk detection (expanded) ----------------
GREETING_RE = re.compile(
    r"\b(hi|hello|hey|hiya|yo|good morning|good afternoon|good evening|morning|afternoon|evening)\b",
    re.I,
)
THANKS_RE = re.compile(r"\b(thanks|thank you|thx|cheers|much appreciated)\b", re.I)
HOW_ARE_RE = re.compile(r"\b(how are you|how r u|how ru|how you|how's it going|how are things)\b", re.I)
BYE_RE = re.compile(r"\b(bye|goodbye|see you|take care|see ya|farewell)\b", re.I)

# ---------------- Expanded direct-query patterns ----------------
WHAT_IS_PATTERNS = [
    re.compile(r"^(?:what(?:'s| is)|whats|what)\s+(?:the\s+)?(.+?)[\?\.]*$", re.I),
    re.compile(r"^(?:who(?:'s| is)|who)\s+(?:is\s+)?(.+?)[\?\.]*$", re.I),
    re.compile(r"^(?:define|definition of|define the)\s+(.+?)[\?\.]*$", re.I),
    re.compile(r"^(?:explain|explanation of|explain the)\s+(.+?)[\?\.]*$", re.I),
    re.compile(r"^(?:tell me about|give me information about|give me info on|info on|information about)\s+(.+?)[\?\.]*$", re.I),
]

PRECAUTIONS_PATTERNS = [
    re.compile(r"(?:(?:suggest|give|show|what are|list|tell me)\s+precautions\s+for|precautions\s+for)\s+(.+?)[\?\.]*$", re.I),
    re.compile(r"(?:how to prevent|how to avoid|ways to prevent|preventing|prevent from)\s+(.+?)[\?\.]*$", re.I),
    re.compile(r"(?:precautions|preventive measures|prevention measures)\s+(?:for|against)?\s*(.+?)[\?\.]*$", re.I),
]

TREATMENT_PATTERNS = [
    re.compile(r"(?:how to treat|treatment for|treatments for|treat)\s+(.+?)[\?\.]*$", re.I),
    re.compile(r"(?:what can I do for|what should I do for|remedies for)\s+(.+?)[\?\.]*$", re.I),
]

SYMPTOMS_PATTERNS = [
    re.compile(r"(?:what are the symptoms of|symptoms of|signs of|what are signs of)\s+(.+?)[\?\.]*$", re.I),
    re.compile(r"(?:i have|i'm having|i am having|i've been having|i have been experiencing|experiencing|suffering from)\s+(.+?)[\?\.]*$", re.I),
    re.compile(r"(?:my|the)\s+(.+?)\s+(hurts|is hurting|is sore|is painful|is itchy|is swollen)[\?\.]*", re.I),
]

ATTRIBUTE_PATTERNS = [
    re.compile(r"(?:is|are|can|does)\s+(.+?)\s+(?:contagious|infectious|serious|dangerous|fatal|common|treatable|curable|chronic)[\?\.]*", re.I),
    re.compile(r"(?:can|could)\s+(.+?)\s+(?:cause|lead to|result in)\s+(.+?)[\?\.]*", re.I),
]

# Matching thresholds
FUZZY_CUTOFF = 0.6
TOKEN_OVERLAP_THRESHOLD = 1

# Engine cache
_ENGINE_CACHE: Dict[str, Any] = {}


# ---------------- Utility helpers ----------------
def normalize_text(t: str) -> str:
    return re.sub(r"\s+", " ", t or "").strip()


def choose_deterministic(lst: List[str], seed_src: str = "") -> str:
    if not lst:
        return ""
    idx = abs(hash(seed_src)) % len(lst) if seed_src else random.randrange(len(lst))
    return lst[idx]


def short_description(description: str, max_sentences: int = 1) -> str:
    if not description:
        return ""
    parts = [s.strip() for s in re.split(r"[\.!?]\s*", description) if s.strip()]
    return (". ".join(parts[:max_sentences]) + ".") if parts else description


def precautions_as_lines(precaution_field: str) -> List[str]:
    if not precaution_field:
        return []
    items: List[str] = []
    for part in str(precaution_field).split(";"):
        for sub in part.split(","):
            s = sub.strip()
            if s and s.lower() not in ("nan", "na"):
                items.append(s)
    # dedupe preserve order
    seen = set()
    out = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def format_info_reply(title: str, description: str, precautions: List[str]) -> str:
    parts: List[str] = []
    if title:
        parts.append(f"{title.strip()}:")
    if description:
        parts.append(short_description(description, max_sentences=2))
    if precautions:
        parts.append("Precautions you can consider:")
        for p in precautions[:6]:
            parts.append(f"- {p}")
    return "\n\n".join(parts)


# ---------------- Topic extraction helpers (robust) ----------------
def _norm_tokens(text: str) -> List[str]:
    text = (text or "").lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    return tokens


def _map_to_candidate(captured: str, candidate_keys: Iterable[str]) -> Optional[str]:
    """
    Map captured topic -> canonical candidate key using:
      1) exact case-insensitive
      2) token overlap
      3) fuzzy match
    """
    if not captured:
        return None
    cap = captured.strip().lower()
    low_map = {c.lower(): c for c in candidate_keys}
    # exact ci
    if cap in low_map:
        return low_map[cap]
    # token overlap
    cap_tokens = set(_norm_tokens(captured))
    best = None
    best_score = 0
    for orig_lower, orig in low_map.items():
        orig_tokens = set(_norm_tokens(orig_lower))
        overlap = len(cap_tokens & orig_tokens)
        if overlap > best_score:
            best_score = overlap
            best = orig
    if best_score >= TOKEN_OVERLAP_THRESHOLD:
        return best
    # fuzzy fallback
    matches = difflib.get_close_matches(cap, list(low_map.keys()), n=1, cutoff=FUZZY_CUTOFF)
    if matches:
        return low_map[matches[0]]
    return None


def extract_topic_from_query(query: str, candidate_keys: Optional[Iterable[str]] = None) -> Optional[str]:
    """
    Attempt to extract a canonical topic/disease string from the user's query.
    Strategy:
      1. Try explicit captures from many patterns (what is, symptoms of, precautions for, etc.).
      2. If capture exists, attempt to map to a candidate key (if provided).
      3. Otherwise, check candidate keys via substring, token-overlap, fuzzy match.
    Returns canonical candidate string (as in candidate_keys) or a captured raw topic (if no candidate_keys).
    """
    q = (query or "").strip()
    if not q:
        return None

    # 1) Check explicit capture patterns
    for pat_list in (WHAT_IS_PATTERNS, SYMPTOMS_PATTERNS, PRECAUTIONS_PATTERNS, TREATMENT_PATTERNS, ATTRIBUTE_PATTERNS):
        for pat in pat_list:
            m = pat.search(q)
            if m:
                for g in m.groups():
                    if g:
                        captured = g.strip()
                        if candidate_keys:
                            mapped = _map_to_candidate(captured, candidate_keys)
                            if mapped:
                                return mapped
                        return captured

    # 2) If candidate_keys provided, try substring match
    if candidate_keys:
        q_low = q.lower()
        for cand in candidate_keys:
            if not cand:
                continue
            if cand.lower() in q_low:
                return cand

    # 3) token-overlap heuristic
    if candidate_keys:
        tokens = set(_norm_tokens(q))
        best = None
        best_score = 0
        for cand in candidate_keys:
            cand_tokens = set(_norm_tokens(cand))
            if not cand_tokens:
                continue
            overlap = len(tokens & cand_tokens)
            if overlap > best_score:
                best = cand
                best_score = overlap
        if best_score >= TOKEN_OVERLAP_THRESHOLD:
            return best

    # 4) fuzzy match against labels
    if candidate_keys:
        low_map = {c.lower(): c for c in candidate_keys}
        nmatches = difflib.get_close_matches(q.lower(), list(low_map.keys()), n=1, cutoff=FUZZY_CUTOFF)
        if nmatches:
            return low_map[nmatches[0]]

    return None


# ---------------- Inference engine loader ----------------
def _get_engine(models_dir: str = "models", threshold: float = 0.5) -> "InferenceEngine":  # type: ignore
    """
    Return cached InferenceEngine instance for given models_dir.
    Loads engine lazily and raises helpful error if not available.
    """
    key = f"{os.path.abspath(models_dir)}|{float(threshold)}"
    if key in _ENGINE_CACHE:
        return _ENGINE_CACHE[key]

    global InferenceEngine
    if InferenceEngine is None:
        # Try to import the class now; try common paths
        tried: List[Tuple[str, str]] = []
        try:
            from inference import InferenceEngine as _IE  # type: ignore
            InferenceEngine = _IE
        except Exception as e1:
            tried.append(("inference", str(e1)))
            try:
                from .inference import InferenceEngine as _IE  # type: ignore
                InferenceEngine = _IE
            except Exception as e2:
                tried.append((".inference", str(e2)))
                raise RuntimeError(
                    f"InferenceEngine not available. Tried imports: {tried}. "
                    "Ensure src/inference.py exists and exports InferenceEngine and your PYTHONPATH includes project root."
                )

    engine = InferenceEngine(models_dir, threshold=threshold)
    _ENGINE_CACHE[key] = engine
    return engine


# ---------------- Conversation formatting ----------------
def format_conversational_reply(out: Dict[str, Any], threshold: float = 0.5) -> str:
    if not isinstance(out, dict):
        return "Sorry, I couldn't process that."
    if out.get("emergency"):
        return out.get("reply", "")

    confidence = float(out.get("confidence", 0.0))
    reply_text = out.get("reply", "") or ""
    description = out.get("description", "") or ""
    precaution_field = out.get("precaution", "") or ""

    if confidence < threshold:
        clar = reply_text or "Could you tell me more about your symptoms?"
        return clar

    parts: List[str] = []
    if reply_text:
        parts.append(reply_text)
    short_desc = short_description(description, max_sentences=1)
    if short_desc and short_desc not in reply_text:
        parts.append(short_desc)
    prec_lines = precautions_as_lines(precaution_field)
    if prec_lines:
        parts.append("Precautions you can try:")
        for p in prec_lines[:4]:
            parts.append(f"- {p}")
    return "\n\n".join(parts)


# ---------------- Stateless handler ----------------
def handle_query_stateless(engine: "InferenceEngine", query: str, threshold: float = 0.5) -> str:  # type: ignore
    q = normalize_text(query)
    if not q:
        return choose_deterministic(FALLBACKS, seed_src="empty")

    # Small-talk
    if GREETING_RE.search(q):
        return choose_deterministic(GREETINGS, seed_src=q)
    if THANKS_RE.search(q):
        return "You're welcome!"
    if HOW_ARE_RE.search(q):
        return "I'm here to help — tell me your symptoms or ask for general advice."
    if BYE_RE.search(q):
        return "Take care!"

    # Prepare candidate keys from engine metadata if available
    desc_map = getattr(engine, "descriptions", {}) or {}
    prec_map = getattr(engine, "precautions", {}) or {}
    candidate_keys = list(desc_map.keys()) if desc_map else list(prec_map.keys()) if prec_map else []

    # 1) Try to extract a topic from the query (handles many phrasings)
    topic = extract_topic_from_query(q, candidate_keys=candidate_keys) if candidate_keys else None
    if topic:
        # If topic maps to an existing canonical key, return info / precautions depending on query intent
        # Determine if user explicitly asked for precautions/treatment/symptoms by checking patterns
        asked_precautions = any(p.search(q) for p in PRECAUTIONS_PATTERNS)
        asked_symptoms = any(p.search(q) for p in SYMPTOMS_PATTERNS)
        asked_treatment = any(p.search(q) for p in TREATMENT_PATTERNS)
        asked_definition = any(p.search(q) for p in WHAT_IS_PATTERNS)

        description = desc_map.get(topic, "")
        precautions = prec_map.get(topic, []) if isinstance(prec_map, dict) else []

        if asked_precautions:
            parts = [f"Precautions for {topic}:"]
            for p in precautions[:8]:
                parts.append(f"- {p}")
            return "\n\n".join(parts)

        if asked_symptoms:
            # Provide short description (which often includes symptoms)
            return format_info_reply(topic, description, precautions)

        if asked_treatment:
            # If no explicit treatment metadata, fall back to classifier
            # but still return description + precaution hint
            reply = f"{topic}: {short_description(description, max_sentences=2)}"
            if precautions:
                reply += " For relief, consider: " + ", ".join(precautions[:3])
            return reply

        # default when topic detected: provide short info (definition + precautions)
        return format_info_reply(topic, description, precautions)

    # 2) No direct topic detected — fall back to classifier inference
    out = engine.infer(q)
    if out.get("emergency"):
        return out.get("reply", "")

    # If classifier low-confidence, return clarifying question (stateless)
    if out.get("confidence", 0.0) < threshold:
        clar = out.get("reply", "") or "Can you tell me how long you've had these symptoms?"
        return clar

    # Otherwise format a conversational reply from classifier output
    return format_conversational_reply(out, threshold)


# ---------------- Public API for direct import ----------------
def get_chatbot_response(query: str, models_dir: str = "models", threshold: float = 0.5) -> Dict[str, Any]:
    """
    Return a structured response dict for a single query.
    Fields: query, intent, confidence, reply, description, precaution, emergency
    """
    engine = _get_engine(models_dir=models_dir, threshold=threshold)
    out = engine.infer(query)
    text_reply = handle_query_stateless(engine, query, threshold=threshold)

    prec = out.get("precaution") or ""
    if isinstance(prec, (list, tuple)):
        prec_str = "; ".join(prec)
    else:
        prec_str = str(prec)

    resp = {
        "query": query,
        "intent": out.get("intent", "unknown"),
        "confidence": float(out.get("confidence", 0.0)),
        "reply": text_reply,
        "description": out.get("description", "") or "",
        "precaution": prec_str,
        "emergency": bool(out.get("emergency", False)),
    }
    return resp


def chatbot_reply(query: str, models_dir: str = "models", threshold: float = 0.5) -> str:
    resp = get_chatbot_response(query, models_dir=models_dir, threshold=threshold)
    return resp.get("reply", "")




# ----------------- CLI Entrypoints -----------------
def run_interactive(models_dir: str, threshold: float):
    engine = _get_engine(models_dir=models_dir, threshold=threshold)
    banner = (
        "Healthcare Chatbot (interactive)\n"
        "Stateless mode — each message is handled independently.\n"
        "Type a question or symptoms and press Enter. Type 'exit' or 'quit' to stop.\n"
    )
    print(banner)
    try:
        while True:
            try:
                user = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break
            if not user:
                continue
            if user.lower() in ("exit", "quit"):
                print("Bye.")
                break
            bot_reply = handle_query_stateless(engine, user, threshold=threshold)
            print()
            print(f"Bot: {bot_reply}")
            print()
    except KeyboardInterrupt:
        print("\nInterrupted. Goodbye.")


def run_single_query(models_dir: str, query: str, threshold: float):
    engine = _get_engine(models_dir=models_dir, threshold=threshold)
    bot_reply = handle_query_stateless(engine, query, threshold=threshold)
    print(f"You: {query}\n")
    print(f"Bot: {bot_reply}\n")


def main():
    p = argparse.ArgumentParser(description="Stateless CLI Healthcare Chatbot.")
    p.add_argument("--models_dir", required=True, help="Directory containing trained artifacts (models/)")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--interactive", action="store_true", help="Start interactive prompt")
    grp.add_argument("--query", type=str, help="Single query string to classify and reply to")
    p.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold (default 0.5)")
    args = p.parse_args()

    if args.interactive:
        run_interactive(args.models_dir, args.threshold)
    else:
        run_single_query(args.models_dir, args.query, args.threshold)


if __name__ == "__main__":
    main()
