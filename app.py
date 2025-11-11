###############################################################
# app.py
"""
Flask UI wrapper for your Medical Chatbot.

- Preserves your backend imports and sanitizer logic (adapted from your Streamlit app).
- Keeps the app stateless: conversation=None is passed to enrich_result calls.
- By default, the visible text returned to the client strips any "Precautions:" paragraph
  (precautions remain available via the "precaution" JSON field).
- Endpoint:
    GET/POST /get?msg=... or form field "msg" -> returns JSON:
      {
        "answer": "<clean explanation (Precautions removed)>",
        "intent": "...",
        "confidence": ...,
        "precaution": [...],     # list (may be empty)
        "emergency": true/false
      }
- Serves index.html at "/". Keep your front-end unchanged. The front-end can choose to
  display precaution data from the "precaution" field if desired.
"""
from __future__ import annotations

import os
import sys
import re
import traceback
import logging
from typing import List, Tuple, Any, Dict, Optional

from flask import Flask, request, jsonify, send_file

# Ensure repo root and src importable (same pattern you used)
ROOT = os.getcwd()
if ROOT not in sys.path:
    sys.path.append(ROOT)
SRC_PATH = os.path.join(ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

# Logging
logger = logging.getLogger("flask_medibot")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(ch)

# --- Preserve your backend imports EXACTLY (do not change) ---
try:
    from src.chatbot import get_chatbot_response  # type: ignore
    logger.info("Imported src.chatbot.get_chatbot_response")
except Exception:
    try:
        from chatbot import get_chatbot_response
        logger.info("Imported chatbot.get_chatbot_response (fallback)")
    except Exception:
        get_chatbot_response = None
        logger.warning("get_chatbot_response not available; endpoint will return an explanatory message.")

# optional LLM enrichment helper (same approach as Streamlit)
_enrich = None
try:
    from src.llm import enrich_result as _enrich  # type: ignore
    logger.info("Imported src.llm.enrich_result")
except Exception:
    try:
        from llm import enrich_result as _enrich
        logger.info("Imported llm.enrich_result (fallback)")
    except Exception:
        _enrich = None
        logger.info("LLM enrich helper not available; continuing without it")

# configuration constants (match Streamlit)
MODELS_DIR = "models"
MAX_CHARS = 600

# -------------------------
# Sanitizer helpers (kept identical / robust)
# -------------------------
_CONJ_RE = re.compile(r'^(and|or|then|also|but)\b[:,;\s]*', flags=re.I)

def _split_candidates(text: str) -> List[str]:
    if not text:
        return []
    t = text.replace("|", ",").replace(";", ",")
    parts = [p.strip() for p in re.split(r',|\n|•|-|\u2022', t) if p.strip()]
    return parts

def _normalize_prec_item(it: str) -> str:
    it = it.strip()
    it = re.sub(r'^[\-\–\—\•\*]+\s*', '', it)
    it = _CONJ_RE.sub('', it)
    it = re.sub(r'\s+', ' ', it)
    it = it.rstrip('.')
    return it.strip()

def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        k = x.lower()
        if k and k not in seen:
            seen.add(k)
            out.append(x)
    return out

def _remove_precisions_in_explanation(prec_list: List[str], explanation: str) -> List[str]:
    expl_norm = re.sub(r'\s+', ' ', (explanation or "")).strip().lower()
    out: List[str] = []
    for p in prec_list:
        p_norm = re.sub(r'\s+', ' ', p).strip().lower()
        if not p_norm:
            continue
        if p_norm in expl_norm:
            continue
        out.append(p)
    return out

def _sanitize_reply(raw_reply: str, fallback_prec: Any) -> Tuple[str, List[str]]:
    """
    Return (reply_text, prec_list).
    Compose an explanation (first paragraph or first two sentences) plus optional Precautions list.
    """
    if not raw_reply:
        return ("", [])

    # Normalize and split into paragraphs
    text = raw_reply.replace("|", ",").strip()
    paras = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

    # Short greeting detection (keep greeting, remove any Precautions lines)
    first_para = paras[0] if paras else ""
    lower_first = first_para.lower()
    if re.fullmatch(r'(?i)(hi|hello|hey|good (day|morning|afternoon|evening)|thanks|thank you|bye|goodbye|take care)([.!]?)', lower_first.strip()):
        lines = [l for l in raw_reply.splitlines() if not re.search(r'precaution', l, flags=re.I)]
        cleaned = "\n".join([ln.strip() for ln in lines if ln.strip()])
        return (cleaned, [])

    # Explanation: first paragraph or first 2 sentences
    explanation = ""
    if paras:
        cand = paras[0]
        sents = [s.strip() for s in re.split(r'(?<=[\.\?\!])\s+', cand) if s.strip()]
        explanation = " ".join(sents[:2]) if sents else cand
    else:
        explanation = text if len(text) < 400 else text[:400] + "..."

    # Build precaution candidates
    prec_candidates: List[str] = []
    for line in raw_reply.splitlines():
        if re.search(r'precaution', line, flags=re.I):
            after = re.sub(r'(?i)precautions?:', '', line).strip()
            prec_candidates.extend(_split_candidates(after))

    if not prec_candidates and len(paras) > 1:
        for para in paras[1:]:
            if re.search(r'\b(rest|stay|hydrate|avoid|keep|monitor|seek|wash|bathe|apply|use)\b', para, flags=re.I):
                prec_candidates.extend(_split_candidates(para))

    if not prec_candidates:
        if isinstance(fallback_prec, (list, tuple)):
            for p in fallback_prec:
                prec_candidates.extend(_split_candidates(str(p)))
        elif isinstance(fallback_prec, str) and fallback_prec.strip():
            prec_candidates.extend(_split_candidates(fallback_prec))

    normed = [_normalize_prec_item(p) for p in prec_candidates if p and p.strip()]
    normed = [p for p in normed if len(p) > 1]
    normed = _dedupe_keep_order(normed)

    final_prec = _remove_precisions_in_explanation(normed, explanation)
    final_prec = [p for p in final_prec if re.search(r'\w', p)]

    # Compose final reply text (Precautions will be included here if needed,
    # but app default will not show that paragraph to the client — see controller)
    parts: List[str] = []
    explanation = explanation.strip()
    if explanation and not explanation.endswith('.'):
        explanation = explanation + '.'
    if explanation:
        parts.append(explanation)

    if final_prec:
        comma_joined = ", ".join(final_prec)
        parts.append("Precautions: " + comma_joined + ".")

    reply = "\n\n".join(parts).strip()
    reply = re.sub(r',\s*', ', ', reply)
    reply = re.sub(r'\s+\.', '.', reply)
    reply = re.sub(r'\s{2,}', ' ', reply)

    return reply, final_prec

# helper: short greeting intent detection (keeps parity with Streamlit)
def _is_short_greeting_intent(intent_str: str) -> bool:
    if not intent_str:
        return False
    s = str(intent_str).lower()
    greetings = ("greeting", "hello", "hi", "thanks", "thank_you", "thank", "goodbye", "bye")
    for g in greetings:
        if g in s:
            return True
    return False

# -------------------------
# Flask app & routes
# -------------------------
app = Flask(__name__)

@app.route("/")
def index():
    # serve the single HTML UI file placed next to this app.py
    index_path = os.path.join(ROOT, "index.html")
    if os.path.exists(index_path):
        return send_file(index_path)
    return "<h3>Place your index.html next to this app.py</h3>", 200

@app.route("/get", methods=["POST", "GET"])
def chat():
    try:
        if request.method == "POST":
            msg = request.form.get("msg", "")
        else:
            msg = request.args.get("msg", "")

        msg = (msg or "").strip()
        if not msg:
            return jsonify({"error": "No message provided."}), 400

        logger.info("User query: %s", msg)

        # Call backend classifier same as Streamlit code
        if get_chatbot_response is None:
            bot_text = "Backend not available for import. Ensure 'src/chatbot.py' is present and exports get_chatbot_response()."
            result: Dict[str, Any] = {"reply": bot_text, "intent": "unknown", "confidence": 0.0, "description": "", "precaution": [], "emergency": False}
        else:
            try:
                result = get_chatbot_response(msg, models_dir=MODELS_DIR, threshold=0.5)
                if not isinstance(result, dict):
                    result = {"reply": str(result), "intent": "unknown", "confidence": 0.0, "description": "", "precaution": [], "emergency": False}
            except Exception as e:
                logger.exception("Error in get_chatbot_response:")
                result = {"reply": f"Error while processing: {e}", "intent": "error", "confidence": 0.0, "description": "", "precaution": [], "emergency": False}

        # Decide whether to call _enrich (LLM) - keep greeting/thanks handled by baseline classifier only
        call_llm = False
        try:
            intent_str = str(result.get("intent", "")).lower()
        except Exception:
            intent_str = ""

        if _enrich is not None and not _is_short_greeting_intent(intent_str):
            call_llm = True

        if call_llm:
            enriched = None
            try:
                # try multiple signatures defensively; stateless by default
                try:
                    enriched = _enrich(result, msg, conversation=None, max_retries=2, prefer_default_model=True)
                except TypeError:
                    try:
                        enriched = _enrich(result, msg, max_retries=2, prefer_default_model=True)
                    except TypeError:
                        enriched = _enrich(result, msg)
            except Exception:
                logger.exception("LLM enrich failed")
                enriched = None

            if enriched and isinstance(enriched, dict) and enriched.get("reply"):
                raw_llm_reply = enriched.get("reply", "")
                fallback_prec = enriched.get("precaution", result.get("precaution", []))
                try:
                    cleaned_reply, prec_list = _sanitize_reply(raw_llm_reply, fallback_prec)
                except Exception:
                    cleaned_reply, prec_list = _sanitize_reply(str(raw_llm_reply), result.get("precaution", []))

                enriched["reply"] = cleaned_reply or enriched.get("reply", "")
                enriched["precaution"] = prec_list if prec_list is not None else enriched.get("precaution", [])
                enriched["emergency"] = result.get("emergency", False)
                enriched["llm_used"] = True

                # merge into result, preserve classifier intent/confidence
                result["reply"] = enriched.get("reply", result.get("reply", ""))
                result["description"] = enriched.get("description", result.get("description", ""))
                result["precaution"] = enriched.get("precaution", result.get("precaution", []))
                result["llm_used"] = True
                logger.info("LLM enriched reply (llm_used=True).")
            else:
                # LLM not helpful; sanitize classifier reply
                cr = result.get("reply", "")
                cleaned, prec_list = _sanitize_reply(cr, result.get("precaution", []))
                if cleaned:
                    result["reply"] = cleaned
                    result["precaution"] = prec_list
                result["llm_used"] = False
                logger.info("LLM did not return usable content; used classifier reply (sanitized).")
        else:
            # No LLM call - sanitize classifier reply for hygiene
            cr = result.get("reply", "")
            try:
                cleaned, prec_list = _sanitize_reply(cr, result.get("precaution", []))
                if cleaned:
                    result["reply"] = cleaned
                    result["precaution"] = prec_list
            except Exception:
                logger.exception("Sanitizer failed on classifier reply; leaving original reply.")

        # emergency metadata logged (UI may use this)
        if result.get("emergency"):
            logger.warning("Emergency flagged in result")

        # --- IMPORTANT: By default we strip the "Precautions:" paragraph from the visible answer.
        # The "precaution" metadata remains available in the JSON response for UI to show if needed.
        try:
            # remove any trailing "Precautions: ..." (case-insensitive, multiline-safe)
            reply_text = re.sub(r'(?is)\n*\s*precautions?:.*$', '', (result.get("reply","") or "")).strip()
        except Exception:
            reply_text = result.get("reply","") or ""

        # Build response JSON (keep types simple)
        response_payload = {
            "answer": reply_text,
            "intent": result.get("intent"),
            "confidence": result.get("confidence"),
            "precaution": result.get("precaution", []),
            "emergency": bool(result.get("emergency", False))
        }

        return jsonify(response_payload)

    except Exception:
        logger.error("Unhandled exception in /get endpoint:\n%s", traceback.format_exc())
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    # Start dev server
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)
