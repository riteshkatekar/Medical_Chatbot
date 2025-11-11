# src/llm.py
"""
LLM enrichment layer (updated).

- Prepares a concise system+context prompt for the LLM
- Calls Groq (via simple HTTP POST) using env vars:
    GROQ_API_KEY, GROQ_MODEL_ID, LLM_TEMPERATURE, LLM_MAX_TOKENS
- Returns dict: {"reply": str, "description": str, "precaution": List[str], "llm_used": bool}
- Formats precautions as comma-separated list in the reply (e.g. "Precautions: a, b, c.")
- Falls back to offline rewrite when API key/model missing or request fails.
"""

import os
import time
import logging
from typing import Optional, Dict, Any, List
import re
import requests

LOG = logging.getLogger("llm")
LOG.addHandler(logging.NullHandler())

SYSTEM_PROMPT = (
    "You are a Medical assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "You are a medical assistant. Keep your answer concise (maximum 3 sentences). Do NOT repeat verbatim any text from the classifier; if you rephrase it, be brief. When listing precautions use short items; avoid pipes '|' and avoid repeating items. If unsure, say 'I don't know'. "
    "Do NOT use phrases like 'I'm sorry to hear...' or 'I'm glad I could help'. Be factual, cautious, and include a short comma-separated precaution list if appropriate."
)

# Groq-compatible endpoint used in prior working code paths
GROQ_CHAT_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")

def _read_env():
    return {
        "api_key": os.getenv("GROQ_API_KEY"),
        "model_id": os.getenv("GROQ_MODEL_ID") or os.getenv("GROQ_MODEL") or "",
        "temperature": float(os.getenv("LLM_TEMPERATURE", "0.0")),
        "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "512")),
        "timeout": int(os.getenv("LLM_HTTP_TIMEOUT", "20")),
    }

def _offline_rewrite(result: Dict[str, Any], user_message: str) -> Dict[str, Any]:
    """Simple conservative fallback that produces a concise reply using classifier output."""
    intent = result.get("intent", "")
    description = result.get("description", "") or ""
    prec = result.get("precaution", []) or []

    if intent and intent.lower() not in ("unknown", "error", "nan", ""):
        # join fallback precautions with commas
        prec_items = []
        if isinstance(prec, (list, tuple)):
            for p in prec:
                if p:
                    p_clean = str(p).replace("|", ",")
                    for part in [s.strip() for s in p_clean.split(",") if s.strip()]:
                        prec_items.append(part)
        elif isinstance(prec, str) and prec.strip():
            for part in [s.strip() for s in prec.replace("|", ",").split(",") if s.strip()]:
                prec_items.append(part)

        prec_items = list(dict.fromkeys(prec_items))  # dedupe preserve order

        reply_parts = []
        if description:
            # keep description as first paragraph
            reply_parts.append(description.splitlines()[0].strip())
        else:
            reply_parts.append(f"Possible condition: {intent}.")

        if prec_items:
            # single-line comma separated
            reply_parts.append("Precautions: " + ", ".join(prec_items[:12]) + ".")

        reply = "\n\n".join(reply_parts)
        return {"reply": reply, "description": description, "precaution": prec_items, "llm_used": False}
    else:
        reply = (
            "I couldn't confidently match your symptom to a condition. "
            "If this is urgent (severe pain, difficulty breathing, fainting) seek emergency care. "
            "Otherwise, can you share more symptoms or how long you've had this?"
        )
        return {"reply": reply, "description": "", "precaution": [], "llm_used": False}

def _build_messages(result: Dict[str, Any], user_message: str, conversation: Optional[List[Dict[str,str]]] = None) -> List[Dict[str,str]]:
    """Construct system + context + user messages for the chat model."""
    messages = []
    messages.append({"role": "system", "content": SYSTEM_PROMPT})

    ctx_parts = []
    intent = result.get("intent", "")
    if intent:
        ctx_parts.append(f"Classifier-predicted condition: {intent}.")

    description = result.get("description", "")
    if description:
        ctx_parts.append(f"Description: {description}")

    prec = result.get("precaution", None)
    if isinstance(prec, (list, tuple)):
        if prec:
            ctx_parts.append("Precautions: " + ", ".join(str(x).strip() for x in prec if x))
    elif isinstance(prec, str) and prec.strip():
        ctx_parts.append("Precautions: " + prec.strip())

    if ctx_parts:
        messages.append({"role": "assistant", "content": "Retrieved context:\n" + "\n".join(ctx_parts)})

    # optional conversation snippet (not used when stateless)
    if conversation:
        N = 6
        conv_texts = []
        for m in (conversation[-N:] if len(conversation) > N else conversation):
            sender = m.get("sender", "user")
            txt = m.get("text", "")
            conv_texts.append(f"{sender}: {txt}")
        messages.append({"role": "assistant", "content": "Conversation context:\n" + "\n".join(conv_texts)})

    user_block = (
        "User message:\n" + user_message.strip() + "\n\n"
        "Instructions to assistant: Use the retrieved context above if applicable. "
        "If context is irrelevant or missing, answer from general medical knowledge but be cautious. "
        "Give a concise 1-3 sentence explanation and a very short comma-separated list of practical precautions. "
        "If you don't know, say 'I don't know' and request more details. "
        "Do not use apologetic stock lines and avoid repeating the exact classifier text verbatim."
    )
    messages.append({"role": "user", "content": user_block})
    return messages

def _call_groq(messages, cfg: Dict[str,Any], max_retries: int = 2) -> str:
    """Call Groq/OpenAI-compatible chat completions endpoint using HTTP POST."""
    headers = {"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"}
    payload = {
        "model": cfg["model_id"],
        "messages": messages,
        "temperature": cfg.get("temperature", 0.0),
        "max_tokens": cfg.get("max_tokens", 512),
        "top_p": 1.0,
        "n": 1,
    }
    for attempt in range(1, max_retries + 1):
        try:
            LOG.info("LLM: sending request (attempt %d)...", attempt)
            resp = requests.post(GROQ_CHAT_URL, headers=headers, json=payload, timeout=cfg.get("timeout", 20))
            if resp.status_code != 200:
                LOG.warning("LLM request failed (attempt %d): %s - %s", attempt, resp.status_code, resp.text[:400])
                if 400 <= resp.status_code < 500 and resp.status_code != 429:
                    break
                time.sleep(1.0 * attempt)
                continue
            data = resp.json()
            choices = data.get("choices") or []
            if choices:
                msg = choices[0].get("message") or choices[0].get("delta") or {}
                content = ""
                if isinstance(msg, dict):
                    content = msg.get("content") or ""
                if not content:
                    content = choices[0].get("text") or ""
                return (content or "").strip()
            LOG.warning("LLM returned unexpected payload; retrying (attempt %d).", attempt)
        except requests.RequestException as e:
            LOG.warning("LLM request exception (attempt %d): %s", attempt, str(e))
            time.sleep(1.0 * attempt)
            continue
    raise RuntimeError("LLM request failed after retries.")

# ----------------------
# Precaution extraction + sanitization helpers
# ----------------------
_WORD_RE = re.compile(r'\w+')

def _split_candidates(text: str) -> List[str]:
    if not text:
        return []
    t = text.replace("|", ",").replace(";", ",")
    parts = [p.strip() for p in re.split(r',|\n|•|-|\u2022', t) if p.strip()]
    return parts

def _normalize_prec_item(it: str) -> str:
    it = it.strip()
    it = re.sub(r'^[\-\–\—\•\*]+\s*', '', it)
    it = re.sub(r'^(and|or|then|also|but)\b[:,;\s]*', '', it, flags=re.I)
    it = re.sub(r'\s+', ' ', it)
    it = it.rstrip('.')
    return it.strip()

def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        k = x.lower()
        if k and k not in seen:
            seen.add(k)
            out.append(x)
    return out

def _token_overlap_ratio(a: str, b: str) -> float:
    a_tokens = set(_WORD_RE.findall(a.lower()))
    b_tokens = set(_WORD_RE.findall(b.lower()))
    if not a_tokens or not b_tokens:
        return 0.0
    inter = a_tokens.intersection(b_tokens)
    union = a_tokens.union(b_tokens)
    # Return overlap relative to shorter token set
    shorter = min(len(a_tokens), len(b_tokens))
    if shorter == 0:
        return 0.0
    return len(inter) / shorter

def _parse_precautions_from_reply(reply: str, fallback_precs: List[str]) -> List[str]:
    """Extract short comma-separated precaution items from a model reply or fallback precs."""
    if not reply and not fallback_precs:
        return []

    text = (reply or "").replace("|", ",")
    parts = []

    lowered = text.lower()
    if "precaution" in lowered:
        for line in [l.strip() for l in text.splitlines() if l.strip()]:
            if "precaution" in line.lower():
                cleaned = line
                for label in ("precautions:", "precaution:", "precautions -", "precaution -"):
                    cleaned = cleaned.replace(label, "")
                    cleaned = cleaned.replace(label.capitalize(), "")
                for candidate in [p.strip() for p in cleaned.split(",") if p.strip()]:
                    parts.append(candidate)
                break

    if not parts:
        tokens = [s.strip() for s in text.replace("\n", ", ").split(",") if s.strip()]
        for t in tokens:
            if 0 < len(t.split()) <= 12:
                parts.append(t)

    if not parts and fallback_precs:
        if isinstance(fallback_precs, (list, tuple)):
            for p in fallback_precs:
                p_clean = str(p).replace("|", ",")
                for candidate in [c.strip() for c in p_clean.split(",") if c.strip()]:
                    parts.append(candidate)
        else:
            for candidate in [c.strip() for c in str(fallback_precs).replace("|", ",").split(",") if c.strip()]:
                parts.append(candidate)

    # normalize and dedupe
    normed = [_normalize_prec_item(p) for p in parts if p and p.strip()]
    normed = [p for p in normed if len(p) > 1]
    normed = _dedupe_keep_order(normed)

    # final cleanup: remove items that are basically contained in each other or are fragments
    cleaned = []
    for p in normed:
        p_lower = p.lower()
        # skip single char garbage
        if len(re.sub(r'\W+', '', p_lower)) <= 1:
            continue
        # skip items that are just identical to previous cleaned
        if p_lower in (c.lower() for c in cleaned):
            continue
        cleaned.append(p)

    return cleaned[:12]

def _remove_precautions_in_explanation(prec_list: List[str], explanation: str) -> List[str]:
    """Remove any precaution items that are strongly repeated in the explanation text."""
    if not prec_list:
        return []
    if not explanation:
        return prec_list
    out = []
    for p in prec_list:
        if not p:
            continue
        ratio = _token_overlap_ratio(p, explanation)
        # if >60% of shorter tokens overlap, consider it redundant and skip
        if ratio >= 0.6:
            continue
        out.append(p)
    return out

def _sanitize_reply(raw_reply: str, fallback_prec) -> (str, List[str]):
    """
    Return (reply_text, prec_list).
    reply_text has explanation as first paragraph, blank line, then "Precautions: a, b, c."
    """
    if not raw_reply:
        return ("", [])

    text = raw_reply.replace("|", ",").strip()
    paras = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

    # If first paragraph looks like a short greeting/thanks, return cleaned greeting only
    if paras:
        first = paras[0].strip()
        if re.fullmatch(r'(?i)(hi|hello|hey|good (day|morning|afternoon|evening)|thanks|thank you|bye|goodbye|take care)([.!]?)', first):
            lines = [l for l in raw_reply.splitlines() if not re.search(r'precaution', l, flags=re.I)]
            cleaned = "\n".join([ln.strip() for ln in lines if ln.strip()])
            return (cleaned, [])

    # Explanation: first paragraph truncated to up to 2 sentences
    explanation = ""
    if paras:
        cand = paras[0]
        sents = [s.strip() for s in re.split(r'(?<=[\.\?\!])\s+', cand) if s.strip()]
        if sents:
            explanation = " ".join(sents[:2]).strip()
        else:
            explanation = cand
    else:
        explanation = text if len(text) < 400 else text[:400] + "..."

    if explanation and not explanation.endswith('.'):
        explanation = explanation + '.'

    # collect precaution candidates
    prec_candidates = []
    for line in raw_reply.splitlines():
        if re.search(r'precaution', line, flags=re.I):
            after = re.sub(r'(?i)precautions?:', '', line).strip()
            prec_candidates.extend(_split_candidates(after))
    if not prec_candidates and len(paras) > 1:
        for para in paras[1:]:
            if re.search(r'\b(rest|stay|hydrate|avoid|keep|monitor|seek|wash|bathe|apply|use|avoid)\b', para, flags=re.I):
                prec_candidates.extend(_split_candidates(para))
    if not prec_candidates:
        if isinstance(fallback_prec, (list, tuple)):
            for p in fallback_prec:
                prec_candidates.extend(_split_candidates(str(p)))
        elif isinstance(fallback_prec, str) and fallback_prec.strip():
            prec_candidates.extend(_split_candidates(fallback_prec))

    prec_list = _parse_precautions_from_reply(", ".join(prec_candidates), [])
    # remove items that repeat explanation strongly
    prec_list = _remove_precautions_in_explanation(prec_list, explanation)
    # final dedupe
    prec_list = _dedupe_keep_order(prec_list)

    # Compose final reply. Precautions must be on new paragraph.
    parts = []
    parts.append(explanation)
    if prec_list:
        parts.append("Precautions: " + ", ".join(prec_list) + ".")
    reply = "\n\n".join(parts).strip()

    # final cleanup
    reply = re.sub(r',\s*', ', ', reply)
    reply = re.sub(r'\s+\.', '.', reply)
    reply = re.sub(r'\s{2,}', ' ', reply)
    return reply, prec_list

# ----------------------
# Main enrich_result
# ----------------------
def enrich_result(result: Dict[str,Any], user_message: str, conversation: Optional[List[Dict[str,str]]] = None,
                  max_retries: int = 2, prefer_default_model: bool = True) -> Dict[str,Any]:
    """
    Main function called by the app.
    - If GROQ_API_KEY or model id missing, returns offline rewrite.
    - Otherwise calls LLM and returns cleaned reply (precautions comma-separated)
    - conversation is optional; when you want stateless, pass conversation=None
    """
    cfg = _read_env()
    if not cfg["api_key"] or not cfg["model_id"]:
        LOG.info("LLM: API key or model id missing; performing offline rewrite.")
        return _offline_rewrite(result, user_message)

    messages = _build_messages(result, user_message, conversation=conversation)
    try:
        content = _call_groq(messages, cfg, max_retries=max_retries)
        if not content:
            LOG.warning("LLM returned empty; falling back to offline rewrite.")
            return _offline_rewrite(result, user_message)

        # sanitize raw content: remove stray pipes and condense whitespace
        content = content.replace("|", ",")
        content = "\n".join([line.strip() for line in content.splitlines() if line.strip()])

        # extract comma-separated precaution items
        fallback_prec = result.get("precaution", [])
        prec_list = _parse_precautions_from_reply(content, fallback_prec)

        # build tidy reply
        # first paragraph / sentence as short explanation
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        short_expl = ""
        if paragraphs:
            first_para = paragraphs[0]
            sentences = [s.strip() for s in re.split(r'(?<=[\.\?\!])\s+', first_para) if s.strip()]
            if sentences:
                short_expl = " ".join(sentences[:2]).strip()
                if not short_expl.endswith('.'):
                    short_expl += '.'
        if not short_expl:
            short_expl = content if len(content) < 300 else content[:300] + "..."

        # sanitize final reply (ensures Precautions on new line, no redundancy)
        try:
            reply_text, parsed_prec = _sanitize_reply(content, prec_list or fallback_prec)
        except Exception as e:
            LOG.exception("Sanitizer failed: %s", e)
            # fallback: minimal construction
            parsed_prec = _dedupe_keep_order(_parse_precautions_from_reply(content, fallback_prec))
            reply_text = short_expl
            if parsed_prec:
                reply_text = reply_text + "\n\n" + "Precautions: " + ", ".join(parsed_prec) + "."

        description = result.get("description", "")
        if not description:
            # take first sentence as description
            description = short_expl.split(".")[0].strip()

        out = {"reply": reply_text, "description": description, "precaution": parsed_prec, "llm_used": True}
        LOG.debug("LLM enrich_result produced llm_used=True")
        return out

    except Exception as e:
        LOG.exception("LLM call failed: %s", e)
        return _offline_rewrite(result, user_message)
