# src/llm_utils.py
import re
from typing import List, Dict, Any

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    return parts

def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for it in items:
        key = it.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(it.strip())
    return out

def _extract_precautions_from_text(reply: str) -> (str, List[str]): # type: ignore
    """
    If reply contains a 'precaution' or 'precautions' section, extract it.
    Returns (main_text_without_prec_section, list_of_precautions_extracted)
    """
    if not reply:
        return "", []
    parts = re.split(r'precaution[s]?:', reply, flags=re.I)
    if len(parts) == 1:
        return reply, []
    main = parts[0].strip()
    prec_raw = parts[1].strip()
    # remove leading punctuation from extracted part
    prec_raw = re.sub(r'^[\s:\-\u2022]+', '', prec_raw).strip()
    # split by newlines, bullets, pipes, semicolons
    items = re.split(r'[\n\r\u2022\|\;]+', prec_raw)
    items = [it.strip() for it in items if it.strip()]
    # further split items that look like " - item" or contain ' | '
    final = []
    for it in items:
        # if there are commas but item is long and comma likely separates items, split
        if ',' in it and len(it) < 80 and it.count(',') >= 1:
            # short comma-list -> split
            parts2 = [p.strip() for p in it.split(',') if p.strip()]
            final.extend(parts2)
        else:
            # try splitting on ' • ' or ' - '
            sub = re.split(r'\s*-\s*|\s*•\s*', it)
            for s in sub:
                if s.strip():
                    final.append(s.strip())
    # clean and dedupe
    final = [re.sub(r'^[\-\u2022\.\s]+', '', s).strip() for s in final]
    final = _dedupe_preserve_order(final)
    return main, final

def _prec_from_field(field: Any) -> List[str]:
    """
    Normalize 'precaution' field if present in result (it may be list or string).
    """
    if field is None:
        return []
    if isinstance(field, list):
        items = []
        for it in field:
            if not isinstance(it, str):
                continue
            # break items that look like comma lists
            if ',' in it and len(it) < 120 and it.count(',') >= 1:
                items.extend([p.strip() for p in it.split(',') if p.strip()])
            else:
                items.append(it.strip())
        return _dedupe_preserve_order([i for i in items if i])
    if isinstance(field, str):
        # similar splitting to above
        parts = re.split(r'[\n\r\u2022\|\;]+', field)
        items = []
        for p in parts:
            if not p.strip():
                continue
            if ',' in p and len(p) < 120 and p.count(',') >= 1:
                items.extend([s.strip() for s in p.split(',') if s.strip()])
            else:
                items.append(p.strip())
        return _dedupe_preserve_order(items)
    return []

def sanitize_llm_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cleans and normalizes the 'reply' and 'precaution' fields inside result dict.
    - Removes duplicate sentences
    - Removes sentences that duplicate a precaution item
    - Normalizes precautions to a deduped list
    - Converts final precautions into a comma-separated string in reply text
    Returns a new dict (does not modify input).
    """
    out = dict(result)  # shallow copy
    reply = str(result.get("reply", "") or "").strip()
    # 1) extract any "Precautions:" block in reply
    main_text, prec_from_reply = _extract_precautions_from_text(reply)

    # 2) split main_text into sentences and dedupe while preserving order
    sentences = _split_sentences(main_text)
    seen = set()
    dedup_sentences = []
    for s in sentences:
        key = re.sub(r'\s+', ' ', s).strip().lower()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        dedup_sentences.append(s.strip())

    # 3) get precautions from result.precaution field too
    prec_field_items = _prec_from_field(result.get("precaution"))
    # combine with extracted ones (reply has higher priority -> place first)
    combined_prec = []
    for p in prec_from_reply + prec_field_items:
        if p and p.strip():
            combined_prec.append(p.strip())
    combined_prec = _dedupe_preserve_order(combined_prec)

    # 4) remove from sentences any sentence that is almost identical to a precaution item
    #    (avoid duplication where body repeats precaution)
    filtered_sentences = []
    prec_keys = {p.lower() for p in combined_prec}
    for s in dedup_sentences:
        s_key = re.sub(r'[^a-z0-9\s]', '', s.lower()).strip()
        # if any precaution token is substring of sentence key (or vice-versa) we drop from sentences
        should_drop = False
        for p in prec_keys:
            pk = re.sub(r'[^a-z0-9\s]', '', p).strip()
            if not pk:
                continue
            if pk in s_key or s_key in pk:
                should_drop = True
                break
        if not should_drop:
            filtered_sentences.append(s)

    # 5) Rebuild reply — ensure concise human-friendly formatting
    # Join sentences with a single space. Then append "Precautions: a, b, c."
    final_reply_parts = []
    if filtered_sentences:
        # ensure first sentence is title-like if original had it (e.g., "Fever:"), we keep as-is
        final_reply_parts.append(" ".join(filtered_sentences))
    # build precautions phrase
    if combined_prec:
        # make sure items are short and comma-separated
        prec_joined = ", ".join(combined_prec)
        final_reply_parts.append(f"Precautions: {prec_joined}")
    final_reply = "\n\n".join(final_reply_parts).strip()

    # fallback to original reply if everything was stripped
    if not final_reply:
        final_reply = reply

    out["reply"] = final_reply
    # also store normalized precautions list
    out["precaution"] = combined_prec
    return out
