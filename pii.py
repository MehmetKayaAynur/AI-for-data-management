"""In-text PII detection & masking for maintenance notes (the *Securing* pillar).

Finds personally identifiable information inside the free-text notes themselves —
person names (NER), phone numbers and e-mails (regex) — and masks them. Unlike the
record-level `mask_person` in pipeline.py (which needs a dedicated person column,
absent here), this works on the note text, where real PII actually appears in the
MaintNet facility data (e.g. "ATTN MIKE BEANE", "CONTACT NUMBER IS 443-573-2802").

NER uses spaCy (en_core_web_sm) when available; otherwise it falls back to
trigger-phrase heuristics, so the module never hard-fails.
"""
import re

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(
    r"\b(?:\(\d{3}\)\s*\d{3}[-.\s]\d{4}"   # (443) 573-2802
    r"|\d{3}[-.\s]\d{3}[-.\s]\d{4}"        # 443-573-2802
    r"|\d{1,2}-\d{4})\b")                  # 6-3315 (internal extension)
# A person name is only accepted in a trusting CONTEXT — names in maintenance notes
# appear after a trigger phrase or an honorific. This gates out technical jargon
# (e.g. "INDUCTION TUBE CLAMP"), which a raw NER pass over title-cased text would
# otherwise mislabel as a person — critical because the aviation set is de-identified.
NAME_TRIGGER_RE = re.compile(
    r"\b(?:ATTN|ATTENTION|CONTACT PERSON|CONTACT NAME|REQUESTED BY|REQUESTOR|ASK FOR)"
    r"\.?:?\s+([A-Z][A-Z']+(?:\s+[A-Z][A-Z']+){0,2})")
# honorific + name (e.g. "SGT. THOMAS", "MR. SMITH")
HONORIFIC_RE = re.compile(
    r"\b(?:SGT|MR|MRS|MS|DR|LT|CAPT|MISS|OFFICER|SIR|MADAM)\.?\s+"
    r"([A-Z][A-Za-z']+(?:\s+[A-Z][A-Za-z']+)?)")

_TRIGGERS = {"attn", "attention", "mr", "mrs", "ms", "miss", "sgt", "dr", "lt",
             "capt", "officer", "contact", "person", "the", "of", "office"}
_STOP = {"urgent", "please", "room", "building", "office"}

_NLP = "uninit"


def _nlp():
    global _NLP
    if _NLP == "uninit":
        try:
            import spacy
            _NLP = spacy.load("en_core_web_sm", disable=["lemmatizer", "tagger", "attribute_ruler"])
        except Exception:
            _NLP = None
    return _NLP


def _clean_name(raw):
    toks = [t for t in re.findall(r"[A-Za-z'.]+", raw) if t]
    while toks and toks[0].lower().strip(".") in _TRIGGERS:
        toks = toks[1:]
    name = " ".join(toks).strip()
    if len(name) < 3 or name.lower() in _STOP:
        return None
    return name


def _is_person(name):
    """Confirm a candidate span is a real name. spaCy validates when available;
    otherwise a 2+ token candidate is accepted."""
    nlp = _nlp()
    if nlp is None:
        return len(name.split()) >= 2
    return any(e.label_ == "PERSON" for e in nlp(name.title()).ents)


def detect(text):
    """Return a list of (type, value) PII findings for one note."""
    text = str(text or "")
    found = []
    for m in EMAIL_RE.finditer(text):
        found.append(("EMAIL", m.group()))
    for m in PHONE_RE.finditer(text):
        found.append(("PHONE", m.group()))
    # names only in a trusting context (trigger phrase), spaCy-validated
    for m in NAME_TRIGGER_RE.finditer(text):
        n = _clean_name(m.group(1))
        if n and _is_person(n):
            found.append(("PERSON", n))
    # honorific + name (high confidence; accept even a single token like "SGT. THOMAS")
    for m in HONORIFIC_RE.finditer(text):
        n = _clean_name(m.group(1))
        if n:
            found.append(("PERSON", n))
    # dedupe, keep order
    seen, out = set(), []
    for typ, val in found:
        k = (typ, val.lower())
        if k not in seen:
            seen.add(k); out.append((typ, val))
    return out


def mask(text):
    """Return (masked_text, findings) with each PII value replaced by [TYPE]."""
    text = str(text or "")
    finds = detect(text)
    masked = text
    for typ, val in finds:
        masked = re.sub(re.escape(val), f"[{typ}]", masked, flags=re.IGNORECASE)
    return masked, finds


def detect_batch(texts):
    """Detection over many notes. Names are gated by trigger context, so spaCy only
    validates short candidate spans (no costly full-text NER pass)."""
    return [detect(t) for t in texts]


if __name__ == "__main__":
    for s in ["ATTN MIKE BEANE. SPRAY FOR BUGS",
              "CONTACT NUMBER IS 443-573-2802",
              "CONTACT PERSON LORETTA BROWN 6-3315",
              "#2 INTAKE GASKET LEAKING"]:
        print(mask(s))
