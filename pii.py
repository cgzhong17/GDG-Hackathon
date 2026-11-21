"""PII guardrail: mask emails, phones, IPs, IBANs/card/account numbers before text leaves the machine."""
import re

_PATTERNS = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    ("IP", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){3,7}(?:[ ]?[A-Z0-9]{1,3})?\b")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("PHONE", re.compile(r"(?<!\w)(?:\+|00)?\d[\d ()/.-]{7,}\d(?!\w)")),
    ("ACCOUNT", re.compile(r"(?i)\b(?:account|acct|konto|kundennummer|customer id|order)\s*(?:no\.?|nr\.?|number|#|:)?\s*[A-Z0-9-]{5,}\b")),
]


def mask_pii(text: str):
    """Return (masked_text, {kind: count})."""
    counts = {}
    for kind, pat in _PATTERNS:
        text, n = pat.subn(f"[{kind}]", text)
        if n:
            counts[kind] = n
    return text, counts
