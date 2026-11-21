"""Reply drafting with the local extractive composer, followed by grounding validation.

Checks run on every draft:
  - every cited id must be one of the retrieved precedents, and a draft that claims
    evidence must cite at least one
  - no money / percent / duration figures that appear neither in the evidence nor the ticket
  - reply language must match the ticket language
PII (emails, phones, IPs, account numbers) is detected and reported for the operator.
"""
import re

import config as C
from composer import compose
from gate import INJECTION, risky_requests
from lang import detect_language
from pii import mask_pii

TID = re.compile(r"TKT-\d{5}")
FIGURE = re.compile(r"(?:[$€£]\s?\d[\d.,]*|\d[\d.,]*\s?(?:%|€|\$|eur|usd|euro|dollars?|prozent|percent)|"
                    r"\b\d+\s?(?:hours?|days?|weeks?|minutes?|stunden|tagen?|wochen|minuten)\b)", re.I)


def validate(draft: dict, precedents: list, lang: str, ticket_text: str) -> list:
    issues = []
    allowed = {p["ticket_id"] for p in precedents}
    in_text = set(TID.findall(draft["reply"]))
    bad = (in_text | set(draft.get("cited_ticket_ids", []))) - allowed
    if bad:
        issues.append(f"cites ticket ids that were not retrieved: {sorted(bad)}")
    if draft.get("evidence_sufficient") and not in_text:
        issues.append("claims evidence but cites no ticket id")
    known = (" ".join(p["answer"] + " " + p["body"] for p in precedents) + " " + ticket_text).lower()
    unsupported = [f for f in FIGURE.findall(draft["reply"]) if f.lower().strip() not in known]
    if unsupported:
        issues.append(f"figures not found in evidence: {unsupported}")
    got = detect_language("", TID.sub("", draft["reply"]))["lang"]
    if got != lang:
        issues.append(f"reply language {got} != ticket language {lang}")
    return issues


def generate(subject: str, body: str, precedents: list, lang: str) -> dict:
    text = f"{subject}\n{body}"
    _, c1 = mask_pii(subject or "")
    _, c2 = mask_pii(body or "")
    pii = {k: c1.get(k, 0) + c2.get(k, 0) for k in set(c1) | set(c2)}
    risky = risky_requests(text)

    if INJECTION.search(text):     # manipulation attempt: safe holding reply only
        draft = compose(subject, body, [], lang, risky=risky)
        draft.update({"validation_issues": [], "model": "safe holding reply (injection)", "pii_detected": pii})
        return draft

    draft = compose(subject, body, precedents, lang, risky=risky, min_precedent_sim=C.MIN_RETRIEVAL_SIM - 0.03)
    draft["validation_issues"] = validate(draft, precedents, lang, text)
    draft.update({"model": "extractive (local)", "pii_detected": pii})
    return draft
