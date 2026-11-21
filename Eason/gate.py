"""Confidence & risk escalation gate.

Decision levels
    ESCALATE_ONCALL  -> high-priority incident / outage / security: page the on-call engineer
    HUMAN_REVIEW     -> low confidence, no precedent, unsupported or risky request
    AUTO_DRAFT_READY -> safe to send the grounded AI draft after a quick glance

Rules (all explicit, every fired rule is reported as a reason):
  R1 high-priority incident, two tiers tuned on val (tune_gate.py) on P(high) * P(Incident):
       >= page threshold (alarm precision >= 60%)  -> ESCALATE_ONCALL
       >= review threshold (HPI recall >= 90%)      -> HUMAN_REVIEW "possible high-priority incident"
  R2 keyword safety net (EN+DE): outage, security incident, healthcare system failure
  R3 classifier unsure: type or queue confidence below floor
  R4 no historical precedent: best retrieval similarity below floor
  R5 request outside what precedents support: refunds/compensation amounts, legal, contract
     guarantees, credentials, plus anything the generator flags as unsupported
  R6 prompt-injection / manipulation attempt in the ticket
"""
import json
import re
from functools import lru_cache

import config as C

SAFETY_NET = {
    "outage": re.compile(
        r"\boutages?\b|ausfall|ausgefallen|\boffline\b|unreachable|not reachable|nicht (?:mehr )?erreichbar|"
        r"\b(?:is|are|went|was|been|goes) down\b|down for (?:all|every)|service unavailable|"
        r"komplett down|totalausfall|störung|stoerung", re.I),
    "security incident": re.compile(
        r"ransomware|malware|cyber[- ]?attac?k|cyberangriff|security (?:incident|breach)|sicherheitsvorfall|"
        r"data (?:breach|leak)|datenleck|datenverletzung|\bhacked\b|gehackt|phishing|unauthori[sz]ed access", re.I),
    "healthcare system failure": re.compile(
        r"(?=.*(?:hospital|patient|\bemr\b|\behr\b|telemedicine|telehealth|krankenhaus|klinik|patienten))"
        r"(?=.*(?:\bdown\b|outage|offline|can(?:no|')t connect|cannot access|not working|fail|"
        r"ausfall|ausgefallen|funktioniert nicht|nicht erreichbar|keine verbindung))", re.I | re.S),
}

RISKY_REQUEST = {
    "refund / compensation demand": re.compile(
        r"refund|reimburse|compensation|money back|chargeback|credit (?:my|our) account|sla credit|"
        r"erstatt|rückerstattung|rueckerstattung|entschädigung|entschaedigung|gutschrift|geld zurück|"
        r"zurückbekomm|zurueckbekomm|zurückzahl|zurueckzahl|zurückerhalten", re.I),
    "legal / contractual guarantee": re.compile(
        r"lawsuit|legal action|lawyer|attorney|sue\b|guarantee|warranty claim|contract(?:ual)? penalt|"
        r"anwalt|klage|rechtliche schritte|garantie|vertragsstrafe|schadensersatz", re.I),
    "pricing / discount commitment": re.compile(
        r"discount|price match|special price|quote me|how much (?:does|will)|rabatt|preisnachlass|angebot über", re.I),
    "credential / account takeover": re.compile(
        r"(?:send|give|tell|share).{0,30}(?:password|passwort|credentials|2fa code|otp)|"
        r"(?:disable|bypass|turn off).{0,20}(?:2fa|mfa|two[- ]factor|authentication)", re.I),
}

INJECTION = re.compile(
    r"ignore (?:all |any |the )?(?:previous|prior|above) (?:instructions|prompts?)|disregard (?:your|the) (?:rules|instructions)|"
    r"system prompt|you are now|act as (?:an? )?(?:admin|developer|dan)|jailbreak|"
    r"ignoriere (?:alle )?(?:vorherigen|bisherigen) anweisungen", re.I)


def risky_requests(text: str) -> list:
    return [name for name, pat in RISKY_REQUEST.items() if pat.search(text)]


@lru_cache(maxsize=2)
def hpi_thresholds(regime: str = "novel"):
    """(review, page) thresholds tuned on val by tune_gate.py, per regime ('novel' | 'near_dup')."""
    try:
        h = json.loads(C.GATE_PATH.read_text())["hpi"][regime]
        return h["review"]["threshold"], h["page"]["threshold"]
    except Exception:
        return 0.125, 0.37


def hpi_score(cls: dict) -> float:
    return cls["priority"]["probabilities"].get("high", 0) * cls["type"]["probabilities"].get("Incident", 0)


def decide(subject: str, body: str, cls: dict, precedents: list, generation: dict | None = None) -> dict:
    text = f"{subject or ''}\n{body or ''}"
    oncall, review = [], []

    s = hpi_score(cls)
    regime = "near_dup" if str(cls["type"].get("source", "")).startswith("precedent") else "novel"
    th_review, th_page = hpi_thresholds(regime)
    if s >= th_page:
        oncall.append(f"R1 high-priority incident: P(high)·P(Incident) = {s:.2f} ≥ page threshold {th_page:.2f}")
    elif s >= th_review:
        review.append(f"R1 possible high-priority incident: P(high)·P(Incident) = {s:.2f} ≥ review threshold {th_review:.2f}")
    for name, pat in SAFETY_NET.items():
        if pat.search(text):
            oncall.append(f"R2 safety net: {name} language detected")

    if cls["type"]["confidence"] < C.MIN_TYPE_CONF:
        review.append(f"R3 low type confidence ({cls['type']['confidence']:.2f} < {C.MIN_TYPE_CONF})")
    if cls["queue"]["confidence"] < C.MIN_QUEUE_CONF:
        review.append(f"R3 low queue confidence ({cls['queue']['confidence']:.2f} < {C.MIN_QUEUE_CONF})")

    n_words = len(text.split())
    if n_words < C.MIN_WORDS:
        review.append(f"R4 too little information to auto-answer ({n_words} words)")
    top_sim = precedents[0]["similarity"] if precedents else 0.0
    if top_sim < C.MIN_RETRIEVAL_SIM:
        review.append(f"R4 no close historical precedent (best similarity {top_sim:.2f} < {C.MIN_RETRIEVAL_SIM})")

    for name, pat in RISKY_REQUEST.items():
        if pat.search(text):
            review.append(f"R5 request needs human approval: {name}")
    if generation:
        for item in generation.get("unsupported_requests") or []:
            review.append(f"R5 not supported by precedents: {item}")
        if generation.get("evidence_sufficient") is False:
            review.append("R5 no precedent resolution is close enough to ground a reply")
        for issue in generation.get("validation_issues") or []:
            review.append(f"R5 draft failed grounding check: {issue}")

    if INJECTION.search(text):
        review.append("R6 possible prompt-injection / manipulation attempt")

    status = "ESCALATE_ONCALL" if oncall else "HUMAN_REVIEW" if review else "AUTO_DRAFT_READY"
    return {"status": status, "reasons": oncall + review, "hpi_score": round(s, 4),
            "hpi_thresholds": {"review": th_review, "page": th_page, "regime": regime}, "top_similarity": top_sim}
