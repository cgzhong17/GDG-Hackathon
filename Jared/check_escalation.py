"""
SentinelDesk - High-Priority Incident escalation gate = model rule + keyword safety net.

1) Model rule (threshold chosen on the validation set):
       escalate if P(high) >= 0.30 AND P(Incident) >= 0.30
2) Keyword safety net (English + German), escalates even if the model is unsure:
       - outage words        (outage, offline, is down, Ausfall, nicht erreichbar ...)
       - security incidents  (ransomware, data breach, Sicherheitsvorfall ...)
       - healthcare + failure (hospital/patient/EMR ... AND down/offline/not working ...)

Run:  python check_escalation.py
"""
import re

try:
    from .predict import predict_ticket      # imported as Jared.check_escalation
except ImportError:
    from predict import predict_ticket       # run as a script

P_HIGH_MIN = 0.30
P_INCIDENT_MIN = 0.30

SAFETY_KEYWORDS = {
    "outage": re.compile(
        r"\boutages?\b|ausfall|ausgefallen|\boffline\b|unreachable|not reachable|"
        r"nicht (mehr )?erreichbar|\b(?:is|are|went|was) down\b|down for (?:all|every)|"
        r"service unavailable", re.I),
    "security incident": re.compile(
        r"ransomware|cyber[- ]?attack|cyberangriff|security incident|sicherheitsvorfall|"
        r"data breach|datenleck|datenverletzung|\bhacked\b|gehackt", re.I),
    "healthcare service failure": re.compile(
        r"(?=.*(?:hospital|patient|\bemr\b|\behr\b|telemedicine|krankenhaus|klinik))"
        r"(?=.*(?:\bdown\b|outage|offline|can(?:no|')t connect|cannot access|not working|failed|"
        r"failure|ausfall|ausgefallen|funktioniert nicht|nicht erreichbar|keine verbindung))",
        re.I | re.S),
}


def high_priority_incident_gate(subject: str, body: str, result: dict) -> tuple[bool, list[str]]:
    reasons = []
    p_high = result["priority"]["probabilities"]["high"]
    p_incident = result["type"]["probabilities"]["Incident"]
    if p_high >= P_HIGH_MIN and p_incident >= P_INCIDENT_MIN:
        reasons.append(f"model: P(high)={p_high:.2f}, P(Incident)={p_incident:.2f}")

    text = f"{subject or ''} {body or ''}"
    for name, pattern in SAFETY_KEYWORDS.items():
        if pattern.search(text):
            reasons.append(f"keyword safety net: {name}")
    return bool(reasons), reasons


if __name__ == "__main__":
    examples = [
        ("VPN down", "Since this morning nobody in the hospital can connect to the VPN. Telemedicine and EMR are offline."),
        ("Server down - production outage", "Our main production server is unreachable. All customers are affected."),
        ("VPN ausgefallen", "Seit heute Morgen ist das VPN im ganzen Krankenhaus ausgefallen. Die Patientenakten sind nicht erreichbar."),
        ("Possible data breach", "We noticed unknown logins and think our customer database was hacked."),
        ("Rechnung doppelt bezahlt", "Ich habe die Rechnung fuer November zweimal bezahlt. Bitte erstatten Sie den Betrag."),
        ("", "Could you send me information about pricing for the enterprise plan?"),
    ]
    for subject, body in examples:
        r = predict_ticket(subject, body)
        escalate, reasons = high_priority_incident_gate(subject, body, r)
        status = "ESCALATE to On-Call Engineer" if escalate else "Auto-Draft Ready"
        print(f"\nSUBJECT : {subject or '(empty)'}")
        print(f"PREDICT : type={r['type']['label']}  queue={r['queue']['label']}  priority={r['priority']['label']}")
        print(f"DECISION: {status}")
        for reason in reasons:
            print(f"          - {reason}")
