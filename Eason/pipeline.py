"""End-to-end SentinelDesk pipeline: ticket in -> triage, gate, precedents, grounded draft."""
import time

from classifier import classify
from gate import decide
from generator import generate
from lang import detect_language
from precedent_labels import apply as apply_precedent_labels
from retriever import embed, retrieve


def triage(subject: str, body: str, burst=None, generate_reply: bool = True) -> dict:
    subject, body = (subject or "").strip(), (body or "").strip()
    if not body and not subject:
        raise ValueError("empty ticket")
    t = {}
    t0 = time.perf_counter()

    lang = detect_language(subject, body)
    t["language_ms"] = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    cls = classify(subject, body)
    t["classify_ms"] = (time.perf_counter() - t1) * 1000

    t1 = time.perf_counter()
    text = f"{subject}\n{body}".strip()
    precedents = retrieve(text)
    t["retrieve_ms"] = (time.perf_counter() - t1) * 1000
    cls = apply_precedent_labels(cls, precedents[0] if precedents else None)

    gen = None
    if generate_reply:
        t1 = time.perf_counter()
        gen = generate(subject, body, precedents, lang["lang"])
        t["generate_ms"] = (time.perf_counter() - t1) * 1000

    decision = decide(subject, body, cls, precedents, gen)

    outage = None
    if burst is not None:
        summary = {"subject": subject or body[:60], "type": cls["type"]["label"], "queue": cls["queue"]["label"]}
        outage = burst.add(embed([text])[0], summary)
        if outage:
            decision["reasons"].insert(0, f"Burst: {outage['size']} similar tickets in the last 30 min")
            decision["status"] = "ESCALATE_ONCALL"

    t["total_ms"] = (time.perf_counter() - t0) * 1000
    return {"language": lang, "classification": cls, "decision": decision, "precedents": precedents,
            "draft": gen, "outage": outage, "timing_ms": {k: round(v, 1) for k, v in t.items()}}
