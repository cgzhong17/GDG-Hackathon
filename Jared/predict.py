"""
SentinelDesk - prediction function for the Dashboard / other teammates.

Usage:
    from predict import predict_ticket
    result = predict_ticket("VPN not working", "Since this morning I cannot connect ...")

"confidence" is calibrated (temperature scaling on data/val.csv) once
evaluate.py has been run; before that it is the raw model probability and
result["calibrated"] is False.
"""
import time
from functools import lru_cache
from pathlib import Path

import joblib
from scipy.special import softmax

MODEL_PATH = Path(__file__).resolve().parent / "models" / "sentinel_classifier.joblib"


@lru_cache(maxsize=1)
def _load():
    # Loaded once, then kept in memory (fast for Streamlit)
    return joblib.load(MODEL_PATH)


def predict_ticket(subject: str, body: str) -> dict:
    bundle = _load()
    t0 = time.perf_counter()

    text = f"{subject or ''}\n{body or ''}".strip()
    X = bundle["vectorizer"].transform([text])

    temps = bundle.get("temperature", {})   # added by evaluate.py
    result = {}
    for target in bundle["targets"]:
        clf = bundle["models"][target]
        probs = softmax(clf.decision_function(X)[0] / temps.get(target, 1.0))
        ranked = sorted(zip(clf.classes_, probs), key=lambda p: p[1], reverse=True)
        result[target] = {
            "label": str(ranked[0][0]),
            "confidence": round(float(ranked[0][1]), 4),
            "probabilities": {str(c): round(float(p), 4) for c, p in ranked},
        }

    result["calibrated"] = bool(temps)
    result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return result


if __name__ == "__main__":
    import json

    examples = [
        ("Server down - production outage",
         "Our main production server has been unreachable since 9am. All customers are affected, please help urgently."),
        ("Rechnung doppelt bezahlt",
         "Guten Tag, ich habe die Rechnung fuer November versehentlich zweimal bezahlt. Koennen Sie mir den Betrag erstatten?"),
        ("", "Could you send me information about pricing for the enterprise plan?"),
    ]
    for subject, body in examples:
        print(f"\nSUBJECT: {subject or '(empty)'}\nBODY:    {body[:80]}...")
        print(json.dumps(predict_ticket(subject, body), indent=2, ensure_ascii=False))
