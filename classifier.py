"""Inference wrapper for the calibrated triage classifier."""
from functools import lru_cache

import joblib
import numpy as np
from scipy.special import softmax

import config as C


@lru_cache(maxsize=1)
def _bundle():
    return joblib.load(C.CLF_PATH)


def predict_batch(texts):
    b = _bundle()
    X = b["vectorizer"].transform(texts)
    out = {}
    for t in b["targets"]:
        clf = b["models"][t]
        out[t] = (clf.classes_, softmax(clf.decision_function(X) / b["temperature"][t], axis=1))
    return out


def classify(subject: str, body: str) -> dict:
    text = f"{subject or ''}\n{body or ''}".strip()
    res = {}
    for t, (classes, p) in predict_batch([text]).items():
        p = p[0]
        order = np.argsort(-p)
        res[t] = {"label": str(classes[order[0]]), "confidence": round(float(p[order[0]]), 4),
                  "probabilities": {str(classes[i]): round(float(p[i]), 4) for i in order}}
    return res
