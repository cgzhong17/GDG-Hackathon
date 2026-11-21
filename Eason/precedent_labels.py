"""Precedent label override: if an almost identical historical ticket exists (e5 cosine >= NEAR_DUP_SIM),
inherit its labels.

The dataset contains many paraphrased variants of the same ticket. When the top precedent is that
close, its labels are ~96-99% correct on val (vs ~57-86% for the text classifier), so we mix them in:

    p_final = acc_t * onehot(precedent label) + (1 - acc_t) * p_classifier

acc_t = precedent-label accuracy measured on val at that similarity (tune_gate.py), so the result is
still a calibrated probability. Novel tickets (no near-duplicate) keep the classifier output unchanged.
Only KB (80% split) labels are ever used.
"""
import json
from functools import lru_cache

import numpy as np

import config as C


@lru_cache(maxsize=1)
def override_acc():
    try:
        return json.loads(C.GATE_PATH.read_text())["precedent_override"]["accuracy"]
    except Exception:
        return {"type": 0.99, "queue": 0.96, "priority": 0.96}


def mix(classes, p, labels, use, acc):
    """Vectorised: classes (k,), p (n,k), labels (n,) precedent labels, use (n,) bool."""
    onehot = (labels[:, None] == classes[None, :]).astype(float)
    out = p.copy()
    out[use] = acc * onehot[use] + (1 - acc) * p[use]
    return out


def apply(cls: dict, precedent: dict | None) -> dict:
    """Single-ticket version on the classify() dict; adds 'source' per target."""
    near = precedent is not None and precedent["similarity"] >= C.NEAR_DUP_SIM
    out = {}
    for t, r in cls.items():
        if not near:
            out[t] = {**r, "source": "classifier"}
            continue
        acc = override_acc()[t]
        probs = {c: (1 - acc) * v + (acc if c == precedent[t] else 0.0) for c, v in r["probabilities"].items()}
        probs = dict(sorted(probs.items(), key=lambda kv: -kv[1]))
        label = next(iter(probs))
        out[t] = {"label": label, "confidence": round(probs[label], 4),
                  "probabilities": {c: round(v, 4) for c, v in probs.items()},
                  "source": f"precedent {precedent['ticket_id']} (sim {precedent['similarity']:.3f})",
                  "classifier_label": r["label"]}
    return out
