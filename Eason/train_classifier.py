"""Step 2: multi-task triage classifier + calibration + escalation-gate threshold.

- One shared TF-IDF (word 1-2 grams + char_wb 3-5 grams, good for German compounds)
- Three class-balanced logistic regressions (type / queue / priority)
- Temperature scaling fitted on VAL -> calibrated confidences
- High-Priority-Incident alarm threshold tuned on VAL for >= HPI_RECALL_TARGET recall
The holdout set is not touched here.
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_union

import config as C


def text_of(df):
    return (df.subject.fillna("") + "\n" + df.body.fillna("")).str.strip()


def ece(p, y, classes, bins=10):
    conf, correct = p.max(1), classes[p.argmax(1)] == y
    edges = np.linspace(0, 1, bins + 1)
    return float(sum(((conf > lo) & (conf <= hi)).mean() * abs(correct[(conf > lo) & (conf <= hi)].mean()
                                                               - conf[(conf > lo) & (conf <= hi)].mean())
                     for lo, hi in zip(edges[:-1], edges[1:]) if ((conf > lo) & (conf <= hi)).any()))


def fit_temperature(scores, y_idx):
    nll = lambda t: -log_softmax(scores / t, axis=1)[np.arange(len(y_idx)), y_idx].mean()
    return float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)


def main():
    tr = pd.read_csv(C.TRAIN_CSV, keep_default_na=False)
    va = pd.read_csv(C.VAL_CSV, keep_default_na=False)
    vec = make_union(
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=200_000),
    )
    t0 = time.time()
    Xtr, Xva = vec.fit_transform(text_of(tr)), vec.transform(text_of(va))
    print(f"features {Xtr.shape[1]:,} ({time.time() - t0:.0f}s)")

    models, temps, val_probs, report = {}, {}, {}, {}
    for target in C.TARGETS:
        t0 = time.time()
        clf = LogisticRegression(max_iter=3000, C=5.0, class_weight="balanced").fit(Xtr, tr[target])
        s = clf.decision_function(Xva)
        y = va[target].to_numpy()
        t = fit_temperature(s, np.searchsorted(clf.classes_, y))
        p = softmax(s / t, axis=1)
        models[target], temps[target], val_probs[target] = clf, t, p
        report[target] = {"val_macro_f1": round(f1_score(y, clf.classes_[p.argmax(1)], average="macro"), 4),
                          "temperature": round(t, 3),
                          "val_ece_raw": round(ece(softmax(s, axis=1), y, clf.classes_), 4),
                          "val_ece_calibrated": round(ece(p, y, clf.classes_), 4)}
        print(target, report[target], f"({time.time() - t0:.0f}s)")

    joblib.dump({"vectorizer": vec, "models": models, "temperature": temps, "targets": C.TARGETS},
                C.CLF_PATH, compress=3)
    C.GATE_PATH.write_text(json.dumps({"classifier_val": report}, indent=2))
    print("saved", C.CLF_PATH.name)

    import tune_gate          # two-tier high-priority-incident thresholds, tuned on val
    tune_gate.main()


if __name__ == "__main__":
    main()
