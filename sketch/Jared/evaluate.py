"""
SentinelDesk - Step 4: calibrate confidence on VAL, then final scores on the 20% TEST.

Input : models/sentinel_classifier.joblib, data/val.csv, data/test.csv
Output: models/sentinel_classifier.joblib   (same file, now with calibration added)
        models/test_metrics.json            (numbers for the README / demo)
        models/test_report.txt              (per-class details)

Calibration = temperature scaling, fitted on the validation set only.
It does NOT change which label is predicted, only how trustworthy the
confidence number is. The test set is used only to measure, never to tune.
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax
from sklearn.metrics import accuracy_score, classification_report, f1_score

ROOT = Path(__file__).resolve().parent.parent   # repo root
DATA = ROOT / "data"
MODELS = Path(__file__).resolve().parent / "models"   # Jared/models
BUNDLE_PATH = MODELS / "sentinel_classifier.joblib"
THRESHOLDS = [0.4, 0.5, 0.6, 0.7, 0.8]

bundle = joblib.load(BUNDLE_PATH)
vec, models, targets = bundle["vectorizer"], bundle["models"], bundle["targets"]

val = pd.read_csv(DATA / "val.csv", keep_default_na=False)
test = pd.read_csv(DATA / "test.csv", keep_default_na=False)
X_val = vec.transform(val["text"])
X_test = vec.transform(test["text"])
print(f"val: {len(val)}  test: {len(test)}")


def ece(probs, y_true, classes, n_bins=10):
    """Expected Calibration Error: average gap between confidence and accuracy."""
    conf = probs.max(axis=1)
    correct = classes[probs.argmax(axis=1)] == y_true
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(total)


def fit_temperature(scores, y_idx):
    """Find T that makes softmax(scores / T) best match the true labels (min log-loss)."""
    def nll(t):
        return -log_softmax(scores / t, axis=1)[np.arange(len(y_idx)), y_idx].mean()
    return float(minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded").x)


# ---- 1) Calibrate each classifier on the VALIDATION set ---------------------
print("\n===== 1) CALIBRATION (fitted on val) =====")
temperatures, cal_info = {}, {}
for target in targets:
    clf = models[target]
    classes = clf.classes_
    y_val = val[target].to_numpy()
    y_idx = np.searchsorted(classes, y_val)

    s_val = clf.decision_function(X_val)
    t = fit_temperature(s_val, y_idx)
    temperatures[target] = t

    before = ece(softmax(s_val, axis=1), y_val, classes)
    after = ece(softmax(s_val / t, axis=1), y_val, classes)
    cal_info[target] = {"temperature": round(t, 3),
                        "val_ece_before": round(before, 4),
                        "val_ece_after": round(after, 4)}
    print(f"{target:9s} T={t:.2f}   ECE before={before:.3f}  after={after:.3f}  (lower is better)")

bundle["temperature"] = temperatures
joblib.dump(bundle, BUNDLE_PATH, compress=3)
print("Calibration saved into", BUNDLE_PATH)


# ---- 2) Confidence thresholds on VAL (for the escalation gate) -------------
print("\n===== 2) CONFIDENCE THRESHOLDS (val) =====")
print("auto = share of tickets above the threshold, acc = accuracy on those tickets")
threshold_table = {}
for target in targets:
    clf = models[target]
    p = softmax(clf.decision_function(X_val) / temperatures[target], axis=1)
    conf = p.max(axis=1)
    correct = clf.classes_[p.argmax(axis=1)] == val[target].to_numpy()
    row, cells = {}, []
    for th in THRESHOLDS:
        m = conf >= th
        acc = float(correct[m].mean()) if m.any() else float("nan")
        row[str(th)] = {"auto_share": round(float(m.mean()), 3), "accuracy": round(acc, 3)}
        cells.append(f">={th}: auto {m.mean():.0%} acc {acc:.0%}")
    threshold_table[target] = row
    print(f"{target:9s} " + " | ".join(cells))


# ---- 3) Final holdout scores on TEST ---------------------------------------
print("\n===== 3) HOLDOUT TEST RESULTS =====")
preds, test_metrics, report_lines = {}, {}, []
for target in targets:
    clf = models[target]
    p = softmax(clf.decision_function(X_test) / temperatures[target], axis=1)
    pred = clf.classes_[p.argmax(axis=1)]
    preds[target] = pred
    y = test[target].to_numpy()
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, average="macro")
    test_metrics[target] = {"accuracy": round(acc, 4), "macro_f1": round(f1, 4),
                            "test_ece": round(ece(p, y, clf.classes_), 4)}
    print(f"{target:9s} accuracy={acc:.3f}  macro-F1={f1:.3f}")
    report_lines += [f"===== {target} =====",
                     classification_report(y, pred, digits=3)]

# High-Priority Incident Recall: of the truly (high + Incident) tickets,
# how many did we catch?
true_hi = (test["priority"] == "high") & (test["type"] == "Incident")
pred_hi = (preds["priority"] == "high") & (preds["type"] == "Incident")
hpi_recall = float((true_hi & pred_hi).sum() / true_hi.sum())
hp_recall = float(((preds["priority"] == "high") & true_hi).sum() / true_hi.sum())
hpi_precision = float((true_hi & pred_hi).sum() / max(pred_hi.sum(), 1))
print(f"\nHigh-Priority Incidents in test: {int(true_hi.sum())}")
print(f"High-Priority Incident Recall (predicted high AND Incident): {hpi_recall:.3f}")
print(f"  ... predicted priority=high (any type):                    {hp_recall:.3f}")
print(f"  precision of the (high AND Incident) alarm:                {hpi_precision:.3f}")

report_lines += ["===== priority confusion matrix (rows = true) =====",
                 pd.crosstab(test["priority"].to_numpy(), preds["priority"],
                             rownames=["true"], colnames=["predicted"]).to_string()]


# ---- 4) Latency: one ticket at a time, like the live demo ------------------
print("\n===== 4) LATENCY (single ticket, 200 test tickets) =====")
times = []
for text in test["text"].head(200):
    t0 = time.perf_counter()
    x = vec.transform([text])
    for target in targets:
        softmax(models[target].decision_function(x) / temperatures[target], axis=1)
    times.append((time.perf_counter() - t0) * 1000)
lat = {"mean_ms": round(float(np.mean(times)), 2),
       "p95_ms": round(float(np.percentile(times, 95)), 2)}
print(f"mean {lat['mean_ms']} ms   p95 {lat['p95_ms']} ms")


# ---- 5) Save --------------------------------------------------------------
results = {
    "test_size": len(test),
    "classification": test_metrics,
    "high_priority_incident": {"count": int(true_hi.sum()),
                               "recall": round(hpi_recall, 4),
                               "recall_priority_only": round(hp_recall, 4),
                               "precision": round(hpi_precision, 4)},
    "calibration": cal_info,
    "val_confidence_thresholds": threshold_table,
    "latency": lat,
}
(MODELS / "test_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
(MODELS / "test_report.txt").write_text("\n".join(report_lines), encoding="utf-8")
print("\nSUCCESS: saved models/test_metrics.json and models/test_report.txt")
