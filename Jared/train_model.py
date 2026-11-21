"""
SentinelDesk - Step 3: train the three classifiers (type, queue, priority).

Input : data/train.csv, data/val.csv     (created by prepare_data.py)
Output: models/sentinel_classifier.joblib
        models/val_metrics.txt

Design: ONE TF-IDF vectorizer (word + character n-grams, works for English and
German) shared by THREE Logistic Regression models. The test set is NOT used here.
"""
import time
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import make_union

TARGETS = ["type", "queue", "priority"]
ROOT = Path(__file__).resolve().parent.parent   # repo root
DATA = ROOT / "data"
MODELS = Path(__file__).resolve().parent / "models"   # Jared/models
MODELS.mkdir(exist_ok=True)

train = pd.read_csv(DATA / "train.csv", keep_default_na=False)
val = pd.read_csv(DATA / "val.csv", keep_default_na=False)
print(f"train: {len(train)}  val: {len(val)}")

# ---- 1) Text -> numbers -------------------------------------------------------
# word 1-2 grams catch phrases ("server down", "Rechnung falsch");
# char 3-5 grams help with German compound words and typos.
vectorizer = make_union(
    TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                    sublinear_tf=True, max_features=200_000),
    TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                    sublinear_tf=True, max_features=200_000),
)
t0 = time.time()
X_train = vectorizer.fit_transform(train["text"])
X_val = vectorizer.transform(val["text"])
print(f"TF-IDF features: {X_train.shape[1]:,}  ({time.time() - t0:.0f}s)")

# ---- 2) One classifier per target ------------------------------------------
models, report_lines = {}, []
for target in TARGETS:
    t0 = time.time()
    clf = LogisticRegression(max_iter=2000, C=5.0, class_weight="balanced")
    clf.fit(X_train, train[target])
    pred = clf.predict(X_val)

    acc = accuracy_score(val[target], pred)
    f1 = f1_score(val[target], pred, average="macro")
    line = f"{target:9s} accuracy={acc:.3f}  macro-F1={f1:.3f}  ({time.time() - t0:.0f}s)"
    print(line)
    report_lines += [f"===== {target} =====", line,
                     classification_report(val[target], pred, digits=3)]
    models[target] = clf

# ---- 3) Save everything the teammates need in one file ---------------------
bundle = {
    "vectorizer": vectorizer,
    "models": models,
    "targets": TARGETS,
    "sklearn_version": sklearn.__version__,
}
joblib.dump(bundle, MODELS / "sentinel_classifier.joblib", compress=3)
(MODELS / "val_metrics.txt").write_text("\n".join(report_lines), encoding="utf-8")

print("\nSUCCESS: saved models/sentinel_classifier.joblib")
print("Per-class details: models/val_metrics.txt")
