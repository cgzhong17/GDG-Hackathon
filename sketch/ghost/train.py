"""Train and evaluate three calibrated ticket classifiers on the official 20k CSV."""
import hashlib
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
CSV = ROOT.parent / "data/dataset-tickets-multi-lang-4-20k.csv"
TARGETS = ["type", "queue", "priority"]
NUMBER = "1"
SAVENAME = f"triage_models{NUMBER}.joblib"

def train() -> None:
    data = pd.read_csv(CSV, usecols=["subject", "body", "language", *TARGETS])
    if data[TARGETS].isna().any().any():
        raise ValueError("Missing training or evaluation labels")
    text = data.subject.fillna("") + "\n" + data.body.fillna("")
    groups = text.str.casefold().str.split().str.join(" ")
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr, te = next(split.split(text, groups=groups))
    assert set(groups.iloc[tr]).isdisjoint(groups.iloc[te])
    result = {
        "sha256": hashlib.sha256(CSV.read_bytes()).hexdigest(),
        "train_rows": len(tr), "test_rows": len(te),
        "unique_texts": groups.nunique(), "tasks": {},
    }
    models = {}
    start = perf_counter()
    for target in TARGETS:
        y = data[target].iloc[tr]
        folds = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
        cv = list(folds.split(text.iloc[tr], y, groups.iloc[tr]))
        pipeline = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
            LogisticRegression(max_iter=1000),
        )
        model = CalibratedClassifierCV(pipeline, method="sigmoid", cv=cv, ensemble=False)
        models[target] = model.fit(text.iloc[tr], y)
        print(f"Trained {target} ({perf_counter() - start:.1f}s elapsed)", flush=True)
    result["fit_seconds"] = round(perf_counter() - start, 2)
    artifact = ROOT / SAVENAME
    joblib.dump(models, artifact, compress=3)
    models = joblib.load(artifact)

    predictions = pd.DataFrame(index=te)
    truth = data.iloc[te]
    for target, model in models.items():
        probabilities = model.predict_proba(text.iloc[te])
        predictions[target] = model.classes_[probabilities.argmax(axis=1)]
        y, pred = truth[target], predictions[target]
        most_common = data[target].iloc[tr].mode().iloc[0]
        result["tasks"][target] = {
            "macro_f1": f1_score(y, pred, average="macro"),
            "accuracy": accuracy_score(y, pred),
            "majority_macro_f1": f1_score(y, [most_common] * len(y), average="macro"),
            "per_class": classification_report(y, pred, output_dict=True, zero_division=0),
        }
    result["language"] = {}
    for lang, subset in truth.groupby("language"):
        result["language"][lang] = {"rows": len(subset), **{
            t: f1_score(subset[t], predictions.loc[subset.index, t],
                        labels=models[t].classes_, average="macro", zero_division=0)
            for t in TARGETS
        }}
    actual = truth["type"].eq("Incident") & truth.priority.eq("high")
    incident = predictions["type"].eq("Incident")
    high = predictions.priority.eq("high")
    result["high_priority_incident"] = {
        "actual": int(actual.sum()), "predicted": int((incident & high).sum()),
        "correct": int((actual & incident & high).sum()),
        "miss_type_only": int((actual & ~incident & high).sum()),
        "miss_priority_only": int((actual & incident & ~high).sum()),
        "miss_both": int((actual & ~incident & ~high).sum()),
    }
    result["all_three_correct_rate"] = float((predictions[TARGETS] == truth[TARGETS]).all(axis=1).mean())
    (ROOT / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        train()
