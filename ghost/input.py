from pathlib import Path

import joblib

model_path = Path(__file__).with_name("triage_models.joblib")
models = joblib.load(model_path)

subject = input("工单标题（可留空）：").strip()
body = input("工单正文：").strip()
text = f"{subject}\n{body}"

for target, model in models.items():
    scores = model.predict_proba([text])[0]
    best = scores.argmax()
    print(f"{target}: {model.classes_[best]} ({scores[best]:.1%})")

