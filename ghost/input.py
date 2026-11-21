from pathlib import Path

import joblib

model_path = Path(__file__).with_name("triage_models1.joblib")
models = joblib.load(model_path)

while True:
    try:
        subject = input("工单标题（可留空，Ctrl+D 退出）：").strip()
        body = input("工单正文：").strip()
    except EOFError:
        print()
        break

    if not subject and not body:
        print("请输入标题或正文。")
        continue
    text = f"{subject}\n{body}"

    for target, model in models.items():
        scores = model.predict_proba([text])[0]
        best = scores.argmax()
        print(f"{target}: {model.classes_[best]} ({scores[best]:.1%})")

