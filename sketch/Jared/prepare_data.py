"""
SentinelDesk - Step 2: clean the raw CSV and make the OFFICIAL 80/20 split.

The split follows the official challenge snippet exactly:
    shuffle with random_state=42 -> ticket_id TKT-00000.. -> first 80% = KB/train, last 20% = test

Input : data/raw_tickets.csv        (created by inspect_data.py)
Output: data/knowledge_base.csv     80%  kb_train_df -> RAG index + model training
        data/train.csv              85% of the KB -> train the classifiers
        data/val.csv                15% of the KB -> tune / calibrate (never train on it)
        data/test.csv               20%  holdout -> ONLY for final evaluation
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent.parent   # repo root
DATA = ROOT / "data"

df = pd.read_csv(DATA / "aa_dataset-tickets-multi-lang-5-2-50-version.csv")
print("Loaded rows:", len(df))

# 1) Official shuffle + ticket IDs (same as the challenge quick-start snippet)
df = df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
df["ticket_id"] = [f"TKT-{i:05d}" for i in range(len(df))]

# 2) Clean text. About 3,800 tickets have no subject -> use "".
df["subject"] = df["subject"].fillna("").astype(str).str.strip()
df["body"] = df["body"].fillna("").astype(str).str.strip()
df["text"] = (df["subject"] + "\n" + df["body"]).str.strip()

# 3) Labels. The official file already has exactly high / medium / low.
for col in ["type", "queue", "priority"]:
    df[col] = df[col].astype(str).str.strip()
df["priority"] = df["priority"].str.lower()
bad = set(df["priority"]) - {"high", "medium", "low"}
if bad:
    raise ValueError(f"Unexpected priority values: {bad}")

keep = ["ticket_id", "subject", "body", "text", "answer",
        "type", "queue", "priority", "language"]
df = df[keep]

# 4) Official 80/20 split: first 80% = knowledge base / training, last 20% = test
split_idx = int(len(df) * 0.8)
kb_train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

# 5) Validation set taken from the 80% only (test is never touched)
train_df, val_df = train_test_split(kb_train_df, test_size=0.15,
                                    random_state=RANDOM_STATE,
                                    stratify=kb_train_df["priority"])

kb_train_df.to_csv(DATA / "knowledge_base.csv", index=False)
train_df.to_csv(DATA / "train.csv", index=False)
val_df.to_csv(DATA / "val.csv", index=False)
test_df.to_csv(DATA / "test.csv", index=False)

print(f"\nknowledge_base: {len(kb_train_df)}  ->  train: {len(train_df)}  val: {len(val_df)}")
print(f"test (holdout): {len(test_df)}")
print("\nPriority share per split:")
print(pd.DataFrame({
    "train": train_df["priority"].value_counts(normalize=True),
    "val": val_df["priority"].value_counts(normalize=True),
    "test": test_df["priority"].value_counts(normalize=True),
}).round(3).to_string())
print("\nSUCCESS: split files saved in data/")
