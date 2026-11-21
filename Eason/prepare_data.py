"""Step 1: clean the official CSV and lock the official 80/20 split.

Split = the challenge quick-start snippet exactly:
    shuffle(random_state=42) -> ticket_id TKT-00000.. -> first 80% KB, last 20% holdout.

Leakage hygiene is enforced by file layout:
    test_inputs.csv  -> ticket_id, subject, body   (what the system is allowed to see)
    test_labels.csv  -> ticket_id, type, queue, priority, language  (scoring only)
    The holdout `answer` column is never written anywhere.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

import config as C


def clean_text(s: pd.Series) -> pd.Series:
    # the 5-2-50 file stores newlines as the literal two characters "\n"
    return (s.fillna("").astype(str)
             .str.replace("\\n", "\n", regex=False)
             .str.replace("\r", "", regex=False)
             .str.strip())


def main():
    df = pd.read_csv(C.RAW_CSV)
    print("raw rows:", len(df))

    df = df.sample(frac=1.0, random_state=C.SEED).reset_index(drop=True)
    df["ticket_id"] = [f"TKT-{i:05d}" for i in range(len(df))]

    for col in ["subject", "body", "answer"]:
        df[col] = clean_text(df[col])
    for col in C.TARGETS:
        df[col] = df[col].astype(str).str.strip()
    df["priority"] = df["priority"].str.lower()
    assert set(df["priority"]) == {"high", "medium", "low"}
    tags = [f"tag_{i}" for i in range(1, 9)]
    df["tags"] = df[tags].apply(lambda r: ", ".join(t for t in r if isinstance(t, str)), axis=1)

    split = int(len(df) * 0.8)
    kb, test = df.iloc[:split].copy(), df.iloc[split:].copy()

    # consistency check with the team's committed split (same ids -> same tickets)
    team_test = C.TEAM_SPLIT_DIR / "test.csv"
    if team_test.exists():
        t = pd.read_csv(team_test, usecols=["ticket_id", "body"])
        same = (t.set_index("ticket_id").body.fillna("").str.replace("\\n", "\n", regex=False).str.strip()
                == test.set_index("ticket_id").body)
        print(f"matches team split: {same.mean():.1%} of holdout ids")

    kb_cols = ["ticket_id", "subject", "body", "answer", "type", "queue", "priority", "language", "tags"]
    kb[kb_cols].to_csv(C.KB_CSV, index=False)
    tr, va = train_test_split(kb[kb_cols], test_size=0.15, random_state=C.SEED,
                              stratify=kb["type"] + "|" + kb["priority"])
    tr.to_csv(C.TRAIN_CSV, index=False)
    va.to_csv(C.VAL_CSV, index=False)

    test[["ticket_id", "subject", "body"]].to_csv(C.TEST_INPUTS_CSV, index=False)
    test[["ticket_id", "type", "queue", "priority", "language"]].to_csv(C.TEST_LABELS_CSV, index=False)

    print(f"KB {len(kb)} (train {len(tr)} / val {len(va)})   holdout {len(test)}")
    print(f"empty subjects: KB {(kb.subject == '').sum()}, holdout {(test.subject == '').sum()}")
    print(f"KB rows without answer (kept for training, skipped by RAG): {(kb.answer == '').sum()}")


if __name__ == "__main__":
    main()
