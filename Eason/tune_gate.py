"""Tune on the VALIDATION split (val is 15% of the KB; its neighbours are searched in the train part only):

1. precedent-label override accuracy at NEAR_DUP_SIM  -> precedent_override.accuracy
2. two-tier high-priority-incident (HPI) gate, score = P(priority=high) * P(type=Incident),
   tuned SEPARATELY for the two regimes because their probability distributions differ:
     near_dup: a near-duplicate precedent exists (override-mixed probabilities)
     novel:    no near-duplicate (pure classifier probabilities) - what live, hand-written tickets hit
   review threshold: highest score keeping HPI recall >= HPI_RECALL_TARGET   -> human review
   page threshold:   lowest score with alarm precision >= PAGE_PRECISION_TARGET -> page on-call
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import config as C
from classifier import predict_batch
from precedent_labels import mix

PAGE_PRECISION_TARGET = 0.60


def val_neighbours(va):
    from retriever import embed, load_kb
    kb = load_kb()
    E = np.load(C.EMB_PATH)
    train_ids = set(pd.read_csv(C.TRAIN_CSV, usecols=["ticket_id"]).ticket_id)
    m = kb.ticket_id.isin(train_ids).to_numpy()
    kb_tr, E_tr = kb[m].reset_index(drop=True), E[m]
    S = embed((va.subject + "\n" + va.body).str.strip().tolist()) @ E_tr.T
    return kb_tr.iloc[S.argmax(1)].reset_index(drop=True), S.max(1)


def tiers(score, truth):
    def stats(th):
        a = score >= th
        return {"threshold": float(th), "val_recall": round(float((a & truth).sum() / truth.sum()), 4),
                "val_precision": round(float((a & truth).sum() / max(a.sum(), 1)), 4),
                "val_alarm_rate": round(float(a.mean()), 4), "n": int(len(score))}
    grid = np.round(np.arange(0.005, 0.95, 0.005), 3)
    review = max(th for th in grid if stats(th)["val_recall"] >= C.HPI_RECALL_TARGET)
    page = min((th for th in grid if stats(th)["val_precision"] >= PAGE_PRECISION_TARGET), default=0.9)
    return {"review": stats(review), "page": stats(max(page, review))}


def main():
    va = pd.read_csv(C.VAL_CSV, keep_default_na=False)
    P = predict_batch((va.subject + "\n" + va.body).str.strip().tolist())
    nb, sim = val_neighbours(va)
    use = sim >= C.NEAR_DUP_SIM

    out = json.loads(C.GATE_PATH.read_text()) if C.GATE_PATH.exists() else {}
    acc, f1, final = {}, {}, {}
    for t in C.TARGETS:
        classes, p = P[t]
        y = va[t].to_numpy()
        acc[t] = round(float((nb[t].to_numpy()[use] == y[use]).mean()), 4)
        final[t] = mix(classes, p, nb[t].to_numpy(), use, acc[t])
        f1[t] = {"classifier": round(f1_score(y, classes[p.argmax(1)], average="macro"), 4),
                 "with_precedent_override": round(f1_score(y, classes[final[t].argmax(1)], average="macro"), 4)}
    out["precedent_override"] = {"near_dup_sim": C.NEAR_DUP_SIM, "val_coverage": round(float(use.mean()), 4),
                                 "accuracy": acc, "val_macro_f1": f1}

    score = final["priority"][:, list(P["priority"][0]).index("high")] * \
        final["type"][:, list(P["type"][0]).index("Incident")]
    truth = ((va.priority == "high") & (va.type == "Incident")).to_numpy()
    out["hpi"] = {"near_dup": tiers(score[use], truth[use]), "novel": tiers(score[~use], truth[~use])}
    C.GATE_PATH.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ["precedent_override", "hpi"]}, indent=2))


if __name__ == "__main__":
    main()
