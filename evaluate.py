"""Holdout evaluation on the 20% test split.

The system only ever receives test_inputs.csv (ticket_id, subject, body).
test_labels.csv is opened AFTER all predictions are made, purely for scoring.

    python evaluate.py              # classifier (+precedent override), gate, language, retrieval, latency
    python evaluate.py --gen 300    # + end-to-end draft checks on 300 holdout tickets
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

import config as C
from classifier import predict_batch
from gate import SAFETY_NET, hpi_thresholds
from lang import detect_language
from precedent_labels import mix, override_acc
from retriever import embed, load_kb

ap = argparse.ArgumentParser()
ap.add_argument("--gen", type=int, default=0)
args = ap.parse_args()

inputs = pd.read_csv(C.TEST_INPUTS_CSV, keep_default_na=False)
texts = (inputs.subject + "\n" + inputs.body).str.strip()
R = {"holdout_rows": len(inputs)}

# ================= predictions (no labels in scope) =================
t0 = time.perf_counter()
probs = predict_batch(texts.tolist())
R["classify_ms_per_ticket_batched"] = round((time.perf_counter() - t0) * 1000 / len(inputs), 3)

t0 = time.perf_counter()
lang_pred = np.array([detect_language(s, b)["lang"] for s, b in zip(inputs.subject, inputs.body)])
R["language_detect_ms_per_ticket"] = round((time.perf_counter() - t0) * 1000 / len(inputs), 3)

kb = load_kb()
E = np.load(C.EMB_PATH)
t0 = time.perf_counter()
Q = embed(texts.tolist())
S = Q @ E.T
R["embed_and_search_ms_per_ticket_batched"] = round((time.perf_counter() - t0) * 1000 / len(inputs), 2)
top5 = np.argsort(-S, axis=1)[:, :5]
top1_sim = S[np.arange(len(S)), top5[:, 0]]
nb = kb.iloc[top5[:, 0]].reset_index(drop=True)
near = top1_sim >= C.NEAR_DUP_SIM

final = {}
for t, (classes, p) in probs.items():
    final[t] = (classes, mix(classes, p, nb[t].to_numpy(), near, override_acc()[t]))


def hpi_alarms(P, regime_near):
    """regime_near: bool per row -> which threshold set applies (all False = classifier-only system)."""
    s = P["priority"][1][:, list(P["priority"][0]).index("high")] * P["type"][1][:, list(P["type"][0]).index("Incident")]
    (rv_n, pg_n), (rv_d, pg_d) = hpi_thresholds("novel"), hpi_thresholds("near_dup")
    review = np.where(regime_near, rv_d, rv_n)
    page = np.where(regime_near, pg_d, pg_n)
    kw = texts.apply(lambda t: any(p.search(t) for p in SAFETY_NET.values())).to_numpy()
    r0 = (P["priority"][0][P["priority"][1].argmax(1)] == "high") & (P["type"][0][P["type"][1].argmax(1)] == "Incident")
    return {"page": r0 | (s >= page) | kw, "human": r0 | (s >= review) | kw}


# ================= scoring (labels loaded only now) =================
labels = pd.read_csv(C.TEST_LABELS_CSV).set_index("ticket_id").loc[inputs.ticket_id].reset_index()
truth_hpi = ((labels.priority == "high") & (labels.type == "Incident")).to_numpy()


def alarm_stats(a):
    return {"recall": round(float((a & truth_hpi).sum() / truth_hpi.sum()), 4),
            "precision": round(float((a & truth_hpi).sum() / max(a.sum(), 1)), 4),
            "alarm_rate": round(float(a.mean()), 4)}


def score(P, regime_near):
    out = {}
    pred = {t: c[p.argmax(1)] for t, (c, p) in P.items()}
    for t in C.TARGETS:
        y = labels[t].to_numpy()
        rep = classification_report(y, pred[t], output_dict=True, zero_division=0)
        out[t] = {"macro_f1": round(f1_score(y, pred[t], average="macro"), 4),
                  "accuracy": round(accuracy_score(y, pred[t]), 4),
                  "per_class_f1": {k: round(v["f1-score"], 3) for k, v in rep.items() if k in set(y)},
                  "per_language_macro_f1": {lg: round(f1_score(y[labels.language == lg], pred[t][labels.language == lg],
                                                               average="macro"), 4) for lg in ["en", "de"]}}
    a = hpi_alarms(P, regime_near)
    argmax = (pred["priority"] == "high") & (pred["type"] == "Incident")
    out["high_priority_incident"] = {"count": int(truth_hpi.sum()),
                                     "argmax labels (pred high AND Incident)": alarm_stats(argmax),
                                     "on-call page (R0 + R1 page tier + R2 keywords)": alarm_stats(a["page"]),
                                     "escalated to a human (page or review tier)": alarm_stats(a["human"])}
    out["all_three_correct"] = round(float(np.all([pred[t] == labels[t].to_numpy() for t in C.TARGETS], axis=0).mean()), 4)
    return out


R["classifier_only"] = score(probs, np.zeros(len(inputs), bool))
R["with_precedent_override"] = score(final, near)
R["with_precedent_override"]["novel_tickets_only (no near-duplicate)"] = {
    "n": int((~near).sum()),
    **{t: round(f1_score(labels[t][~near], final[t][0][final[t][1][~near].argmax(1)], average="macro"), 4)
       for t in C.TARGETS},
    "hpi_escalated_to_human_recall": round(float(
        (hpi_alarms(final, near)["human"] & truth_hpi)[~near].sum() / truth_hpi[~near].sum()), 4)}
R["with_precedent_override"]["coverage (near-duplicate found)"] = round(float(near.mean()), 4)
R["with_precedent_override"]["precedent_label_accuracy_on_covered"] = {
    t: round(float((nb[t].to_numpy()[near] == labels[t].to_numpy()[near]).mean()), 4) for t in C.TARGETS}

R["language_detection"] = {
    "agreement_with_dataset_label": round(float((lang_pred == labels.language.to_numpy()).mean()), 4),
    "confusion (rows=label, cols=detected)": pd.crosstab(labels.language, lang_pred).to_dict(orient="index"),
    "note": "all disagreements are 'de'-labelled rows whose text is English (dataset label noise)"}

lab = labels
R["retrieval"] = {
    "top1_similarity_mean": round(float(top1_sim.mean()), 4),
    "top1_similarity_p05": round(float(np.percentile(top1_sim, 5)), 4),
    "share_below_min_sim": round(float((top1_sim < C.MIN_RETRIEVAL_SIM).mean()), 4),
    "top1_same_queue": round(float((nb.queue.to_numpy() == lab.queue.to_numpy()).mean()), 4),
    "top1_same_type": round(float((nb.type.to_numpy() == lab.type.to_numpy()).mean()), 4),
    "top5_any_same_queue": round(float(np.mean([(kb.queue.to_numpy()[top5[i]] == lab.queue[i]).any()
                                               for i in range(len(lab))])), 4),
    "cross_lingual_share_of_top5": round(float((kb.language.to_numpy()[top5] != lab.language.to_numpy()[:, None]).mean()), 4),
}

# ================= optional: end-to-end drafts =================
if args.gen:
    from pipeline import triage
    gsample = inputs.sample(n=args.gen, random_state=7)
    rows = []
    for s, b in zip(gsample.subject, gsample.body):
        out = triage(s, b)
        d = out["draft"]
        rows.append({"valid": not d["validation_issues"], "cites": bool(d["cited_ticket_ids"]),
                     "translated": any(x.get("translated_from") for x in d.get("sentences", [])),
                     "lang_ok": not any("language" in i for i in d["validation_issues"]),
                     "fig_ok": not any("figures" in i for i in d["validation_issues"]),
                     "status": out["decision"]["status"], "ms": out["timing_ms"]["total_ms"]})
    g = pd.DataFrame(rows)
    R["generation"] = {"n": len(g), "engine": "extractive (local)",
                       "passes_all_grounding_checks": round(g.valid.mean(), 4),
                       "grounded_draft_with_citations": round(g.cites.mean(), 4),
                       "uses_cross_lingual_precedent": round(g.translated.mean(), 4),
                       "language_match": round(g.lang_ok.mean(), 4),
                       "no_unsupported_figures": round(g.fig_ok.mean(), 4),
                       "gate_status": g.status.value_counts(normalize=True).round(3).to_dict(),
                       "end_to_end_ms_mean": round(g.ms.mean(), 1), "end_to_end_ms_p95": round(g.ms.quantile(.95), 1)}

C.METRICS_PATH.write_text(json.dumps(R, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(R, indent=2, ensure_ascii=False))
