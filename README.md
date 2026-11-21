# SentinelDesk — Multilingual Incident Triage & Precedent-Grounded Reply Copilot

Paste a raw ticket (subject optional + body, English or German). SentinelDesk:

1. **Classifies** type / queue / priority with calibrated confidence.
2. **Gates** it: `🚨 Escalate to On-Call Engineer`, `🧑‍💼 Human Review`, or `✅ Auto-Draft Ready`, with every fired rule listed as a reason.
3. **Retrieves** the top-5 most similar *historical* tickets (multilingual embeddings, EN↔DE).
4. **Drafts a reply in the customer's language** where **every sentence is taken from a cited historical resolution** (`[TKT-xxxxx]`), so it cannot invent policies, prices, guarantees or integrations.

Runs **fully offline** — no ticket text ever leaves the machine (see [Privacy](#privacy--pii)).

## Quick start

The trained classifier, gate thresholds and knowledge-base embeddings are committed in `artifacts/`, so the console runs straight after install:

```bash
pip install -r requirements.txt
streamlit run app.py         # live triage console (first query ~25 s while models load, then ~300 ms)
```

The first run downloads three small open models from Hugging Face (`intfloat/multilingual-e5-small`, `Helsinki-NLP/opus-mt-en-de`, `opus-mt-de-en`); after that everything runs offline.

Full rebuild from the raw CSV (~35 min on a laptop CPU):

```bash
python prepare_data.py       # official 80/20 split -> artifacts/
python retriever.py          # embed the 80% knowledge base (~20 min on CPU)
python train_classifier.py   # classifier + calibration, then tune_gate.py (val only)
python evaluate.py --gen 300 # holdout metrics -> artifacts/test_metrics.json
```

Data: `data/aa_dataset-tickets-multi-lang-5-2-50-version.csv` (official recommended file, 28,587 tickets). All generated files go to `artifacts/` (git-ignored).

## Repository layout

```
GDG-Hackathon/
├── *.py, README.md, requirements.txt   final system (this README)
├── data/                               official dataset (5-2-50 version)
├── artifacts/                          generated: split, models, embeddings, metrics (git-ignored)
└── sketch/                             earlier team explorations, kept for reference only (not used by the final system)
    ├── ghost/   first baseline (20k file, word TF-IDF + calibrated LogReg)
    ├── Jared/   official split + word/char TF-IDF classifier + keyword escalation rule
    ├── tens/    data exploration
    └── data/    20k file and the team's earlier split CSVs
```

## Architecture

```
ticket ─► language detection (lingua EN/DE + stop-word fallback)
       ─► TF-IDF word+char ─► 3 class-balanced LogReg ─► temperature-scaled probs ─┐
       ─► multilingual-e5-small ─► cosine top-5 over KB (80% split only) ──────────┤
                                                                                   ▼
          extractive composer: precedent answer sentences ─► filters ─► rank ─► translate (opus-mt) ─► cite
                                                                                   ▼
          escalation gate R1–R6 ─► status + reasons            burst detector ─► outage banner
```

| File | Role |
|---|---|
| `prepare_data.py` | Official split (`sample(frac=1, random_state=42)`, `TKT-xxxxx`, first 80% = KB). Writes holdout as `test_inputs.csv` (id, subject, body) + `test_labels.csv`; the holdout `answer` is never written. |
| `train_classifier.py` | Shared TF-IDF (word 1–2 + char_wb 3–5) → LogReg ×3, temperature scaling on val. |
| `tune_gate.py` | Precedent-override accuracy + two-tier high-priority-incident thresholds, on val. |
| `precedent_labels.py` | Inherit labels from a near-duplicate KB ticket (cosine ≥ 0.97). |
| `retriever.py` | e5 embeddings of KB subject+body; asserts no holdout id is indexed. |
| `composer.py` | Extractive, citation-by-construction reply composer. |
| `generator.py` | Runs the composer, then grounding validation. |
| `gate.py` | Escalation rules R1–R6. |
| `lang.py`, `pii.py`, `burst.py` | Language detection, PII detection, outage burst detection. |
| `evaluate.py` | Holdout evaluation (labels read only after predictions). |
| `app.py` | Streamlit operator console. |

## Escalation gate

| Rule | Fires when | Result |
|---|---|---|
| R0 | predicted `priority == high` AND predicted `type == Incident` (the challenge's explicit rule) | 🚨 On-call |
| R1 page | `P(high)·P(Incident)` ≥ page threshold (val precision ≥ 60%; separate thresholds for near-duplicate vs novel tickets) | 🚨 On-call |
| R2 | outage / security / healthcare-failure keywords (EN+DE) | 🚨 On-call |
| R1 review | `P(high)·P(Incident)` ≥ review threshold (val HPI recall ≥ 90%) | 🧑‍💼 Review |
| R3 | type confidence < 0.45 or queue confidence < 0.25 | 🧑‍💼 Review |
| R4 | best precedent similarity < 0.88, or < 6 words | 🧑‍💼 Review |
| R5 | refund/compensation, legal/guarantee, pricing, credential requests; no groundable precedent; draft fails a grounding check | 🧑‍💼 Review |
| R6 | prompt-injection patterns (reply replaced by a safe holding message) | 🧑‍💼 Review |
| Burst | ≥ 3 similar tickets within 30 min | 🚨 On-call + outage banner |

The 0.88 similarity floor is data-driven: in-domain validation tickets have top-1 similarity ≥ 0.918 (1st percentile), off-topic queries score 0.75–0.83.

## Grounded reply composer — why extractive

Every body sentence is copied from (or machine-translated from) an official historical answer and tagged with its ticket id. Filters drop sentences that would be unsafe to reuse for a different customer:

- placeholders (`<name>`, `<tel_num>`, `[Name]` …), greetings and sign-offs
- **commitments**: money, %, deadlines, refunds, guarantees, discounts
- **foreign entities**: product names / acronyms / versions the customer never mentioned (prevents "invented integrations")
- capability claims ("supports", "integrates with" …), context-dependent sentences, restatements of the other customer's issue

Remaining sentences are ranked by e5 similarity to the ticket (weighted by precedent similarity), de-duplicated, and 2–4 are kept. If the customer asks for refunds/legal/pricing, a fixed line says a specialist will follow up and nothing is confirmed. A post-check verifies cited ids ⊂ retrieved ids, no unsupported figures, and reply language = ticket language.

### Design trade-off: traceability over fluency

We deliberately chose an extractive composer over a generative LLM. The cost is honest to state:

- Drafts can read **stiffer** than an LLM reply, and some dataset answers are themselves awkwardly phrased.
- Sentences come from *similar* past tickets, so for an unusual ticket the draft can be **less specific** than the customer's exact question (it tends to ask for the right diagnostics rather than solve the precise case).
- Only the fixed opening / closing / "specialist will follow up" lines are templated; everything else is a historical sentence.

What we get in return: **every content sentence is traceable to a cited historical resolution**, so the draft cannot fabricate prices, refunds, SLAs, legal guarantees or product integrations, and it works with zero API dependency or cost. Drafts are meant for agent review — the gate sends anything risky, novel or unsupported to a human, and the operator sees the cited precedents next to the draft.

## Privacy / PII

The challenge's PII bonus asks for masking *before ticket text is sent to an external LLM API*. SentinelDesk sends **nothing** externally: classification, retrieval, translation and drafting all run locally on open models, so customer data never leaves the machine — a stronger guarantee than masking. `pii.py` still detects emails, phone numbers, IP addresses, IBAN/card and account numbers, and the console shows what was found so an operator can redact before forwarding a ticket.

## Holdout results (20% split, 5,718 tickets)

Full numbers: `artifacts/test_metrics.json` (re-generate with `python evaluate.py --gen 300`).

### Triage (macro-F1)

| | type | queue | priority | all 3 correct |
|---|---|---|---|---|
| Majority-class baseline | 0.14 | 0.05 | 0.19 | – |
| Classifier only (TF-IDF + LogReg) | 0.866 | 0.569 | 0.617 | 36.3% |
| **+ precedent label override (final)** | **0.920** | **0.717** | **0.774** | **65.0%** |
| ↳ novel tickets only (no near-duplicate, 39% of holdout) | 0.812 | 0.332 | 0.461 | – |

**Precedent label override — read this.** The dataset contains many paraphrased variants of the same ticket. When the top retrieved KB ticket has cosine ≥ 0.97 (61% of holdout), its labels are 96.5–99.7% correct, so we mix them into the probabilities (`p = acc·onehot + (1−acc)·p_clf`, `acc` measured on val). Only KB (80%) labels are used; nothing from the holdout. The "novel tickets" row is the honest estimate for brand-new, hand-written tickets — queue and priority are hard from text alone in this dataset, which is exactly why the escalation gate is tuned for recall.

### High-Priority Incident recall (995 true high-priority Incidents)

| | recall | precision | alarm rate |
|---|---|---|---|
| Argmax labels (pred high AND Incident) | 81.7% | 77.2% | 18.4% |
| Paged to on-call (R0 + R1 page tier + R2 keywords) | 85.0% | 53.0% | 27.9% |
| **Escalated to a human (page or review tier)** | **95.4%** | 46.8% | 35.4% |
| ↳ novel tickets only | 88.7% | – | – |

Gate thresholds are tuned on val separately for near-duplicate vs novel tickets (their probability distributions differ).

### Retrieval, reply, language, latency

| Metric | Value |
|---|---|
| Top-1 precedent same queue / same type | 76.9% / 92.1% |
| Top-5 contains same-queue precedent | 89.4% |
| Cross-lingual share of top-5 precedents | 16.7% |
| Drafts with valid citations (300 holdout tickets) | 100% |
| Drafts passing all grounding checks (ids ⊂ retrieved, no unsupported figures, language) | 100% |
| Reply language = ticket language | 100% |
| Language detection vs dataset label | 88.3% — every disagreement is a `de`-labelled row whose text (and official answer) is English; 0 English tickets detected as German |
| End-to-end latency (classify + retrieve + compose, CPU) | mean 212 ms, p95 308 ms |
| Gate outcome on 300 holdout tickets | 45.0% auto-draft · 26.3% human review · 28.7% on-call |

Classifier-only val calibration (ECE): type 0.016, queue 0.024, priority 0.017 after temperature scaling.

## Leakage controls

- Split identical to the official snippet (and to the team's earlier split in `sketch/data` — checked 100% id match).
- Classifier fit on `train` (85% of KB), calibration + thresholds on `val` (15% of KB); holdout used once, for scoring.
- RAG index = KB only; `retriever._index()` asserts no holdout id is present.
- Holdout `answer` column is never written to disk; `evaluate.py` reads labels only after all predictions.
