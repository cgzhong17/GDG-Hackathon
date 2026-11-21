"""Shared paths and tunable constants for SentinelDesk."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW_CSV = HERE / "data" / "aa_dataset-tickets-multi-lang-5-2-50-version.csv"
TEAM_SPLIT_DIR = HERE / "sketch" / "data"   # earlier team split, used only for an optional consistency check
ART = HERE / "artifacts"                # everything generated lives here (git-ignored)
ART.mkdir(exist_ok=True)

KB_CSV = ART / "kb.csv"                 # 80% historical knowledge base (answers allowed)
TRAIN_CSV = ART / "train.csv"           # 85% of KB -> fit classifiers
VAL_CSV = ART / "val.csv"               # 15% of KB -> calibration + gate thresholds
TEST_INPUTS_CSV = ART / "test_inputs.csv"   # 20% holdout: ticket_id, subject, body ONLY
TEST_LABELS_CSV = ART / "test_labels.csv"   # 20% holdout labels, read only for scoring
CLF_PATH = ART / "classifier.joblib"
GATE_PATH = ART / "gate_thresholds.json"
EMB_PATH = ART / "kb_embeddings.npy"
EMB_IDS_PATH = ART / "kb_embedding_ids.npy"
METRICS_PATH = ART / "test_metrics.json"

TARGETS = ["type", "queue", "priority"]
SEED = 42

EMBED_MODEL = "intfloat/multilingual-e5-small"
TOP_K = 5

# Escalation thresholds (gate_thresholds.json written by train_classifier.py overrides the HPI ones)
HPI_RECALL_TARGET = 0.90        # tune the high-priority-incident alarm to catch >= 90% on val
MIN_TYPE_CONF = 0.45            # below this the classifier is unsure about the ticket type
MIN_QUEUE_CONF = 0.25           # queue has 10 classes, so a lower floor
MIN_RETRIEVAL_SIM = 0.88        # e5 cosine; val in-domain p1 = 0.918, off-topic queries 0.75-0.83
MIN_WORDS = 6                   # shorter tickets carry too little information to auto-answer
NEAR_DUP_SIM = 0.97             # top precedent this close = paraphrase of the same ticket -> inherit its labels
