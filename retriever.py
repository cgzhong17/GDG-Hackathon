"""Step 3: multilingual precedent retrieval over the 80% knowledge base only.

intfloat/multilingual-e5-small embeds EN and DE into one space, so a German ticket
can retrieve English precedents (cross-lingual bonus). Index text = subject + body;
the stored `answer` is what we show and ground the reply on.

Build once:  python retriever.py      (writes artifacts/kb_embeddings.npy)
"""
import time
from functools import lru_cache

import numpy as np
import pandas as pd

import config as C


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(C.EMBED_MODEL, device="cpu")
    m.max_seq_length = 256
    return m


def embed(texts, batch_size=64, show=False):
    # e5 convention: symmetric similarity -> "query: " prefix on both sides
    return _model().encode([f"query: {t}" for t in texts], batch_size=batch_size,
                           normalize_embeddings=True, show_progress_bar=show,
                           convert_to_numpy=True).astype(np.float32)


@lru_cache(maxsize=1)
def load_kb():
    kb = pd.read_csv(C.KB_CSV, keep_default_na=False)
    kb = kb[kb.answer.str.len() > 0].reset_index(drop=True)      # need an answer to cite
    return kb


@lru_cache(maxsize=1)
def _index():
    kb = load_kb()
    ids = np.load(C.EMB_IDS_PATH, allow_pickle=True)
    assert (ids == kb.ticket_id.to_numpy()).all(), "embedding index out of sync with kb.csv - rebuild"
    holdout = set(pd.read_csv(C.TEST_INPUTS_CSV, usecols=["ticket_id"]).ticket_id)
    assert holdout.isdisjoint(ids), "holdout ticket found in the RAG index"
    return kb, np.load(C.EMB_PATH)


def retrieve(text: str, k: int = C.TOP_K):
    """Return top-k distinct precedents: list of dicts with ticket_id, similarity, answer, labels."""
    kb, emb = _index()
    q = embed([text])[0]
    sims = emb @ q
    order = np.argsort(-sims)[: k * 4]
    out, seen = [], set()
    for i in order:
        row = kb.iloc[i]
        key = " ".join(row.body.lower().split())[:200]
        if key in seen:
            continue
        seen.add(key)
        out.append({"ticket_id": row.ticket_id, "similarity": round(float(sims[i]), 4),
                    "language": row.language, "subject": row.subject, "body": row.body,
                    "answer": row.answer, "type": row.type, "queue": row.queue,
                    "priority": row.priority, "tags": row.tags})
        if len(out) == k:
            break
    return out


def build():
    kb = load_kb()
    texts = (kb.subject + "\n" + kb.body).str.strip().tolist()
    t0 = time.time()
    emb = embed(texts, show=True)
    np.save(C.EMB_PATH, emb)
    np.save(C.EMB_IDS_PATH, kb.ticket_id.to_numpy().astype(object))
    print(f"indexed {len(kb)} KB tickets in {time.time() - t0:.0f}s -> {C.EMB_PATH.name}")


if __name__ == "__main__":
    build()
