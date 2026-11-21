"""Live outage / burst detection: several recent incoming tickets about the same failure -> banner."""
import time
from collections import deque

import numpy as np

WINDOW_SEC = 30 * 60
SIM = 0.90          # e5 cosine: burst demo pairs 0.91-0.94 (incl. EN<->DE), unrelated tickets <= 0.885
MIN_CLUSTER = 3     # this ticket + 2 similar recent ones


class BurstDetector:
    def __init__(self):
        self.recent = deque(maxlen=500)     # (ts, embedding, ticket summary)

    def add(self, emb: np.ndarray, summary: dict, now: float | None = None) -> dict | None:
        now = now or time.time()
        while self.recent and now - self.recent[0][0] > WINDOW_SEC:
            self.recent.popleft()
        similar = [s for ts, e, s in self.recent if float(e @ emb) >= SIM]
        self.recent.append((now, emb, summary))
        if len(similar) + 1 >= MIN_CLUSTER:
            return {"size": len(similar) + 1, "tickets": similar + [summary]}
        return None

    def reset(self):
        self.recent.clear()
