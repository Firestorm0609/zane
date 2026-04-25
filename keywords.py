"""Keyword model: learns words correlated with pump/rug outcomes."""
import logging
import math
import re
from collections import defaultdict
from contextlib import closing

from .config import ML_LABEL_WINDOW
from .db import db_conn
from .utils import now_ts

log = logging.getLogger(__name__)


class KeywordModel:
    MIN_WORD_SUPPORT = 5

    def __init__(self):
        self._weights: dict[str, float] = {}
        self._base_rate = 0.5
        self._n_samples = 0
        self._updated_at = 0

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r'\b[a-zA-Z]{3,15}\b', text.lower())

    def score(self, text: str) -> float:
        if not self._weights:
            return 0.0
        words = self._tokenize(text)
        if not words:
            return 0.0
        raw = sum(self._weights.get(w, 0.0) for w in words)
        return raw / math.sqrt(len(words))

    def positive_hits(self, text: str) -> int:
        return sum(1 for w in self._tokenize(text) if self._weights.get(w, 0.0) > 0.1)

    def negative_hits(self, text: str) -> int:
        return sum(1 for w in self._tokenize(text) if self._weights.get(w, 0.0) < -0.1)

    def top_keywords(self, n: int = 30, positive_only: bool = False,
                     negative_only: bool = False) -> list[tuple[str, float]]:
        items = list(self._weights.items())
        if positive_only:   items = [(w, v) for w, v in items if v > 0]
        elif negative_only: items = [(w, v) for w, v in items if v < 0]
        return sorted(items, key=lambda x: abs(x[1]), reverse=True)[:n]

    def learn_from_db(self) -> None:
        with closing(db_conn()) as conn:
            rows = conn.execute("""
                SELECT s.description_text, lb.outcome
                FROM signals s
                JOIN lookbacks lb ON lb.signal_id = s.id
                WHERE lb.window_label = ?
                  AND lb.checked = 1
                  AND lb.outcome IS NOT NULL
                  AND s.description_text IS NOT NULL
                  AND s.description_text != ''
            """, (ML_LABEL_WINDOW,)).fetchall()

        if len(rows) < 50:
            return

        pump_words: dict[str, int] = defaultdict(int)
        rug_words:  dict[str, int] = defaultdict(int)
        pump_docs = rug_docs = 0

        for row in rows:
            words = set(self._tokenize(row["description_text"] or ""))
            is_pump = row["outcome"] in ("PUMP", "MOON", "UP")
            if is_pump:
                pump_docs += 1
                for w in words: pump_words[w] += 1
            else:
                rug_docs += 1
                for w in words: rug_words[w] += 1

        if pump_docs == 0 or rug_docs == 0:
            return

        base_rate = pump_docs / (pump_docs + rug_docs)
        all_words = set(pump_words) | set(rug_words)
        weights: dict[str, float] = {}
        for word in all_words:
            pc = pump_words[word]
            rc = rug_words[word]
            total = pc + rc
            if total < self.MIN_WORD_SUPPORT:
                continue
            word_pump_rate = pc / total
            lift = (word_pump_rate - base_rate) / (base_rate + 1e-8)
            weights[word] = max(-2.0, min(2.0, lift))

        self._weights    = weights
        self._base_rate  = base_rate
        self._n_samples  = len(rows)
        self._updated_at = now_ts()
        log.info("KeywordModel: %d words learned | base pump rate %.1f%% | n=%d",
                 len(weights), base_rate * 100, len(rows))

    def status(self) -> dict:
        return {
            "n_words":      len(self._weights),
            "base_rate":    self._base_rate,
            "n_samples":    self._n_samples,
            "updated_at":   self._updated_at,
            "top_positive": self.top_keywords(5, positive_only=True),
            "top_negative": self.top_keywords(5, negative_only=True),
        }
