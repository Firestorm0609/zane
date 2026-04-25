"""Rolling market context for percentile/zscore features."""
import math
from .config import MARKET_CACHE_TTL_SEC, MAX_MARKET_CTX_ENTRIES
from .utils import now_ts


class MarketContext:
    def __init__(self, window_sec: int = 86400):
        self._window = window_sec
        self._mcs: list[tuple[int, float]] = []
        self._replies: list[tuple[int, int]] = []
        self._mc_cache: list[float] = []
        self._reply_cache: list[float] = []
        self._cache_expires = 0.0

    def _refresh_cache(self) -> None:
        import time
        t = time.time()
        if t < self._cache_expires:
            return
        self._mc_cache    = [v for _, v in self._mcs]
        self._reply_cache = [float(v) for _, v in self._replies]
        self._cache_expires = t + MARKET_CACHE_TTL_SEC

    def update(self, mc: float, replies: int, *, track_replies: bool = True) -> None:
        ts = now_ts()
        self._mcs.append((ts, mc))
        if track_replies:
            self._replies.append((ts, replies))
        if len(self._mcs) > MAX_MARKET_CTX_ENTRIES or (
            len(self._mcs) > 0 and len(self._mcs) % 200 == 0
        ):
            self._prune()
            self._cache_expires = 0.0

    def _prune(self) -> None:
        cutoff = now_ts() - self._window
        self._mcs     = [(t, v) for t, v in self._mcs     if t > cutoff]
        self._replies = [(t, v) for t, v in self._replies if t > cutoff]
        if len(self._mcs) > MAX_MARKET_CTX_ENTRIES:
            self._mcs = self._mcs[-MAX_MARKET_CTX_ENTRIES:]
        if len(self._replies) > MAX_MARKET_CTX_ENTRIES:
            self._replies = self._replies[-MAX_MARKET_CTX_ENTRIES:]

    @staticmethod
    def _percentile(vals: list[float], x: float) -> float:
        if not vals:
            return 0.5
        return sum(1 for v in vals if v <= x) / len(vals)

    @staticmethod
    def _zscore(vals: list[float], x: float) -> float:
        if len(vals) < 2:
            return 0.0
        mu  = sum(vals) / len(vals)
        var = sum((v - mu) ** 2 for v in vals) / len(vals)
        sigma = math.sqrt(var) + 1e-8
        return max(-4.0, min(4.0, (x - mu) / sigma))

    def percentile_mc(self, mc: float) -> float:
        self._refresh_cache()
        return self._percentile(self._mc_cache, mc)

    def zscore_mc(self, mc: float) -> float:
        self._refresh_cache()
        return self._zscore(self._mc_cache, mc)

    def percentile_replies(self, replies: int) -> float:
        self._refresh_cache()
        return self._percentile(self._reply_cache, float(replies))

    def summary(self) -> dict:
        mcs  = [v for _, v in self._mcs]
        reps = [v for _, v in self._replies]
        if not mcs:
            return {"samples": 0, "mc_median": 0, "mc_mean": 0, "replies_median": 0}
        sorted_mcs = sorted(mcs)
        n = len(sorted_mcs)
        return {
            "samples":        n,
            "mc_p25":         sorted_mcs[n // 4],
            "mc_median":      sorted_mcs[n // 2],
            "mc_p75":         sorted_mcs[min(3 * n // 4, n - 1)],
            "mc_mean":        sum(mcs) / n,
            "replies_median": sorted(reps)[len(reps) // 2] if reps else 0,
            "window_hours":   self._window // 3600,
        }
