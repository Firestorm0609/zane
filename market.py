"""Rolling market context for percentile/zscore features."""
import bisect
import math
import time as _time

from .config import MARKET_CACHE_TTL_SEC, MAX_MARKET_CTX_ENTRIES
from .utils import now_ts


class MarketContext:
    """Tracks rolling market-cap and reply-count distributions.

    Percentile lookups use bisect on a sorted shadow list — O(log n)
    instead of the previous O(n) linear scan.  The sorted lists are
    rebuilt only when the cache expires or after a prune, not on every
    call.
    """

    def __init__(self, window_sec: int = 86400):
        self._window = window_sec

        # Raw time-series: list of (timestamp, value)
        self._mcs: list[tuple[int, float]] = []
        self._replies: list[tuple[int, int]] = []

        # Cache: flat value lists rebuilt on expiry
        self._mc_cache: list[float] = []
        self._mc_sorted: list[float] = []   # always sorted — for bisect
        self._reply_cache: list[float] = []
        self._reply_sorted: list[float] = []
        self._cache_expires = 0.0

    def _refresh_cache(self) -> None:
        t = _time.time()
        if t < self._cache_expires:
            return
        self._mc_cache     = [v for _, v in self._mcs]
        self._mc_sorted    = sorted(self._mc_cache)
        self._reply_cache  = [float(v) for _, v in self._replies]
        self._reply_sorted = sorted(self._reply_cache)
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
            self._cache_expires = 0.0   # force cache rebuild after prune

    def _prune(self) -> None:
        cutoff = now_ts() - self._window
        self._mcs     = [(t, v) for t, v in self._mcs     if t > cutoff]
        self._replies = [(t, v) for t, v in self._replies if t > cutoff]
        if len(self._mcs) > MAX_MARKET_CTX_ENTRIES:
            self._mcs = self._mcs[-MAX_MARKET_CTX_ENTRIES:]
        if len(self._replies) > MAX_MARKET_CTX_ENTRIES:
            self._replies = self._replies[-MAX_MARKET_CTX_ENTRIES:]

    @staticmethod
    def _percentile_sorted(sorted_vals: list[float], x: float) -> float:
        """O(log n) percentile using a pre-sorted list and bisect."""
        n = len(sorted_vals)
        if n == 0:
            return 0.5
        return bisect.bisect_right(sorted_vals, x) / n

    @staticmethod
    def _zscore(vals: list[float], x: float) -> float:
        if len(vals) < 2:
            return 0.0
        mu    = sum(vals) / len(vals)
        var   = sum((v - mu) ** 2 for v in vals) / len(vals)
        sigma = math.sqrt(var) + 1e-8
        return max(-4.0, min(4.0, (x - mu) / sigma))

    def percentile_mc(self, mc: float) -> float:
        self._refresh_cache()
        return self._percentile_sorted(self._mc_sorted, mc)

    def zscore_mc(self, mc: float) -> float:
        self._refresh_cache()
        return self._zscore(self._mc_cache, mc)

    def percentile_replies(self, replies: int) -> float:
        self._refresh_cache()
        return self._percentile_sorted(self._reply_sorted, float(replies))

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

