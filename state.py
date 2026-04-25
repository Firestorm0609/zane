"""BotState (in-memory) and BlacklistCache."""
import asyncio
import logging
import threading
import time
from collections import OrderedDict, deque
from contextlib import closing

from .config import (
    BLACKLIST_CACHE_TTL_SEC, MAX_GRADUATED_ENTRIES, MAX_SEEN_ENTRIES,
    SEEN_TTL_SEC,
)
from .db import db_conn

log = logging.getLogger(__name__)


class BotState:
    def __init__(self, paper_enabled: bool = False):
        self._seen: "OrderedDict[str, float]" = OrderedDict()
        self._seen_lock = asyncio.Lock()
        self._last_prune = 0.0
        self.alerts: dict[int, int] = {}
        self.paper_chats: set[int] = set()
        self._paper_on = paper_enabled
        self._pe_lock = asyncio.Lock()
        self.last_coin_ts = time.time()
        self.stream_dead_alerted = False
        self.stream_dead_alert_at = 0.0
        self._graduated_order: deque = deque(maxlen=MAX_GRADUATED_ENTRIES)
        self._graduated: set[str] = set()
        self._graduated_lock = asyncio.Lock()

    @property
    def paper_enabled(self) -> bool:
        return self._paper_on

    async def set_paper_enabled(self, val: bool) -> None:
        async with self._pe_lock:
            self._paper_on = val

    async def seen_recently(self, mint: str) -> bool:
        async with self._seen_lock:
            t = time.time()
            self._prune_seen_locked(t)
            return mint in self._seen and (t - self._seen[mint]) < SEEN_TTL_SEC

    async def mark_seen(self, mint: str) -> None:
        async with self._seen_lock:
            self._seen[mint] = time.time()
            self._seen.move_to_end(mint)

    def _prune_seen_locked(self, t: float) -> None:
        if t - self._last_prune < 60:
            return
        self._last_prune = t
        cutoff = t - SEEN_TTL_SEC
        while self._seen:
            mint, ts = next(iter(self._seen.items()))
            if ts < cutoff:
                self._seen.popitem(last=False)
            else:
                break
        while len(self._seen) > MAX_SEEN_ENTRIES:
            self._seen.popitem(last=False)

    async def add_graduated(self, mint: str) -> bool:
        """Return True if newly added (not duplicate). FIFO eviction."""
        async with self._graduated_lock:
            if mint in self._graduated:
                return False
            # BUG FIX: original code peeked deque[0] before append, but maxlen
            # eviction happens during append, so set was getting out of sync
            # when capacity was exactly hit. Use len check + manual evict.
            if len(self._graduated_order) >= MAX_GRADUATED_ENTRIES:
                oldest = self._graduated_order.popleft()
                self._graduated.discard(oldest)
            self._graduated_order.append(mint)
            self._graduated.add(mint)
            return True

    def load(self) -> None:
        self.alerts.clear()
        self.paper_chats.clear()
        with closing(db_conn()) as conn:
            for r in conn.execute("SELECT * FROM chat_settings").fetchall():
                cid = int(r["chat_id"])
                if int(r["alerts_enabled"]) == 1:
                    self.alerts[cid] = int(r["threshold"])
                if int(r["paper_reports_enabled"]) == 1:
                    self.paper_chats.add(cid)


class BlacklistCache:
    """Thread-safe in-memory creator blacklist cache."""

    def __init__(self, ttl: int = BLACKLIST_CACHE_TTL_SEC):
        self._set: set[str] = set()
        self._expires: float = 0.0
        self._lock = threading.Lock()
        self._ttl = ttl

    def _refresh_locked(self) -> None:
        try:
            with closing(db_conn()) as conn:
                rows = conn.execute("SELECT creator FROM creator_blacklist").fetchall()
            self._set = {r["creator"] for r in rows if r["creator"]}
            self._expires = time.time() + self._ttl
        except Exception as e:
            log.warning("BlacklistCache refresh failed: %s", e)
            self._expires = time.time() + 10

    def contains(self, creator: str) -> bool:
        if not creator:
            return False
        with self._lock:
            if time.time() >= self._expires:
                self._refresh_locked()
            return creator in self._set

    def invalidate(self) -> None:
        with self._lock:
            self._expires = 0.0


blacklist_cache = BlacklistCache()


def is_creator_blacklisted(creator: str) -> bool:
    return blacklist_cache.contains(creator)
