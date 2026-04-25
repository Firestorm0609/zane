"""Token-bucket rate limiter and shared instances."""
import asyncio
import time

from .config import RPC_RATE_PER_SEC


class RateLimiter:
    def __init__(self, rate: float, per: float = 1.0):
        self._rate = rate
        self._per  = per
        self._tokens = rate
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(
                self._rate,
                self._tokens + elapsed * (self._rate / self._per)
            )
            self._last_refill = now
            if self._tokens < 1.0:
                wait = (1.0 - self._tokens) * (self._per / self._rate)
                await asyncio.sleep(wait)
                self._tokens = 0.0
            else:
                self._tokens -= 1.0


pump_api_limiter = RateLimiter(rate=10, per=1.0)
rpc_limiter      = RateLimiter(rate=RPC_RATE_PER_SEC, per=1.0)
