"""Real-time pump.fun trade stream (NATS over WebSocket).

Why this exists: the exit engine polls every 20s, and a memecoin can give back
its entire peak inside one cycle — a trailing stop that only looks every 20s
is a trailing stop in name only. pump.fun pushes every trade to a per-mint
NATS subject, so we can react in well under a second.

Details, all verified against the live site:
  * endpoint:      wss://unified-prod.nats.realtime.pump.fun/
  * credentials:   the public read-only "subscriber" account that the site
                   ships to every anonymous visitor
  * subject:       unifiedTradeEvent.lite.<mint>  — PER MINT; wildcards are
                   rejected server-side ("Authorization Violation"), so we
                   subscribe only to mints we actually hold
  * payload:       a JSON *string* containing JSON:
                   {mint, tx, slotIndexId, userAddress, type: buy|sell,
                    baseAmount (tokens), amountSol, amountUsd, priceUsd, ...}
                   → price in SOL = amountSol / baseAmount

Hand-rolled NATS protocol (INFO / CONNECT / SUB / MSG / PING / PONG) on top of
aiohttp — no nats-py dependency.

This is an exit accelerator, not a replacement: the 20s poll loop stays as the
safety net for anything the stream can't cover (non-pump mints, mints that
migrated off-program, a dropped socket).
"""
import asyncio
import json
import logging
import re
import time
from typing import Any, Callable, Iterable, Optional

import aiohttp

import config

log = logging.getLogger("pump_nats")

# Must match what the site's own client sends, field for field. The server
# rejects a CONNECT that omits headers/no_responders with
# "-ERR 'Authorization Violation'" even when the credentials are right.
CONNECT_TEMPLATE = (
    'CONNECT {"verbose":false,"pedantic":false,"tls_required":false,"name":"",'
    '"lang":"pumpfun-worker","version":"1","protocol":1,"headers":true,'
    '"no_responders":true,"user":"%s","pass":"%s"}'
)

RECONNECT_MIN_S = 2.0
RECONNECT_MAX_S = 60.0
# How long the socket may sit silent before we send our own keepalive PING. The
# read loop wakes every 0.25s regardless (see _session), so this is a timer, not
# a wakeup interval.
IDLE_PING_S = 15.0


def _text(msg: Any) -> str:
    data = getattr(msg, "data", msg)
    if isinstance(data, str):
        return data
    if isinstance(data, (bytes, bytearray)):
        return data.decode("utf-8", "replace")
    return ""


async def discover_credentials(timeout: float = 15.0) -> Optional[tuple[str, str]]:
    """Re-scrape (url, password) from pump.fun's injected runtime config.

    The password is served to every visitor and could rotate at any time, so on
    an auth failure we re-read it instead of hard-coding one forever.
    """
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get("https://pump.fun/",
                             headers={"User-Agent": "Mozilla/5.0"},
                             timeout=aiohttp.ClientTimeout(total=timeout)) as r:
                html = (await r.text()).replace('\\"', '"')
    except Exception as e:
        log.warning("nats credential discovery failed: %s", e)
        return None
    for name in ("UNIFIED", "CORE", "ADVANCED"):
        i = html.find(f'"{name}":{{')
        if i < 0:
            continue
        seg = html[i:i + 600]
        url = re.search(r'"servers":"([^"]+)"', seg)
        pw = re.search(r'"pass":"([^"]+)"', seg)
        if url and pw:
            found = (url.group(1).rstrip("/") + "/", pw.group(1))
            log.info("nats credentials discovered (%s)", name)
            return found
    return None


class PumpNats:
    """Subscribes to per-mint trade subjects and calls back with each trade.

    ``on_trade(mint, price_sol, sol_amount, is_buy)`` is invoked synchronously
    and must be cheap — it runs inside the socket read loop.
    """

    def __init__(self, on_trade: Callable[..., None],
                 url: Optional[str] = None, password: Optional[str] = None):
        self.on_trade = on_trade
        self.url = url or config.RT_NATS_URL
        self.password = password or config.RT_NATS_PASS
        self.connected = False
        self.trades = 0
        self.last_trade_ts = 0.0
        self.reconnects = 0
        self._wanted: set[str] = set()
        self._subbed: dict[str, str] = {}     # mint -> sid
        self._next_sid = 1
        self._stop = False

    # ---- subscription bookkeeping (called from the sync loop) ----
    def set_wanted(self, mints: Iterable[str]) -> None:
        self._wanted = {m for m in mints if m}

    def status(self) -> str:
        if not self.connected:
            return "disconnected"
        age = time.time() - self.last_trade_ts if self.last_trade_ts else -1
        return (f"connected · {len(self._subbed)} subscribed · {self.trades} trades"
                + (f" · last {age:.0f}s ago" if age >= 0 else ""))

    async def run(self) -> None:
        """Connect forever, reconnecting with backoff. Never raises."""
        backoff = RECONNECT_MIN_S
        while not self._stop:
            try:
                await self._session()
                backoff = RECONNECT_MIN_S
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.connected = False
                log.warning("nats stream down (%s) — retrying in %.0fs",
                            str(e)[:120], backoff)
                # a rejected password is the one failure we can self-heal
                if "Authorization" in str(e):
                    creds = await discover_credentials()
                    if creds:
                        self.url, self.password = creds
                        backoff = RECONNECT_MIN_S
                        continue
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, RECONNECT_MAX_S)
                continue
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, RECONNECT_MAX_S)

    async def _session(self) -> None:
        async with aiohttp.ClientSession() as s:
            async with s.ws_connect(self.url, heartbeat=None, max_msg_size=0) as ws:
                await asyncio.wait_for(ws.receive(), 10)          # INFO
                await ws.send_str((CONNECT_TEMPLATE % ("subscriber", self.password)) + "\r\n")
                self.connected = True
                self.reconnects += 1
                self._subbed.clear()
                log.info("nats connected (%s) — subscribed to %d held mint(s)",
                         self.url, len(self._wanted))
                await self._sync_subs(ws)
                last_ping = time.time()
                while True:
                    # Sync BEFORE parking the loop. _wanted is set by other
                    # tasks (exits.py, entry_probe.py), and _sync_subs used to
                    # be reachable only from the idle timeout below — a busy
                    # stream never times out, so a new mint's SUB waited for the
                    # first lull, which is backwards: the mint gets added
                    # because it is hot. Checking every iteration takes frame
                    # pacing out of it entirely.
                    if self._wanted != set(self._subbed):
                        await self._sync_subs(ws)
                        continue
                    try:
                        # Still a short wait: a fresh mint's own stream is empty
                        # by definition, so nothing on this socket may wake us,
                        # and set_wanted() can land at any moment. This bounds
                        # that case to 0.25s; the check above bounds the busy
                        # one to a single frame.
                        frame = _text(await asyncio.wait_for(ws.receive(), 0.25))
                    except asyncio.TimeoutError:
                        if time.time() - last_ping >= IDLE_PING_S:
                            await ws.send_str("PING\r\n")   # keepalive
                            last_ping = time.time()
                        continue
                    if not frame:
                        continue
                    if frame.startswith("MSG"):
                        self._dispatch(frame)
                    elif frame.startswith("PING"):
                        await ws.send_str("PONG\r\n")   # server keepalive
                    elif "-ERR" in frame:
                        raise RuntimeError(frame.strip()[:160])

    async def _sync_subs(self, ws) -> None:
        """SUB new held mints, UNSUB ones we no longer hold."""
        for mint in self._wanted - set(self._subbed):
            sid = str(self._next_sid)
            self._next_sid += 1
            self._subbed[mint] = sid
            await ws.send_str(f"SUB unifiedTradeEvent.lite.{mint} {sid}\r\n")
            log.info("nats subscribed mint=%s", mint[:12])
        for mint in set(self._subbed) - self._wanted:
            await ws.send_str(f"UNSUB {self._subbed.pop(mint)}\r\n")
            log.info("nats unsubscribed mint=%s", mint[:12])

    def _dispatch(self, frame: str) -> None:
        head, _, payload = frame.partition("\r\n")
        parts = head.split()
        if len(parts) < 3:
            return
        subject = parts[1]
        try:
            data = json.loads(payload)
            if isinstance(data, str):        # the payload is a JSON-encoded string
                data = json.loads(data)
        except Exception:
            return
        if not isinstance(data, dict):
            return
        try:
            base = float(data.get("baseAmount") or 0)      # tokens
            sol = float(data.get("amountSol") or 0)        # SOL paid/received
        except (TypeError, ValueError):
            return
        if base <= 0:
            return
        self.trades += 1
        self.last_trade_ts = time.time()
        mint = data.get("mint") or subject.rsplit(".", 1)[-1]
        try:
            self.on_trade(mint, sol / base, sol, data.get("type") == "buy")
        except Exception:
            log.exception("on_trade handler failed mint=%s", mint[:12])
