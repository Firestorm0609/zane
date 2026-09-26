"""Real-time pump.fun callout stream (NATS over WebSocket) — the push path.

Why this exists: the HTTP /callout/list endpoint serves a callout ~10s after it
was created (p90 ~29s, worst 30s — measured over every logged callout). That is
not a polling problem and never was: polling faster cannot fetch a record the
endpoint hasn't published yet. The phone app knows instantly because it doesn't
poll at all — it subscribes to pump.fun's own NATS firehose and receives the
callout the moment it is written. Our HTTP poller was simply not listening.

Measured created->us on this path: median 0.05s, worst 0.30s, and ZERO requests
against the HTTP allowance (so it cannot 429, and the multi-mirror rotation
becomes a safety net instead of the mechanism).

Details, all verified against the live site:
  * endpoint:    wss://prod-v2.nats.realtime.pump.fun/    (the app's CORE cluster)
  * credentials: the anonymous read-only "subscriber" account pump.fun ships to
                 every visitor in the page's runtime config. Discovered on every
                 connect (pump.fun can rotate it), never hard-coded.
  * subject:     calloutCreated.<mint>.<chainId>   — platform-wide, ~50/min, so
                 the filter (tracked caller) is applied client-side.
  * payload:     one JSON object. The at-call numbers are nested TWO levels deep
                 (the outer callout is the caller's position record):
                   d.author.walletAddress / d.author.userId  -> who called
                   d.coin.mint / d.coin.chainId              -> what
                   d.callout.callout.calloutId               -> id (dedupe key)
                   d.callout.callout.calloutMarketCap        -> MC AT CALL
                   d.callout.callout.multiplier              -> run-up since
                   d.callout.callout.thesis                  -> the thesis

Deliberately normalized to the exact /callout/list item shape and handed to the
SAME Poller.ingest_external() the HTTP poller and the browser relay already use.
Shared cursor, shared dedupe, shared fanout — so a pushed callout can never
double-alert, and nothing downstream needs to know which path found it.

Hand-rolled NATS protocol (INFO / CONNECT / SUB / MSG / PING / PONG) on top of
aiohttp, same as pump_nats.py — no new dependency.

The HTTP poller stays on as the safety net: if this socket drops, ingest stops
refreshing and normal polling resumes by itself (see RELAY_TRUST_S).
"""
import asyncio
import json
import logging
import re
import time
from collections import deque
from datetime import datetime
from typing import Any, Awaitable, Callable, Optional

import aiohttp

import config

log = logging.getLogger("pump_callouts")

# Must match what the site's own client sends, field for field: the server
# rejects a CONNECT that omits headers/no_responders even when the anonymous
# credentials are correct.
CONNECT_TEMPLATE = (
    'CONNECT {"verbose":false,"pedantic":false,"tls_required":false,"name":"",'
    '"lang":"pumpfun-worker","version":"1","protocol":1,"headers":true,'
    '"no_responders":true,"user":"%s","pass":"%s"}'
)

RECONNECT_MIN_S = 2.0
RECONNECT_MAX_S = 60.0
KEEPALIVE_S = 20.0          # idle PING keeps the edge from culling the socket
CALLER_REFRESH_S = 30.0     # tracked-caller cache; a new /track starts matching
LATENCY_KEEP = 20


def _text(msg: Any) -> str:
    data = getattr(msg, "data", msg)
    if isinstance(data, str):
        return data
    if isinstance(data, (bytes, bytearray)):
        return data.decode("utf-8", "replace")
    return ""


async def discover_credentials(timeout: float = 15.0
                               ) -> Optional[tuple[str, str]]:
    """Re-scrape (url, password) for the CORE cluster from the live site.

    The password is served to every visitor and can rotate at any time, so it is
    re-read on every connect instead of being pinned in config. CORE matters:
    the app subscribes there (NatsInstance.Core = prod-v2), and it is the only
    cluster that carries the callout stream.
    """
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get("https://pump.fun/",
                             headers={"User-Agent": "Mozilla/5.0 Chrome/131"},
                             timeout=aiohttp.ClientTimeout(total=timeout)) as r:
                html = (await r.text()).replace('\\"', '"')
    except Exception as e:
        log.warning("push: nats credential discovery failed: %s", e)
        return None
    i = html.find('"CORE":{')
    if i < 0:
        log.warning("push: CORE cluster missing from the runtime config")
        return None
    seg = html[i:i + 800]
    url = re.search(r'"servers":"([^"]+)"', seg)
    pw = re.search(r'"pass":"([^"]+)"', seg)
    if not (url and pw):
        log.warning("push: could not read CORE url/pass from the runtime config")
        return None
    return url.group(1).rstrip("/") + "/", pw.group(1)


def _iso_ms(value: Any) -> int:
    """'2026-09-26T06:05:33.266Z' -> epoch ms (0 if unusable)."""
    if not isinstance(value, str):
        return 0
    try:
        return int(datetime.fromisoformat(value.replace("Z", "+00:00"))
                   .timestamp() * 1000)
    except ValueError:
        return 0


def normalize(d: dict) -> Optional[dict]:
    """One push event -> one /callout/list item (identical field names).

    Reads the nested at-call record, not the outer position record: only
    d.callout.callout carries calloutMarketCap (the 18k, not the later 34k),
    multiplier (the run-up) and the thesis.
    """
    if not isinstance(d, dict):
        return None
    author = d.get("author") or {}
    coin = d.get("coin") or {}
    outer = d.get("callout") or {}
    inner = outer.get("callout") or {}
    cid = inner.get("calloutId") or d.get("id") or ""
    mint = coin.get("mint") or outer.get("coinMint") or ""
    if not cid or not mint:
        return None
    created = inner.get("createdAt")
    if not isinstance(created, (int, float)) or created <= 0:
        created = _iso_ms(d.get("createdAt"))
    # The push payload prices in USD; the HTTP item splits SOL and USD. Keep the
    # USD value where the HTTP item puts it and leave the SOL field at 0 rather
    # than mislabelling one as the other (derive_mcap_usd reads that field as
    # SOL, and a wrong unit there is a wrong mcap filter).
    usd_price = inner.get("calloutPrice") or 0.0
    mult = inner.get("multiplier") or 1.0
    return {
        "calloutId": cid,
        "userId": author.get("walletAddress") or outer.get("walletAddress") or "",
        "user_uuid": author.get("userId") or "",
        "userName": author.get("userName") or "",
        "profileImage": author.get("profileImage") or "",
        "coinMint": mint,
        "marketCap": float(inner.get("calloutMarketCap") or 0.0),
        "calloutPrice": 0.0,
        "calloutPriceUsd": float(usd_price or 0.0),
        "multiple": float(mult or 1.0),
        "maxMultiplier": float(inner.get("maxMultiplier") or mult or 1.0),
        "thesis": inner.get("thesis") or "",
        "createdAt": int(created or 0),
        "chainId": coin.get("chainId"),
        "calloutType": outer.get("type") or "",
        # not present on the push payload; format_callout omits them when None
        # rather than printing a fake "0 likes"
        "likes": None,
        "viewCount": None,
        "updates": [],
        "updateCount": 0,
        "source": "push",
    }


class PumpCallouts:
    """Subscribes calloutCreated.> and feeds matched callouts to the ingest sink.

    ``sink(caller_id, page)`` is Poller.ingest_external — the same entry point
    the HTTP poller and the browser relay use. Runs forever: any socket failure
    reconnects with backoff, and an auth rejection re-discovers the credentials
    first (the anonymous password can rotate).
    """

    def __init__(self, db: Any = None,
                 sink: Optional[Callable[[str, dict], Awaitable[dict]]] = None,
                 wallets: Optional[list[str]] = None,
                 url: Optional[str] = None,
                 password: Optional[str] = None,
                 subject: Optional[str] = None):
        self.db = db
        self.sink = sink
        self.url = url or config.PUMP_PUSH_NATS_URL or None
        self.password = password or config.PUMP_PUSH_NATS_PASS or None
        self.subject = subject or config.PUMP_PUSH_SUBJECT
        self.connected = False
        self.events = 0            # platform-wide, unfiltered
        self.matched = 0           # from a tracked caller -> handed to the sink
        self.last_event_ts = 0.0
        self.last_match_ts = 0.0
        self.reconnects = 0
        self._latencies: deque[float] = deque(maxlen=LATENCY_KEEP)
        self._static: Optional[list[str]] = wallets
        self._callers: dict[str, str] = {}
        self._callers_ts = 0.0
        self._tasks: set[asyncio.Task] = set()
        self._stop = False

    # ---- tracked-caller filter ----
    async def _tracked(self) -> dict[str, str]:
        """{wallet-or-uuid -> stored caller_id} for everything enabled.

        Matching on the stored id (not the wallet) is what makes the sink call
        correct for both storage styles: wallet ids match
        author.walletAddress, uuid ids match author.userId.
        """
        if self._static is not None:
            return {w: w for w in self._static}
        now = time.time()
        if self._callers and now - self._callers_ts < CALLER_REFRESH_S:
            return self._callers
        try:
            rows = await self.db.get_callers()
        except Exception:
            log.exception("push: caller refresh failed")
            return self._callers
        keys = {str(r.get("caller_id") or "").strip(): str(r.get("caller_id") or "").strip()
                for r in rows}
        keys.pop("", None)
        if set(keys) != set(self._callers):
            log.info("push filter: tracking %d caller(s): %s", len(keys),
                     ", ".join(sorted(k[:8] for k in keys)) or "none")
        self._callers, self._callers_ts = keys, now
        return keys

    # ---- dispatch ----
    async def _handle(self, d: dict) -> bool:
        """True if this event was from a tracked caller and reached the sink."""
        author = d.get("author") or {}
        keys = await self._tracked()
        caller = keys.get(author.get("walletAddress") or "")
        if not caller:
            caller = keys.get(author.get("userId") or "")
        if not caller:
            if "*" not in keys:      # dry-run --all
                return False
            caller = author.get("walletAddress") or author.get("userId") or "?"
        item = normalize(d)
        if not item:
            log.warning("push: unparseable callout event id=%s", str(d.get("id"))[:12])
            return False
        latency = max(0.0, time.time() - item["createdAt"] / 1000.0)
        self._latencies.append(latency)
        self.matched += 1
        self.last_match_ts = time.time()
        if self.sink is None:
            log.info("push match caller=%s mint=%s mc=$%.0f in %.2fs (%s) type=%s",
                     caller[:8], item["coinMint"][:12], item["marketCap"],
                     latency, item["calloutType"] or "callout",
                     item["calloutId"][:8])
            return True
        try:
            res = await self.sink(caller, {"callouts": [item]})
        except Exception:
            log.exception("push: ingest failed caller=%s", caller[:8])
            return False
        # one line per real callout: the source tag is what makes the push-vs-
        # HTTP comparison readable in the log without guessing
        log.info("push %s caller=%s mint=%s mc=$%.0f posted %.2fs ago "
                 "(ingest ok=%s new=%s)",
                 item["calloutId"][:12], caller[:8], item["coinMint"][:12],
                 item["marketCap"], latency, (res or {}).get("ok"),
                 (res or {}).get("new"))
        return True

    def _dispatch(self, frame: str) -> None:
        _head, _, payload = frame.partition("\r\n")
        try:
            d = json.loads(payload)
        except Exception:
            return
        self.events += 1
        self.last_event_ts = time.time()
        # keep a reference: a task with no strong ref can be garbage-collected
        # mid-await, and the filter/sink work is not instant (one DB read)
        task = asyncio.create_task(self._handle(d))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def status(self) -> str:
        if not self.connected:
            return "disconnected"
        avg = (sum(self._latencies) / len(self._latencies)) if self._latencies else 0.0
        age = time.time() - self.last_match_ts if self.last_match_ts else -1
        return (f"connected · {self.events} events · {self.matched} matched"
                + (f" · avg push latency {avg:.2f}s" if self._latencies else "")
                + (f" · last match {age:.0f}s ago" if age >= 0 else ""))

    # ---- connection ----
    async def run(self) -> None:
        """Connect forever with backoff. Never raises."""
        backoff = RECONNECT_MIN_S
        while not self._stop:
            try:
                await self._session()
                backoff = RECONNECT_MIN_S
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.connected = False
                log.warning("push stream down (%s) — retrying in %.0fs",
                            str(e)[:120], backoff)
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
        url, password = self.url, self.password
        if not (url and password):
            creds = await discover_credentials()
            if not creds:
                raise RuntimeError("no CORE nats credentials discovered")
            url, password = creds
            self.url, self.password = creds
        async with aiohttp.ClientSession() as s:
            async with s.ws_connect(url, heartbeat=None, max_msg_size=0) as ws:
                await asyncio.wait_for(ws.receive(), 10)      # INFO
                await ws.send_str(
                    (CONNECT_TEMPLATE % ("subscriber", password)) + "\r\n")
                await ws.send_str(f"SUB {self.subject} 1\r\n")
                self.connected = True
                self.reconnects += 1
                log.info("push stream connected (%s) — SUB %s", url, self.subject)
                last_beat = time.time()
                while True:
                    try:
                        frame = _text(await asyncio.wait_for(ws.receive(), 5))
                    except asyncio.TimeoutError:
                        if time.time() - last_beat >= KEEPALIVE_S:
                            last_beat = time.time()
                            await ws.send_str("PING\r\n")
                            # a silent socket is THE failure mode here: report it
                            # rather than looking alive while seeing nothing
                            if (self.events and time.time() - self.last_event_ts > 120):
                                log.warning("push: no events for %.0fs — link "
                                            "may be stale, reconnecting",
                                            time.time() - self.last_event_ts)
                                raise RuntimeError("push stream went quiet")
                        continue
                    if not frame:
                        continue
                    if frame.startswith("MSG"):
                        self._dispatch(frame)
                    elif frame.startswith("PING"):
                        await ws.send_str("PONG\r\n")
                    elif "-ERR" in frame:
                        raise RuntimeError(frame.strip()[:160])

    def stop(self) -> None:
        self._stop = True


async def _dry_run(wallets: list[str]) -> None:
    """`python3 pump_callouts.py --dry-run --wallet <addr|*>`

    Exercises discovery, connect, SUB, filter and normalize against the LIVE
    stream, printing the exact page that would be handed to ingest_external —
    without touching the bot, the DB, or anyone's Telegram.
    """
    client = PumpCallouts(wallets=wallets or ["*"])
    log.info("dry run: watching %s", ", ".join(client._static or []))
    try:
        await client.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="pump.fun callout push stream")
    ap.add_argument("--dry-run", action="store_true",
                    help="print matched callouts instead of ingesting them")
    ap.add_argument("--wallet", action="append", default=[],
                    help="wallet (or uuid) to match; '*' matches every callout")
    args = ap.parse_args()
    asyncio.run(_dry_run(args.wallet))
