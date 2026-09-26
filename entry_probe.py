"""entry_probe.py — what did a callout actually cost, at trade time?

The recorded "MC at call" is not an entry anyone gets: on MD's last calls the
first tradeable second was already 1.6-2.2x the call price. That gap — and how
it trends — is the only honest way to judge whether a caller is picking up
traction, and it is also the number an exec path (mirrorbot -> trojan, or our
own buy) has to be compared against.

Measured from the two push streams and nothing else:
  * calloutCreated.> on CORE      -> the moment a callout exists (and its price)
  * unifiedTradeEvent.lite.<mint> -> every trade of that mint, as it happens

Zero HTTP and zero RPC on the hot path, so it cannot 429, and it sees the first
trades instead of reconstructing them from block history later. Each callout
gets a ladder: the first trade (what anyone could actually pay), snapshots at
+1/+5/+15/+30/+60s, and the peak inside the window. One JSONL line per event in
entry_probe.jsonl, so the gap can be trended per caller over time.

    .venv/bin/python entry_probe.py --caller 6qudAN2kV8mtCcYJxb5QQ6Vr15itdHHdeVbYm99NKMhy
    .venv/bin/python entry_probe.py --all --window 60      # platform-wide sample
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from typing import Optional

import aiohttp

import pump_callouts
import pump_nats

log = logging.getLogger("entry_probe")

SUPPLY = 1_000_000_000
STATUS_S = 60.0


async def sol_usd() -> float:
    """One SOL/USD read to turn the stream's SOL prices into the feed's USD.

    Only needed once per run: a call's ladder lasts minutes and a few tenths of
    a percent of drift in the rate cannot change a 2x multiple.
    """
    async with aiohttp.ClientSession() as s:
        for url, pick in (
            ("https://api.binance.com/api/v3/ticker/price?symbol=SOLUSDT",
             lambda d: float(d["price"])),
            ("https://api.coingecko.com/api/v3/simple/price?ids=solana&vs_currencies=usd",
             lambda d: float(d["solana"]["usd"])),
        ):
            try:
                async with s.get(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
                    return pick(await r.json())
            except Exception as e:
                log.warning("sol/usd via %s failed: %s", url.split("/")[2], e)
    log.warning("no SOL/USD available — ladders report multiples only")
    return 0.0


class Ladder:
    """One callout: the reference price plus every trade of its mint."""

    def __init__(self, item: dict, sol: float, out):
        self.mint = item.get("coinMint") or ""
        self.caller = (item.get("userId") or "")[:8]
        self.cid = item.get("calloutId") or ""
        self.name = item.get("userName") or ""
        self.call_mc = float(item.get("marketCap") or 0)
        self.call_usd_price = float(item.get("calloutPriceUsd") or 0)
        self.call_sol_price = (self.call_usd_price / sol) if (sol and self.call_usd_price) else 0.0
        self.created = (item.get("createdAt") or 0) / 1000.0
        self.arrived = time.time()
        self.sol = sol
        self.out = out
        self.trades: list[tuple[float, float, float, bool]] = []   # t, mult, sol, buy
        self.peak = 0.0
        self.peak_t = 0.0
        self.buys = 0
        self.sells = 0
        self.done = False

    def mult(self, price_sol: float) -> float:
        return price_sol / self.call_sol_price if self.call_sol_price else 0.0

    def mc(self, price_sol: float) -> float:
        return price_sol * self.sol * SUPPLY if self.sol else 0.0

    def record(self, price_sol: float, sol_amount: float, is_buy: bool) -> None:
        if self.done:
            return
        t = time.time() - self.created
        m = self.mult(price_sol)
        self.trades.append((t, m, sol_amount, is_buy))
        if is_buy:
            self.buys += 1
        else:
            self.sells += 1
        if m > self.peak:
            self.peak, self.peak_t = m, t
        if len(self.trades) == 1:
            log.info("ENTRY %s %s first trade +%.2fs @ %.2fx (MC $%s, %s, %.3f SOL)",
                     self.mint[:12], self.caller, t, m, f"{self.mc(price_sol):,.0f}",
                     "buy" if is_buy else "sell", sol_amount)
            self.emit({"kind": "first", "t": round(t, 3), "mult": round(m, 3),
                       "call_mc": self.call_mc})

    def at(self, secs: float) -> Optional[float]:
        """Multiple of the trade closest to +secs (so a quiet minute isn't 0)."""
        before = [x for x in self.trades if x[0] <= secs]
        return before[-1][1] if before else None

    def finish(self) -> None:
        if self.done:
            return
        self.done = True
        last = self.trades[-1][1] if self.trades else 0.0
        snaps = {f"+{s:g}s": (round(v, 2) if (v := self.at(s)) else None)
                 for s in (1, 5, 15, 30, 60)}
        log.info("WINDOW %s %s peak %.2fx @ +%.0fs · now %.2fx · %d buys/%d sells",
                 self.mint[:12], self.caller, self.peak, self.peak_t, last,
                 self.buys, self.sells)
        self.emit({"kind": "window", "peak": round(self.peak, 3),
                   "peak_t": round(self.peak_t, 1), "now": round(last, 3),
                   "buys": self.buys, "sells": self.sells, "snaps": snaps})

    def emit(self, extra: dict) -> None:
        row = {"ts": round(time.time(), 3), "mint": self.mint,
               "calloutId": self.cid, "caller": self.caller, "name": self.name,
               "call_mc": self.call_mc, "trades": len(self.trades), **extra}
        self.out.write(json.dumps(row) + "\n")
        self.out.flush()


class Probe:
    def __init__(self, callers: list[str], window: float, cap: int, out):
        self.callers = callers
        self.window = window
        self.cap = cap
        self.out = out
        self.sol = 0.0
        self.ladders: dict[str, Ladder] = {}
        self.seen = 0
        self.trades_client = pump_nats.PumpNats(self.on_trade)
        self.callouts_client = pump_callouts.PumpCallouts(
            wallets=callers, sink=self.sink)

    async def sink(self, caller: str, page: dict) -> dict:
        """pump_callouts hands us matched callouts; start a ladder for each."""
        started = 0
        for item in page.get("callouts") or []:
            mint = item.get("coinMint") or ""
            if not mint or mint.startswith("0x") or len(mint) < 32:
                continue                     # the trade stream is Solana-only
            if mint in self.ladders:
                continue
            if len(self.ladders) >= self.cap:
                oldest = min(self.ladders.values(), key=lambda l: l.arrived)
                oldest.finish()
                del self.ladders[oldest.mint]
            lad = Ladder(item, self.sol, self.out)
            self.ladders[mint] = lad
            self.seen += 1
            self.trades_client.set_wanted(set(self.ladders))
            asyncio.create_task(self._retire(lad))
            started += 1
        return {"ok": True, "new": started}

    async def _retire(self, lad: Ladder) -> None:
        await asyncio.sleep(self.window)
        lad.finish()
        if self.ladders.get(lad.mint) is lad:
            del self.ladders[lad.mint]
            self.trades_client.set_wanted(set(self.ladders))

    def on_trade(self, mint: str, price_sol: float, sol_amount: float,
                 is_buy: bool) -> None:
        lad = self.ladders.get(mint)
        if lad:
            lad.record(price_sol, sol_amount, is_buy)

    async def heartbeat(self) -> None:
        while True:
            await asyncio.sleep(STATUS_S)
            log.info("probe: %d callouts seen · %d live ladders · callout %s · trades %s",
                     self.seen, len(self.ladders),
                     self.callouts_client.status(), self.trades_client.status())

    async def run(self) -> None:
        self.sol = await sol_usd()
        log.info("entry probe: SOL/USD %.2f · window %.0fs · callers %s",
                 self.sol, self.window,
                 ", ".join(c[:8] for c in self.callers) or "none")
        await asyncio.gather(self.callouts_client.run(), self.trades_client.run(),
                             self.heartbeat())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--caller", action="append", default=[],
                    help="caller wallet/uuid to watch (repeatable)")
    ap.add_argument("--all", action="store_true",
                    help="every callout on the platform (a sample, not a filter)")
    ap.add_argument("--window", type=float, default=180.0,
                    help="seconds to keep trading each mint (default 180)")
    ap.add_argument("--cap", type=int, default=6,
                    help="max mints subscribed at once (default 6)")
    ap.add_argument("--out", default="entry_probe.jsonl")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    callers = ["*"] if args.all else args.caller
    if not callers:
        ap.error("need --caller <addr> (repeatable) or --all")
    with open(args.out, "a", encoding="utf-8") as out:
        asyncio.run(Probe(callers, args.window, args.cap, out).run())


if __name__ == "__main__":
    main()
