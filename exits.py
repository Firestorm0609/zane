"""Exit engine: auto take-profit and stop-loss for open positions.

Runs every ~20s over all open positions. Price source:
- pump.fun tokens: PumpPortal trade-local is price-only... so we use Jupiter
  quotes for everything (uniform, works for pump + off-platform mints).
- Positions record entry SOL amount; TP/SL are evaluated on current SOL value
  of the held tokens vs entry cost.
"""
import asyncio
import json
import logging
import os
import time
from typing import Any, Optional

import aiohttp

import config
import jupiter
import pump_nats
import trader
from db import DB

log = logging.getLogger("exits")

# per-user exit config stored on the caller row: max_multiple = take-profit (x),
# stop_multiple = stop-loss (x, expressed as multiple, e.g. 0.5 = -50%)
DEFAULT_TP_X = 2.0
DEFAULT_SL_X = 0.5
CHECK_INTERVAL_S = 20

# A trailing stop only arms once the position has actually run — otherwise a
# dip from entry would tighten the stop above the configured SL.
TRAIL_ARM_X = float(os.environ.get("TRAIL_ARM_X", "1.2"))
# A zero balance usually means "the buy tx hasn't landed yet", not "we hold
# nothing" — so it's re-checked quickly instead of being cached for a full TTL.
# Caching it for RT_TOKEN_TTL_S would blind the realtime path for the first
# minute of a trade, which is exactly when a fresh snipe can double.
ZERO_BAL_TTL_S = 5.0
# Where the breakeven stop actually sits once armed. Exiting at exactly 1.0x is
# still a NET LOSS — pump.fun charges ~1% each way plus the priority fee on both
# legs, so a 0.1 SOL buy with a 0.003 fee needs ~1.08x just to break even.
BE_STOP_X = float(os.environ.get("BE_STOP_X", "1.0"))


def caller_rules(caller: Optional[dict]) -> tuple[float, float, float, float]:
    """(tp_x, sl_x, trail_pct, be_x) for a caller row.

    Stored semantics: None/0 = use the default, negative = disabled,
    positive = that value. Shared by the poll loop and the realtime path so
    both always agree on what the rules are.
    """
    c = caller or {}
    tp_raw, sl_raw = c.get("max_multiple"), c.get("stop_multiple")
    tp_x = tp_raw if (tp_raw and tp_raw > 0) else (DEFAULT_TP_X if not tp_raw else 0)
    sl_x = sl_raw if (sl_raw and sl_raw > 0) else (DEFAULT_SL_X if not sl_raw else 0)
    return tp_x, sl_x, float(c.get("trail_pct") or 0), float(c.get("be_multiple") or 0)


def evaluate_exit(multiple: float, peak: float, *, tp_x: float, sl_x: float,
                  trail_pct: float = 0.0, be_x: float = 0.0,
                  tp_done: bool = False) -> tuple[str, str]:
    """Decide what an open position should do at `multiple` given its `peak`.

    Precedence: take-profit first, then the tightest armed stop out of
      * the hard stop-loss (sl_x),
      * breakeven — 1.0x once the position has reached be_x,
      * a trailing stop trail_pct% below the peak once it armed.
    Returns (action, detail) where action is "" when the position is held.
    """
    peak = max(peak, multiple)

    if tp_x > 0 and multiple >= tp_x and not tp_done:
        return "take-profit", f"{multiple:.2f}x ≥ TP {tp_x}x"

    stop, reason = (sl_x, "stop-loss") if sl_x > 0 else (0.0, "")
    if be_x > 0 and peak >= be_x and BE_STOP_X > stop:
        stop, reason = BE_STOP_X, "breakeven"
    if trail_pct > 0 and peak >= TRAIL_ARM_X:
        trail = peak * (1 - trail_pct / 100.0)
        if trail > stop:
            stop, reason = trail, "trailing-stop"

    if stop > 0 and multiple <= stop:
        gave_back = f" (peak {peak:.2f}x)" if peak > multiple else ""
        return reason, f"{multiple:.2f}x ≤ {reason} {stop:.2f}x{gave_back}"
    return "", ""


async def get_token_decimals(mint: str) -> int:
    """Fetch mint decimals via RPC getAccountInfo of the mint."""
    body = json.dumps({
        "jsonrpc": "2.0", "id": 1, "method": "getAccountInfo",
        "params": [mint, {"encoding": "jsonParsed"}],
    })
    async with aiohttp.ClientSession() as s:
        async with s.post(config.SOLANA_RPC, data=body,
                          headers={"Content-Type": "application/json"}) as resp:
            data = await resp.json()
    try:
        return int(data["result"]["value"]["data"]["parsed"]["info"]["decimals"])
    except (TypeError, KeyError):
        raise RuntimeError(f"cannot fetch decimals for {mint}")


class ExitEngine:
    def __init__(self, db: DB, notify):
        self.db = db
        self.notify = notify
        self._decimals_cache: dict[str, int] = {}
        # one lock per position: the 20s poll and the realtime stream both walk
        # the same positions, and neither may sell twice
        self._pos_locks: dict[str, asyncio.Lock] = {}
        # realtime path state (rebuilt by _rt_sync every few seconds)
        self._rt: dict[str, dict[str, Any]] = {}
        self._tokens_cache: dict[str, tuple[float, float]] = {}
        self._rt_tasks: set[asyncio.Task] = set()
        self.nats = (pump_nats.PumpNats(self.on_trade)
                     if config.RT_EXITS_ENABLED else None)

    def _lock(self, callout_id: str) -> asyncio.Lock:
        lk = self._pos_locks.get(callout_id)
        if lk is None:
            lk = self._pos_locks[callout_id] = asyncio.Lock()
        return lk

    async def _decimals(self, mint: str) -> int:
        if mint not in self._decimals_cache:
            self._decimals_cache[mint] = await get_token_decimals(mint)
        return self._decimals_cache[mint]

    async def _wallet(self, tg_id: int):
        w = await self.db.load_wallet(tg_id)
        return trader.keypair_from_secret(w[1]) if w else None

    async def check_once(self):
        positions = await self.db.get_all_open_positions()
        if not positions:
            return
        for p in positions:
            try:
                async with self._lock(p["callout_id"]):
                    await self._check_position(p)
            except Exception:
                log.exception("exit check failed pos=%s mint=%s", p["id"], p["mint"])

    async def _check_position(self, p: dict[str, Any], peak_hint: float = 0.0):
        tg = p["tg_id"]
        caller = await self.db.get_caller(tg, p["caller_id"])
        tp_x, sl_x, trail_pct, be_x = caller_rules(caller)
        if tp_x <= 0 and sl_x <= 0 and trail_pct <= 0 and be_x <= 0:
            return  # exits disabled for this caller

        kp = await self._wallet(tg)
        if not kp:
            return
        pubkey = str(kp.pubkey())

        # current token balance (covers partial manual sells too)
        bal = await trader.get_token_balance(pubkey, p["mint"])
        if bal <= 0:
            # Grace period: a just-bought position can read 0 on a stale RPC,
            # or the buy tx may still be landing (confirm_tx times out at 20s
            # but the tx can confirm later).
            # Never auto-close within 10 minutes of opening.
            age = time.time() - p["opened_at"]
            if age < 600:
                log.info("pos %s balance 0 but only %.0fs old — skipping this cycle",
                         p["id"], age)
                return
            # Before closing, check whether the buy tx ever landed on-chain.
            # If it did (and the balance is still 0), tokens were sold/dumped
            # elsewhere; if it failed or vanished, the buy never filled.
            landed = await trader.signature_landed(p["buy_sig"])
            if landed is False:
                # buy failed/never landed — the position record is a ghost
                await self.db.abandon_position(tg, p["callout_id"],
                                               reason="buy-not-confirmed")
                log.warning("pos %s closed: buy tx never confirmed (%s)",
                            p["id"], p["buy_sig"][:16])
                await self.notify(tg, f"ℹ️ Position on <code>{p['mint'][:12]}…</code> "
                                      "closed — buy tx never confirmed on-chain "
                                      "(no funds spent).")
                return
            # landed True (sold manually elsewhere) or None (unknown/timeout —
            # err on the side of keeping the record and retrying next cycle)
            if landed is None:
                log.info("pos %s balance 0, buy tx status unknown — retrying next cycle",
                         p["id"])
                return
            await self.db.mark_sold(tg, p["callout_id"], "manual-or-drained", 0.0)
            await self.notify(tg, f"ℹ️ Position on <code>{p['mint'][:12]}…</code> "
                                  "closed (no token balance found).")
            return

        # current SOL value of the whole balance
        decimals = await self._decimals(p["mint"])
        try:
            sol_value = await jupiter.quote_sol_out(p["mint"], bal, decimals)
        except Exception as e:
            log.warning("quote failed mint=%s: %s", p["mint"][:8], e)
            return  # illiquid/undelisted — skip this cycle

        entry = p["buy_amount_sol"]
        if entry <= 0:
            return
        multiple = sol_value / entry

        # tp_done: the caller's TP already fired once (partial exit) — the
        # remainder only exits via SL or manual sell, never a second TP.
        tp_done = bool(p.get("tp_done"))

        # peak tracking: the trailing stop measures the distance from the high
        # water mark, so it has to survive between check cycles. peak_hint comes
        # from the realtime stream, which can see a top the poll loop missed.
        prev_peak = float(p.get("peak_multiple") or 0)
        peak = max(prev_peak, multiple, peak_hint)
        if peak > prev_peak:
            await self.db.set_position_peak(tg, p["callout_id"], peak)

        # Guards: a trigger <= 0 means DISABLED — without them, tp_x=0 makes
        # `multiple >= tp_x` always true (instant sell at entry) and a 0 stop
        # fires whenever sol_value is 0.
        action, detail = evaluate_exit(
            multiple, peak, tp_x=tp_x, sl_x=sl_x, trail_pct=trail_pct,
            be_x=be_x, tp_done=tp_done)
        if action:
            await self._execute_exit(p, kp, bal, decimals, action, detail)

    async def _execute_exit(self, p: dict[str, Any], kp, bal: float, decimals: int,
                            reason: str, detail: str):
        tg = p["tg_id"]
        # honor the caller's slippage setting; widen 3x and retry once on failure
        caller = await self.db.get_caller(tg, p["caller_id"])
        slip = float((caller or {}).get("slippage") or 0)
        slip_bps = int(slip * 100) if slip > 0 else 300
        # TP sells the caller's configured % (default 100%); SL always exits fully
        sell_pct = 100.0
        if reason == "take-profit":
            sell_pct = float((caller or {}).get("tp_sell_pct") or 100)
            sell_pct = min(max(sell_pct, 1.0), 100.0)
        try:
            sig = await jupiter.sell(kp, p["mint"], bal, decimals, percent=sell_pct,
                                     slippage_bps=slip_bps)
        except Exception as e:
            log.exception("exit sell failed pos=%s", p["id"])
            if slip_bps < 900:
                try:
                    sig = await jupiter.sell(kp, p["mint"], bal, decimals,
                                             percent=sell_pct, slippage_bps=900)
                    log.warning("exit retry at 9%% slippage succeeded pos=%s", p["id"])
                except Exception:
                    await self.notify(tg, f"❌ {reason} sell failed for <code>{p['mint'][:12]}…</code>: {e}")
                    return
            else:
                await self.notify(tg, f"❌ {reason} sell failed for <code>{p['mint'][:12]}…</code>: {e}")
                return
        # value what we actually sold (fresh quote on the sold slice)
        try:
            proceeds = await jupiter.quote_sol_out(
                p["mint"], bal * sell_pct / 100.0, decimals)
        except Exception:
            proceeds = 0.0
        # tokens moved: make the realtime path re-read the balance
        self._tokens_cache.pop(p["mint"], None)
        rt = self._rt.get(p["mint"])
        if rt:
            if sell_pct < 100.0:
                rt["tp_done"] = True          # no second TP on the remainder
                rt["tokens"] = 0
            else:
                self._rt.pop(p["mint"], None)
        if sell_pct >= 100.0:
            await self.db.mark_sold(tg, p["callout_id"], sig, proceeds)
            pnl = proceeds - p["buy_amount_sol"]
        else:
            # partial TP: record slice, shrink basis, keep position open
            await self.db.apply_partial_exit(tg, p["callout_id"], sig,
                                             proceeds, sell_pct)
            pnl = proceeds - p["buy_amount_sol"] * sell_pct / 100.0
        emoji = "🟢" if pnl >= 0 else "🔴"
        if sell_pct < 100:
            # name the rules that still guard the remainder, so it's obvious
            # the position is being managed and not just left open
            guards = ["SL"]
            if float((caller or {}).get("trail_pct") or 0) > 0:
                guards.append(f"trail {float(caller['trail_pct']):g}%")
            if float((caller or {}).get("be_multiple") or 0) > 0:
                guards.append("breakeven")
            rest_txt = (f"\n• rest ({100 - sell_pct:g}%) rides — "
                        f"{' + '.join(guards)} still active")
        else:
            rest_txt = ""
        await self.notify(
            tg,
            f"{emoji} <b>{reason.upper()}</b> — <code>{p['mint'][:12]}…{p['mint'][-6:]}</code>\n"
            f"• {detail}\n"
            f"• sold {sell_pct:g}% → out ≈{proceeds:.4f} SOL ({pnl:+.4f})\n"
            f"• tx: <code>{sig}</code>{rest_txt}")

    # ---------- realtime path (NATS trade stream) ----------
    async def _tokens(self, p: dict[str, Any]) -> float:
        """On-chain token balance, cached briefly (one RPC per position/min)."""
        mint = p["mint"]
        hit = self._tokens_cache.get(mint)
        if hit:
            ttl = config.RT_TOKEN_TTL_S if hit[0] > 0 else ZERO_BAL_TTL_S
            if time.time() - hit[1] < ttl:
                return hit[0]
        kp = await self._wallet(p["tg_id"])
        bal = 0.0
        if kp:
            try:
                bal = await trader.get_token_balance(str(kp.pubkey()), mint)
            except Exception as e:
                log.warning("rt token balance failed mint=%s: %s", mint[:8], e)
        self._tokens_cache[mint] = (bal, time.time())
        return bal

    async def _rt_sync(self) -> None:
        """Keep subscriptions + the rule cache in step with open positions.

        Everything the hot path needs is precomputed here so on_trade is pure
        arithmetic — no DB, no RPC, no awaits.
        """
        while True:
            try:
                positions = await self.db.get_all_open_positions()
                table: dict[str, dict[str, Any]] = {}
                for p in positions:
                    tokens = await self._tokens(p)
                    if tokens <= 0:
                        continue
                    caller = await self.db.get_caller(p["tg_id"], p["caller_id"])
                    tp_x, sl_x, trail_pct, be_x = caller_rules(caller)
                    if tp_x <= 0 and sl_x <= 0 and trail_pct <= 0 and be_x <= 0:
                        continue
                    table[p["mint"]] = {
                        "callout_id": p["callout_id"],
                        "basis": float(p["buy_amount_sol"] or 0),
                        "tokens": tokens,
                        "peak": float(p.get("peak_multiple") or 0),
                        "tp_x": tp_x, "sl_x": sl_x,
                        "trail_pct": trail_pct, "be_x": be_x,
                        "tp_done": bool(p.get("tp_done")),
                        "last_trigger": 0.0,
                    }
                self._rt = table
                if self.nats:
                    self.nats.set_wanted(table.keys())
                live = {p["callout_id"] for p in positions}
                for key in [k for k in self._pos_locks if k not in live]:
                    self._pos_locks.pop(key, None)
            except Exception:
                log.exception("rt sync failed")
            await asyncio.sleep(config.RT_SYNC_INTERVAL_S)

    def on_trade(self, mint: str, price_sol: float, sol_amount: float,
                 is_buy: bool) -> None:
        """Hot path: every trade on a held mint. Sync and fast by design."""
        p = self._rt.get(mint)
        if not p or p["basis"] <= 0 or p["tokens"] <= 0:
            return
        if sol_amount < config.RT_MIN_TRADE_SOL:
            return  # dust print — don't let it move a stop
        try:
            multiple = p["tokens"] * price_sol / p["basis"]
        except (TypeError, ZeroDivisionError):
            return
        if multiple <= 0:
            return
        peak = max(p["peak"], multiple)
        p["peak"] = peak                      # in-memory high-water mark
        action, detail = evaluate_exit(
            multiple, peak, tp_x=p["tp_x"], sl_x=p["sl_x"],
            trail_pct=p["trail_pct"], be_x=p["be_x"], tp_done=p["tp_done"])
        if not action:
            return
        now = time.time()
        if now - p["last_trigger"] < config.RT_TRIGGER_COOLDOWN_S:
            return
        p["last_trigger"] = now
        log.info("realtime %s on %s at %.2fx (stream) — confirming",
                 action, mint[:12], multiple)
        self._spawn(self._rt_confirm(mint, peak, detail))

    async def _rt_confirm(self, mint: str, peak_hint: float, detail: str) -> None:
        """Re-verify with a real quote before selling.

        The stream gives the last traded price; a single outlier print is not
        proof. Re-running the normal check uses a live quote, so a stop that
        only the stream would have tripped is correctly ignored.
        """
        try:
            positions = await self.db.get_all_open_positions()
            p = next((x for x in positions if x["mint"] == mint), None)
            if not p:
                return
            async with self._lock(p["callout_id"]):
                await self._check_position(p, peak_hint=peak_hint)
        except Exception:
            log.exception("realtime confirm failed mint=%s", mint[:12])

    def _spawn(self, coro) -> None:
        """Keep a reference so the task can't be garbage-collected mid-flight."""
        task = asyncio.ensure_future(coro)
        self._rt_tasks.add(task)
        task.add_done_callback(self._rt_tasks.discard)

    async def run(self):
        rt = []
        if self.nats:
            rt.append(asyncio.create_task(self.nats.run()))
            rt.append(asyncio.create_task(self._rt_sync()))
        log.info("exit engine started (interval %ds, realtime=%s)",
                 CHECK_INTERVAL_S, "on" if self.nats else "off")
        try:
            while True:
                try:
                    await self.check_once()
                except Exception:
                    log.exception("exit cycle error")
                await asyncio.sleep(CHECK_INTERVAL_S)
        finally:
            for t in rt:
                t.cancel()
