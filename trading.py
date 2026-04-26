"""Paper trading: positions, sizing, monitor loop."""
import asyncio
import logging
from contextlib import closing
from dataclasses import dataclass
from typing import Optional

import aiohttp

from .config import (
    CONFIDENCE_GATE_STD, MAX_MARKET_CAP, PAPER_DAILY_LOSS_LIMIT_PCT,
    PAPER_ENTRY_SCORE, PAPER_FEE_PCT, PAPER_LOSS_STREAK_PAUSE,
    PAPER_MAX_CONCURRENT, PAPER_MAX_POSITION_PCT, PAPER_MINT_COOLDOWN_SEC,
    PAPER_POLL_INTERVAL_SEC, PAPER_SLIPPAGE_PCT, PAPER_STATS_LOOKBACK,
    PAPER_STOP_LOSS_PCT, PAPER_TAKE_PROFIT_PCT, PAPER_TIME_STOP_SEC,
)
from .db import db_conn, db_write
from .enrichment import fetch_coin_mc
from .state import BotState, blacklist_cache, is_creator_blacklisted
from .storage import save_paper_snapshot
from .utils import now_ts, safe_float, safe_int
from .wallet import PaperWallet, daily_pnl_usd, recent_loss_streak

log = logging.getLogger(__name__)


@dataclass
class OpenTrade:
    id: int
    mint: str
    name: str
    symbol: str
    entry_time: int
    entry_mc: float
    position_size_usd: float


def get_open_trades() -> list[OpenTrade]:
    with closing(db_conn()) as conn:
        rows = conn.execute(
            "SELECT * FROM paper_trades WHERE status='OPEN' ORDER BY entry_time"
        ).fetchall()
    return [
        OpenTrade(
            int(r["id"]), r["mint"], r["name"] or "", r["symbol"] or "",
            int(r["entry_time"]), float(r["entry_mc"]), float(r["position_size_usd"]),
        )
        for r in rows
    ]


def last_trade_time_for_mint(mint: str) -> Optional[int]:
    with closing(db_conn()) as conn:
        r = conn.execute(
            "SELECT COALESCE(MAX(entry_time),0) AS t FROM paper_trades WHERE mint=?",
            (mint,),
        ).fetchone()
    t = int(r["t"]) if r else 0
    return t if t > 0 else None


def calc_position_size(result: dict) -> float:
    balance = PaperWallet.get_balance()
    if balance <= 0:
        return 0.0
    score = safe_int(result.get("score", 0))
    prob  = safe_float(result.get("probability", 0.0))
    score_factor = max(0.5, min(1.0, (score - 5) / 5.0))
    prob_factor  = max(0.5, min(1.0, prob * 1.5))
    max_size = balance * (PAPER_MAX_POSITION_PCT / 100.0)
    size = max_size * score_factor * prob_factor
    return round(min(size, balance * 0.95), 2)


def paper_entry_allowed(state: BotState, coin: dict, result: dict) -> tuple[bool, str]:
    if not state.paper_enabled:
        return False, "paper disabled"
    if int(result.get("score", 0)) < PAPER_ENTRY_SCORE:
        return False, "score below entry"
    mint = coin.get("mint", "")
    if not mint:
        return False, "no mint"
    lt = last_trade_time_for_mint(mint)
    if lt and (now_ts() - lt) < PAPER_MINT_COOLDOWN_SEC:
        return False, "mint cooldown"
    mc = safe_float(coin.get("usd_market_cap"))
    if mc <= 0:
        return False, "invalid mc"
    if mc > MAX_MARKET_CAP:
        return False, "mc too high"
    cv_std = safe_float(result.get("ml_cv_auc_std", 0.0))
    if cv_std > CONFIDENCE_GATE_STD:
        return False, f"low confidence (std={cv_std:.3f})"

    balance = PaperWallet.get_balance()
    if balance < 10:
        return False, "insufficient balance"

    streak = recent_loss_streak()
    if streak >= PAPER_LOSS_STREAK_PAUSE:
        return False, f"loss streak ({streak} losses)"

    daily = daily_pnl_usd()
    starting = PaperWallet.get_starting()
    daily_loss_pct = abs(daily) / starting * 100 if (daily < 0 and starting > 0) else 0
    if daily_loss_pct >= PAPER_DAILY_LOSS_LIMIT_PCT:
        return False, f"daily loss limit hit ({daily_loss_pct:.1f}%)"

    creator = (coin.get("creator") or coin.get("user")
               or coin.get("traderPublicKey") or "")
    if creator and is_creator_blacklisted(creator):
        return False, "creator blacklisted"

    return True, "ok"


def open_paper_trade(coin: dict, position_size_usd: float = 0.0) -> bool:
    """Atomic: deduct balance + insert OPEN trade in single transaction."""
    mint = coin.get("mint", "")
    if not mint or position_size_usd <= 0:
        return False

    raw_mc = safe_float(coin.get("usd_market_cap"))
    if raw_mc <= 0:
        return False
    entry_mc  = raw_mc * (1 + PAPER_SLIPPAGE_PCT / 100.0)
    entry_fee = position_size_usd * (PAPER_FEE_PCT / 100.0)
    net_size  = position_size_usd - entry_fee
    name      = coin.get("name", "")
    symbol    = coin.get("symbol", "")

    def _atomic():
        with closing(db_conn()) as conn:
            try:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT balance_usd FROM paper_wallet WHERE id=1"
                ).fetchone()
                bal = safe_float(row["balance_usd"]) if row else 0.0
                if bal < position_size_usd:
                    conn.execute("ROLLBACK")
                    return False, "insufficient balance"

                exists = conn.execute(
                    "SELECT 1 FROM paper_trades WHERE mint=? AND status='OPEN'",
                    (mint,),
                ).fetchone()
                if exists:
                    conn.execute("ROLLBACK")
                    return False, "already open"

                open_count = conn.execute(
                    "SELECT COUNT(*) AS c FROM paper_trades WHERE status='OPEN'"
                ).fetchone()["c"]
                if open_count >= PAPER_MAX_CONCURRENT:
                    conn.execute("ROLLBACK")
                    return False, "max concurrent"

                conn.execute("""
                    INSERT INTO paper_trades
                        (mint,name,symbol,entry_time,entry_mc,status,position_size_usd)
                    VALUES (?,?,?,?,?,'OPEN',?)
                """, (mint, name, symbol, now_ts(), entry_mc, net_size))
                conn.execute(
                    "UPDATE paper_wallet SET balance_usd=balance_usd-?, updated_at=? WHERE id=1",
                    (position_size_usd, now_ts()),
                )
                conn.execute("COMMIT")
                return True, "ok"
            except Exception as e:
                try: conn.execute("ROLLBACK")
                except Exception: pass
                log.error("open_paper_trade tx failed: %s", e)
                return False, str(e)

    opened, reason = db_write(_atomic)
    if opened:
        log.info("PAPER OPEN | %s | size=$%.2f fee=$%.2f balance=$%.2f",
                 name or mint[:8], position_size_usd, entry_fee,
                 PaperWallet.get_balance())
    else:
        log.debug("PAPER open rejected | %s | %s", mint[:8], reason)
    return opened


def close_trade(trade: OpenTrade, exit_mc: float, reason: str) -> None:
    """Atomic: close trade row + credit balance in single transaction."""
    effective_exit = (exit_mc * (1 - PAPER_SLIPPAGE_PCT / 100.0)) if exit_mc > 0 else 0
    pnl_pct = (((effective_exit - trade.entry_mc) / trade.entry_mc) * 100
               if trade.entry_mc > 0 else 0.0)
    gross_proceeds = trade.position_size_usd * (1 + pnl_pct / 100.0)
    exit_fee       = gross_proceeds * (PAPER_FEE_PCT / 100.0)
    net_proceeds   = max(0.0, gross_proceeds - exit_fee)
    pnl_usd        = net_proceeds - trade.position_size_usd

    def _atomic():
        with closing(db_conn()) as conn:
            try:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT status FROM paper_trades WHERE id=?", (trade.id,),
                ).fetchone()
                if not row or row["status"] != "OPEN":
                    conn.execute("ROLLBACK")
                    return False
                conn.execute(
                    "UPDATE paper_trades SET exit_time=?,exit_mc=?,pnl_pct=?,pnl_usd=?,"
                    "reason=?,status='CLOSED' WHERE id=? AND status='OPEN'",
                    (now_ts(), effective_exit, pnl_pct, pnl_usd, reason, trade.id),
                )
                conn.execute(
                    "UPDATE paper_wallet SET balance_usd=balance_usd+?, updated_at=? WHERE id=1",
                    (net_proceeds, now_ts()),
                )
                conn.execute("COMMIT")
                return True
            except Exception as e:
                try: conn.execute("ROLLBACK")
                except Exception: pass
                log.error("close_trade tx failed: %s", e)
                return False

    closed = db_write(_atomic)
    if closed:
        maybe_auto_blacklist_creator(trade.mint)
        log.info("PAPER CLOSE | %s | %+.2f%% pnl=$%.2f balance=$%.2f | %s",
                 trade.name or trade.mint[:8], pnl_pct, pnl_usd,
                 PaperWallet.get_balance(), reason)


# ---------- Creator history / auto-blacklist ----------

def record_creator_token(creator: str, mint: str) -> None:
    if not creator or not mint:
        return
    def _w():
        with closing(db_conn()) as conn, conn:
            conn.execute(
                "INSERT OR IGNORE INTO creator_history(creator,mint,seen_at) VALUES(?,?,?)",
                (creator, mint, now_ts()))
    db_write(_w)


def maybe_auto_blacklist_creator(mint: str) -> None:
    with closing(db_conn()) as conn:
        cr = conn.execute(
            "SELECT creator FROM creator_history WHERE mint=?", (mint,),
        ).fetchone()
    if not cr:
        return
    creator = cr["creator"]
    if not creator or is_creator_blacklisted(creator):
        return

    with closing(db_conn()) as conn:
        rows = conn.execute("""
            SELECT outcome FROM creator_history
            WHERE creator=? AND outcome IS NOT NULL
            ORDER BY seen_at DESC LIMIT 3
        """, (creator,)).fetchall()

    if len(rows) < 3:
        return

    bad_outcomes = {"RUG", "DOWN"}
    if all(r["outcome"] in bad_outcomes for r in rows):
        def _w():
            with closing(db_conn()) as conn, conn:
                conn.execute(
                    "INSERT OR IGNORE INTO creator_blacklist(creator,reason,added_at,auto_added) "
                    "VALUES(?,?,?,1)",
                    (creator, "3 consecutive rugs", now_ts()))
        db_write(_w)
        blacklist_cache.invalidate()
        log.warning("AUTO-BLACKLIST creator %s (3 consecutive rugs)", creator[:8])


# ---------- Entry helpers ----------

def maybe_open_paper_trade(state: BotState, coin: dict, result: dict) -> None:
    ok, reason = paper_entry_allowed(state, coin, result)
    if ok:
        size = calc_position_size(result)
        if size <= 0:
            log.debug("PAPER SKIP | %s | size=0", (coin.get("mint") or "?")[:8])
            return
        if open_paper_trade(coin, position_size_usd=size):
            log.info("PAPER OPEN signal | %s mc=%.2f size=$%.2f",
                     coin.get("name"), safe_float(coin.get("usd_market_cap")), size)
    else:
        log.debug("PAPER SKIP | %s | %s", (coin.get("mint") or "?")[:8], reason)


def adaptive_sl_tp() -> tuple[float, float]:
    with closing(db_conn()) as conn:
        rows = conn.execute(
            "SELECT pnl_pct FROM paper_trades WHERE status='CLOSED' "
            "ORDER BY exit_time DESC LIMIT 20"
        ).fetchall()
    if len(rows) < 10:
        return PAPER_STOP_LOSS_PCT, PAPER_TAKE_PROFIT_PCT
    wins = sum(1 for r in rows if safe_float(r["pnl_pct"]) > 0)
    win_rate = wins / len(rows)
    if win_rate < 0.40:
        sl = max(10.0, PAPER_STOP_LOSS_PCT * 0.75)
        tp = PAPER_TAKE_PROFIT_PCT
    elif win_rate > 0.60:
        sl = PAPER_STOP_LOSS_PCT
        tp = min(100.0, PAPER_TAKE_PROFIT_PCT * 1.25)
    else:
        sl = PAPER_STOP_LOSS_PCT
        tp = PAPER_TAKE_PROFIT_PCT
    return sl, tp


def maybe_close_paper_trades_for_coin(coin: dict) -> None:
    mint = coin.get("mint", "")
    current_mc = safe_float(coin.get("usd_market_cap"))
    if not mint or current_mc <= 0:
        return
    sl_pct, tp_pct = adaptive_sl_tp()
    for t in get_open_trades():
        if t.mint != mint:
            continue
        pnl_pct = (((current_mc - t.entry_mc) / t.entry_mc) * 100
                   if t.entry_mc > 0 else 0.0)
        age_sec = now_ts() - t.entry_time
        if pnl_pct <= -abs(sl_pct):
            close_trade(t, current_mc, f"STOP_LOSS_{sl_pct:.1f}%")
        elif pnl_pct >= abs(tp_pct):
            close_trade(t, current_mc, f"TAKE_PROFIT_{tp_pct:.1f}%")
        elif age_sec >= PAPER_TIME_STOP_SEC:
            close_trade(t, current_mc, f"TIME_STOP_{PAPER_TIME_STOP_SEC}s")


async def paper_monitor_loop() -> None:
    while True:
        try:
            trades = get_open_trades()
            if trades:
                sl_pct, tp_pct = adaptive_sl_tp()
                async with aiohttp.ClientSession() as session:
                    for t in trades:
                        mc = await fetch_coin_mc(session, t.mint)
                        if mc is None:
                            if now_ts() - t.entry_time >= PAPER_TIME_STOP_SEC:
                                close_trade(t, t.entry_mc, "TIME_STOP_NO_DATA")
                            continue

                        save_paper_snapshot(t.id, t.mint, mc)

                        pnl_pct = (((mc - t.entry_mc) / t.entry_mc) * 100
                                   if t.entry_mc > 0 else 0.0)
                        age_sec = now_ts() - t.entry_time

                        if pnl_pct <= -abs(sl_pct):
                            close_trade(t, mc, f"STOP_LOSS_{sl_pct:.1f}%")
                        elif pnl_pct >= abs(tp_pct):
                            close_trade(t, mc, f"TAKE_PROFIT_{tp_pct:.1f}%")
                        elif age_sec >= PAPER_TIME_STOP_SEC:
                            close_trade(t, mc, f"TIME_STOP_{PAPER_TIME_STOP_SEC}s")
        except Exception as e:
            log.error("paper_monitor_loop: %s", e)
        await asyncio.sleep(PAPER_POLL_INTERVAL_SEC)


def paper_stats() -> dict:
    with closing(db_conn()) as conn:
        closed = conn.execute(
            "SELECT pnl_pct, pnl_usd FROM paper_trades WHERE status='CLOSED' "
            "ORDER BY exit_time DESC LIMIT ?", (PAPER_STATS_LOOKBACK,),
        ).fetchall()
        open_n = conn.execute(
            "SELECT COUNT(*) AS c FROM paper_trades WHERE status='OPEN'"
        ).fetchone()["c"]

    closed_chrono = list(reversed(closed))
    n = len(closed_chrono)
    wins = sum(1 for r in closed_chrono if safe_float(r["pnl_pct"]) > 0)
    equity = peak = max_dd = 0.0
    for r in closed_chrono:
        equity += safe_float(r["pnl_usd"])
        peak    = max(peak, equity)
        max_dd  = max(max_dd, peak - equity)
    return {
        # NOTE: callers must set "paper_enabled" from BotState; this dict
        # intentionally omits it so accidental direct usage is obvious.
        "open_positions":   int(open_n),
        "closed_positions": n,
        "wins": wins, "losses": n - wins,
        "win_rate":         wins / n * 100 if n else 0.0,
        "total_pnl_usd":    sum(safe_float(r["pnl_usd"]) for r in closed_chrono),
        "avg_pnl_pct":      sum(safe_float(r["pnl_pct"]) for r in closed_chrono) / n if n else 0.0,
        "best_pnl_pct":     max((safe_float(r["pnl_pct"]) for r in closed_chrono), default=0.0),
        "worst_pnl_pct":    min((safe_float(r["pnl_pct"]) for r in closed_chrono), default=0.0),
        "max_drawdown_usd": max_dd,
    }

