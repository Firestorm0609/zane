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

# ---------- adaptive_params TTL cache ----------
# Called from both the real-time stream path and paper_monitor_loop.
# A 30-second cache eliminates redundant DB queries without staling the signal.
import time as _time
_adaptive_cache: "tuple[float, float, int] | None" = None
_adaptive_cache_ts: float = 0.0
_ADAPTIVE_CACHE_TTL = 30.0


@dataclass
class OpenTrade:
    id: int
    mint: str
    name: str
    symbol: str
    entry_time: int
    entry_mc: float
    position_size_usd: float
    # Dynamic exit parameters computed at entry
    entry_score: int = 0
    entry_prob: float = 0.0
    highest_mc: float = 0.0
    trailing_stop_price: float = 0.0
    dynamic_sl_pct: float = 0.0
    dynamic_tp_pct: float = 0.0
    dynamic_time_stop: int = 0


def get_open_trades() -> list[OpenTrade]:
    with closing(db_conn()) as conn:
        rows = conn.execute(
            "SELECT * FROM paper_trades WHERE status='OPEN' ORDER BY entry_time"
        ).fetchall()
    return [
        OpenTrade(
            id=int(r["id"]), mint=r["mint"], name=r["name"] or "",
            symbol=r["symbol"] or "", entry_time=int(r["entry_time"]),
            entry_mc=float(r["entry_mc"]),
            position_size_usd=float(r["position_size_usd"]),
            entry_score=int(r["entry_score"] or 0),
            entry_prob=float(r["entry_prob"] or 0.0),
            highest_mc=float(r["highest_mc"] or r["entry_mc"]),
            trailing_stop_price=float(r["trailing_stop_price"] or r["entry_mc"]),
            dynamic_sl_pct=float(r["dynamic_sl_pct"] or 0.0),
            dynamic_tp_pct=float(r["dynamic_tp_pct"] or 0.0),
            dynamic_time_stop=int(r["dynamic_time_stop"] or 0),
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


def _get_entry_risk_state(mint: str) -> dict:
    """Single DB round-trip for all paper_entry_allowed risk checks."""
    cutoff = now_ts() - 86400
    with closing(db_conn()) as conn:
        wallet = conn.execute(
            "SELECT balance_usd, starting_usd FROM paper_wallet WHERE id=1"
        ).fetchone()
        last_trade = conn.execute(
            "SELECT COALESCE(MAX(entry_time), 0) AS t FROM paper_trades WHERE mint=?",
            (mint,),
        ).fetchone()
        streak_rows = conn.execute(
            "SELECT pnl_pct FROM paper_trades WHERE status='CLOSED' "
            "ORDER BY exit_time DESC LIMIT 5"
        ).fetchall()
        daily_row = conn.execute(
            "SELECT COALESCE(SUM(pnl_usd), 0) AS s FROM paper_trades "
            "WHERE status='CLOSED' AND exit_time >= ?",
            (cutoff,),
        ).fetchone()
    return {
        "balance":         safe_float(wallet["balance_usd"])  if wallet else 0.0,
        "starting":        safe_float(wallet["starting_usd"]) if wallet else PAPER_STARTING_BALANCE_USD,
        "last_trade_time": int(last_trade["t"]) if last_trade and last_trade["t"] else None,
        "streak_rows":     streak_rows,
        "daily_pnl":       safe_float(daily_row["s"]) if daily_row else 0.0,
    }


def paper_entry_allowed(state: BotState, coin: dict, result: dict) -> tuple[bool, str]:
    if not state.paper_enabled:
        return False, "paper disabled"
    if int(result.get("score", 0)) < PAPER_ENTRY_SCORE:
        return False, "score below entry"
    mint = coin.get("mint", "")
    if not mint:
        return False, "no mint"
    mc = safe_float(coin.get("usd_market_cap"))
    if mc <= 0:
        return False, "invalid mc"
    if mc > MAX_MARKET_CAP:
        return False, "mc too high"
    cv_std = safe_float(result.get("ml_cv_auc_std", 0.0))
    if cv_std > CONFIDENCE_GATE_STD:
        return False, f"low confidence (std={cv_std:.3f})"

    risk = _get_entry_risk_state(mint)

    lt = risk["last_trade_time"]
    if lt and (now_ts() - lt) < PAPER_MINT_COOLDOWN_SEC:
        return False, "mint cooldown"
    if risk["balance"] < 10:
        return False, "insufficient balance"

    # Only count the leading (most-recent) run of losses
    streak = 0
    for r in risk["streak_rows"]:
        if safe_float(r["pnl_pct"]) < 0:
            streak += 1
        else:
            break
    if streak >= PAPER_LOSS_STREAK_PAUSE:
        return False, f"loss streak ({streak} losses)"

    starting = risk["starting"]
    daily = risk["daily_pnl"]
    daily_loss_pct = abs(daily) / starting * 100 if (daily < 0 and starting > 0) else 0
    if daily_loss_pct >= PAPER_DAILY_LOSS_LIMIT_PCT:
        return False, f"daily loss limit hit ({daily_loss_pct:.1f}%)"

    creator = (coin.get("creator") or coin.get("user")
               or coin.get("traderPublicKey") or "")
    if creator and is_creator_blacklisted(creator):
        return False, "creator blacklisted"

    return True, "ok"


def open_paper_trade(
    coin: dict, position_size_usd: float = 0.0,
    result: dict | None = None,
    market_ctx: "MarketContext | None" = None,
    bot: "Bot | None" = None,
) -> "OpenTrade | None":
    """Atomic: deduct balance + insert OPEN trade with dynamic exit params.
    Returns the OpenTrade if opened, else None."""
    mint = coin.get("mint", "")
    if not mint or position_size_usd <= 0:
        return None

    raw_mc = safe_float(coin.get("usd_market_cap"))
    if raw_mc <= 0:
        return None
    entry_mc  = raw_mc * (1 + PAPER_SLIPPAGE_PCT / 100.0)
    entry_fee = position_size_usd * (PAPER_FEE_PCT / 100.0)
    net_size  = position_size_usd - entry_fee
    name      = coin.get("name", "")
    symbol    = coin.get("symbol", "")

    # Compute dynamic exit parameters for this specific trade
    if result is not None:
        sl, tp, time_sec = compute_dynamic_exit_params(coin, result, market_ctx)
        score = safe_int(result.get("score", 0))
        prob  = safe_float(result.get("probability", 0.0))
    else:
        sl, tp, time_sec = PAPER_STOP_LOSS_PCT, PAPER_TAKE_PROFIT_PCT, PAPER_TIME_STOP_SEC
        score = 0
        prob  = 0.0

    trade_row: dict = {}

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
                    return (False, "insufficient balance")

                exists = conn.execute(
                    "SELECT 1 FROM paper_trades WHERE mint=? AND status='OPEN'",
                    (mint,),
                ).fetchone()
                if exists:
                    conn.execute("ROLLBACK")
                    return (False, "already open")

                open_count = conn.execute(
                    "SELECT COUNT(*) AS c FROM paper_trades WHERE status='OPEN'"
                ).fetchone()["c"]
                if open_count >= PAPER_MAX_CONCURRENT:
                    conn.execute("ROLLBACK")
                    return (False, "max concurrent")

                ts = now_ts()
                conn.execute("""
                    INSERT INTO paper_trades
                        (mint,name,symbol,entry_time,entry_mc,status,position_size_usd,
                         entry_score,entry_prob,highest_mc,trailing_stop_price,
                         dynamic_sl_pct,dynamic_tp_pct,dynamic_time_stop)
                    VALUES (?,?,?,?,?,'OPEN',?,?,?,?,?,?,?,?)
                """, (mint, name, symbol, ts, entry_mc,
                      net_size,
                      score, prob, entry_mc, entry_mc,
                      sl, tp, time_sec))
                trade_row["id"] = conn.execute(
                    "SELECT last_insert_rowid() AS id"
                ).fetchone()["id"]
                conn.execute(
                    "UPDATE paper_wallet SET balance_usd=balance_usd-?, updated_at=? WHERE id=1",
                    (position_size_usd, ts),
                )
                conn.execute("COMMIT")
                return (True, "ok")
            except Exception as e:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                log.error("open_paper_trade tx failed: %s", e)
                return (False, str(e))

    opened, reason = db_write(_atomic)
    if opened:
        log.info("PAPER OPEN | %s | size=$%.2f fee=$%.2f balance=$%.2f | SL=%.1f%% TP=%.1f%% time=%ds (score=%d prob=%.2f)",
                 name or mint[:8], position_size_usd, entry_fee,
                 PaperWallet.get_balance(), sl, tp, time_sec, score, prob)
        trade = OpenTrade(
            id=trade_row.get("id", 0), mint=mint, name=name, symbol=symbol,
            entry_time=now_ts(), entry_mc=entry_mc,
            position_size_usd=net_size,
            entry_score=score, entry_prob=prob,
            highest_mc=entry_mc, trailing_stop_price=entry_mc,
            dynamic_sl_pct=sl, dynamic_tp_pct=tp,
            dynamic_time_stop=time_sec,
        )
        # Send notification + pin (best-effort, fire-and-forget)
        if bot is not None:
            try:
                import asyncio
                from .alerts import send_trade_opened
                asyncio.ensure_future(
                    send_trade_opened(bot, coin, trade, None))
            except Exception as e:
                log.debug("trade open notify spawn failed: %s", e)
        return trade
    else:
        log.debug("PAPER open rejected | %s | %s", mint[:8], reason)
        return None


def close_trade(trade: OpenTrade, exit_mc: float, reason: str,
               bot: "Bot | None" = None) -> None:
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
        # NOTE: maybe_auto_blacklist_creator is intentionally NOT called here.
        # At close time, creator_history.outcome is still NULL (the lookback loop
        # hasn't run yet). The auto-blacklist check is triggered from
        # lookback.py after the outcome label is written.
        log.info("PAPER CLOSE | %s | %+.2f%% pnl=$%.2f balance=$%.2f | %s",
                 trade.name or trade.mint[:8], pnl_pct, pnl_usd,
                 PaperWallet.get_balance(), reason)
        # Send close notification (best-effort)
        if bot is not None:
            try:
                import asyncio
                from .alerts import send_trade_closed
                asyncio.ensure_future(
                    send_trade_closed(bot, trade, exit_mc, reason))
            except Exception as e:
                log.debug("trade close notify spawn failed: %s", e)


# ---------- Dynamic exit parameter computation ----------

def compute_dynamic_exit_params(
    coin: dict,
    result: dict,
    market_ctx: "MarketContext | None" = None,
) -> tuple[float, float, int]:
    """Compute per-trade SL%, TP%, and time-stop seconds from coin features.

    Decisions are based on:
      - Score / probability (high confidence → wider TP, tighter SL)
      - Market-cap percentile (late entries get tighter parameters)
      - Coin age (fresh coins get more room)
      - Reply momentum (high engagement → longer time, wider TP)
      - Recent market volatility (not yet available at signal time — default)
    Returns (sl_pct, tp_pct, time_stop_sec).
    """
    score = safe_int(result.get("score", 0))
    prob  = safe_float(result.get("probability", 0.0))
    mc    = safe_float(coin.get("usd_market_cap"))
    replies = safe_int(coin.get("reply_count", 0))

    # --- Base values from config (used as anchors) ---
    base_sl = PAPER_STOP_LOSS_PCT
    base_tp = PAPER_TAKE_PROFIT_PCT
    base_time = PAPER_TIME_STOP_SEC

    # --- Score/prob scaling ---
    # Score 10 → TP +80%, SL -30%.  Score 5 → TP -40%, SL +20%
    score_factor = (score - 5) / 5.0  # -1.0 … +1.0
    prob_factor  = (prob - 0.5) * 2.0  # -1.0 … +1.0
    confidence   = max(-1.0, min(1.0, 0.6 * score_factor + 0.4 * prob_factor))

    # --- MC percentile: later entries in the distribution get tighter exits ---
    mc_pct = 0.5
    if market_ctx is not None:
        try:
            mc_pct = market_ctx.percentile_mc(mc)
        except Exception:
            pass
    # High percentile (>0.8) → tighter TP, tighter time. Low → more room.
    mc_factor = 1.0 - (mc_pct - 0.5) * 1.0  # 0.5→1.0, 1.0→0.5

    # --- Coin age: fresh coins are more volatile → wider exit bands ---
    created_raw = coin.get("created_timestamp")
    age_factor = 1.0
    if created_raw:
        created_ts = safe_int(created_raw)
        if created_ts > 1_000_000_000_000:
            created_ts //= 1000
        age_sec = now_ts() - created_ts
        # <30 min → 1.4x room.  >2 hr → 0.8x room.
        age_factor = max(0.6, min(1.4, 1.4 - (age_sec / 7200.0) * 0.6))

    # --- Reply momentum: high engagement → give more time/room ---
    mc_k = max(mc, 1.0)
    replies_per_kmc = replies / (mc_k / 1000.0)
    engagement_factor = max(0.6, min(1.4, 0.8 + (replies_per_kmc / 50.0) * 0.6))

    # --- Compute SL ---
    # High confidence → tighter SL (we trust the signal more).
    sl = base_sl * (1.0 + 0.3 * confidence) * mc_factor * age_factor
    sl = max(base_sl * 0.3, min(base_sl * 1.5, sl))

    # --- Compute TP ---
    # High confidence → wider TP. Low confidence → tight TP.
    tp = base_tp * (1.0 + 0.8 * confidence) * mc_factor * engagement_factor
    tp = max(base_tp * 0.5, min(base_tp * 2.5, tp))

    # --- Compute time stop ---
    # High engagement + high confidence → more time to run.
    # High MC percentile (late entry) → less time.
    time_mult = (1.0 + 0.5 * confidence) * mc_factor * engagement_factor
    time_sec = int(base_time * max(0.3, min(2.5, time_mult)))

    return round(sl, 1), round(tp, 1), time_sec


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

def maybe_open_paper_trade(state: BotState, coin: dict, result: dict,
                            market_ctx: "MarketContext | None" = None,
                            bot: "Bot | None" = None) -> None:
    ok, reason = paper_entry_allowed(state, coin, result)
    if ok:
        size = calc_position_size(result)
        if size <= 0:
            log.debug("PAPER SKIP | %s | size=0", (coin.get("mint") or "?")[:8])
            return
        if open_paper_trade(coin, position_size_usd=size,
                             result=result, market_ctx=market_ctx, bot=bot):
            log.info("PAPER OPEN signal | %s mc=%.2f size=$%.2f",
                     coin.get("name"), safe_float(coin.get("usd_market_cap")), size)
    else:
        log.debug("PAPER SKIP | %s | %s", (coin.get("mint") or "?")[:8], reason)


def adaptive_params() -> tuple[float, float, int]:
    """Return (sl_pct, tp_pct, time_stop_sec) scaled continuously to recent performance.

    Result is cached for _ADAPTIVE_CACHE_TTL seconds to avoid a DB
    round-trip on every coin that enters the monitoring pipeline.
    """
    global _adaptive_cache, _adaptive_cache_ts
    now_mono = _time.monotonic()
    if _adaptive_cache is not None and (now_mono - _adaptive_cache_ts) < _ADAPTIVE_CACHE_TTL:
        return _adaptive_cache

    with closing(db_conn()) as conn:
        rows = conn.execute(
            "SELECT pnl_pct, reason FROM paper_trades WHERE status='CLOSED' "
            "ORDER BY exit_time DESC LIMIT 20"
        ).fetchall()

    if len(rows) < 10:
        return PAPER_STOP_LOSS_PCT, PAPER_TAKE_PROFIT_PCT, PAPER_TIME_STOP_SEC

    pnls = [safe_float(r["pnl_pct"]) for r in rows]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = len(wins) / len(pnls)

    # Edge: average win minus average loss magnitude, normalised by TP baseline
    avg_win  = sum(wins)  / len(wins)  if wins   else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    edge_raw = avg_win + avg_loss  # avg_loss is already negative
    edge = max(-1.0, min(1.0, edge_raw / max(PAPER_TAKE_PROFIT_PCT, 1.0)))

    # Blend: win_rate centred at 0.5, edge already centred at 0
    wr_signal = (win_rate - 0.5) * 2.0   # maps 0–1 → -1 to +1
    score = 0.6 * wr_signal + 0.4 * edge  # weighted blend, still -1 to +1

    # Time-stop rate — how often are we closing on time rather than SL/TP?
    time_stopped = sum(1 for r in rows if (r["reason"] or "").startswith("TIME_STOP"))
    time_stop_rate = time_stopped / len(rows)

    # --- SL: tighten when score < 0 (poor performance), floor at 50% of default ---
    sl_mult = 1.0 + min(0.0, score) * 0.5   # score=-1 → mult=0.5; score≥0 → mult=1.0
    sl = max(PAPER_STOP_LOSS_PCT * 0.5, PAPER_STOP_LOSS_PCT * sl_mult)

    # --- TP: widen when score > 0 (good performance), ceiling at 2× default ---
    tp_mult = 1.0 + max(0.0, score) * 1.0   # score=+1 → mult=2.0; score≤0 → mult=1.0
    tp = min(PAPER_TAKE_PROFIT_PCT * 2.0, PAPER_TAKE_PROFIT_PCT * tp_mult)

    # --- Time stop: shorten on poor performance, lengthen if time-stops are common
    #     and performance is good (coins keep moving after we exit) ---
    if score < 0:
        # Poor perf — exit stale trades faster, proportional to how bad it is
        time_mult = 1.0 + score * 0.5       # score=-1 → mult=0.5
    elif time_stop_rate > 0.25 and score >= 0:
        # Many trades hitting time stop while winning — give them more room
        time_mult = 1.0 + score * 1.0       # score=+1 → mult=2.0
    else:
        time_mult = 1.0

    time_sec = int(max(
        300,                                  # floor: 5 minutes
        min(4 * 3600,                         # ceiling: 4 hours
            PAPER_TIME_STOP_SEC * time_mult)
    ))

    result = round(sl, 1), round(tp, 1), time_sec
    _adaptive_cache = result
    _adaptive_cache_ts = _time.monotonic()
    return result


def adaptive_sl_tp() -> tuple[float, float]:
    """Backwards-compatible shim for callers that only need SL/TP."""
    sl, tp, _ = adaptive_params()
    return sl, tp


def _update_trailing_stop(trade: OpenTrade, current_mc: float) -> None:
    """Raise trailing stop when price moves in-the-money.

    Trail distance = dynamic_sl_pct% below the highest MC seen.
    Persists highest_mc and trailing_stop_price to DB.
    """
    if current_mc <= 0:
        return
    if current_mc > trade.highest_mc:
        trade.highest_mc = current_mc
        sl_pct = trade.dynamic_sl_pct or PAPER_STOP_LOSS_PCT
        trade.trailing_stop_price = current_mc * (1.0 - sl_pct / 100.0)
        def _w():
            with closing(db_conn()) as conn, conn:
                conn.execute(
                    "UPDATE paper_trades SET highest_mc=?, trailing_stop_price=? "
                    "WHERE id=? AND status='OPEN'",
                    (trade.highest_mc, trade.trailing_stop_price, trade.id),
                )
        db_write(_w)


def _check_exit_conditions(
    trade: OpenTrade, current_mc: float,
) -> tuple[bool, str]:
    """Evaluate all exit conditions for a trade given current MC.

    Priority: hard SL > trailing stop > TP > time stop.
    Uses per-trade dynamic params; falls back to adaptive_params().
    """
    sl_pct   = trade.dynamic_sl_pct   or PAPER_STOP_LOSS_PCT
    tp_pct   = trade.dynamic_tp_pct   or PAPER_TAKE_PROFIT_PCT
    time_sec = trade.dynamic_time_stop or PAPER_TIME_STOP_SEC

    if trade.entry_mc <= 0:
        return False, ""

    pnl_pct = ((current_mc - trade.entry_mc) / trade.entry_mc) * 100.0
    age_sec  = now_ts() - trade.entry_time

    # 1. Hard stop-loss (from entry)
    if current_mc <= trade.entry_mc * (1.0 - sl_pct / 100.0):
        return True, f"STOP_LOSS_{sl_pct:.1f}%"

    # 2. Trailing stop (only active when in profit)
    if trade.highest_mc > trade.entry_mc and trade.trailing_stop_price > 0:
        if current_mc <= trade.trailing_stop_price:
            trail_pnl = ((trade.trailing_stop_price - trade.entry_mc)
                        / trade.entry_mc) * 100.0
            return True, f"TRAILING_STOP_{trail_pnl:+.1f}%"

    # 3. Take profit
    if pnl_pct >= abs(tp_pct):
        return True, f"TAKE_PROFIT_{tp_pct:.1f}%"

    # 4. Time stop
    if age_sec >= time_sec:
        return True, f"TIME_STOP_{time_sec}s"

    return False, ""


def maybe_close_paper_trades_for_coin(coin: dict, bot: "Bot | None" = None) -> None:
    """Evaluate exit conditions for a specific coin using per-trade dynamic params."""
    mint = coin.get("mint", "")
    current_mc = safe_float(coin.get("usd_market_cap"))
    if not mint or current_mc <= 0:
        return
    for t in get_open_trades():
        if t.mint != mint:
            continue
        _update_trailing_stop(t, current_mc)
        should_close, reason = _check_exit_conditions(t, current_mc)
        if should_close:
            close_trade(t, current_mc, reason, bot=bot)


async def paper_monitor_loop(bot: "Bot | None" = None) -> None:
    """Monitor open trades using per-trade dynamic exits + trailing stops."""
    while True:
        try:
            trades = get_open_trades()
            if trades:
                async with aiohttp.ClientSession() as session:
                    for t in trades:
                        mc = await fetch_coin_mc(session, t.mint)
                        if mc is None:
                            # No data — close only if time stop has passed
                            if now_ts() - t.entry_time >= (
                                t.dynamic_time_stop or PAPER_TIME_STOP_SEC
                            ):
                                close_trade(t, t.entry_mc, "TIME_STOP_NO_DATA", bot=bot)
                            continue

                        save_paper_snapshot(t.id, t.mint, mc)
                        _update_trailing_stop(t, mc)

                        should_close, reason = _check_exit_conditions(t, mc)
                        if should_close:
                            close_trade(t, mc, reason, bot=bot)
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



