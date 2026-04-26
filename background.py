"""Background loops: watchdog, backups, retries, notifications."""
import asyncio
import json
import logging
import sqlite3
import time
from collections import defaultdict, deque
from contextlib import closing

import aiohttp
from telegram import Bot
from telegram.error import TelegramError

from .config import (
    BLACKLIST_CACHE_TTL_SEC, DB_BACKUP_INTERVAL_SEC, DB_BACKUP_PATH,
    DEAD_LETTER_MAX_RETRIES, DEAD_LETTER_RETRY_SEC, OUTCOME_NOTIFY_ENABLED,
    OUTCOME_NOTIFY_MIN_PCT, PUMP_FRONT, STREAM_DEAD_ALERT_SEC,
    STREAM_DEAD_COOLDOWN_SEC,
)
from .db import db_conn, db_write
from .enrichment import enrich_from_pumpfun, fetch_coin_mc
from .market import MarketContext
from .processor import process_coin
from .scoring import ScoringEngine
from .state import BotState, blacklist_cache
from .utils import (
    fmt_duration, fmt_pct, fmt_usd, mdbold, mdcode, now_ts, safe_float, safe_int,
)

log = logging.getLogger(__name__)


# ---------- Watchlist monitor ----------

async def watchlist_monitor_loop(bot: Bot) -> None:
    await asyncio.sleep(180)
    while True:
        try:
            with closing(db_conn()) as conn:
                rows = conn.execute(
                    "SELECT DISTINCT mint, name, symbol FROM watchlist"
                ).fetchall()
                chats = conn.execute(
                    "SELECT chat_id, mint, added_at FROM watchlist"
                ).fetchall()

            if rows:
                chat_map: dict[str, list[int]] = defaultdict(list)
                # Track the earliest added_at per mint for a correct entry baseline.
                added_at_map: dict[str, int] = {}
                for c in chats:
                    m = c["mint"]
                    chat_map[m].append(int(c["chat_id"]))
                    t = safe_int(c["added_at"])
                    if m not in added_at_map or t < added_at_map[m]:
                        added_at_map[m] = t

                # Pre-load all entry snapshots in one executor call (non-blocking).
                def _fetch_entry_snaps(mint_list, at_map):
                    snaps = {}
                    with closing(db_conn()) as conn:
                        for m in mint_list:
                            added = at_map.get(m, 0)
                            row = conn.execute(
                                "SELECT market_cap FROM price_snapshots "
                                "WHERE mint=? AND created_at >= ? "
                                "ORDER BY created_at ASC LIMIT 1",
                                (m, added),
                            ).fetchone()
                            snaps[m] = safe_float(row["market_cap"]) if row else 0.0
                    return snaps

                loop = asyncio.get_running_loop()
                mint_list = [r["mint"] for r in rows]
                entry_snaps: dict[str, float] = await loop.run_in_executor(
                    None, _fetch_entry_snaps, mint_list, added_at_map
                )

                async with aiohttp.ClientSession() as session:
                    for row in rows:
                        mint = row["mint"]
                        mc = await fetch_coin_mc(session, mint)
                        if mc is None:
                            continue
                        name = row["name"] or row["symbol"] or mint[:8]

                        entry_mc = entry_snaps.get(mint, 0.0)
                        pct_str = ""
                        if entry_mc > 0:
                            pct = ((mc - entry_mc) / entry_mc) * 100
                            pe = "ðŸŸ¢" if pct > 0 else ("ðŸ”´" if pct < 0 else "âšª")
                            pct_str = f" {pe} {mdcode(fmt_pct(pct, 1, signed=True))}"

                        text = (
                            f"ðŸ‘ {mdbold(name)} watchlist update\n"
                            f"MC {mdcode(fmt_usd(mc))}{pct_str}\n"
                            f"ðŸ”— [Pump\\.fun]({PUMP_FRONT}/{mint})"
                        )
                        for chat_id in chat_map.get(mint, []):
                            try:
                                await bot.send_message(
                                    chat_id=chat_id, text=text,
                                    parse_mode="MarkdownV2",
                                    disable_web_page_preview=True,
                                )
                            except TelegramError as e:
                                log.error("watchlist_monitor %s: %s", chat_id, e)
        except Exception as e:
            log.error("watchlist_monitor_loop: %s", e)
        await asyncio.sleep(15 * 60)


# ---------- Outcome notifications ----------

async def outcome_notify_loop(bot: Bot, state: BotState) -> None:
    await asyncio.sleep(120)
    notified_order: deque = deque(maxlen=10000)
    notified_set: set[int] = set()

    while True:
        try:
            if OUTCOME_NOTIFY_ENABLED and state.alerts:
                with closing(db_conn()) as conn:
                    rows = conn.execute("""
                        SELECT lb.id, lb.mint, lb.pct_change, lb.outcome,
                               lb.window_label,
                               s.name, s.symbol, s.score, s.market_cap_at_signal
                        FROM lookbacks lb
                        JOIN signals s ON s.id = lb.signal_id
                        WHERE lb.checked = 1
                          AND lb.outcome IN ('PUMP','MOON','RUG')
                          AND lb.pct_change IS NOT NULL
                          AND ABS(lb.pct_change) >= ?
                        ORDER BY lb.check_at DESC
                        LIMIT 50
                    """, (OUTCOME_NOTIFY_MIN_PCT,)).fetchall()

                for row in rows:
                    rid = int(row["id"])
                    if rid in notified_set:
                        continue
                    if len(notified_order) >= notified_order.maxlen:
                        evicted = notified_order[0]
                        notified_set.discard(evicted)
                    notified_order.append(rid)
                    notified_set.add(rid)

                    pct     = safe_float(row["pct_change"])
                    outcome = row["outcome"] or "?"
                    name    = row["name"] or row["symbol"] or (row["mint"] or "?")[:8]
                    score   = safe_int(row["score"])
                    mc      = safe_float(row["market_cap_at_signal"])
                    window  = row["window_label"] or "?"
                    mint    = row["mint"] or ""

                    if outcome in ("PUMP", "MOON"):
                        emoji = "ðŸš€" if outcome == "MOON" else "ðŸ“Œ"
                    else:
                        emoji = "ðŸ’€"

                    text = "\n".join([
                        f"{emoji} {mdbold('Outcome Alert')}",
                        f"{mdbold(name)} scored {mdcode(f'{score}/10')} at signal time",
                        f"Result \\({window}\\): {mdbold(outcome)} "
                        f"{mdcode(fmt_pct(pct, 1, signed=True))}",
                        f"Entry MC: {mdcode(fmt_usd(mc))}",
                        f"ðŸ”— [Pump\\.fun]({PUMP_FRONT}/{mint})" if mint else "",
                    ])

                    for chat_id in list(state.alerts.keys()):
                        try:
                            await bot.send_message(
                                chat_id=chat_id, text=text,
                                parse_mode="MarkdownV2",
                                disable_web_page_preview=True,
                            )
                        except TelegramError as e:
                            log.error("outcome_notify %s: %s", chat_id, e)
        except Exception as e:
            log.error("outcome_notify_loop: %s", e)
        await asyncio.sleep(300)


# ---------- Stream watchdog ----------

async def stream_watchdog_loop(bot: Bot, state: BotState) -> None:
    await asyncio.sleep(60)
    while True:
        try:
            dead_sec = time.time() - state.last_coin_ts
            now_t = time.time()
            if (dead_sec > STREAM_DEAD_ALERT_SEC
                    and not state.stream_dead_alerted
                    and (now_t - state.stream_dead_alert_at) > STREAM_DEAD_COOLDOWN_SEC):
                msg = (
                    f"âš ï¸ {mdbold('Stream Warning')}\n"
                    f"No coins received for {mdcode(fmt_duration(int(dead_sec)))}\\. "
                    f"WebSocket may be down\\."
                )
                for chat_id in list(state.alerts.keys()):
                    try:
                        await bot.send_message(
                            chat_id=chat_id, text=msg, parse_mode="MarkdownV2")
                    except TelegramError as e:
                        log.error("watchdog alert failed %s: %s", chat_id, e)
                state.stream_dead_alerted = True
                state.stream_dead_alert_at = now_t
                log.warning("Stream dead for %.0fs â€” alert sent", dead_sec)
        except Exception as e:
            log.error("stream_watchdog_loop: %s", e)
        await asyncio.sleep(60)


# ---------- DB backup ----------

async def db_backup_loop() -> None:
    await asyncio.sleep(300)
    while True:
        try:
            def _backup():
                with closing(db_conn()) as src:
                    backup_conn = sqlite3.connect(DB_BACKUP_PATH, timeout=30)
                    with backup_conn:
                        src.backup(backup_conn)
                    backup_conn.close()
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _backup)
            log.info("Database backed up to %s", DB_BACKUP_PATH)
        except Exception as e:
            log.error("db_backup_loop: %s", e)
        await asyncio.sleep(DB_BACKUP_INTERVAL_SEC)


# ---------- Blacklist cache refresh ----------

async def blacklist_refresh_loop() -> None:
    while True:
        await asyncio.sleep(BLACKLIST_CACHE_TTL_SEC)
        try:
            blacklist_cache.invalidate()
        except Exception as e:
            log.error("blacklist_refresh_loop: %s", e)


# ---------- Dead letter retry ----------

async def dead_letter_retry_loop(
    bot: Bot, engine: ScoringEngine,
    market_ctx: MarketContext, state: BotState,
) -> None:
    await asyncio.sleep(DEAD_LETTER_RETRY_SEC)
    loop = asyncio.get_running_loop()
    while True:
        try:
            cutoff = now_ts() - DEAD_LETTER_RETRY_SEC
            with closing(db_conn()) as conn:
                rows = conn.execute(
                    "SELECT id, mint, raw_data FROM dead_letters "
                    "WHERE (retry_count IS NULL OR retry_count < ?) "
                    "AND (last_retry_at IS NULL OR last_retry_at < ?) "
                    "ORDER BY created_at ASC LIMIT 20",
                    (DEAD_LETTER_MAX_RETRIES, cutoff),
                ).fetchall()

            if rows:
                async with aiohttp.ClientSession() as session:
                    for row in rows:
                        try:
                            raw = json.loads(row["raw_data"] or "{}")
                            mint = row["mint"] or raw.get("mint", "")
                            if not mint:
                                continue

                            coin, err = await enrich_from_pumpfun(raw, session)
                            ts_now = now_ts()

                            if err:
                                def _inc(rid=row["id"], ts=ts_now):
                                    with closing(db_conn()) as c, c:
                                        c.execute(
                                            "UPDATE dead_letters SET "
                                            "retry_count=COALESCE(retry_count,0)+1,"
                                            "last_retry_at=? WHERE id=?",
                                            (ts, rid))
                                await loop.run_in_executor(None, db_write, _inc)
                            else:
                                await process_coin(
                                    coin, bot, engine, market_ctx, state)
                                def _done(rid=row["id"]):
                                    with closing(db_conn()) as c, c:
                                        c.execute(
                                            "DELETE FROM dead_letters WHERE id=?",
                                            (rid,))
                                await loop.run_in_executor(None, db_write, _done)
                                log.info("Dead letter retried OK: %s", mint[:8])
                        except Exception as e:
                            log.error("dead_letter_retry row %s: %s", row["id"], e)
        except Exception as e:
            log.error("dead_letter_retry_loop: %s", e)
        await asyncio.sleep(DEAD_LETTER_RETRY_SEC)


