"""Outcome labeling and ML retraining loops."""
import asyncio
import concurrent.futures
import logging
from contextlib import closing

import aiohttp

from .config import (LOOKBACK_WINDOWS, ML_AVAILABLE, RETRAIN_EVERY_SEC)
from .db import db_conn, db_write
from .enrichment import fetch_coin_mc
from .scoring import ScoringEngine
from .utils import now_ts, safe_float

log = logging.getLogger(__name__)

train_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="trainer",
)


def label_outcome(pct: float) -> str:
    if pct >= 200: return "MOON"
    if pct >= 50:  return "PUMP"
    if pct >= 10:  return "UP"
    if pct >= -20: return "STALE"
    if pct >= -50: return "DOWN"
    return "RUG"


def schedule_lookbacks(signal_id: int, mint: str) -> None:
    ts = now_ts()
    def _w():
        with closing(db_conn()) as conn, conn:
            for label, offset in LOOKBACK_WINDOWS:
                conn.execute(
                    "INSERT INTO lookbacks(signal_id,mint,window_label,check_at) "
                    "VALUES(?,?,?,?)",
                    (signal_id, mint, label, ts + offset))
    db_write(_w)


async def process_due_lookbacks() -> bool:
    """Process up to 50 due lookbacks.

    Returns True if the batch was full (caller should loop immediately to
    drain remaining backlog rather than waiting 60 s).
    """
    # Deferred import to avoid circular dependency at module level.
    from .trading import maybe_auto_blacklist_creator

    ts = now_ts()
    with closing(db_conn()) as conn:
        due = conn.execute("""
            SELECT lb.id, lb.mint, lb.window_label, lb.check_at,
                   s.market_cap_at_signal
            FROM lookbacks lb
            JOIN signals s ON s.id = lb.signal_id
            WHERE lb.checked=0 AND lb.check_at<=?
            LIMIT 50
        """, (ts,)).fetchall()
    if not due:
        return False

    batch_full = len(due) == 50
    if batch_full:
        log.warning(
            "Lookback backlog: processed a full batch of 50 rows — "
            "more may be pending; draining immediately."
        )

    loop = asyncio.get_running_loop()

    async with aiohttp.ClientSession() as session:
        for row in due:
            try:
                mc = await fetch_coin_mc(session, row["mint"])
                if mc is None:
                    if ts - row["check_at"] > 7200:
                        def _expire(rid=row["id"]):
                            with closing(db_conn()) as c, c:
                                c.execute(
                                    "UPDATE lookbacks SET checked=1 WHERE id=?", (rid,))
                        await loop.run_in_executor(None, db_write, _expire)
                    continue

                entry_mc = safe_float(row["market_cap_at_signal"])
                if entry_mc <= 0:
                    def _skip(rid=row["id"]):
                        with closing(db_conn()) as c, c:
                            c.execute(
                                "UPDATE lookbacks SET checked=1 WHERE id=?", (rid,))
                    await loop.run_in_executor(None, db_write, _skip)
                    continue

                pct = ((mc - entry_mc) / entry_mc) * 100
                outcome = label_outcome(pct)

                def _update(rid=row["id"], mint=row["mint"],
                            _mc=mc, _pct=pct, _out=outcome):
                    with closing(db_conn()) as c, c:
                        c.execute(
                            "UPDATE lookbacks SET checked=1,mc_at_check=?,pct_change=?,"
                            "outcome=? WHERE id=?",
                            (_mc, _pct, _out, rid))
                        c.execute(
                            "INSERT INTO price_snapshots(mint,market_cap,created_at) "
                            "VALUES(?,?,?)",
                            (mint, _mc, now_ts()))
                        c.execute(
                            "UPDATE creator_history SET outcome=? "
                            "WHERE mint=? AND outcome IS NULL",
                            (_out, mint))
                await loop.run_in_executor(None, db_write, _update)

                # Trigger auto-blacklist NOW — outcome is committed in the DB
                # so maybe_auto_blacklist_creator sees real labelled data.
                await loop.run_in_executor(
                    None, maybe_auto_blacklist_creator, row["mint"])

                log.info("LOOKBACK %-5s | %s | %-5s | %+.1f%%",
                         row["window_label"], row["mint"][:8], outcome, pct)
            except Exception as e:
                log.error("lookback row %s failed: %s", row["id"], e)

    return batch_full


async def lookback_loop() -> None:
    while True:
        try:
            # Drain backlog: if the batch was full, loop immediately
            # instead of waiting 60 s for the next tick.
            while await process_due_lookbacks():
                await asyncio.sleep(0)   # yield to event loop between bursts
        except Exception as e:
            log.error("lookback_loop: %s", e)
        await asyncio.sleep(60)


async def training_loop(engine: ScoringEngine) -> None:
    if not ML_AVAILABLE:
        log.info("training_loop disabled — sklearn not installed")
        return
    await asyncio.sleep(300)
    while True:
        try:
            loop = asyncio.get_running_loop()
            trained = await loop.run_in_executor(train_executor, engine.train)
            if trained:
                log.info("Model retrained successfully")
        except Exception as e:
            log.error("training_loop: %s", e)
        await asyncio.sleep(RETRAIN_EVERY_SEC)

