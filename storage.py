"""Signal/snapshot/dead-letter persistence."""
import json
import logging
import os
from contextlib import closing

from .config import DEAD_LETTER_FALLBACK, DEAD_LETTER_FALLBACK_MAX_BYTES
from .db import db_conn, db_write
from .utils import now_ts, safe_float, safe_int

log = logging.getLogger(__name__)


def save_signal(coin: dict, result: dict) -> int:
    def _w():
        with closing(db_conn()) as conn, conn:
            cur = conn.execute("""
                INSERT INTO signals(mint, name, symbol, score, probability, ml_probability,
                    ml_cv_auc_std, recommendation, summary, red_flags, market_cap_at_signal,
                    reply_count, has_twitter, has_telegram, has_website, description_text,
                    feature_vector, scoring_mode, created_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                coin.get("mint", ""),
                coin.get("name", ""),
                coin.get("symbol", ""),
                int(result.get("score", 0)),
                float(result.get("probability", 0)),
                result.get("ml_probability"),
                result.get("ml_cv_auc_std"),
                result.get("recommendation", ""),
                result.get("summary", ""),
                "; ".join(result.get("red_flags", [])),
                safe_float(coin.get("usd_market_cap")),
                safe_int(coin.get("reply_count")),
                1 if coin.get("twitter")  else 0,
                1 if coin.get("telegram") else 0,
                1 if coin.get("website")  else 0,
                (coin.get("description") or "").strip(),
                json.dumps(result.get("feature_vector", [])),
                result.get("mode", ""),
                now_ts(),
            ))
            return cur.lastrowid
    return db_write(_w)


def save_snapshot(coin: dict) -> None:
    mint = coin.get("mint")
    if not mint:
        return
    mc = safe_float(coin.get("usd_market_cap"))
    if mc <= 0:
        return
    def _w():
        with closing(db_conn()) as conn, conn:
            conn.execute(
                "INSERT INTO price_snapshots(mint,market_cap,created_at) VALUES(?,?,?)",
                (mint, mc, now_ts()))
    db_write(_w)


def save_paper_snapshot(trade_id: int, mint: str, mc: float) -> None:
    def _w():
        with closing(db_conn()) as conn, conn:
            conn.execute(
                "INSERT INTO paper_mc_snapshots(trade_id,mint,market_cap,created_at) "
                "VALUES(?,?,?,?)",
                (trade_id, mint, mc, now_ts()))
    db_write(_w)


def _rotate_dead_letter_fallback() -> None:
    try:
        if os.path.exists(DEAD_LETTER_FALLBACK):
            size = os.path.getsize(DEAD_LETTER_FALLBACK)
            if size > DEAD_LETTER_FALLBACK_MAX_BYTES:
                rotated = DEAD_LETTER_FALLBACK + ".1"
                if os.path.exists(rotated):
                    os.remove(rotated)
                os.rename(DEAD_LETTER_FALLBACK, rotated)
                log.info("Rotated dead letter fallback (%d bytes -> %s)", size, rotated)
    except Exception as e:
        log.warning("rotate fallback failed: %s", e)


def save_dead_letter(mint: str, raw_data: dict, error: str) -> None:
    _MAX_RAW_BYTES = 8_000   # prevent bloated dead-letter rows

    def _w():
        serialised = json.dumps(raw_data, default=str)
        if len(serialised) > _MAX_RAW_BYTES:
            # Store only the essential fields needed for retry
            trimmed = {k: raw_data.get(k) for k in (
                "mint", "name", "symbol", "description",
                "twitter", "telegram", "website",
                "reply_count", "usd_market_cap", "creator",
            )}
            serialised = json.dumps(trimmed, default=str)
        with closing(db_conn()) as conn, conn:
            conn.execute(
                "INSERT INTO dead_letters(mint,raw_data,error,created_at) VALUES(?,?,?,?)",
                (mint, serialised, str(error)[:1000], now_ts()))
    try:
        db_write(_w)
    except Exception as e:
        log.error("save_dead_letter failed, writing to file: %s", e)
        try:
            _rotate_dead_letter_fallback()
            with open(DEAD_LETTER_FALLBACK, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "mint": mint, "raw_data": raw_data,
                    "error": str(error)[:500], "ts": now_ts(),
                }) + "\n")
        except Exception as e2:
            log.error("Fallback file write also failed: %s", e2)

