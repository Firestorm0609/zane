"""SQLite connection, schema, migrations, and write serialization."""
import logging
import sqlite3
import threading
from contextlib import closing
from typing import Any, Callable

from .config import DB_PATH, PAPER_STARTING_BALANCE_USD
from .utils import now_ts

log = logging.getLogger(__name__)

_db_write_lock = threading.Lock()


def db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def db_write(fn: Callable[[], Any]) -> Any:
    """Serialize writes across threads (SQLite WAL still benefits)."""
    with _db_write_lock:
        return fn()


def init_db() -> None:
    with closing(db_conn()) as conn, conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS chat_settings (
            chat_id INTEGER PRIMARY KEY,
            alerts_enabled INTEGER NOT NULL DEFAULT 0,
            threshold INTEGER NOT NULL DEFAULT 7,
            paper_reports_enabled INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS bot_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mint TEXT NOT NULL,
            name TEXT, symbol TEXT,
            score INTEGER,
            probability REAL,
            ml_probability REAL,
            ml_cv_auc_std REAL,
            recommendation TEXT,
            summary TEXT,
            red_flags TEXT,
            market_cap_at_signal REAL,
            reply_count INTEGER,
            has_twitter INTEGER, has_telegram INTEGER, has_website INTEGER,
            description_text TEXT,
            feature_vector TEXT,
            scoring_mode TEXT,
            created_at INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS price_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mint TEXT NOT NULL,
            market_cap REAL NOT NULL,
            created_at INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mint TEXT NOT NULL, name TEXT, symbol TEXT,
            entry_time INTEGER NOT NULL, entry_mc REAL NOT NULL,
            exit_time INTEGER, exit_mc REAL,
            pnl_pct REAL, pnl_usd REAL, reason TEXT,
            status TEXT NOT NULL, position_size_usd REAL NOT NULL,
            entry_score INTEGER, entry_prob REAL,
            highest_mc REAL, trailing_stop_price REAL,
            dynamic_sl_pct REAL, dynamic_tp_pct REAL,
            dynamic_time_stop INTEGER
        );

        CREATE TABLE IF NOT EXISTS paper_mc_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id INTEGER NOT NULL,
            mint TEXT NOT NULL,
            market_cap REAL NOT NULL,
            created_at INTEGER NOT NULL,
            FOREIGN KEY(trade_id) REFERENCES paper_trades(id)
        );

        CREATE TABLE IF NOT EXISTS lookbacks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id INTEGER NOT NULL,
            mint TEXT NOT NULL,
            window_label TEXT NOT NULL,
            check_at INTEGER NOT NULL,
            checked INTEGER NOT NULL DEFAULT 0,
            mc_at_check REAL,
            pct_change REAL,
            outcome TEXT,
            FOREIGN KEY(signal_id) REFERENCES signals(id)
        );

        CREATE TABLE IF NOT EXISTS dead_letters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mint TEXT,
            raw_data TEXT,
            error TEXT,
            created_at INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS paper_wallet (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            balance_usd REAL NOT NULL,
            starting_usd REAL NOT NULL,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS pinned_alerts (
            chat_id INTEGER NOT NULL,
            message_id INTEGER NOT NULL,
            mint TEXT,
            pinned_at INTEGER NOT NULL,
            PRIMARY KEY (chat_id, message_id)
        );

        CREATE TABLE IF NOT EXISTS creator_blacklist (
            creator TEXT PRIMARY KEY,
            reason TEXT,
            added_at INTEGER NOT NULL,
            auto_added INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS creator_history (
            creator TEXT NOT NULL,
            mint TEXT NOT NULL,
            outcome TEXT,
            seen_at INTEGER NOT NULL,
            PRIMARY KEY (creator, mint)
        );

        CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            mint TEXT NOT NULL,
            name TEXT, symbol TEXT,
            added_at INTEGER NOT NULL,
            UNIQUE(chat_id, mint)
        );

        CREATE INDEX IF NOT EXISTS idx_signals_mint_time   ON signals(mint, created_at);
        CREATE INDEX IF NOT EXISTS idx_signals_created_at  ON signals(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_prices_mint_time    ON price_snapshots(mint, created_at);
        CREATE INDEX IF NOT EXISTS idx_trades_status       ON paper_trades(status);
        CREATE INDEX IF NOT EXISTS idx_trades_mint         ON paper_trades(mint);
        CREATE INDEX IF NOT EXISTS idx_trades_exit_time    ON paper_trades(exit_time DESC);
        CREATE INDEX IF NOT EXISTS idx_lookbacks_due       ON lookbacks(checked, check_at);
        CREATE INDEX IF NOT EXISTS idx_lookbacks_signal    ON lookbacks(signal_id);
        CREATE INDEX IF NOT EXISTS idx_paper_snaps_trade   ON paper_mc_snapshots(trade_id);
        CREATE INDEX IF NOT EXISTS idx_paper_snaps_mint    ON paper_mc_snapshots(mint, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_dead_letters_mint   ON dead_letters(mint);
        CREATE INDEX IF NOT EXISTS idx_dead_letters_time   ON dead_letters(created_at);
        CREATE INDEX IF NOT EXISTS idx_watchlist_chat      ON watchlist(chat_id);
        CREATE INDEX IF NOT EXISTS idx_watchlist_mint      ON watchlist(mint);
        CREATE INDEX IF NOT EXISTS idx_creator_history_creator ON creator_history(creator);
        """)

        conn.execute("""
            INSERT OR IGNORE INTO paper_wallet(id, balance_usd, starting_usd, created_at, updated_at)
            VALUES (1, ?, ?, ?, ?)
        """, (PAPER_STARTING_BALANCE_USD, PAPER_STARTING_BALANCE_USD, now_ts(), now_ts()))

        # Migrations
        for table, col, ddl in [
            ("signals",      "ml_cv_auc_std",  "REAL"),
            ("signals",      "ml_probability", "REAL"),
            ("signals",      "scoring_mode",   "TEXT"),
            ("dead_letters", "retry_count",    "INTEGER DEFAULT 0"),
            ("dead_letters", "last_retry_at",  "INTEGER"),
            ("paper_trades", "entry_score",    "INTEGER"),
            ("paper_trades", "entry_prob",     "REAL"),
            ("paper_trades", "highest_mc",     "REAL"),
            ("paper_trades", "trailing_stop_price", "REAL"),
            ("paper_trades", "dynamic_sl_pct", "REAL"),
            ("paper_trades", "dynamic_tp_pct", "REAL"),
            ("paper_trades", "dynamic_time_stop", "INTEGER"),
        ]:
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")
                log.info("Migration: added %s.%s", table, col)
            except sqlite3.OperationalError:
                pass


# ---- Convenience helpers ----

def upsert_chat(chat_id: int, alerts_enabled=None, threshold=None,
                paper_reports_enabled=None) -> None:
    from .config import DEFAULT_THRESHOLD

    def _write():
        with closing(db_conn()) as conn, conn:
            conn.execute("""
                INSERT INTO chat_settings(chat_id, alerts_enabled, threshold, paper_reports_enabled)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(chat_id) DO UPDATE SET
                    alerts_enabled        = COALESCE(?, alerts_enabled),
                    threshold             = COALESCE(?, threshold),
                    paper_reports_enabled = COALESCE(?, paper_reports_enabled)
            """, (
                chat_id,
                alerts_enabled if alerts_enabled is not None else 0,
                threshold if threshold is not None else DEFAULT_THRESHOLD,
                paper_reports_enabled if paper_reports_enabled is not None else 0,
                # COALESCE args — None keeps the existing column value
                alerts_enabled,
                threshold,
                paper_reports_enabled,
            ))
    db_write(_write)


def set_state(key: str, value: str) -> None:
    def _write():
        with closing(db_conn()) as conn, conn:
            conn.execute(
                "INSERT INTO bot_state(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value))
    db_write(_write)


def get_state(key: str, default: str = "") -> str:
    with closing(db_conn()) as conn:
        r = conn.execute("SELECT value FROM bot_state WHERE key=?", (key,)).fetchone()
        return r["value"] if r else default

