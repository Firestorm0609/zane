"""SQLite persistence: users, wallets (encrypted), callers, positions."""
import asyncio
import json
import logging
import os
import sqlite3
import time
from typing import Any, Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

import config

log = logging.getLogger("db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    tg_id INTEGER PRIMARY KEY,
    created_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS wallets (
    tg_id INTEGER PRIMARY KEY REFERENCES users(tg_id),
    label TEXT NOT NULL,
    enc_iv BLOB NOT NULL,
    enc_sk BLOB NOT NULL,
    pubkey TEXT NOT NULL,
    imported_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS callers (
    tg_id INTEGER NOT NULL,
    caller_id TEXT NOT NULL,
    label TEXT DEFAULT '',
    enabled INTEGER NOT NULL DEFAULT 1,
    buy_sol REAL NOT NULL DEFAULT 0.01,
    max_multiple REAL NOT NULL DEFAULT 0,
    stop_multiple REAL NOT NULL DEFAULT 0,
    added_at INTEGER NOT NULL,
    last_callout_id TEXT DEFAULT '',
    PRIMARY KEY (tg_id, caller_id)
);
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tg_id INTEGER NOT NULL,
    caller_id TEXT NOT NULL,
    callout_id TEXT NOT NULL,
    mint TEXT NOT NULL,
    buy_sig TEXT NOT NULL,
    buy_amount_sol REAL NOT NULL,
    tokens_received REAL NOT NULL DEFAULT 0,
    entry_price REAL NOT NULL DEFAULT 0,
    sold_sig TEXT DEFAULT '',
    sold_amount_sol REAL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'open',
    opened_at INTEGER NOT NULL,
    closed_at INTEGER DEFAULT 0,
    UNIQUE (tg_id, callout_id)
);
CREATE TABLE IF NOT EXISTS dedupe (
    callout_id TEXT PRIMARY KEY,
    seen_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS access_codes (
    code TEXT PRIMARY KEY,
    created_by INTEGER NOT NULL,
    created_at INTEGER NOT NULL,
    used_by INTEGER NOT NULL DEFAULT 0,
    used_at INTEGER NOT NULL DEFAULT 0,
    source TEXT NOT NULL DEFAULT 'manual'  -- manual | stars
);
CREATE TABLE IF NOT EXISTS subscriptions (
    tg_id INTEGER PRIMARY KEY,
    unlocked_at INTEGER NOT NULL,
    access_code TEXT DEFAULT '',           -- code used, or 'stars:<charge_id>'
    active INTEGER NOT NULL DEFAULT 1
);
"""


class DB:
    def __init__(self, path: str = config.DB_PATH):
        self.path = path
        self._lock = asyncio.Lock()
        self._conn: Optional[sqlite3.Connection] = None

    async def connect(self):
        # check_same_thread=False: connect() runs in a worker thread via to_thread,
        # while queries run on the loop thread. Access is serialized by self._lock.
        self._conn = await asyncio.to_thread(
            lambda: sqlite3.connect(self.path, check_same_thread=False)
        )
        self._conn.row_factory = sqlite3.Row
        await asyncio.to_thread(self._conn.executescript, _SCHEMA)
        await asyncio.to_thread(self._conn.commit)
        await self._migrate()

    async def _migrate(self):
        """Add columns introduced after the initial schema."""
        migrations = [
            "ALTER TABLE callers ADD COLUMN min_mcap REAL NOT NULL DEFAULT 0",
            "ALTER TABLE callers ADD COLUMN max_mcap REAL NOT NULL DEFAULT 0",
            "ALTER TABLE callers ADD COLUMN slippage REAL NOT NULL DEFAULT 0",
            "ALTER TABLE callers ADD COLUMN tp_sell_pct REAL NOT NULL DEFAULT 100",
            "ALTER TABLE callers ADD COLUMN priority_fee REAL NOT NULL DEFAULT 0",
            "ALTER TABLE callers ADD COLUMN trail_pct REAL NOT NULL DEFAULT 0",
            "ALTER TABLE callers ADD COLUMN be_multiple REAL NOT NULL DEFAULT 0",
            "ALTER TABLE callers ADD COLUMN max_entry_multiple REAL NOT NULL DEFAULT 0",
            "ALTER TABLE positions ADD COLUMN peak_multiple REAL NOT NULL DEFAULT 0",
            "ALTER TABLE positions ADD COLUMN tp_done INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE positions ADD COLUMN sold_pct REAL NOT NULL DEFAULT 0",
            "ALTER TABLE positions ADD COLUMN proceeds_sol REAL NOT NULL DEFAULT 0",
        ]
        for sql in migrations:
            try:
                await asyncio.to_thread(self._conn.execute, sql)
                await asyncio.to_thread(self._conn.commit)
            except sqlite3.OperationalError as e:
                if "duplicate column" in str(e).lower() or "already exists" in str(e).lower():
                    pass  # expected on an already-migrated db
                else:
                    log.warning("migration failed: %s — %s", sql[:60], e)

    def _q(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        assert self._conn is not None
        cur = self._conn.execute(sql, params)
        rows = cur.fetchall()
        cur.close()
        return rows

    def _w(self, sql: str, params: tuple = ()) -> None:
        assert self._conn is not None
        cur = self._conn.execute(sql, params)
        cur.close()
        self._conn.commit()

    # ---- users ----
    async def ensure_user(self, tg_id: int):
        async with self._lock:
            self._w("INSERT OR IGNORE INTO users (tg_id, created_at) VALUES (?, ?)",
                    (tg_id, int(time.time())))

    async def ensure_owner(self, tg_id: int):
        """Owner bypasses the paywall.

        The poller and exit engine read their work through paid-gate JOINs
        (get_callers / get_all_open_positions), so without a subscription row
        the owner's own callers are never polled and their positions are never
        managed. Mint a permanent row at startup so those queries include the
        owner.
        """
        async with self._lock:
            self._w("INSERT OR IGNORE INTO users (tg_id, created_at) VALUES (?, ?)",
                    (tg_id, int(time.time())))
            self._w(
                "INSERT INTO subscriptions (tg_id, unlocked_at, access_code, active) "
                "VALUES (?, ?, 'owner', 1) "
                "ON CONFLICT(tg_id) DO UPDATE SET active = 1",
                (tg_id, int(time.time())))

    # ---- wallets ----
    async def save_wallet(self, tg_id: int, label: str, secret_key_hex: str, pubkey: str):
        """Encrypt secret key with AES-256-GCM before it touches disk."""
        if not config.WALLET_ENC_KEY:
            raise RuntimeError("WALLET_ENC_KEY is not set — refusing to store a wallet")
        key = bytes.fromhex(config.WALLET_ENC_KEY)
        aes = AESGCM(key)
        iv = os.urandom(12)
        ct = aes.encrypt(iv, secret_key_hex.encode(), None)
        async with self._lock:
            self._w(
                "INSERT OR REPLACE INTO wallets (tg_id, label, enc_iv, enc_sk, pubkey, imported_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (tg_id, label, iv, ct, pubkey, int(time.time())),
            )

    async def load_wallet(self, tg_id: int) -> Optional[tuple[str, str]]:
        """Returns (label, secret_key_hex) or None."""
        async with self._lock:
            rows = self._q("SELECT label, enc_iv, enc_sk FROM wallets WHERE tg_id = ?", (tg_id,))
        if not rows:
            return None
        r = rows[0]
        key = bytes.fromhex(config.WALLET_ENC_KEY)
        aes = AESGCM(key)
        sk = aes.decrypt(bytes(r["enc_iv"]), bytes(r["enc_sk"]), None).decode()
        return r["label"], sk

    async def wallet_pubkey(self, tg_id: int) -> Optional[str]:
        async with self._lock:
            rows = self._q("SELECT pubkey FROM wallets WHERE tg_id = ?", (tg_id,))
        return rows[0]["pubkey"] if rows else None

    async def delete_wallet(self, tg_id: int):
        async with self._lock:
            self._w("DELETE FROM wallets WHERE tg_id = ?", (tg_id,))

    # ---- callers ----
    async def add_caller(self, tg_id: int, caller_id: str, label: str, buy_sol: float,
                         max_multiple: float, stop_multiple: float):
        async with self._lock:
            self._w(
                "INSERT OR REPLACE INTO callers (tg_id, caller_id, label, enabled, buy_sol, "
                "max_multiple, stop_multiple, added_at, last_callout_id) "
                "VALUES (?, ?, ?, 1, ?, ?, ?, ?, '')",
                (tg_id, caller_id, label, buy_sol, max_multiple, stop_multiple, int(time.time())),
            )

    async def remove_caller(self, tg_id: int, caller_id: str):
        async with self._lock:
            self._w("DELETE FROM callers WHERE tg_id = ? AND caller_id = ?", (tg_id, caller_id))

    async def set_caller_enabled(self, tg_id: int, caller_id: str, enabled: bool):
        async with self._lock:
            self._w("UPDATE callers SET enabled = ? WHERE tg_id = ? AND caller_id = ?",
                    (1 if enabled else 0, tg_id, caller_id))

    async def set_caller_buy_sol(self, tg_id: int, caller_id: str, buy_sol: float):
        async with self._lock:
            self._w("UPDATE callers SET buy_sol = ? WHERE tg_id = ? AND caller_id = ?",
                    (buy_sol, tg_id, caller_id))

    async def set_caller_label(self, tg_id: int, caller_id: str, label: str):
        """Rename a caller (shown in the caller list)."""
        async with self._lock:
            self._w("UPDATE callers SET label = ? WHERE tg_id = ? AND caller_id = ?",
                    (label, tg_id, caller_id))

    async def set_caller_tpsl(self, tg_id: int, caller_id: str,
                              max_multiple: float, stop_multiple: float):
        async with self._lock:
            self._w("UPDATE callers SET max_multiple = ?, stop_multiple = ? "
                    "WHERE tg_id = ? AND caller_id = ?",
                    (max_multiple, stop_multiple, tg_id, caller_id))

    async def set_caller_mcap(self, tg_id: int, caller_id: str,
                              min_mcap: float, max_mcap: float):
        async with self._lock:
            self._w("UPDATE callers SET min_mcap = ?, max_mcap = ? "
                    "WHERE tg_id = ? AND caller_id = ?",
                    (min_mcap, max_mcap, tg_id, caller_id))

    async def set_caller_slippage(self, tg_id: int, caller_id: str, slippage_pct: float):
        """Store slippage as percent (0 = venue defaults)."""
        async with self._lock:
            self._w("UPDATE callers SET slippage = ? WHERE tg_id = ? AND caller_id = ?",
                    (slippage_pct, tg_id, caller_id))

    async def set_caller_priority_fee(self, tg_id: int, caller_id: str, fee_sol: float):
        """Priority fee per buy, in SOL. 0 = global default; <0 = disabled."""
        async with self._lock:
            self._w("UPDATE callers SET priority_fee = ? WHERE tg_id = ? AND caller_id = ?",
                    (fee_sol, tg_id, caller_id))

    async def set_caller_tp_sell_pct(self, tg_id: int, caller_id: str, pct: float):
        """What % of the position a TP trigger sells (100 = exit fully)."""
        async with self._lock:
            self._w("UPDATE callers SET tp_sell_pct = ? WHERE tg_id = ? AND caller_id = ?",
                    (pct, tg_id, caller_id))

    async def set_caller_trail(self, tg_id: int, caller_id: str, trail_pct: float):
        """Trailing-stop distance in percent below the peak. 0 = off."""
        async with self._lock:
            self._w("UPDATE callers SET trail_pct = ? WHERE tg_id = ? AND caller_id = ?",
                    (trail_pct, tg_id, caller_id))

    async def set_caller_breakeven(self, tg_id: int, caller_id: str, be_multiple: float):
        """Move the stop to entry once the position reaches be_multiple. 0 = off."""
        async with self._lock:
            self._w("UPDATE callers SET be_multiple = ? WHERE tg_id = ? AND caller_id = ?",
                    (be_multiple, tg_id, caller_id))

    async def set_caller_max_entry_multiple(self, tg_id: int, caller_id: str, mult: float):
        """Skip callouts already up more than `mult` since the call. 0 = off."""
        async with self._lock:
            self._w("UPDATE callers SET max_entry_multiple = ? "
                    "WHERE tg_id = ? AND caller_id = ?",
                    (mult, tg_id, caller_id))

    async def set_position_peak(self, tg_id: int, callout_id: str, peak: float):
        """Highest multiple this position has reached — the trailing stop needs it."""
        async with self._lock:
            self._w("UPDATE positions SET peak_multiple = ? "
                    "WHERE tg_id = ? AND callout_id = ?",
                    (peak, tg_id, callout_id))

    async def apply_partial_exit(self, tg_id: int, callout_id: str, sig: str,
                                 proceeds_sol: float, pct_sold: float):
        """Record a partial TP: shrink the cost basis proportionally, keep open.

        tp_done=1 stops the exit engine from re-triggering TP on the remainder;
        SL keeps working on the shrunken basis (same price level as before).
        """
        async with self._lock:
            self._w(
                "UPDATE positions SET tp_done = 1, "
                "sold_pct = sold_pct + ?, proceeds_sol = proceeds_sol + ?, "
                "buy_amount_sol = buy_amount_sol * ? "
                "WHERE tg_id = ? AND callout_id = ?",
                (pct_sold, proceeds_sol, max(0.0, 1 - pct_sold / 100.0),
                 tg_id, callout_id))

    async def get_callers(self, tg_id: Optional[int] = None) -> list[dict[str, Any]]:
        async with self._lock:
            if tg_id is None:
                # paid gate: only followers with an active subscription
                # (owner implicitly subscribed — row created lazily in main)
                rows = self._q(
                    "SELECT c.* FROM callers c "
                    "JOIN subscriptions s ON s.tg_id = c.tg_id AND s.active = 1 "
                    "WHERE c.enabled = 1")
            else:
                rows = self._q("SELECT * FROM callers WHERE tg_id = ?", (tg_id,))
        return [dict(r) for r in rows]

    async def get_caller(self, tg_id: int, caller_id: str) -> Optional[dict[str, Any]]:
        async with self._lock:
            rows = self._q("SELECT * FROM callers WHERE tg_id = ? AND caller_id = ?",
                           (tg_id, caller_id))
        return dict(rows[0]) if rows else None

    async def get_caller_by_mint(self, tg_id: int, mint: str) -> Optional[dict[str, Any]]:
        """Find the caller whose open position holds this mint (for /sell settings)."""
        async with self._lock:
            rows = self._q(
                "SELECT c.* FROM callers c JOIN positions p "
                "ON p.tg_id = c.tg_id AND p.caller_id = c.caller_id "
                "WHERE p.tg_id = ? AND p.mint = ? AND p.status = 'open' LIMIT 1",
                (tg_id, mint))
        return dict(rows[0]) if rows else None

    async def update_caller_cursor(self, tg_id: int, caller_id: str, last_callout_id: str):
        async with self._lock:
            self._w("UPDATE callers SET last_callout_id = ? WHERE tg_id = ? AND caller_id = ?",
                    (last_callout_id, tg_id, caller_id))

    # ---- positions ----
    async def open_position(self, tg_id: int, caller_id: str, callout_id: str, mint: str,
                            buy_sig: str, buy_amount_sol: float, tokens: float, entry_price: float):
        async with self._lock:
            self._w(
                "INSERT OR IGNORE INTO positions (tg_id, caller_id, callout_id, mint, buy_sig, "
                "buy_amount_sol, tokens_received, entry_price, status, opened_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open', ?)",
                (tg_id, caller_id, callout_id, mint, buy_sig, buy_amount_sol, tokens,
                 entry_price, int(time.time())),
            )

    async def mark_sold(self, tg_id: int, callout_id: str, sig: str, proceeds_sol: float):
        async with self._lock:
            self._w(
                "UPDATE positions SET status='sold', sold_sig=?, sold_amount_sol=?, closed_at=? "
                "WHERE tg_id=? AND callout_id=?",
                (sig, proceeds_sol, int(time.time()), tg_id, callout_id),
            )

    async def abandon_position(self, tg_id: int, callout_id: str, reason: str = ""):
        """Close a position record that never actually became a position
        (e.g. buy tx never confirmed) — distinct from mark_sold so PnL stats
        can exclude these (sold_amount_sol stays 0)."""
        async with self._lock:
            self._w(
                "UPDATE positions SET status='abandoned', sold_sig=?, closed_at=? "
                "WHERE tg_id=? AND callout_id=? AND status='open'",
                (reason, int(time.time()), tg_id, callout_id),
            )

    async def get_open_positions(self, tg_id: int) -> list[dict[str, Any]]:
        async with self._lock:
            rows = self._q("SELECT * FROM positions WHERE tg_id = ? AND status = 'open'", (tg_id,))
        return [dict(r) for r in rows]

    async def get_position_by_id(self, pos_id: int) -> Optional[dict[str, Any]]:
        async with self._lock:
            rows = self._q("SELECT * FROM positions WHERE id = ?", (pos_id,))
        return dict(rows[0]) if rows else None

    async def get_position_by_mint(self, tg_id: int, mint: str) -> Optional[dict[str, Any]]:
        async with self._lock:
            rows = self._q(
                "SELECT * FROM positions WHERE tg_id = ? AND mint = ? AND status = 'open'",
                (tg_id, mint))
        return dict(rows[0]) if rows else None

    async def get_all_open_positions(self) -> list[dict[str, Any]]:
        """Paid gate: only positions of actively-subscribed users are managed."""
        async with self._lock:
            rows = self._q(
                "SELECT p.* FROM positions p "
                "JOIN subscriptions s ON s.tg_id = p.tg_id AND s.active = 1 "
                "WHERE p.status = 'open'")
        return [dict(r) for r in rows]

    # ---- dedupe ----
    async def seen(self, callout_id: str) -> bool:
        async with self._lock:
            rows = self._q("SELECT 1 FROM dedupe WHERE callout_id = ?", (callout_id,))
        return bool(rows)

    async def mark_seen(self, callout_id: str):
        async with self._lock:
            self._w("INSERT OR IGNORE INTO dedupe (callout_id, seen_at) VALUES (?, ?)",
                    (callout_id, int(time.time())))

    async def prune_dedupe(self, older_than_s: int = 86400 * 3):
        async with self._lock:
            self._w("DELETE FROM dedupe WHERE seen_at < ?", (int(time.time()) - older_than_s,))

    # ---- access (one-time unlock codes + Stars purchases) ----
    async def create_access_code(self, code: str, created_by: int,
                                 source: str = "manual") -> bool:
        async with self._lock:
            try:
                self._w(
                    "INSERT INTO access_codes (code, created_by, created_at, source) "
                    "VALUES (?, ?, ?, ?)",
                    (code.upper(), created_by, int(time.time()), source))
                return True
            except sqlite3.IntegrityError:
                return False

    async def redeem_access_code(self, code: str, tg_id: int) -> bool:
        """One-time redeem: flips access_codes.used_by and inserts a subscription
        atomically. Returns False if the code is unknown or already used."""
        code = code.upper().strip()
        now = int(time.time())
        async with self._lock:
            rows = self._q(
                "SELECT used_by FROM access_codes WHERE code = ?", (code,))
            if not rows or rows[0]["used_by"]:
                return False
            self._w("UPDATE access_codes SET used_by = ?, used_at = ? WHERE code = ?",
                    (tg_id, now, code))
            self._w(
                "INSERT INTO subscriptions (tg_id, unlocked_at, access_code, active) "
                "VALUES (?, ?, ?, 1) "
                "ON CONFLICT(tg_id) DO UPDATE SET active = 1, access_code = excluded.access_code",
                (tg_id, now, code))
            return True

    async def unlock_via_stars(self, tg_id: int, charge_id: str):
        """Mark a user unlocked after a successful Stars payment."""
        async with self._lock:
            self._w(
                "INSERT INTO subscriptions (tg_id, unlocked_at, access_code, active) "
                "VALUES (?, ?, ?, 1) "
                "ON CONFLICT(tg_id) DO UPDATE SET active = 1, access_code = excluded.access_code",
                (tg_id, int(time.time()), f"stars:{charge_id}"))

    async def is_unlocked(self, tg_id: int) -> bool:
        async with self._lock:
            rows = self._q(
                "SELECT active FROM subscriptions WHERE tg_id = ? AND active = 1",
                (tg_id,))
        return bool(rows)

    async def unlocked_ids(self) -> list[int]:
        async with self._lock:
            rows = self._q("SELECT tg_id FROM subscriptions WHERE active = 1")
        return [r["tg_id"] for r in rows]

    async def list_codes(self, limit: int = 20) -> list[dict[str, Any]]:
        async with self._lock:
            rows = self._q(
                "SELECT code, created_by, created_at, used_by, used_at, source "
                "FROM access_codes ORDER BY created_at DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]

    async def list_subs(self, limit: int = 50) -> list[dict[str, Any]]:
        async with self._lock:
            rows = self._q(
                "SELECT tg_id, unlocked_at, access_code FROM subscriptions "
                "WHERE active = 1 ORDER BY unlocked_at DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]
