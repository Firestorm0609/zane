"""Paper wallet: balance, equity, risk gates."""
from contextlib import closing

from .config import PAPER_STARTING_BALANCE_USD
from .db import db_conn, db_write
from .utils import now_ts, safe_float


class PaperWallet:
    @staticmethod
    def get_balance() -> float:
        with closing(db_conn()) as conn:
            r = conn.execute("SELECT balance_usd FROM paper_wallet WHERE id=1").fetchone()
            return safe_float(r["balance_usd"]) if r else 0.0

    @staticmethod
    def get_starting() -> float:
        with closing(db_conn()) as conn:
            r = conn.execute("SELECT starting_usd FROM paper_wallet WHERE id=1").fetchone()
            return safe_float(r["starting_usd"]) if r else PAPER_STARTING_BALANCE_USD

    @staticmethod
    def reset(new_balance: float = None) -> None:
        if new_balance is None:
            new_balance = PAPER_STARTING_BALANCE_USD
        def _w():
            with closing(db_conn()) as conn, conn:
                conn.execute(
                    "UPDATE paper_wallet SET balance_usd=?, starting_usd=?, "
                    "created_at=?, updated_at=? WHERE id=1",
                    (new_balance, new_balance, now_ts(), now_ts()))
        db_write(_w)

    @staticmethod
    def equity() -> dict:
        balance = PaperWallet.get_balance()
        with closing(db_conn()) as conn:
            opens = conn.execute("""
                SELECT t.id, t.entry_mc, t.position_size_usd, t.mint,
                       (SELECT s.market_cap FROM paper_mc_snapshots s
                        WHERE s.mint = t.mint
                        ORDER BY s.created_at DESC LIMIT 1) AS cur_mc
                FROM paper_trades t WHERE t.status='OPEN'
            """).fetchall()
        unrealized = 0.0
        positions_value = 0.0
        for r in opens:
            entry_mc = safe_float(r["entry_mc"])
            size     = safe_float(r["position_size_usd"])
            if entry_mc <= 0 or size <= 0:
                continue
            cur_mc = safe_float(r["cur_mc"]) if r["cur_mc"] is not None else entry_mc
            pnl_pct = ((cur_mc - entry_mc) / entry_mc) * 100
            position_value = size * (1 + pnl_pct / 100)
            positions_value += position_value
            unrealized      += (position_value - size)
        return {
            "balance":         balance,
            "positions_value": positions_value,
            "unrealized":      unrealized,
            "total_equity":    balance + positions_value,
            "starting":        PaperWallet.get_starting(),
        }


def recent_loss_streak(limit: int = 5) -> int:
    with closing(db_conn()) as conn:
        rows = conn.execute(
            "SELECT pnl_pct FROM paper_trades WHERE status='CLOSED' "
            "ORDER BY exit_time DESC LIMIT ?", (limit,)
        ).fetchall()
    streak = 0
    for r in rows:
        if safe_float(r["pnl_pct"]) < 0:
            streak += 1
        else:
            break
    return streak


def daily_pnl_usd() -> float:
    cutoff = now_ts() - 86400
    with closing(db_conn()) as conn:
        r = conn.execute(
            "SELECT COALESCE(SUM(pnl_usd), 0) AS s FROM paper_trades "
            "WHERE status='CLOSED' AND exit_time >= ?", (cutoff,)
        ).fetchone()
    return safe_float(r["s"]) if r else 0.0
