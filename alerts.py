"""Telegram message building and sending."""
import asyncio
import logging
from contextlib import closing
from typing import Optional

from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import BadRequest, RetryAfter, TelegramError

from .config import (
    HIGH_CONVICTION_MAX_STD, HIGH_CONVICTION_PROB, HIGH_CONVICTION_SCORE,
    PIN_HIGH_CONVICTION, PUMP_FRONT,
)
from .db import db_conn, db_write, upsert_chat
from .state import BotState
from .utils import (
    REC_EMOJI, fmt_prob, fmt_usd, mdbold, mdcode, mditalic,
    now_ts, safe_float, safe_int, score_emoji, strip_md2, validate_url, fmt_duration,
)

log = logging.getLogger(__name__)


def build_message(coin: dict, result: dict) -> str:
    name    = coin.get("name", "Unknown") or "Unknown"
    symbol  = coin.get("symbol", "???") or "???"
    mint    = coin.get("mint", "")
    mc      = safe_float(coin.get("usd_market_cap"))
    score   = safe_int(result.get("score", 0))
    replies = safe_int(coin.get("reply_count"))
    ml_p    = result.get("ml_probability")
    f_p     = safe_float(result.get("formula_probability", 0))
    std     = safe_float(result.get("ml_cv_auc_std", 0.0))
    mode    = result.get("mode", "") or ""
    red_f   = result.get("red_flags", []) or []

    tw_url = validate_url(coin.get("twitter"), social=True)
    tg_url = validate_url(coin.get("telegram"), social=True)
    wb_url = validate_url(coin.get("website"))
    socials_parts = []
    if tw_url: socials_parts.append("🐦 Twitter")
    if tg_url: socials_parts.append("✈️ Telegram")
    if wb_url: socials_parts.append("🌐 Website")
    socials = " · ".join(socials_parts)

    rec   = result.get("recommendation", "") or ""
    rec_e = REC_EMOJI.get(rec, "")

    prob_line = f"📈 {mdcode(fmt_prob(result.get('probability', 0)))} pump prob"
    if ml_p is not None:
        std_str = f" ±{mdcode(f'{std:.2f}')}" if std > 0 else ""
        prob_line += (
            f" \\(ML {mdcode(fmt_prob(ml_p))}{std_str} · "
            f"formula {mdcode(fmt_prob(f_p))}\\)"
        )
    else:
        prob_line += (
            f" \\(formula {mdcode(fmt_prob(f_p))} · "
            f"{mditalic('ML accumulating')}\\)"
        )

    lines = [
        f"{score_emoji(score)} {mdbold(f'Score: {score}/10')}  {rec_e}".rstrip(),
        f"{mdbold(name)} \\({mdcode('$' + symbol)}\\)",
        "",
        prob_line,
        f"⚙️ Mode: {mditalic(mode) or '—'}",
        f"💬 Replies {mdcode(replies)}",
    ]
    if red_f:
        lines.append(f"🚩 {mditalic('; '.join(red_f))}")
    if socials:
        lines.append("")
        lines.append(socials)
    lines += [
        "",
        f"💰 {mdcode(fmt_usd(mc, 0))} market cap",
        f"🪙 {mdcode(mint)}",
    ]
    if mint:
        lines.append(f"🔗 [Pump\\.fun]({PUMP_FRONT}/{mint})")
    return "\n".join(line for line in lines if line is not None)


def build_keyboard(coin: dict) -> Optional[InlineKeyboardMarkup]:
    mint = coin.get("mint", "")
    if not mint:
        return None
    rows: list[list[InlineKeyboardButton]] = [
        [InlineKeyboardButton("🔗 Open on Pump.fun", url=f"{PUMP_FRONT}/{mint}")]
    ]
    tw = validate_url(coin.get("twitter"), social=True)
    tg = validate_url(coin.get("telegram"), social=True)
    wb = validate_url(coin.get("website"))
    if tw: rows.append([InlineKeyboardButton("🐦 Twitter",  url=tw)])
    if tg: rows.append([InlineKeyboardButton("✈️ Telegram", url=tg)])
    if wb: rows.append([InlineKeyboardButton("🌐 Website",  url=wb)])
    return InlineKeyboardMarkup(rows)


async def _send_with_fallback(
    bot: Bot, chat_id: int, text: str, kb: Optional[InlineKeyboardMarkup],
):
    try:
        return await bot.send_message(
            chat_id=chat_id, text=text,
            parse_mode="MarkdownV2", reply_markup=kb,
            disable_web_page_preview=False,
        )
    except BadRequest as e:
        if "can't parse entities" in str(e).lower() or "parse" in str(e).lower():
            log.warning("MD2 parse failed for chat %s: %s — sending plain", chat_id, e)
            try:
                return await bot.send_message(
                    chat_id=chat_id, text=strip_md2(text),
                    reply_markup=kb, disable_web_page_preview=False,
                )
            except TelegramError as e2:
                log.error("Plain fallback also failed %s: %s", chat_id, e2)
                raise
        else:
            raise


async def send_trade_opened(
    bot: Bot, coin: dict, trade: "OpenTrade", state: BotState,
) -> None:
    """Notify all chats with paper_reports_enabled and pin the message."""
    from .config import PUMP_FRONT
    from .db import db_write
    from .utils import now_ts

    name   = coin.get("name", "Unknown") or "Unknown"
    symbol = coin.get("symbol", "???") or "???"
    mint   = coin.get("mint", "")
    mc     = trade.entry_mc
    size   = trade.position_size_usd
    score  = trade.entry_score
    prob   = trade.entry_prob
    sl     = trade.dynamic_sl_pct or 20.0
    tp     = trade.dynamic_tp_pct or 35.0
    tsec   = trade.dynamic_time_stop or 14400

    lines = [
        f"ðŸ" + "‡" + " {mdbold('PAPER TRADE OPENED')}",
        f"{mdbold(name)} \\({mdcode('$' + symbol)}\\)",
        "",
        f"ðŸ" + "’" + " Size: {mdcode(fmt_usd(size, 2))}",
        f"ðŸ" + "’" + " Entry MC: {mdcode(fmt_usd(mc, 0))}",
        f"ðŸ" + "“" + " Score: {mdcode(str(score) + '/10')}  "
        f"Prob: {mdcode(fmt_prob(prob))}",
        "",
        f"ðŸ" + "š" + " SL: {mdcode(f'{sl:.1f}%')}  "
        f"TP: {mdcode(f'{tp:.1f}%')}  "
        f"Time: {mdcode(fmt_duration(tsec))}",
    ]
    if mint:
        lines.append(f"ðŸ" + "ª™" + " {mdcode(mint)}")
        lines.append(f"ðŸ" + "”—" + " [Pump\\.fun]({PUMP_FRONT}/{mint})")

    text = "\n".join(lines)
    kb_rows = [[InlineKeyboardButton("ðŸ" + "”—" + " Open on Pump.fun",
                                     url=f"{PUMP_FRONT}/{mint}")]] if mint else []
    kb = InlineKeyboardMarkup(kb_rows) if kb_rows else None

    with closing(db_conn()) as conn:
        rows = conn.execute(
            "SELECT chat_id FROM chat_settings WHERE paper_reports_enabled=1"
        ).fetchall()
    chats = [int(r["chat_id"]) for r in rows]

    for chat_id in chats:
        try:
            sent = await _send_with_fallback(bot, chat_id, text, kb)
            if sent is not None:
                try:
                    await bot.pin_chat_message(
                        chat_id, sent.message_id, disable_notification=True,
                    )
                    def _save_pin(tid=trade.id, cid=chat_id,
                                  mid=sent.message_id, m=mint):
                        with closing(db_conn()) as conn, conn:
                            conn.execute(
                                "INSERT OR IGNORE INTO pinned_trades"
                                "(chat_id,message_id,trade_id,mint,pinned_at) "
                                "VALUES(?,?,?,?,?)",
                                (cid, mid, tid, m, now_ts()))
                    db_write(_save_pin)
                except TelegramError as e:
                    log.debug("pin trade msg failed %s: %s", chat_id, e)
        except TelegramError as e:
            log.error("trade open notify failed %s: %s", chat_id, e)


async def send_trade_closed(
    bot: Bot, trade: "OpenTrade", exit_mc: float, reason: str,
) -> None:
    """Notify all chats with paper_reports_enabled about a closed trade."""
    from .utils import now_ts

    name   = trade.name or "Unknown"
    symbol = trade.symbol or "???"
    pnl_usd = (trade.position_size_usd
               * ((exit_mc * (1 - 0.02) - trade.entry_mc) / trade.entry_mc)
               if trade.entry_mc > 0 else 0)
    pnl_pct = ((exit_mc - trade.entry_mc) / trade.entry_mc * 100
               if trade.entry_mc > 0 else 0)

    sign = "+" if pnl_pct >= 0 else ""
    arrow = "ðŸ" + "š" + "—" if pnl_pct >= 0 else "ðŸ" + "š" + "“"
    color = "ðŸ" + "Ÿ" if pnl_pct >= 0 else "ðŸ" + "”"

    lines = [
        f"{color} {mdbold('PAPER TRADE CLOSED')}",
        f"{mdbold(name)} \\({mdcode('$' + symbol)}\\)",
        "",
        f"{arrow} P&L: {mdcode(f'{sign}{pnl_pct:.1f}%')}  "
        f"${mdcode(f'{pnl_usd:+.2f}')}",
        f"ðŸ" + "’”" + " Reason: {mditalic(reason)}",
        f"ðŸ" + "’" + " Duration: "
        f"{mdcode(fmt_duration(now_ts() - trade.entry_time))}",
    ]

    text = "\n".join(lines)
    kb = None

    with closing(db_conn()) as conn:
        rows = conn.execute(
            "SELECT chat_id FROM chat_settings WHERE paper_reports_enabled=1"
        ).fetchall()
    chats = [int(r["chat_id"]) for r in rows]

    for chat_id in chats:
        try:
            await _send_with_fallback(bot, chat_id, text, kb)
            # Unpin the corresponding open-trade message
            try:
                with closing(db_conn()) as conn:
                    pin_row = conn.execute(
                        "SELECT message_id FROM pinned_trades "
                        "WHERE chat_id=? AND trade_id=?",
                        (chat_id, trade.id),
                    ).fetchone()
                if pin_row:
                    try:
                        await bot.unpin_chat_message(
                            chat_id, pin_row["message_id"])
                    except TelegramError:
                        pass
                    def _del_pin(cid=chat_id, mid=pin_row["message_id"]):
                        with closing(db_conn()) as conn, conn:
                            conn.execute(
                                "DELETE FROM pinned_trades "
                                "WHERE chat_id=? AND message_id=?",
                                (cid, mid))
                    db_write(_del_pin)
            except Exception:
                pass
        except TelegramError as e:
            log.error("trade close notify failed %s: %s", chat_id, e)


async def send_alert(bot: Bot, coin: dict, result: dict, state: BotState) -> None:
    if not state.alerts:
        return
    score = safe_int(result.get("score", 0))
    prob  = safe_float(result.get("probability", 0))
    std   = safe_float(result.get("ml_cv_auc_std", 0.0))
    is_high_conviction = (
        PIN_HIGH_CONVICTION
        and score >= HIGH_CONVICTION_SCORE
        and prob  >= HIGH_CONVICTION_PROB
        and std   <= HIGH_CONVICTION_MAX_STD
    )
    try:
        text = build_message(coin, result)
        if is_high_conviction:
            text = f"⭐ {mdbold('HIGH CONVICTION')} ⭐\n\n" + text
    except Exception as e:
        log.error("build_message failed: %s", e)
        return
    kb = build_keyboard(coin)

    # Snapshot the dict before iteration.  Callback handlers can mutate
    # state.alerts between awaits; the snapshot prevents mid-loop surprises.
    # Chats to disable are collected and removed AFTER the loop completes.
    alerts_snapshot = list(state.alerts.items())
    stale_chats: list[int] = []

    for chat_id, threshold in alerts_snapshot:
        if score < threshold:
            continue
        try:
            sent = await _send_with_fallback(bot, chat_id, text, kb)
            log.info("Alert → %s | %s | %s/10", chat_id, coin.get("name"), score)
            if is_high_conviction and sent is not None:
                try:
                    await bot.pin_chat_message(
                        chat_id, sent.message_id, disable_notification=True,
                    )
                    def _save_pin():
                        with closing(db_conn()) as conn, conn:
                            conn.execute(
                                "INSERT OR IGNORE INTO pinned_alerts"
                                "(chat_id,message_id,mint,pinned_at) VALUES(?,?,?,?)",
                                (chat_id, sent.message_id, coin.get("mint", ""), now_ts()))
                    db_write(_save_pin)
                except TelegramError as e:
                    log.debug("pin failed %s: %s", chat_id, e)
        except RetryAfter as e:
            log.warning("Rate limited on chat %s — sleeping %.0fs",
                        chat_id, e.retry_after)
            await asyncio.sleep(e.retry_after)
            try:
                await _send_with_fallback(bot, chat_id, text, kb)
            except TelegramError as retry_err:
                log.error("Retry send failed %s: %s", chat_id, retry_err)
        except TelegramError as e:
            log.error("Send failed %s: %s", chat_id, e)
            err_str = str(e).lower()
            if any(kw in err_str for kw in
                   ("blocked", "kicked", "chat not found", "deactivated")):
                stale_chats.append(chat_id)
                log.warning("Marking chat %s as unreachable", chat_id)

    # Remove stale chats after the loop — not during it
    for chat_id in stale_chats:
        state.alerts.pop(chat_id, None)
        try:
            upsert_chat(chat_id, alerts_enabled=0)
        except Exception:
            pass
        log.warning("Removed unreachable chat %s from alerts", chat_id)

