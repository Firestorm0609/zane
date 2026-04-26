"""Telegram message building and sending."""
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
    now_ts, safe_float, safe_int, score_emoji, strip_md2, validate_url,
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

    tw_url = validate_url(coin.get("twitter"))
    tg_url = validate_url(coin.get("telegram"))
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
    tw = validate_url(coin.get("twitter"))
    tg = validate_url(coin.get("telegram"))
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

    for chat_id, threshold in list(state.alerts.items()):
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
            import asyncio
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
                state.alerts.pop(chat_id, None)
                try:
                    upsert_chat(chat_id, alerts_enabled=0)
                except Exception:
                    pass
                log.warning("Removed unreachable chat %s", chat_id)

