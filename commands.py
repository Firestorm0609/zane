"""Slash-command handlers."""
import asyncio
import logging
from collections import defaultdict
from contextlib import closing
from typing import Optional

import aiohttp
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import BadRequest, TelegramError
from telegram.ext import ContextTypes

from .alerts import build_keyboard, build_message
from .config import (
    DEFAULT_THRESHOLD, HTTP_TIMEOUT_SEC, MIN_TRAIN_SAMPLES, ML_AVAILABLE,
    ML_LABEL_WINDOW, PAPER_STARTING_BALANCE_USD, PUMP_THRESHOLD_PCT,
    RUG_THRESHOLD_PCT,
)
from .db import db_conn, db_write, set_state, upsert_chat
from .enrichment import enrich_with_rpc, fetch_mc_momentum_from_db
from .keyboards import MENU_HEADER, main_menu_keyboard
from .lookback import train_executor
from .market import MarketContext
from .scoring import ScoringEngine
from .state import BotState, blacklist_cache
from .ui_text import (
    format_top_performers, query_top_performers, text_features,
    text_health, text_help, text_keywords, text_market, text_model,
    text_monitor_status, text_outcomes, text_paper_report,
    text_paper_status, text_scoring_mode, text_snapshot, text_stats,
    text_wallet,
)
from .utils import (
    fmt_duration, fmt_pct, fmt_prob, fmt_usd,
    mdbold, mdcode, mditalic, now_ts, safe_float, safe_int, strip_md2,
)
from .wallet import PaperWallet

log = logging.getLogger(__name__)

PM = "MarkdownV2"


def _state(ctx: ContextTypes.DEFAULT_TYPE) -> BotState:
    return ctx.bot_data["state"]


async def _reply(update: Update, text: str,
                 kb: Optional[InlineKeyboardMarkup] = None) -> None:
    try:
        await update.message.reply_text(text, parse_mode=PM, reply_markup=kb)
    except BadRequest as e:
        if "parse" in str(e).lower():
            log.warning("MD2 parse failed on reply: %s", e)
            try:
                await update.message.reply_text(strip_md2(text), reply_markup=kb)
            except TelegramError as e2:
                log.error("Plain reply failed: %s", e2)
        else:
            log.debug("reply BadRequest: %s", e)


async def do_train(engine: ScoringEngine) -> str:
    """Shared training handler — used by both /train and the Train button."""
    if not ML_AVAILABLE:
        return f"⚠️ {mdbold('ML not available')} — install scikit-learn, joblib, numpy"
    try:
        loop    = asyncio.get_running_loop()
        trained = await loop.run_in_executor(train_executor, engine.train)
        if trained:
            s = engine.status()
            return (
                f"✅ {mdbold('Model trained!')}\n"
                f"CV AUC: {mdcode(str(round(s['cv_auc'],3)))} "
                f"±{mdcode(str(round(s['cv_auc_std'],3)))} \\| "
                f"ML weight: {mdcode(str(round(s['ml_weight']*100))+'%')}\n"
                f"BUY ≥ {mdcode(fmt_prob(s['buy_threshold']))} \\| "
                f"WATCH ≥ {mdcode(fmt_prob(s['watch_threshold']))}"
            )
        with closing(db_conn()) as conn:
            n = conn.execute(
                "SELECT COUNT(*) AS c FROM lookbacks "
                "WHERE window_label=? AND checked=1",
                (ML_LABEL_WINDOW,),
            ).fetchone()["c"]
        return f"⚠️ Need {mdcode(MIN_TRAIN_SAMPLES)} samples, have {mdcode(n)}"
    except Exception as e:
        return f"❌ {mdcode(str(e))}"


# ---------- Basic ----------


from .config import ALLOWED_CHAT_IDS

async def _check_allowed(update: Update) -> bool:
    if not ALLOWED_CHAT_IDS:
        return True
    chat_id = update.effective_chat.id
    if chat_id not in ALLOWED_CHAT_IDS:
        await update.message.reply_text("⛔ Unauthorized.")
        return False
    return True

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    await _reply(
        update,
        f"🤖 {mdbold('Pump.fun Dynamic Monitor v1.1')}\n\n"
        f"Tap {mdbold('Menu')} below to control the bot without typing any commands\\.",
        kb=InlineKeyboardMarkup([[
            InlineKeyboardButton("📋 Open Menu", callback_data="menu"),
        ]]),
    )


async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    await _reply(update, text_help())


async def cmd_menu(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    await _reply(update, MENU_HEADER, kb=main_menu_keyboard())


# ---------- Monitor ----------

async def cmd_monitor_on(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    state = _state(ctx)
    cid   = update.effective_chat.id
    th    = state.alerts.get(cid, DEFAULT_THRESHOLD)
    state.alerts[cid] = th
    upsert_chat(cid, alerts_enabled=1, threshold=th)
    await _reply(update, f"🟢 {mdbold('Alerts ON')} — threshold {mdcode(f'{th}/10')}")


async def cmd_monitor_off(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    state = _state(ctx)
    state.alerts.pop(update.effective_chat.id, None)
    upsert_chat(update.effective_chat.id, alerts_enabled=0)
    await _reply(update, f"🔴 {mdbold('Alerts OFF')}")


async def cmd_monitor_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    await _reply(update, text_monitor_status(update.effective_chat.id, _state(ctx)))


async def cmd_set_threshold(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    state = _state(ctx)
    cid   = update.effective_chat.id
    if not ctx.args or not ctx.args[0].lstrip("-").isdigit():
        await _reply(update, f"Usage: {mdcode('/set_threshold <1-10>')}")
        return
    val = int(ctx.args[0])
    if not 1 <= val <= 10:
        await _reply(update, "Must be 1–10\\.")
        return
    state.alerts[cid] = val
    upsert_chat(cid, threshold=val, alerts_enabled=1)
    await _reply(update, f"✅ Threshold → {mdcode(f'{val}/10')}")


# ---------- Info ----------

async def cmd_scoring_mode(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    await _reply(update, text_scoring_mode(ctx.bot_data["engine"]))


async def cmd_features(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    await _reply(update, text_features(ctx.bot_data["engine"]))


async def cmd_keywords(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    await _reply(update, text_keywords(ctx.bot_data["engine"]))


async def cmd_market(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    await _reply(update, text_market(ctx.bot_data["market_ctx"]))


async def cmd_outcomes(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    await _reply(update, text_outcomes())


async def cmd_model(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    await _reply(update, text_model(ctx.bot_data["engine"]))


async def cmd_train(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    engine: ScoringEngine = ctx.bot_data["engine"]
    await _reply(update, "🏋 Training\\.\\.\\.")
    await _reply(update, await do_train(engine))


async def cmd_snapshot(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    await _reply(update, text_snapshot())


# ---------- Paper ----------

async def cmd_paper_on(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    state = _state(ctx)
    await state.set_paper_enabled(True)
    set_state("paper_engine_enabled", "1")
    cid = update.effective_chat.id
    state.paper_chats.add(cid)
    upsert_chat(cid, paper_reports_enabled=1)
    await _reply(update, f"✅ {mdbold('Paper trading ON')}")


async def cmd_paper_off(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    state = _state(ctx)
    await state.set_paper_enabled(False)
    set_state("paper_engine_enabled", "0")
    cid = update.effective_chat.id
    state.paper_chats.discard(cid)
    upsert_chat(cid, paper_reports_enabled=0)
    await _reply(update, f"❌ {mdbold('Paper trading OFF')}")


async def cmd_paper_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    await _reply(update, text_paper_status(_state(ctx)))


async def cmd_paper_report(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    await _reply(update, text_paper_report(_state(ctx)))


# ---------- Diagnostics ----------

async def cmd_health(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    await _reply(update, text_health(_state(ctx), ctx.bot_data["engine"]))


async def cmd_stats(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    await _reply(update, text_stats(_state(ctx), ctx.bot_data["engine"]))


# ---------- Wallet ----------

async def cmd_wallet(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    await _reply(update, text_wallet())


async def cmd_wallet_reset(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    amount = PAPER_STARTING_BALANCE_USD
    if ctx.args and ctx.args[0].replace(".", "").isdigit():
        amount = float(ctx.args[0])
    PaperWallet.reset(amount)
    await _reply(
        update,
        f"✅ Wallet reset to {mdcode(fmt_usd(amount, 2))}\n"
        f"{mditalic('Note: open positions remain — close them manually if needed.')}",
    )


# ---------- Score / backtest / watchlist ----------

async def cmd_score(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    if not ctx.args or len(ctx.args[0]) < 32:
        await _reply(update, f"Usage: {mdcode('/score <mint_address>')}")
        return
    mint = ctx.args[0].strip()
    await _reply(update, f"🔍 Fetching {mdcode(mint[:8])}\\.\\.\\.")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"https://frontend-api-v3.pump.fun/coins/{mint}",
                timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT_SEC),
            ) as resp:
                if resp.status != 200:
                    await _reply(update, f"❌ pump\\.fun returned HTTP {mdcode(resp.status)}")
                    return
                data = await resp.json(content_type=None)

            if not isinstance(data, dict) or not data.get("mint"):
                await _reply(update, "❌ Coin not found")
                return

            engine: ScoringEngine = ctx.bot_data["engine"]

            coin = {**data, "mint": mint}
            # Enrich with on-chain data so on-chain features are populated,
            # matching the live pipeline behaviour.
            coin = await enrich_with_rpc(coin, session)

        loop = asyncio.get_running_loop()
        coin["_mc_momentum_pct"] = await loop.run_in_executor(
            None, fetch_mc_momentum_from_db, mint,
        )
        result = engine.score(coin)
        text = build_message(coin, result)
        kb = build_keyboard(coin)
        await _reply(
            update,
            text + "\n\n" + mditalic("\\(manual scoring — not saved\\)"),
            kb=kb,
        )
    except Exception as e:
        await _reply(update, f"❌ {mdcode(str(e))}")


async def cmd_backtest(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    await _reply(update, "📊 Running backtest\\.\\.\\.")
    try:
        with closing(db_conn()) as conn:
            rows = conn.execute("""
                SELECT s.score, lb.outcome, lb.pct_change
                FROM signals s
                JOIN lookbacks lb ON lb.signal_id = s.id
                WHERE lb.window_label = ?
                  AND lb.checked = 1
                  AND lb.outcome IS NOT NULL
                ORDER BY s.score
            """, (ML_LABEL_WINDOW,)).fetchall()

        if not rows:
            await _reply(update, mditalic("No labeled data yet — check back later"))
            return

        buckets: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            score = safe_int(r["score"])
            pct   = safe_float(r["pct_change"])
            if   score >= 9: bucket = "9-10 🔥"
            elif score >= 7: bucket = "7-8 ⭐"
            elif score >= 5: bucket = "5-6 👍"
            else:            bucket = "1-4 🤔"
            buckets[bucket].append(pct)

        lines = [
            f"🎯 {mdbold('Backtest Results')}",
            f"{mditalic(f'Label window: {ML_LABEL_WINDOW} | {len(rows)} total labeled signals')}",
            "",
        ]

        for bucket in ["9-10 🔥", "7-8 ⭐", "5-6 👍", "1-4 🤔"]:
            vals = buckets.get(bucket, [])
            if not vals:
                continue
            n = len(vals)
            pumps = sum(1 for v in vals if v >= PUMP_THRESHOLD_PCT)
            rugs  = sum(1 for v in vals if v <= RUG_THRESHOLD_PCT)
            avg_pct = sum(vals) / n
            lines += [
                f"{mdbold(bucket)}",
                f"Signals {mdcode(n)} \\| "
                f"Pump rate {mdcode(f'{pumps/n*100:.1f}%')} \\| "
                f"Rug rate {mdcode(f'{rugs/n*100:.1f}%')}",
                f"Avg outcome {mdcode(fmt_pct(avg_pct, 1, signed=True))}",
                "",
            ]
        await _reply(update, "\n".join(lines))
    except Exception as e:
        await _reply(update, f"❌ {mdcode(str(e))}")


async def cmd_watch(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    if not ctx.args or len(ctx.args[0]) < 32:
        await _reply(update, f"Usage: {mdcode('/watch <mint_address>')}")
        return
    mint = ctx.args[0].strip()
    chat_id = update.effective_chat.id

    with closing(db_conn()) as conn:
        sig = conn.execute(
            "SELECT name, symbol FROM signals WHERE mint=? "
            "ORDER BY created_at DESC LIMIT 1",
            (mint,),
        ).fetchone()
    name   = sig["name"]   if sig else ""
    symbol = sig["symbol"] if sig else ""

    try:
        def _w():
            with closing(db_conn()) as conn, conn:
                conn.execute(
                    "INSERT OR IGNORE INTO watchlist"
                    "(chat_id,mint,name,symbol,added_at) VALUES(?,?,?,?,?)",
                    (chat_id, mint, name, symbol, now_ts()))
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, db_write, _w)
        await _reply(update, f"👁 Watching {mdbold(name or mint[:8])}")
    except Exception as e:
        await _reply(update, f"❌ {mdcode(str(e))}")


async def cmd_unwatch(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    if not ctx.args:
        await _reply(update, f"Usage: {mdcode('/unwatch <mint_address>')}")
        return
    mint = ctx.args[0].strip()
    chat_id = update.effective_chat.id

    def _w():
        with closing(db_conn()) as conn, conn:
            conn.execute(
                "DELETE FROM watchlist WHERE chat_id=? AND mint=?",
                (chat_id, mint))
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, db_write, _w)
    await _reply(update, f"✅ Removed {mdcode(mint[:8])} from watchlist")


async def cmd_watchlist(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    chat_id = update.effective_chat.id
    with closing(db_conn()) as conn:
        rows = conn.execute(
            "SELECT mint, name, symbol, added_at FROM watchlist "
            "WHERE chat_id=? ORDER BY added_at DESC",
            (chat_id,),
        ).fetchall()
    if not rows:
        await _reply(update, mditalic("Your watchlist is empty. Use /watch <mint>"))
        return

    from .config import PUMP_FRONT
    lines = [f"👁 {mdbold('Your Watchlist')} \\({len(rows)} coins\\)", ""]
    for r in rows:
        name = r["name"] or r["symbol"] or (r["mint"] or "?")[:8]
        mint = r["mint"] or ""
        age = fmt_duration(now_ts() - safe_int(r["added_at"]))
        lines.append(
            f"• {mdbold(name)} {mdcode(mint[:8])} \\| added {mdcode(age)} ago"
        )
        if mint:
            lines.append(f"   🔗 [Pump\\.fun]({PUMP_FRONT}/{mint})")
    await _reply(update, "\n".join(lines))


# ---------- Blacklist / top ----------

async def cmd_blacklist(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    args = ctx.args or []
    if not args:
        with closing(db_conn()) as conn:
            rows = conn.execute(
                "SELECT creator, reason, auto_added, added_at FROM creator_blacklist "
                "ORDER BY added_at DESC LIMIT 30"
            ).fetchall()
        if not rows:
            await _reply(update, mditalic("No blacklisted creators."))
            return
        lines = [f"🚫 {mdbold('Creator Blacklist')} \\({len(rows)}\\)", ""]
        for r in rows:
            tag = "🤖" if r["auto_added"] else "👤"
            age = fmt_duration(now_ts() - safe_int(r["added_at"]))
            lines.append(
                f"{tag} {mdcode(r['creator'][:12])} — "
                f"{mditalic(r['reason'] or 'manual')} — {mdcode(age)} ago"
            )
        lines.append("")
        lines.append(mditalic(
            "Usage: /blacklist add <wallet> [reason] | /blacklist remove <wallet>"))
        await _reply(update, "\n".join(lines))
        return

    action = args[0].lower()
    if action == "add" and len(args) >= 2:
        creator = args[1]
        reason  = " ".join(args[2:]) or "manual"
        def _w():
            with closing(db_conn()) as conn, conn:
                conn.execute(
                    "INSERT OR REPLACE INTO creator_blacklist"
                    "(creator,reason,added_at,auto_added) VALUES(?,?,?,0)",
                    (creator, reason, now_ts()))
        db_write(_w)
        blacklist_cache.invalidate()
        await _reply(update, f"🚫 Blacklisted {mdcode(creator[:12])}")
    elif action == "remove" and len(args) >= 2:
        creator = args[1]
        def _w():
            with closing(db_conn()) as conn, conn:
                conn.execute(
                    "DELETE FROM creator_blacklist WHERE creator=?",
                    (creator,))
        db_write(_w)
        blacklist_cache.invalidate()
        await _reply(update, f"✅ Removed {mdcode(creator[:12])} from blacklist")
    else:
        await _reply(update, mditalic(
            "Usage: /blacklist [add|remove] <wallet> [reason]"))


async def cmd_top(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await _check_allowed(update): return
    days = 7
    if ctx.args and ctx.args[0].isdigit():
        days = max(1, min(30, int(ctx.args[0])))
    rows = query_top_performers(days=days)
    await _reply(update, format_top_performers(rows, days))

