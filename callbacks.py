"""Inline button callbacks."""
import logging

from telegram import Update
from telegram.error import BadRequest, TelegramError
from telegram.ext import ContextTypes

from .config import DEFAULT_THRESHOLD
from .db import set_state, upsert_chat
from .keyboards import (
    MENU_HEADER, back_keyboard, main_menu_keyboard, threshold_keyboard,
)
from .market import MarketContext
from .scoring import ScoringEngine
from .state import BotState
from .ui_text import (
    format_top_performers, query_top_performers, text_features,
    text_keywords, text_market, text_model, text_monitor_status,
    text_outcomes, text_paper_report, text_paper_status,
    text_scoring_mode, text_snapshot, text_stats, text_wallet,
)
from .utils import mdbold, mdcode, strip_md2
from .commands import do_train

log = logging.getLogger(__name__)
PM = "MarkdownV2"


async def handle_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    try:
        await query.answer()
    except Exception:
        pass

    data = query.data
    cid  = update.effective_chat.id
    state:  BotState       = ctx.bot_data["state"]
    engine: ScoringEngine  = ctx.bot_data["engine"]
    mctx:   MarketContext  = ctx.bot_data["market_ctx"]

    async def show(text: str, kb=None):
        target_kb = kb or back_keyboard()
        try:
            await query.edit_message_text(text, parse_mode=PM, reply_markup=target_kb)
        except BadRequest as e:
            err = str(e).lower()
            if "not modified" in err:
                return
            if "parse" in err:
                log.warning("MD2 parse failed in callback %s: %s", data, e)
                try:
                    await query.edit_message_text(
                        strip_md2(text), reply_markup=target_kb)
                except TelegramError as e2:
                    log.error("Plain callback fallback failed: %s", e2)
                return
            log.debug("edit_message_text: %s", e)
        except TelegramError as e:
            log.debug("edit_message_text telegram: %s", e)

    try:
        if data == "menu":
            await show(MENU_HEADER, kb=main_menu_keyboard())

        elif data == "close_menu":
            try:
                await query.message.delete()
            except Exception:
                pass

        elif data == "monitor_on":
            th = state.alerts.get(cid, DEFAULT_THRESHOLD)
            state.alerts[cid] = th
            upsert_chat(cid, alerts_enabled=1, threshold=th)
            await show(f"🟢 {mdbold('Alerts ON')} — threshold {mdcode(f'{th}/10')}")

        elif data == "monitor_off":
            state.alerts.pop(cid, None)
            upsert_chat(cid, alerts_enabled=0)
            await show(f"🔴 {mdbold('Alerts OFF')}")

        elif data == "monitor_status":
            await show(text_monitor_status(cid, state))

        elif data == "threshold_menu":
            current = state.alerts.get(cid, DEFAULT_THRESHOLD)
            await show(
                f"🎚 {mdbold('Set Alert Threshold')}\n"
                f"Current: {mdcode(f'{current}/10')}\n"
                f"Choose new minimum score:",
                kb=threshold_keyboard(),
            )

        elif data.startswith("set_threshold_"):
            try:
                val = int(data.split("_")[-1])
            except ValueError:
                await show("❌ Invalid threshold")
                return
            if not 1 <= val <= 10:
                await show("❌ Must be 1–10")
                return
            state.alerts[cid] = val
            upsert_chat(cid, threshold=val, alerts_enabled=1)
            await show(
                f"✅ Threshold set to {mdcode(f'{val}/10')}\n"
                f"Alerts are {mdbold('ON')}\\."
            )

        elif data == "scoring_mode": await show(text_scoring_mode(engine))
        elif data == "features":     await show(text_features(engine))
        elif data == "keywords":     await show(text_keywords(engine))
        elif data == "market":       await show(text_market(mctx))
        elif data == "outcomes":     await show(text_outcomes())
        elif data == "model":        await show(text_model(engine))
        elif data == "snapshot":     await show(text_snapshot())
        elif data == "stats":        await show(text_stats(state, engine))
        elif data == "wallet":       await show(text_wallet())

        elif data == "top":
            rows = query_top_performers(days=7)
            await show(format_top_performers(rows, 7))

        elif data == "train":
            try:
                await query.edit_message_text("🏋 Training\\.\\.\\.", parse_mode=PM)
            except Exception:
                pass
            await show(await do_train(engine))

        elif data == "paper_on":
            await state.set_paper_enabled(True)
            set_state("paper_engine_enabled", "1")
            state.paper_chats.add(cid)
            upsert_chat(cid, paper_reports_enabled=1)
            await show(f"✅ {mdbold('Paper trading ON')}")

        elif data == "paper_off":
            await state.set_paper_enabled(False)
            set_state("paper_engine_enabled", "0")
            state.paper_chats.discard(cid)
            upsert_chat(cid, paper_reports_enabled=0)
            await show(f"❌ {mdbold('Paper trading OFF')}")

        elif data == "paper_status": await show(text_paper_status(state))
        elif data == "paper_report": await show(text_paper_report(state))

        else:
            try:
                await query.answer("Unknown action", show_alert=True)
            except Exception:
                pass

    except Exception as e:
        log.error("handle_callback error for %s: %s", data, e, exc_info=True)
