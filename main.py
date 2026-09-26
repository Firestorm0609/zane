"""Entry point: python main.py"""
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

from telegram import BotCommand
from telegram.error import RetryAfter
from telegram.ext import Application

import config
from pump_auth import PumpAuth
import callouts
from db import DB
from exits import ExitEngine
from handlers import Handlers
from poller import Poller

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("main")

# ---- Telegram branding -------------------------------------------------------
# setMyName/setMyCommands are hard rate-limited by Telegram. Re-applying them on
# every boot earned a ~22h flood-control window, and each retry EXTENDS it, so:
#   * only call when the value actually changed, and
#   * cache the applied values + any RetryAfter window in a state file.
BOT_DISPLAY_NAME = "ZANE ⚡️"
TELEGRAM_PROFILE_STATE = ".telegram-profile.json"
BOT_COMMANDS = [
    BotCommand("start", "⚡ Main menu — wallet, callers, positions"),
    BotCommand("unlock", "🔑 Unlock the bot with your access code"),
    BotCommand("help", "📖 Full guide: setup, commands, how it works"),
    BotCommand("wallet", "💼 Import, view or remove your trading wallet"),
    BotCommand("balance", "💰 Show wallet SOL balance"),
    BotCommand("addcaller", "➕ Follow a caller: /addcaller <wallet> [label]"),
    BotCommand("callers", "📣 Manage callers — buy size, TP/SL, mcap filter"),
    BotCommand("stats", "📈 Caller win-rate: /stats <caller_id>"),
    BotCommand("buy", "🛒 Manual buy: /buy <mint> [amount_in_SOL]"),
    BotCommand("sell", "💸 Manual sell: /sell <mint> [percent]"),
    BotCommand("positions", "📊 Open positions with sell buttons"),
]


def commands_sig(commands) -> str:
    """Stable fingerprint of a command menu, so no-op updates can be skipped."""
    return "\n".join(f"{c.command}\t{c.description}" for c in commands)


async def sync_telegram_branding(bot, state_path=TELEGRAM_PROFILE_STATE) -> dict:
    """Apply the bot name + slash-command menu, avoiding pointless API calls.

    Telegram flood-limits setMyName/setMyCommands hard, and a failed attempt
    *extends* the cooldown (~22h in practice), so the values last applied and
    any RetryAfter deadline are cached in ``state_path``. Returns that state.
    """
    try:
        state = json.loads(Path(state_path).read_text())
    except Exception:
        state = {}

    now = time.time()
    dirty = False

    async def apply(field: str, value: str, call) -> bool:
        """Apply one profile field. False if flood control blocked it."""
        nonlocal dirty
        if state.get(field) == value:
            return True
        try:
            await call()
            state[field] = value
            dirty = True
            log.info("telegram %s registered", field)
            return True
        except RetryAfter as e:
            state["blocked_until"] = now + e.retry_after
            dirty = True
            log.warning("telegram flood control on %s — skipping for %.0f min "
                        "(already correct server-side)", field, e.retry_after / 60)
            return False
        except Exception as e:
            log.warning("telegram %s update failed (non-fatal): %s", field, e)
            return True

    blocked_until = float(state.get("blocked_until") or 0)
    if blocked_until > now:
        log.info("telegram branding skipped — flood control, %.0f min left",
                 (blocked_until - now) / 60)
    elif await apply("commands", commands_sig(BOT_COMMANDS),
                     lambda: bot.set_my_commands(BOT_COMMANDS)):
        # commands first: they're functional, the name is only cosmetic
        await apply("name", BOT_DISPLAY_NAME,
                    lambda: bot.set_my_name(BOT_DISPLAY_NAME))

    if dirty:
        try:
            Path(state_path).write_text(json.dumps(state))
        except Exception as e:
            log.warning("could not save telegram branding state: %s", e)
    return state


async def main():
    if not config.TELEGRAM_BOT_TOKEN:
        print("❌ TELEGRAM_BOT_TOKEN missing. Copy .env.example to .env and fill it in.")
        sys.exit(1)
    if not config.WALLET_ENC_KEY:
        print("❌ WALLET_ENC_KEY missing (needed to encrypt imported wallets).")
        print("   Generate one: python -c \"import os; print(os.urandom(32).hex())\"")
        sys.exit(1)

    db = DB()
    await db.connect()
    # owner bypasses the paywall — without this subscription row the paid-gate
    # JOINs in get_callers()/get_all_open_positions() exclude the owner, so
    # their callers are never polled and positions never managed
    await db.ensure_owner(config.BOT_OWNER_ID)
    log.info("owner %s unlocked (paywall bypass)", config.BOT_OWNER_ID)

    client = callouts.CalloutClient()

    # ---- pump.fun feed-mode auth (Privy token forwarded by userscript) ----
    pump_auth = PumpAuth()
    has_pump_auth = pump_auth.load()
    client.attach_auth(pump_auth)

    async def activate_feed_mode():
        """Opt-in only (FEED_MODE_ENABLED=1): follow every caller with the bot
        account so ONE /following-feed request could cover them all. Disabled by
        default because that endpoint serves a stale 25-item sample and missed
        most callouts; detection polls each caller's list instead."""
        if not config.FEED_MODE_ENABLED:
            return
        if client.bot_uuid:
            return
        try:
            uuid = await client.init_feed()
            log.info("feed mode ON — bot account uuid %s", uuid)
            for c in await db.get_callers():
                try:
                    cuuid = await client.resolve_uuid(c["caller_id"])
                    if await client.follow(cuuid):
                        log.info("followed %s", c["caller_id"][:12])
                except Exception as e:
                    log.warning("follow %s failed: %s", c["caller_id"][:12], e)
        except Exception as e:
            log.warning("feed mode init failed (%s) — per-caller polling; "
                        "will retry when the next token arrives", e)

    pump_auth.on_token = activate_feed_mode
    pump_auth_task = asyncio.create_task(pump_auth.run_ingest())

    app = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()
    handlers = Handlers(db, client)
    handlers.register(app)

    async def notify(tg_id: int, text: str, reply_markup=None):
        log.info("notify tg=%s: %s", tg_id, text.replace("\n", " | ")[:120])
        try:
            await app.bot.send_message(tg_id, text, parse_mode="HTML",
                                       reply_markup=reply_markup)
        except Exception as e:
            # blocked bot, chat deleted, network hiccup — never take down the
            # caller of notify (a fanout buy, an exit, a feed loop)
            log.warning("telegram send failed tg=%s: %s", tg_id, e)

    poller = Poller(db, client, notify)
    handlers.poller = poller  # so wallet import/remove invalidates the cache

    # browser relay: pump_forwarder.user.js fetches callout pages from the
    # user's own IP (this host's is throttled by pump.fun) and POSTs them to
    # /callouts on the ingest server, which feeds them into the poller
    pump_auth.callout_sink = poller.ingest_external

    # feed mode: use the bot account (JWT) to follow all callers, then one
    # /callout/feed request covers them all — enables ~2s polling.
    # Auth = Privy token saved by pumpfarm --set-token / forwarded live by
    # pump_forwarder.user.js (static PUMP_AUTH_TOKEN env still works).
    if has_pump_auth or config.PUMP_AUTH_TOKEN:
        await activate_feed_mode()
    else:
        log.info("no pump.fun auth yet — waiting for pump_forwarder.user.js "
                 "(or `python3 pumpfarm.py --set-token`); per-caller polling "
                 "until then; feed mode will activate automatically on the "
                 "first forwarded token")

    exits_engine = ExitEngine(db, notify)

    async def on_shutdown(app):
        pump_auth_task.cancel()
        poll_task.cancel()
        exits_task.cancel()
        await client.close()
        log.info("shutdown complete")

    app.post_shutdown = on_shutdown

    log.info("starting bot — detection=%s, polling callouts every %.1fs",
             "feed" if config.FEED_MODE_ENABLED else "per-caller",
             config.POLL_INTERVAL_S)
    # rsplit drops any user:pass@ so credentials never reach the log
    log.info("pump.fun egress: %s",
             config.PUMP_PROXY.rsplit("@", 1)[-1] if config.PUMP_PROXY
             else "direct (PUMP_PROXY unset)")
    await app.initialize()

    # start background trading/feed tasks only NOW: app.bot must be
    # initialized before any task sends messages through it (the poller can
    # fire an alert seconds after starting)
    poll_task = asyncio.create_task(poller.run(config.POLL_INTERVAL_S))
    exits_task = asyncio.create_task(exits_engine.run())

    # brand the bot + slash-command menu (skips no-ops and flood-control windows)
    await sync_telegram_branding(app.bot)

    await app.start()
    await app.updater.start_polling(allowed_updates=["message", "callback_query"])
    # run forever
    await asyncio.Event().wait()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
