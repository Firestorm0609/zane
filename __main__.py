"""Entry point: orchestrates app startup and graceful shutdown."""
import asyncio
import logging
import logging.handlers
import sys

from telegram.ext import (
    ApplicationBuilder, CallbackQueryHandler, CommandHandler, ContextTypes,
)

from .background import (
    db_backup_loop, dead_letter_retry_loop,
    outcome_notify_loop, stream_watchdog_loop, watchlist_monitor_loop,
)
from .config import BOT_TOKEN, LOG_PATH, MODEL_VERSION, PAPER_ENABLED_DEFAULT
from .db import get_state, init_db, set_state
from .features import FEATURES
from .callbacks import handle_callback
from .commands import (
    cmd_backtest, cmd_blacklist, cmd_features, cmd_health, cmd_help,
    cmd_keywords, cmd_market, cmd_menu, cmd_model, cmd_monitor_off,
    cmd_monitor_on, cmd_monitor_status, cmd_outcomes, cmd_paper_off,
    cmd_paper_on, cmd_paper_report, cmd_paper_status, cmd_score,
    cmd_scoring_mode, cmd_set_threshold, cmd_snapshot, cmd_start,
    cmd_stats, cmd_top, cmd_train, cmd_unwatch, cmd_wallet,
    cmd_wallet_reset, cmd_watch, cmd_watchlist,
)
from .keywords import KeywordModel
from .lookback import lookback_loop, train_executor, training_loop
from .market import MarketContext
from .processor import init_semaphore
from .scoring import ScoringEngine
from .state import BotState
from .stream import get_active_tasks, stream
from .trading import paper_monitor_loop
from .utils import now_ts


# ---------- Logging ----------

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.handlers.RotatingFileHandler(
            LOG_PATH, maxBytes=10 * 1024 * 1024,
            backupCount=5, encoding="utf-8",
        ),
    ],
)
log = logging.getLogger(__name__)


# ---------- Error handler ----------

async def error_handler(update: object, ctx: ContextTypes.DEFAULT_TYPE):
    log.error("Telegram handler error: %s", ctx.error, exc_info=ctx.error)


# ---------- Main ----------

async def run() -> None:
    init_db()

    # Create semaphore inside the running event loop (safe on all Python versions)
    init_semaphore()

    market_ctx    = MarketContext()
    keyword_model = KeywordModel()
    engine        = ScoringEngine(FEATURES, keyword_model, market_ctx)

    engine.load()
    keyword_model.learn_from_db()
    set_state("bot_started_at", str(now_ts()))

    paper_init = get_state(
        "paper_engine_enabled",
        "1" if PAPER_ENABLED_DEFAULT else "0",
    ) == "1"
    state = BotState(paper_enabled=paper_init)
    state.load()

    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.bot_data["engine"]     = engine
    app.bot_data["market_ctx"] = market_ctx
    app.bot_data["state"]      = state

    handlers = [
        ("start",          cmd_start),
        ("help",           cmd_help),
        ("menu",           cmd_menu),
        ("monitor_on",     cmd_monitor_on),
        ("monitor_off",    cmd_monitor_off),
        ("monitor_status", cmd_monitor_status),
        ("set_threshold",  cmd_set_threshold),
        ("scoring_mode",   cmd_scoring_mode),
        ("features",       cmd_features),
        ("keywords",       cmd_keywords),
        ("market",         cmd_market),
        ("outcomes",       cmd_outcomes),
        ("model",          cmd_model),
        ("train",          cmd_train),
        ("snapshot",       cmd_snapshot),
        ("paper_on",       cmd_paper_on),
        ("paper_off",      cmd_paper_off),
        ("paper_status",   cmd_paper_status),
        ("paper_report",   cmd_paper_report),
        ("paper_reports_on", cmd_paper_reports_on),
        ("paper_reports_off", cmd_paper_reports_off),
        ("health",         cmd_health),
        ("score",          cmd_score),
        ("backtest",       cmd_backtest),
        ("watch",          cmd_watch),
        ("unwatch",        cmd_unwatch),
        ("watchlist",      cmd_watchlist),
        ("stats",          cmd_stats),
        ("last",           cmd_last),
        ("wallet",         cmd_wallet),
        ("wallet_reset",   cmd_wallet_reset),
        ("blacklist",      cmd_blacklist),
        ("top",            cmd_top),
    ]
    for name, fn in handlers:
        app.add_handler(CommandHandler(name, fn))

    app.add_handler(CallbackQueryHandler(handle_callback))
    app.add_error_handler(error_handler)

    background_tasks: list[asyncio.Task] = []
    async with app:
        await app.start()
        await app.updater.start_polling(drop_pending_updates=True)
        log.info(
            "Bot running v%s | paper=%s | scoring=%s | features=%d",
            MODEL_VERSION, state.paper_enabled, engine.mode_label, len(FEATURES),
        )

        background_tasks = [
            asyncio.create_task(
                stream(app.bot, engine, market_ctx, state), name="stream"),
            asyncio.create_task(lookback_loop(),         name="lookback"),
            asyncio.create_task(training_loop(engine),   name="training"),
            asyncio.create_task(paper_monitor_loop(app.bot), name="paper_monitor"),
            asyncio.create_task(
                stream_watchdog_loop(app.bot, state),    name="stream_watchdog"),
            asyncio.create_task(
                dead_letter_retry_loop(app.bot, engine, market_ctx, state),
                name="dead_letter_retry"),
            asyncio.create_task(db_backup_loop(),        name="db_backup"),
            asyncio.create_task(
                outcome_notify_loop(app.bot, state),     name="outcome_notify"),
            asyncio.create_task(
                watchlist_monitor_loop(app.bot),         name="watchlist_monitor"),
            # blacklist_refresh_loop removed: BlacklistCache already auto-refreshes
            # via its internal TTL on every lookup — a separate loop was redundant.
        ]

        try:
            await asyncio.gather(*background_tasks)
        except asyncio.CancelledError:
            pass
        finally:
            for t in background_tasks:
                if not t.done():
                    t.cancel()
            for t in background_tasks:
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass

            active = get_active_tasks()
            if active:
                log.info("Waiting for %d in-flight enrich tasks…", len(active))
                for t in list(active):
                    t.cancel()
                await asyncio.gather(*list(active), return_exceptions=True)

            try:
                await app.updater.stop()
            except Exception as e:
                log.warning("Error stopping updater during shutdown: %s", e)
            try:
                await app.stop()
            except Exception as e:
                log.warning("Error stopping app during shutdown: %s", e)

            train_executor.shutdown(wait=False)


def main() -> None:
    if not BOT_TOKEN:
        print("ERROR: TELEGRAM_BOT_TOKEN not set in .env", file=sys.stderr)
        sys.exit(1)
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        log.info("Shutting down")


if __name__ == "__main__":
    main()

