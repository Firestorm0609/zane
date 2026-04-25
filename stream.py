"""PumpPortal WebSocket stream consumer."""
import asyncio
import json
import logging

import aiohttp
from telegram import Bot
from telegram.error import TelegramError

from .config import PUMP_FRONT, PUMP_PORTAL_URI
from .enrichment import (
    enrich_from_pumpfun, enrich_with_rpc, fetch_mc_momentum_from_db,
    _normalize_pumpportal,
)
from .market import MarketContext
from .processor import process_coin
from .scoring import ScoringEngine
from .state import BotState
from .storage import save_dead_letter
from .utils import mdbold, mdcode

log = logging.getLogger(__name__)

_active_tasks: set["asyncio.Task"] = set()


def _track(task: "asyncio.Task") -> "asyncio.Task":
    _active_tasks.add(task)
    task.add_done_callback(_active_tasks.discard)
    return task


def get_active_tasks() -> set:
    return _active_tasks


async def _enrich_and_process(
    coin: dict, bot: Bot, engine: ScoringEngine,
    market_ctx: MarketContext, state: BotState,
) -> None:
    try:
        async with aiohttp.ClientSession() as session:
            coin, enrich_err = await enrich_from_pumpfun(coin, session)
            if enrich_err:
                log.debug("enrich %s: %s — using partial data",
                          (coin.get("mint", "?") or "?")[:8], enrich_err)
                save_dead_letter(coin.get("mint", ""), coin, enrich_err)

            coin = await enrich_with_rpc(coin, session)

        mint = coin.get("mint", "")
        if mint:
            loop = asyncio.get_running_loop()
            coin["_mc_momentum_pct"] = await loop.run_in_executor(
                None, fetch_mc_momentum_from_db, mint
            )

        await process_coin(coin, bot, engine, market_ctx, state)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        log.error("_enrich_and_process %s: %s",
                  (coin.get("mint", "?") or "?")[:8], e, exc_info=True)
        save_dead_letter(coin.get("mint", ""), coin, str(e))


async def stream(
    bot: Bot, engine: ScoringEngine,
    market_ctx: MarketContext, state: BotState,
) -> None:
    delay = 5
    while True:
        session = None
        try:
            log.info("Connecting to PumpPortal %s", PUMP_PORTAL_URI)
            session = aiohttp.ClientSession()
            async with session.ws_connect(
                PUMP_PORTAL_URI, heartbeat=30, max_msg_size=0,
            ) as ws:
                await ws.send_str(json.dumps({"method": "subscribeNewToken"}))
                log.info("Subscribed to PumpPortal new-token stream")
                delay = 5

                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        raw = msg.data
                    elif msg.type == aiohttp.WSMsgType.BINARY:
                        try:
                            raw = msg.data.decode("utf-8")
                        except Exception as e:
                            log.debug("binary decode: %s", e)
                            continue
                    elif msg.type in (
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.ERROR,
                        aiohttp.WSMsgType.CLOSE,
                    ):
                        log.warning("WS closed/error: %s", msg)
                        break
                    else:
                        continue

                    try:
                        data = json.loads(raw)
                    except Exception:
                        continue

                    if not isinstance(data, dict):
                        continue

                    if "errors" in data or data.get("message") == "error":
                        log.error("WS server reported error: %s", data)
                        continue

                    mint = data.get("mint")
                    if not mint:
                        continue

                    # Raydium graduation
                    if data.get("txType") == "graduated" or data.get("raydiumPool"):
                        is_new = await state.add_graduated(mint)
                        if not is_new:
                            continue
                        name = data.get("name") or data.get("symbol") or mint[:8]
                        log.info("GRADUATION | %s | %s", mint[:8], name)
                        grad_text = (
                            f"🎓 {mdbold('Raydium Graduation!')}\n"
                            f"{mdbold(name)} has graduated from pump\\.fun to Raydium\\!\n"
                            f"🪙 {mdcode(mint)}\n"
                            f"🔗 [Pump\\.fun]({PUMP_FRONT}/{mint})"
                        )
                        for chat_id in list(state.alerts.keys()):
                            try:
                                await bot.send_message(
                                    chat_id=chat_id, text=grad_text,
                                    parse_mode="MarkdownV2",
                                    disable_web_page_preview=True,
                                )
                            except TelegramError as e:
                                log.error("graduation alert %s: %s", chat_id, e)
                        continue

                    if await state.seen_recently(mint):
                        continue

                    coin = _normalize_pumpportal(data)
                    _track(asyncio.create_task(
                        _enrich_and_process(coin, bot, engine, market_ctx, state),
                        name=f"enrich_{mint[:8]}",
                    ))

        except asyncio.CancelledError:
            if session and not session.closed:
                await session.close()
            raise
        except Exception as e:
            log.warning("WS dropped: %s — retry in %ss", e, delay)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60)
        finally:
            if session and not session.closed:
                try:
                    await session.close()
                except Exception:
                    pass
