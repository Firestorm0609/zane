"""pump.fun API + Solana RPC enrichment."""
import asyncio
import logging
from collections import defaultdict
from contextlib import closing
from typing import Any, Optional

import aiohttp

from .config import (
    BUNDLE_SLOT_THRESHOLD, ENRICH_DELAY_SEC, HTTP_TIMEOUT_SEC, RPC_ENABLED,
    RPC_TIMEOUT_SEC, SOLANA_RPC_URL,
)
from .db import db_conn
from .ratelimit import pump_api_limiter, rpc_limiter
from .utils import now_ts, safe_float, safe_int, validate_url

log = logging.getLogger(__name__)


# ---------- pump.fun frontend API ----------

def _normalize_pumpportal(data: dict) -> dict:
    def _social(v):
        s = (v or "").strip() if isinstance(v, str) else ""
        return validate_url(s)
    return {
        "mint":              data.get("mint", ""),
        "name":              data.get("name", ""),
        "symbol":            data.get("symbol", ""),
        "description":       data.get("description", ""),
        "twitter":           _social(data.get("twitter")),
        "telegram":          _social(data.get("telegram")),
        "website":           _social(data.get("website")),
        "reply_count":       data.get("reply_count", 0),
        "usd_market_cap":    0,
        "created_timestamp": data.get("created_timestamp"),
        "creator":           data.get("creator") or data.get("traderPublicKey") or "",
    }


async def fetch_coin_mc(session: aiohttp.ClientSession, mint: str) -> Optional[float]:
    await pump_api_limiter.acquire()
    try:
        async with session.get(
            f"https://frontend-api-v3.pump.fun/coins/{mint}",
            timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT_SEC),
        ) as resp:
            if resp.status == 200:
                data = await resp.json(content_type=None)
                if isinstance(data, dict):
                    mc = safe_float(data.get("usd_market_cap"))
                    return mc if mc > 0 else None
    except asyncio.TimeoutError:
        log.debug("fetch_coin_mc timeout %s", mint[:8])
    except Exception as e:
        log.debug("fetch_coin_mc %s: %s", mint[:8], e)
    return None


async def enrich_from_pumpfun(
    coin: dict, session: aiohttp.ClientSession,
) -> tuple[dict, Optional[str]]:
    mint = coin.get("mint", "")
    if not mint:
        return coin, None

    await asyncio.sleep(ENRICH_DELAY_SEC)
    await pump_api_limiter.acquire()

    try:
        async with session.get(
            f"https://frontend-api-v3.pump.fun/coins/{mint}",
            timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT_SEC),
        ) as resp:
            if resp.status == 200:
                api_data = await resp.json(content_type=None)
                if isinstance(api_data, dict) and api_data.get("mint"):
                    merged = {**coin}
                    for k, v in api_data.items():
                        if v is not None:
                            merged[k] = v
                    log.debug("Enriched %s | mc=$%.0f replies=%s",
                              mint[:8],
                              safe_float(merged.get("usd_market_cap")),
                              merged.get("reply_count"))
                    return merged, None
                return coin, None
            elif resp.status == 404:
                return coin, None
            elif resp.status in (429, 503, 502, 504):
                log.warning("pump.fun API degraded: HTTP %s — backing off", resp.status)
                await asyncio.sleep(5)
                return coin, f"HTTP {resp.status} (degraded)"
            else:
                return coin, f"HTTP {resp.status}"
    except asyncio.TimeoutError:
        return coin, "timeout"
    except Exception as e:
        return coin, str(e)


# ---------- Solana RPC ----------

async def _rpc_post(
    session: aiohttp.ClientSession, method: str, params: list,
) -> tuple[Any, Optional[str]]:
    await rpc_limiter.acquire()
    try:
        async with session.post(
            SOLANA_RPC_URL,
            json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
            timeout=aiohttp.ClientTimeout(total=RPC_TIMEOUT_SEC),
            headers={"Content-Type": "application/json"},
        ) as resp:
            if resp.status == 200:
                data = await resp.json(content_type=None)
                if "error" in data:
                    return None, str(data["error"])
                return data.get("result"), None
            return None, f"HTTP {resp.status}"
    except asyncio.TimeoutError:
        return None, "rpc_timeout"
    except Exception as e:
        return None, str(e)


async def rpc_mint_authorities(
    session: aiohttp.ClientSession, mint: str,
) -> dict[str, Optional[bool]]:
    result, err = await _rpc_post(
        session, "getAccountInfo",
        [mint, {"encoding": "jsonParsed", "commitment": "confirmed"}],
    )
    unknown = {"mint_auth_revoked": None, "freeze_auth_revoked": None}
    if err or result is None:
        log.debug("rpc_mint_authorities %s: %s", mint[:8], err)
        return unknown
    try:
        info = (result.get("value") or {})
        parsed = info.get("data", {}).get("parsed", {})
        minfo = parsed.get("info", {})
        return {
            "mint_auth_revoked":   minfo.get("mintAuthority") is None,
            "freeze_auth_revoked": minfo.get("freezeAuthority") is None,
        }
    except Exception as e:
        log.debug("rpc_mint_authorities parse %s: %s", mint[:8], e)
        return unknown


async def rpc_top_holder_concentration(
    session: aiohttp.ClientSession, mint: str,
) -> Optional[float]:
    result, err = await _rpc_post(
        session, "getTokenLargestAccounts",
        [mint, {"commitment": "confirmed"}],
    )
    if err or result is None:
        log.debug("rpc_top_holder_concentration %s: %s", mint[:8], err)
        return None
    try:
        accounts = result.get("value") or []
        if not accounts:
            return None
        amounts = [safe_float(a.get("uiAmount", 0)) for a in accounts]
        total = sum(amounts)
        if total <= 0:
            return None
        top5 = sum(sorted(amounts, reverse=True)[:5])
        return min(1.0, top5 / total)
    except Exception as e:
        log.debug("rpc_top_holder_concentration parse %s: %s", mint[:8], e)
        return None


async def rpc_bundle_score(
    session: aiohttp.ClientSession, mint: str,
) -> Optional[float]:
    result, err = await _rpc_post(
        session, "getSignaturesForAddress",
        [mint, {"limit": 20, "commitment": "confirmed"}],
    )
    if err or result is None:
        log.debug("rpc_bundle_score %s: %s", mint[:8], err)
        return None
    try:
        sigs = result if isinstance(result, list) else []
        if len(sigs) < 3:
            return 0.0
        slot_counts: dict[int, int] = defaultdict(int)
        for s in sigs:
            slot = s.get("slot")
            if slot:
                slot_counts[slot] += 1
        max_same = max(slot_counts.values(), default=0)
        if max_same >= BUNDLE_SLOT_THRESHOLD:
            return 1.0
        if max_same == 2:
            return 0.5
        return 0.0
    except Exception as e:
        log.debug("rpc_bundle_score parse %s: %s", mint[:8], e)
        return None


async def rpc_creator_tx_count(
    session: aiohttp.ClientSession, creator: str,
) -> Optional[int]:
    if not creator:
        return None
    result, err = await _rpc_post(
        session, "getSignaturesForAddress",
        [creator, {"limit": 200, "commitment": "confirmed"}],
    )
    if err or result is None:
        log.debug("rpc_creator_tx_count %s: %s", creator[:8], err)
        return None
    try:
        sigs = result if isinstance(result, list) else []
        return len(sigs)
    except Exception as e:
        log.debug("rpc_creator_tx_count parse %s: %s", creator[:8], e)
        return None


async def enrich_with_rpc(coin: dict, session: aiohttp.ClientSession) -> dict:
    if not RPC_ENABLED:
        return coin

    mint = coin.get("mint", "")
    creator = (coin.get("creator") or coin.get("user")
               or coin.get("traderPublicKey") or "")
    if not mint:
        return coin

    try:
        auth_result, conc_result, bundle_result, hist_result = await asyncio.gather(
            rpc_mint_authorities(session, mint),
            rpc_top_holder_concentration(session, mint),
            rpc_bundle_score(session, mint),
            rpc_creator_tx_count(session, creator),
            return_exceptions=True,
        )

        if isinstance(auth_result, dict):
            coin["_rpc_mint_auth_revoked"]   = auth_result.get("mint_auth_revoked")
            coin["_rpc_freeze_auth_revoked"] = auth_result.get("freeze_auth_revoked")
        if isinstance(conc_result, (int, float)):
            coin["_rpc_top5_concentration"] = float(conc_result)
        if isinstance(bundle_result, (int, float)):
            coin["_rpc_bundle_score"] = float(bundle_result)
        if isinstance(hist_result, int):
            coin["_rpc_creator_tx_count"] = hist_result

        log.debug(
            "RPC enriched %s | mint_auth=%s freeze=%s top5=%s bundle=%s creator_txs=%s",
            mint[:8],
            coin.get("_rpc_mint_auth_revoked"),
            coin.get("_rpc_freeze_auth_revoked"),
            coin.get("_rpc_top5_concentration"),
            coin.get("_rpc_bundle_score"),
            coin.get("_rpc_creator_tx_count"),
        )
    except Exception as e:
        log.warning("enrich_with_rpc %s: %s", mint[:8], e)

    return coin


# ---------- DB-derived momentum ----------

def fetch_mc_momentum_from_db(mint: str, window_sec: int = 1800) -> float:
    """Sync — must be called via run_in_executor from async code."""
    cutoff = now_ts() - window_sec
    try:
        with closing(db_conn()) as conn:
            rows = conn.execute(
                "SELECT market_cap FROM price_snapshots "
                "WHERE mint=? AND created_at>=? ORDER BY created_at ASC LIMIT 20",
                (mint, cutoff),
            ).fetchall()
        if len(rows) < 2:
            return 0.0
        first_mc = safe_float(rows[0]["market_cap"])
        last_mc  = safe_float(rows[-1]["market_cap"])
        if first_mc <= 0:
            return 0.0
        return ((last_mc - first_mc) / first_mc) * 100.0
    except Exception as e:
        log.debug("fetch_mc_momentum_from_db %s: %s", mint[:8], e)
        return 0.0
