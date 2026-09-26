"""Jupiter swap support for off-platform tokens (Raydium, Meteora, etc.).

Flow: quote -> /swap (build tx) -> sign locally -> send via own RPC.
Uses lite-api (free tier, ~60 req/min per IP).
"""
import asyncio
import base64
import logging
from typing import Any, Optional

import aiohttp
from solders.keypair import Keypair

import config
import trader

log = logging.getLogger("jupiter")

JUP_BASE = "https://lite-api.jup.ag/swap/v1"
WSOL = "So11111111111111111111111111111111111111112"


class JupiterError(RuntimeError):
    pass


async def get_quote(input_mint: str, output_mint: str, raw_amount: int,
                    slippage_bps: int = 100) -> dict[str, Any]:
    url = f"{JUP_BASE}/quote"
    params = {
        "inputMint": input_mint,
        "outputMint": output_mint,
        "amount": str(raw_amount),
        "slippageBps": str(slippage_bps),
    }
    async with aiohttp.ClientSession() as s:
        async with s.get(url, params=params) as resp:
            if resp.status == 429:
                raise JupiterError("jupiter rate limited (429)")
            if resp.status != 200:
                text = await resp.text()
                raise JupiterError(f"quote {resp.status}: {text[:200]}")
            return await resp.json()


async def build_swap_tx(quote: dict[str, Any], user_pubkey: str,
                        priority_lamports: int = 1_000_000) -> str:
    """POST /swap -> unsigned versioned transaction (base64)."""
    body = {
        "quoteResponse": quote,
        "userPublicKey": user_pubkey,
        "wrapAndUnwrapSol": True,
        "dynamicComputeUnitLimit": True,
        "prioritizationFeeLamports": {
            "priorityLevelWithMaxLamports": {
                "maxLamports": priority_lamports,
                "priorityLevel": "high",
            }
        },
    }
    async with aiohttp.ClientSession() as s:
        async with s.post(f"{JUP_BASE}/swap", json=body) as resp:
            if resp.status == 429:
                raise JupiterError("jupiter rate limited (429)")
            if resp.status != 200:
                text = await resp.text()
                raise JupiterError(f"swap {resp.status}: {text[:200]}")
            data = await resp.json()
            tx = data.get("swapTransaction")
            if not tx:
                raise JupiterError(f"no swapTransaction in response: {str(data)[:200]}")
            return tx


async def _check_balance_before(pubkey: str, amount_sol: float):
    balance = await trader.get_balance_sol(pubkey)
    if balance - amount_sol < config.MIN_SOL_LEFT:
        raise RuntimeError(
            f"insufficient SOL: balance {balance:.4f}, need to keep {config.MIN_SOL_LEFT}")


async def buy(keypair: Keypair, mint: str, amount_sol: float,
              slippage_bps: int = 100, wait_confirm: bool = True) -> str:
    """Buy any SPL token with SOL via Jupiter. Returns signature.

    `wait_confirm=False` returns the moment the tx is submitted (the sniping
    path) instead of polling for it to land.
    """
    if config.MAX_BUY_SOL > 0 and amount_sol > config.MAX_BUY_SOL:
        raise ValueError(f"buy amount {amount_sol} exceeds MAX_BUY_SOL {config.MAX_BUY_SOL}")
    pubkey = str(keypair.pubkey())
    # balance read runs CONCURRENTLY with the quote — it must not sit in front
    # of the swap build on the sniping path.
    balance_task = asyncio.create_task(_check_balance_before(pubkey, amount_sol))
    raw_lamports = int(amount_sol * 1e9)
    try:
        quote = await get_quote(WSOL, mint, raw_lamports, slippage_bps)
        await balance_task
    except BaseException:
        if balance_task.done():
            balance_task.exception()  # already failed — retrieve so nothing warns
        else:
            balance_task.cancel()
        raise
    unsigned = await build_swap_tx(quote, pubkey)
    signed = trader.sign_tx(unsigned, str(keypair))
    sig = await trader.send_tx(signed)
    if wait_confirm:
        conf = await trader.confirm_tx(sig)
        if conf["found"] is False:
            log.warning("buy tx %s not confirmed within timeout — may still land",
                        sig[:16])
    else:
        trader.confirm_in_background(sig)
    return sig


async def sell(keypair: Keypair, mint: str, token_balance: float, decimals: int,
               percent: float = 100.0, slippage_bps: int = 300) -> str:
    """Sell a percentage of the held token balance back to SOL. Returns signature."""
    if not 0 < percent <= 100:
        raise ValueError("percent must be 0-100")
    if token_balance <= 0:
        raise JupiterError("no token balance to sell")
    pubkey = str(keypair.pubkey())
    raw_amount = int(token_balance * percent / 100.0 * (10 ** decimals))
    if raw_amount <= 0:
        raise JupiterError("raw sell amount rounds to zero")
    quote = await get_quote(mint, WSOL, raw_amount, slippage_bps)
    unsigned = await build_swap_tx(quote, pubkey)
    signed = trader.sign_tx(unsigned, str(keypair))
    sig = await trader.send_tx(signed)
    await trader.confirm_tx(sig)
    return sig


async def quote_sol_out(mint: str, token_balance: float, decimals: int) -> float:
    """How much SOL the current token balance would fetch (for exit pricing)."""
    raw_amount = int(token_balance * (10 ** decimals))
    if raw_amount <= 0:
        raise JupiterError("raw amount rounds to zero")
    quote = await get_quote(mint, WSOL, raw_amount, slippage_bps=1000)
    return int(quote["outAmount"]) / 1e9
