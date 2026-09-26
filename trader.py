"""Trading via PumpPortal trade-local: build unsigned tx, sign locally, send via own RPC."""
import asyncio
import base64
import json
import logging
from typing import Any, Optional

import aiohttp
from base58 import b58decode
from solders.keypair import Keypair
from solders.transaction import VersionedTransaction

import config

log = logging.getLogger("trader")

PUMP_PORTAL_HEADERS = {"Accept": "application/json", "User-Agent": "pump-callout-bot/1.0"}


async def build_buy_tx(mint: str, amount_sol: float, priority_fee: Optional[float] = None,
                       slippage: str = "10") -> dict[str, Any]:
    """POST /api/trade-local -> unsigned tx (base64) + rate-limited jito bundles."""
    body: dict[str, Any] = {
        "publicKey": None,  # filled by caller below via keypair pubkey
        "action": "buy",
        "mint": mint,
        "denominatedInSol": "true",
        "amount": amount_sol,
        "slippage": slippage,
    }
    if priority_fee:
        body["priorityFee"] = priority_fee
    return body


async def get_unsigned_tx(keypair_pubkey: str, mint: str, action: str, amount: float,
                          denominated_in_sol: bool = True, slippage: str = "10",
                          priority_fee: Optional[float] = None) -> str:
    """Fetch an unsigned transaction (base64) from PumpPortal for local signing."""
    body: dict[str, Any] = {
        "publicKey": keypair_pubkey,
        "action": action,           # "buy" | "sell"
        "mint": mint,
        "denominatedInSol": "true" if denominated_in_sol else "false",
        "amount": amount,
        "slippage": slippage,
        "pool": "pump",
    }
    if priority_fee:
        body["priorityFee"] = priority_fee
    async with aiohttp.ClientSession(headers=PUMP_PORTAL_HEADERS) as s:
        async with s.post(config.PUMP_TRADE_URL, json=body) as resp:
            if resp.status != 200:
                text = await resp.text()
                hint = ""
                if resp.status == 400:
                    hint = (" — usually means the token is not on a pump.fun "
                            "bonding curve (check venue)")
                raise RuntimeError(f"trade-local {resp.status}: {text[:200]}{hint}")
            raw = await resp.read()
            return base64.b64encode(raw).decode()


def sign_tx(unsigned_b64: str, secret_key_b58: str) -> str:
    """Sign the unsigned tx locally — the private key never leaves this machine."""
    kp = Keypair.from_base58_string(secret_key_b58)
    raw = base64.b64decode(unsigned_b64)
    msg = VersionedTransaction.from_bytes(raw).message
    signed = VersionedTransaction(msg, [kp])
    return base64.b64encode(bytes(signed)).decode()


async def send_tx(signed_b64: str) -> str:
    """Broadcast the signed tx to every configured RPC at once.

    The first endpoint to accept it wins (so a slow RPC never delays a snipe),
    but the copies already in flight are never cancelled — whichever endpoint
    includes it first wins the race. Free redundancy, no paid RPC needed.
    """
    body = json.dumps({
        "jsonrpc": "2.0", "id": 1,
        "method": "sendTransaction",
        "params": [signed_b64, {"encoding": "base64",
                                "skipPreflight": True, "maxRetries": 3}],
    })

    async def _one(url: str) -> str:
        async with aiohttp.ClientSession() as s:
            async with s.post(url, data=body,
                              headers={"Content-Type": "application/json"}) as resp:
                data = await resp.json()
                if "error" in data:
                    raise RuntimeError(f"{url}: {data['error']}")
                return data["result"]

    urls = config.SOLANA_RPC_BROADCAST or [config.SOLANA_RPC]
    tasks = [asyncio.create_task(_one(u)) for u in urls]
    try:
        while tasks:
            done, tasks = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                if t.exception() is None:
                    return t.result()
    finally:
        # let stragglers finish (a copy still in flight can still land);
        # consume their exceptions so nothing warns
        for t in tasks:
            t.add_done_callback(lambda x: x.exception())
    raise RuntimeError("sendTransaction failed on every RPC")


async def confirm_tx(sig: str, timeout_s: float = 20.0, interval_s: float = 2.0) -> dict:
    """Poll getSignatureStatuses until the tx lands or timeout.

    Returns {"found": bool, "err": dict|None, "slot": int|None}.
    Raises RuntimeError on on-chain failure so callers don't record dead positions.
    """
    import asyncio
    import json
    deadline = asyncio.get_event_loop().time() + timeout_s
    while asyncio.get_event_loop().time() < deadline:
        body = json.dumps({
            "jsonrpc": "2.0", "id": 1,
            "method": "getSignatureStatuses",
            "params": [[sig], {"searchTransactionHistory": False}],
        })
        async with aiohttp.ClientSession() as s:
            async with s.post(config.SOLANA_RPC, data=body,
                              headers={"Content-Type": "application/json"}) as resp:
                data = await resp.json()
        value = (data.get("result") or {}).get("value") or [None]
        st = value[0]
        if st is not None:
            if st.get("err"):
                err = st["err"]
                code = None
                try:
                    code = err["InstructionError"][1]["Custom"]
                except Exception:
                    pass
                hint = " (likely slippage — price moved past tolerance)" if code == 6005 else ""
                raise RuntimeError(f"tx failed on-chain: {err}{hint}")
            return {"found": True, "err": None, "slot": st.get("slot")}
        await asyncio.sleep(interval_s)
    return {"found": False, "err": None, "slot": None}  # unknown — don't crash callers


async def signature_landed(sig: str, timeout_s: float = 15.0) -> Optional[bool]:
    """Did this signature ever land on-chain?

    Returns True (confirmed), False (failed on-chain or not found after
    processing — note RPC keeps unprocessed sigs ~1-2 min), or None (RPC
    error / still unprocessed — unknown, caller should retry later).
    """
    body = json.dumps({
        "jsonrpc": "2.0", "id": 1,
        "method": "getSignatureStatuses",
        "params": [[sig], {"searchTransactionHistory": True}],
    })
    deadline = asyncio.get_event_loop().time() + timeout_s
    while asyncio.get_event_loop().time() < deadline:
        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(config.SOLANA_RPC, data=body,
                                  headers={"Content-Type": "application/json"}) as resp:
                    data = await resp.json()
            value = (data.get("result") or {}).get("value") or [None]
            st = value[0]
            if st is not None:
                return not bool(st.get("err"))
            # None: not yet processed. Poll briefly; unprocessed sigs expire
            # from this view ~2min after send.
            await asyncio.sleep(2.0)
        except Exception:
            await asyncio.sleep(2.0)
    return None  # unknown — don't let the caller act destructively


async def get_balance_sol(pubkey: str) -> float:
    import json
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "getBalance",
                       "params": [pubkey]})
    async with aiohttp.ClientSession() as s:
        async with s.post(config.SOLANA_RPC, data=body,
                          headers={"Content-Type": "application/json"}) as resp:
            data = await resp.json()
            if "error" in data:
                raise RuntimeError(f"RPC error: {data['error']}")
            return data["result"]["value"] / 1e9


_bg_tasks: set[asyncio.Task] = set()


def confirm_in_background(sig: str):
    """Don't block the sniping path on confirmation — poll on-chain in the
    background and just log the outcome. If the tx actually failed, the exit
    engine reaps the ghost position via `signature_landed`."""
    async def _run():
        try:
            await confirm_tx(sig)
        except Exception as e:
            log.warning("background confirm %s: %s", sig[:16], e)
    t = asyncio.create_task(_run())
    _bg_tasks.add(t)
    t.add_done_callback(_bg_tasks.discard)


async def buy(keypair: Keypair, mint: str, amount_sol: float,
              slippage: str = "10", priority_fee: Optional[float] = None,
              wait_confirm: bool = True) -> str:
    """Full buy flow: fetch unsigned -> sign locally -> send. Returns signature.

    `wait_confirm=False` returns the moment the tx is submitted (the sniping
    path) instead of polling up to 20s for it to land.
    """
    if config.MAX_BUY_SOL > 0 and amount_sol > config.MAX_BUY_SOL:
        raise ValueError(f"buy amount {amount_sol} exceeds MAX_BUY_SOL {config.MAX_BUY_SOL}")
    pubkey = str(keypair.pubkey())
    # fire the balance read and the PumpPortal build CONCURRENTLY — the RPC
    # round trip must not sit in front of the tx build on the sniping path.
    balance_task = asyncio.create_task(get_balance_sol(pubkey))
    try:
        unsigned = await get_unsigned_tx(pubkey, mint, "buy", amount_sol,
                                         denominated_in_sol=True, slippage=slippage,
                                         priority_fee=priority_fee)
        balance = await balance_task
    except BaseException:
        if balance_task.done():
            balance_task.exception()  # already failed — retrieve so nothing warns
        else:
            balance_task.cancel()
        raise
    if balance - amount_sol < config.MIN_SOL_LEFT:
        raise RuntimeError(
            f"insufficient SOL: balance {balance:.4f}, need to keep {config.MIN_SOL_LEFT}")
    signed = sign_tx(unsigned, str(keypair))
    sig = await send_tx(signed)
    if wait_confirm:
        conf = await confirm_tx(sig)
        if conf["found"] is False:
            log.warning("buy tx %s not confirmed within timeout — may still land",
                        sig[:16])
    else:
        confirm_in_background(sig)
    return sig


async def sell(keypair: Keypair, mint: str, percent: float = 100.0,
               slippage: str = "10", priority_fee: Optional[float] = None) -> str:
    """Sell a percentage of the token balance (denominatedInSol=false means token amount)."""
    if not 0 < percent <= 100:
        raise ValueError("percent must be 0-100")
    pubkey = str(keypair.pubkey())
    token_balance = await get_token_balance(pubkey, mint)
    if token_balance <= 0:
        raise RuntimeError("no token balance to sell")
    amount = token_balance * percent / 100.0
    unsigned = await get_unsigned_tx(pubkey, mint, "sell", amount,
                                     denominated_in_sol=False, slippage=slippage,
                                     priority_fee=priority_fee)
    signed = sign_tx(unsigned, str(keypair))
    sig = await send_tx(signed)
    await confirm_tx(sig)
    return sig


TOKEN_PROGRAM = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
TOKEN_2022_PROGRAM = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"


async def get_token_balance(pubkey: str, mint: str) -> float:
    """Read token balance across BOTH the classic Token program and Token-2022.

    NOTE: RPC getTokenAccountsByOwner rejects a filter containing both `mint`
    and `programId` — so we query each program by programId only and match
    the mint client-side. Retries briefly for RPC staleness right after a buy.
    """
    import asyncio
    import json
    for attempt in range(3):
        total = 0.0
        for program in (TOKEN_PROGRAM, TOKEN_2022_PROGRAM):
            body = json.dumps({
                "jsonrpc": "2.0", "id": 1, "method": "getTokenAccountsByOwner",
                "params": [pubkey, {"programId": program},
                           {"encoding": "jsonParsed"}],
            })
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.post(config.SOLANA_RPC, data=body,
                                      headers={"Content-Type": "application/json"}) as resp:
                        data = await resp.json()
                accounts = data.get("result", {}).get("value", [])
                for acc in accounts:
                    info = acc["account"]["data"]["parsed"]["info"]
                    if info["mint"] != mint:
                        continue
                    total += float(info["tokenAmount"]["uiAmount"] or 0)
            except Exception:
                continue
        if total > 0 or attempt == 2:
            return total
        await asyncio.sleep(2.0)  # retry — likely RPC staleness
    return 0.0


BASE58_ALPHABET = set("123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz")


def keypair_from_secret(secret: str) -> Keypair:
    """Accept base58 (Phantom export), 128-char hex (stored format), or [u8;64] JSON.

    Validates input strictly — solders' Rust parser panics on bad base58,
    which would crash the whole bot.
    """
    secret = secret.strip()
    if secret.startswith("["):
        import json
        arr = json.loads(secret)
        if not isinstance(arr, list) or len(arr) != 64 or not all(
                isinstance(x, int) and 0 <= x <= 255 for x in arr):
            raise ValueError("JSON key must be an array of 64 bytes 0-255")
        return Keypair.from_bytes(bytes(arr))
    if len(secret) == 128:
        try:
            return Keypair.from_bytes(bytes.fromhex(secret))
        except ValueError:
            raise ValueError("128-char key is not valid hex — re-import the wallet")
    if not (87 <= len(secret) <= 90):
        raise ValueError(
            f"base58 private key should be ~87-90 chars, got {len(secret)}")
    bad = set(secret) - BASE58_ALPHABET
    if bad:
        raise ValueError(
            f"invalid base58 character(s): {', '.join(sorted(bad))} "
            "(0, O, I, l are not in base58 — copy the key again from Phantom)")
    return Keypair.from_base58_string(secret)
