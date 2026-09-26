"""pump.fun token capture utilities (Privy era).

pump.fun signs users in with Privy (auth.privy.io) and accepts a Privy
identity token as the Authorization Bearer on frontend-api-v3.pump.fun.
There is no refresh endpoint we can automate, so a valid token has to come
from a real logged-in browser. Two ways to get it in:

  1. Tampermonkey (recommended, automatic):
       install pump_forwarder.user.js, keep pump.fun open in a tab.
       The script forwards the token to the bot's ingest endpoint
       (PUMP_INGEST_PORT, default 8766) whenever it changes.

  2. Manual (works right now, no extension):
       a. log in on pump.fun in your browser
       b. DevTools -> Network -> click any request to frontend-api-v3.pump.fun
       c. copy the value after "Bearer " in the Authorization request header
       d. python3 pumpfarm.py --set-token "eyJ..."

Commands:
  python3 pumpfarm.py --check        validate the saved token against the API
  python3 pumpfarm.py --set-token T  store a token copied from DevTools

(There is no headless capture here anymore: Playwright/Chromium were removed
from this box, so a fresh token must come from the userscript or DevTools.)
"""
import argparse
import asyncio
import json
import logging
import os
import time

import config

log = logging.getLogger("pumpfarm")


def _load_file() -> dict:
    try:
        with open(config.PUMP_AUTH_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_file(data: dict):
    tmp = config.PUMP_AUTH_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, config.PUMP_AUTH_FILE)
    try:
        os.chmod(config.PUMP_AUTH_FILE, 0o600)
    except OSError:
        pass


def _is_jwt(t: str) -> bool:
    return isinstance(t, str) and t.startswith("eyJ") and len(t) > 100


async def _validate_token(token: str) -> tuple[int, str]:
    """Live-check against GET /auth/my-profile. Returns (status, userId)."""
    import aiohttp
    async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15)) as s:
        async with s.get(
                f"{config.PUMP_CALLOUT_BASE}/auth/my-profile",
                headers={"Authorization": f"Bearer {token}",
                         "Origin": "https://pump.fun",
                         "Accept": "application/json",
                         "Referer": "https://pump.fun/"}) as resp:
            uid = ""
            if resp.status == 200:
                try:
                    d = await resp.json()
                    uid = d.get("userId") or d.get("uuid") or ""
                except Exception:
                    pass
            return resp.status, uid


async def cmd_check() -> int:
    data = _load_file()
    tok = data.get("token") or ""
    if not _is_jwt(tok):
        print(f"❌ no token saved in {config.PUMP_AUTH_FILE}")
        print("   Fix: install pump_forwarder.user.js in Tampermonkey and open "
              "pump.fun (auto), or:")
        print("   DevTools → Network → frontend-api request → copy the "
              "Authorization Bearer value, then:")
        print('   python3 pumpfarm.py --set-token "eyJ..."')
        return 1
    age_h = (time.time() - float(data.get("captured_at") or 0)) / 3600
    print(f"token loaded ({len(tok)} chars, captured {age_h:.1f}h ago) — "
          "validating against pump.fun…")
    status, uid = await _validate_token(tok)
    if status == 200:
        print(f"✅ token WORKS — authenticated as user {uid or '(unknown id)'}")
        print("   restart the bot to enable feed mode (~2s polls)")
        return 0
    if status == 401:
        print("❌ token REJECTED (401) — expired or revoked.")
        print("   Refresh the pump.fun tab (the userscript re-forwards "
              "automatically), or grab a fresh Bearer token and --set-token again.")
        return 1
    print(f"❌ unexpected status {status} (rate limit? Cloudflare?) — try again "
          "in a minute")
    return 1


def cmd_set_token(token: str) -> int:
    token = token.strip().strip('"')
    if not _is_jwt(token):
        print("❌ that doesn't look like a JWT (expected a long string starting "
              "with 'eyJ'). Copy the value after 'Bearer ' in the Authorization "
              "header of a frontend-api-v3.pump.fun request.")
        return 1
    _save_file({"token": token, "captured_at": time.time()})
    print(f"✅ saved to {config.PUMP_AUTH_FILE} — validating…")
    status, uid = asyncio.run(_validate_token(token))
    if status == 200:
        print(f"✅ token WORKS — authenticated as {uid or '(unknown id)'}")
        print("   restart the bot to enable feed mode (~2s polls)")
        return 0
    if status == 401:
        print("⚠️ saved, but the API rejected it (401) — token already expired? "
              "Grab a fresh one.")
        return 1
    print(f"⚠️ saved, but validation returned {status} (network/Cloudflare?) — "
          "the bot will still try it.")
    return 0


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="pump.fun token utilities")
    ap.add_argument("--check", action="store_true",
                    help="validate the saved token against pump.fun")
    ap.add_argument("--set-token", metavar="JWT",
                    help="store a Privy token copied from DevTools")
    args = ap.parse_args()

    if args.check:
        raise SystemExit(asyncio.run(cmd_check()))
    if args.set_token:
        raise SystemExit(cmd_set_token(args.set_token))
    ap.print_help()


if __name__ == "__main__":
    main()
