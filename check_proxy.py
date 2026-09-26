#!/usr/bin/env python3
"""Check whether a proxy is a usable egress for pump.fun polling.

Why this exists: this host's IP is throttled by pump.fun (13 requests per 600s
window, then Retry-After 600s), which is why callout detection needs a different
egress. Before committing a proxy to PUMP_PROXY, test it — free proxy tiers hand
out datacenter IPs and datacenter IPs get flagged individually, so the same
provider can give you a clean IP or a dead one.

Usage:
    python3 check_proxy.py http://user:pass@host:port
    python3 check_proxy.py socks5://user:pass@host:port
    python3 check_proxy.py --probe 5 http://user:pass@host:port

--probe N fires N pump.fun requests to measure how many the IP allows before it
429s. Left at 1 by default on purpose: probing hard can flag a fresh IP you were
about to use, and the bot learns the real budget on its own anyway.
"""
import argparse
import asyncio
import re
import sqlite3
import sys
import time
from typing import Optional

import aiohttp

import config

# Same headers the bot sends — a proxy that only passes for a bare GET is not
# necessarily a proxy the poller can use.
HEADERS = {
    "Origin": "https://pump.fun",
    "Accept": "application/json",
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://pump.fun/",
}
IP_ECHO = "https://api.ipify.org?format=json"


def redact(proxy: Optional[str]) -> str:
    """Drop the user:pass@ from a proxy URL before printing.

    These strings end up in terminals, logs and screenshots; the credentials in
    them are a paid resource worth not leaking.
    """
    if not proxy:
        return "DIRECT (this host)"
    return re.sub(r"//[^@/]+@", "//***@", proxy)


def tracked_caller(db_path: Optional[str] = None) -> Optional[str]:
    """First enabled caller from the bot's DB, so the probe hits a real path."""
    try:
        con = sqlite3.connect(db_path or config.DB_PATH)
        row = con.execute(
            "SELECT caller_id FROM callers WHERE enabled = 1 LIMIT 1").fetchone()
        con.close()
        return row[0] if row else None
    except Exception:
        return None


def is_socks(proxy: str) -> bool:
    return proxy.lower().startswith(("socks4", "socks5"))


def connector_for(proxy: str):
    """SOCKS needs a connector; aiohttp's proxy= kwarg only does HTTP."""
    if not is_socks(proxy):
        return None
    try:
        from aiohttp_socks import ProxyConnector
        return ProxyConnector.from_url(proxy)
    except ImportError:
        raise SystemExit("socks proxy given but aiohttp_socks is missing — "
                         "pip install aiohttp-socks")


async def fetch(session, url: str, proxy: Optional[str]) -> tuple[int, str, float]:
    """One GET. Returns (status, body-or-error, seconds). Never raises."""
    started = time.monotonic()
    try:
        kw = {"proxy": proxy} if proxy and not is_socks(proxy) else {}
        async with session.get(url, **kw) as resp:
            body = await resp.text()
            return resp.status, body, time.monotonic() - started
    except Exception as e:
        return 0, f"{type(e).__name__}: {e}", time.monotonic() - started


async def check(proxy: Optional[str], caller: str, probe: int) -> dict:
    label = redact(proxy)
    print(f"\n=== {label}")
    result = {"proxy": proxy, "ok": False, "status": "DEAD", "allowed": 0}
    connector = connector_for(proxy) if proxy else None

    async with aiohttp.ClientSession(headers=HEADERS, connector=connector,
                                     timeout=aiohttp.ClientTimeout(total=20)) as s:
        status, body, dt = await fetch(s, IP_ECHO, proxy)
        if status != 200:
            print(f"  exit IP : FAILED — {body[:100]}")
            print("  verdict : DEAD (the proxy itself refused: bad creds, quota, "
                  "or unusable endpoint)")
            return result
        import json
        try:
            ip = json.loads(body).get("ip", "?")
        except Exception:
            ip = "?"
        print(f"  exit IP : {ip}  ({dt:.1f}s)")

        url = f"{config.PUMP_CALLOUT_BASE}/callout/list/{caller}?limit=5&sortBy=TIMESTAMP&sortOrder=desc"
        allowed = 0
        for i in range(1, max(1, probe) + 1):
            status, body, dt = await fetch(s, url, proxy)
            if status == 429:
                print(f"  req {i}   : 429 — already throttled")
                result["status"] = "THROTTLED"
                break
            if status == 403 and "pump.fun/blocked" in body:
                print(f"  req {i}   : 403 — pump.fun's own block page "
                      f"(static.pump.fun/blocked)")
                print("             This is an explicit IP blocklist, NOT a rate "
                      "limit: no amount of rotation fixes it, and a headless\n"
                      "             browser can't solve it either. Known-proxy "
                      "ranges (free/datacenter pools) live here.")
                result["status"] = "BLOCKED"
                break
            if status != 200:
                print(f"  req {i}   : HTTP {status} ({body[:60]})")
                result["status"] = f"HTTP {status}"
                break
            try:
                n = len(json.loads(body).get("callouts", []))
            except Exception:
                n = -1
            allowed += 1
            print(f"  req {i}   : 200 · {n} callouts · {len(body)} bytes · {dt:.2f}s")

    result["ok"] = allowed > 0
    result["allowed"] = allowed
    if allowed >= max(1, probe) and result["ok"]:
        result["status"] = "CLEAN"
        print(f"  verdict : CLEAN — {allowed} request(s) served, no throttling")
    elif result["status"] == "DEAD":
        print("  verdict : DEAD — proxy answered the echo but nothing usable")
    return result


async def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("proxies", nargs="*",
                    help="http://user:pass@host:port or socks5://... "
                         "(omit to test this host directly)")
    ap.add_argument("--caller", help="caller wallet to probe (default: first in bot.db)")
    ap.add_argument("--probe", type=int, default=1,
                    help="requests to fire per proxy (default 1; higher measures "
                         "the allowance but can flag a fresh IP)")
    ap.add_argument("--skip-pump", action="store_true",
                    help="only test reachability + exit IP, never touch pump.fun")
    args = ap.parse_args()

    caller = args.caller or tracked_caller()
    if not args.skip_pump and not caller:
        sys.exit("no enabled caller found in the DB — pass --caller <wallet>")
    if caller:
        print(f"probing with caller {caller[:12]}…")

    targets = args.proxies or [None]
    results = []
    for p in targets:
        if args.skip_pump:
            connector = connector_for(p) if p else None
            async with aiohttp.ClientSession(headers=HEADERS, connector=connector,
                                             timeout=aiohttp.ClientTimeout(total=20)) as s:
                status, body, dt = await fetch(s, IP_ECHO, p)
                print(f"\n=== {redact(p)}\n  exit IP : "
                      + (body[:80] if status == 200 else f"FAILED — {body[:80]}"))
            continue
        results.append(await check(p, caller, args.probe))

    if len(results) > 1:
        print("\n=== ranking (cleanest first) ===")
        order = {"CLEAN": 0, "THROTTLED": 1, "HTTP 403": 2, "BLOCKED": 3, "DEAD": 4}
        for r in sorted(results, key=lambda r: (order.get(r.get("status"), 9),
                                                -r.get("allowed", 0))):
            name = redact(r["proxy"])
            print(f"  {r.get('status', '?'):9s} {name[:58]}")
        winner = next((r for r in results if r.get("status") == "CLEAN"), None)
        print(f"\nset this in .env:  PUMP_PROXY={winner['proxy']}" if winner
              else "\nnone usable. BLOCKED = on pump.fun's IP blocklist (free "
                   "proxy pools live there);\nDEAD = the proxy provider refused "
                   "it. A fresh IP not on the blocklist is what you need.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
