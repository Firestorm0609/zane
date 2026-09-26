"""Find pump.fun callers with the most CONSISTENT 2x callouts.

Source: GET /callout/leaderboard — pump.fun's own per-profile aggregate stats
(totalCallouts, pct2xOrMore, avgMultiple, medianMultiple, topCallouts). It
returns 50 profiles in one request, so discovery costs one call instead of
hundreds of coin lookups. NOTE: this endpoint needs auth (401 without a Bearer
token) — the token is read from .pump-auth.json, which the browser forwarder
(pump_forwarder.user.js) keeps fresh.

Two traps this script already handles: `pct2xOrMore` is a FRACTION (0.23 =
23%) while its sibling `onePointFiveXPercent` is a percentage (35.91); and the
leaderboard is MULTICHAIN, so EVM (`0x…`) profiles rank high on callouts the
bot could never trade.

Ranking: raw 2x-rate is a trap — 2/2 (100%) is noise, 30/50 (60%) is an edge.
Profiles are ranked by the WILSON LOWER BOUND of their 2x rate, which rewards
rate AND sample size. A deep dive then re-verifies the top profiles against
their real last-50 callout history and reports how much of that lands inside
YOUR market-cap window — a caller who only calls $300k coins is useless when
your filter is $5k-24k.

Rate budget: this shares the 60 req/min per-IP limit with the running bot.
Use a slow --pacing (the default), or stop the bot first.

Usage:
    .venv/bin/python scan_callers.py                    # leaderboard + top 10
    .venv/bin/python scan_callers.py --deep 20 --pacing 6
    .venv/bin/python scan_callers.py --pages 4          # more profiles to rank
"""
import argparse
import asyncio
import json
import math
import os
import sqlite3
import sys
import time

os.environ.setdefault("WALLET_ENC_KEY", "0" * 32)
import config  # noqa: E402
import callouts as callouts_mod  # noqa: E402
from callouts import CalloutClient  # noqa: E402

LEADERBOARD_PAGE = 50
HISTORY_LIMIT = 50
Z = 1.96                      # 95% confidence
DEFAULT_WINDOW = (5_000.0, 24_000.0)
MIN_CALLS_TO_RANK = 20        # below this the sample can't support a claim


def wilson_lower(wins: int, n: int, z: float = Z) -> float:
    """Lower bound of the 2x rate at 95% confidence.

    This is what separates an edge from a lucky streak: 3/3 scores ~44% vs
    30/50 at ~46% — same raw 100% vs 60%, but the honest bound shows only one
    of those is repeatable.
    """
    if n <= 0:
        return 0.0
    p = wins / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return max(0.0, (centre - margin) / denom)


def my_window(db_path: str = "bot.db") -> tuple[float, float]:
    """Market-cap window taken from the callers already configured in the bot."""
    try:
        c = sqlite3.connect(db_path)
        row = c.execute("SELECT min_mcap, max_mcap FROM callers "
                        "WHERE min_mcap > 0 OR max_mcap > 0 LIMIT 1").fetchone()
        if row and (row[0] or row[1]):
            return (row[0] or 0.0, row[1] or 0.0)
    except Exception:
        pass
    return DEFAULT_WINDOW


def followed_wallets(db_path: str = "bot.db") -> set[str]:
    try:
        c = sqlite3.connect(db_path)
        return {r[0] for r in c.execute("SELECT caller_id FROM callers")}
    except Exception:
        return set()


def load_token() -> str:
    try:
        with open(config.PUMP_AUTH_FILE) as f:
            return (json.load(f) or {}).get("token") or ""
    except Exception:
        return config.PUMP_AUTH_TOKEN or ""


def _median(xs: list[float]) -> float:
    return sorted(xs)[len(xs) // 2] if xs else 0.0


async def fetch_leaderboard(client: CalloutClient, pages: int) -> list[dict]:
    rows, seen = [], set()
    for page in range(pages):
        status, data = await client._paced_get(
            f"{config.PUMP_CALLOUT_BASE}/callout/leaderboard",
            {"limit": str(LEADERBOARD_PAGE), "offset": str(page * LEADERBOARD_PAGE)})
        if status != 200 or not isinstance(data, dict):
            print(f"      page {page}: HTTP {status} — stopping", flush=True)
            break
        items = data.get("callouts") or []
        if not items:
            break
        fresh = 0
        for it in items:
            wallet = it.get("primaryWallet") or it.get("userId") or ""
            if wallet and wallet not in seen:
                seen.add(wallet)
                rows.append(it)
                fresh += 1
        print(f"      page {page}: {len(items)} profiles ({fresh} new)", flush=True)
        if fresh == 0:
            break
    return rows


def is_solana(wallet: str) -> bool:
    """The bot can only trade Solana mint addresses, so EVM callers are useless.

    The leaderboard is multichain: an EVM profile shows up looking spectacular
    (odd multiples, tiny sample) but its callouts are `0x…` tokens that
    PumpPortal/Jupiter can't touch.
    """
    return bool(wallet) and not wallet.lower().startswith("0x")


def score_row(it: dict, mine: set[str]) -> dict:
    """Leaderboard-only view of a profile (comparable across all profiles)."""
    tops = it.get("topCallouts") or []
    top_mults = [float(t.get("multiple") or 0) for t in tops]
    top_mcs = [float(t.get("marketCap") or 0) for t in tops if t.get("marketCap")]
    n = int(it.get("totalCallouts") or 0)
    # SCALE TRAP: pct2xOrMore is a FRACTION (0.23 = 23%), while its siblings
    # onePointFiveXPercent / onePointTwoXPercent are percentages (35.91).
    # Reading it as a percent silently zeroes every score. Be tolerant of both
    # so a future server-side change can't quietly break the ranking again.
    raw = float(it.get("pct2xOrMore") or 0)
    rate2x = raw / 100.0 if raw > 1.0 else raw
    wallet = it.get("primaryWallet") or it.get("userId") or ""
    return {
        "wallet": wallet,
        "uuid": it.get("user_uuid") or "",
        "n": n,
        "rate2x": rate2x,
        "wins2x": round(rate2x * n),
        "wilson": wilson_lower(round(rate2x * n), n),
        "avg": round(float(it.get("avgMultiple") or 0), 2),
        "median": round(float(it.get("medianMultiple") or 0), 2),
        "best": round(max(top_mults), 2) if top_mults else 0.0,
        "med_mc": _median(top_mcs),
        "followed": wallet in mine,
    }


async def deep_dive(client: CalloutClient, wallet: str) -> dict:
    """Verify a profile from its real last-50 callouts (keys prefixed v_)."""
    out: dict = {}
    prof = await client.caller_profile(wallet)
    out["name"] = prof.get("username") or ""
    out["followers"] = prof.get("followers") or 0
    out["x"] = prof.get("x_username") or ""
    try:
        data = await client.list_callouts(wallet, limit=HISTORY_LIMIT)
    except Exception as e:
        print(f"      ! {wallet[:10]}… {e}", flush=True)
        return out
    hist = data.get("callouts") or []
    if not hist:
        return out
    mults = [float(co.get("maxMultiplier") or 0) for co in hist]
    tss = [co.get("createdAt") or 0 for co in hist]
    span_d = (max(tss) - min(tss)) / 86_400_000 if len(tss) > 1 else 0.0
    mcs = [float(co.get("marketCap") or 0) for co in hist if co.get("marketCap")]
    lo, hi = WINDOW
    pairs = [(float(co.get("marketCap") or 0), m)
             for co, m in zip(hist, mults) if co.get("marketCap")]
    in_win = [m for mc, m in pairs if lo <= mc <= hi]
    wins = sum(1 for m in mults if m >= 2.0)
    win_wins = sum(1 for m in in_win if m >= 2.0)
    out.update({
        "v_n": len(mults),
        "v_wins2x": wins,
        "v_rate2x": wins / len(mults),
        "v_wilson": wilson_lower(wins, len(mults)),
        "v_avg": round(sum(mults) / len(mults), 2),
        "v_median": round(_median(mults), 2),
        "v_best": round(max(mults), 2),
        "v_span_d": round(span_d, 1),
        "v_calls_per_week": round(len(mults) / span_d * 7, 1) if span_d > 0.5 else None,
        "v_med_mc": _median(mcs),
        "v_in_window_n": len(in_win),
        "v_in_window_wins": win_wins,
        "v_in_window_rate2x": (win_wins / len(in_win)) if in_win else None,
        "v_last_call_h": round((time.time() * 1000 - max(tss)) / 3_600_000, 1) if tss else None,
    })
    return out


def cell(x, fmt: str = "{:g}", dash: str = "?") -> str:
    return fmt.format(x) if x not in (None, 0, "") else dash


async def main() -> None:
    global WINDOW
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=1,
                    help="leaderboard pages; the endpoint appears to IGNORE offset "
                         "and always returns the same 50 profiles")
    ap.add_argument("--deep", type=int, default=10, help="profiles to deep-dive")
    ap.add_argument("--pacing", type=float, default=6.0,
                    help="seconds between requests (shares the bot's 60/min budget)")
    ap.add_argument("--min-calls", type=int, default=MIN_CALLS_TO_RANK,
                    help="ignore profiles with fewer total callouts (sample size)")
    args = ap.parse_args()

    WINDOW = my_window()
    lo, hi = WINDOW
    mine = followed_wallets()
    client = CalloutClient()
    callouts_mod.MIN_REQUEST_INTERVAL_S = args.pacing
    token = load_token()
    client.set_static_token(token)
    print(f"auth: {'token loaded' if token else 'NO TOKEN — authenticated endpoints will 401'}")
    print(f"target window: ${lo:,.0f}-{hi:,.0f} · already following {len(mine)} wallet(s) · "
          f"pacing {args.pacing:g}s\n", flush=True)

    try:
        print(f"[1/3] leaderboard ({args.pages} page(s))…", flush=True)
        board = await fetch_leaderboard(client, args.pages)
        if not board:
            print("\nno leaderboard data — the pump.fun token has probably expired "
                  "(check with `pumpfarm.py --check`).\nThe scan needs auth; "
                  "/coins and /callout/top alone are not enough.")
            return

        skipped_evm = 0
        ranked = []
        for it in board:
            wallet = it.get("primaryWallet") or it.get("userId") or ""
            if not is_solana(wallet):
                skipped_evm += 1
                continue
            if int(it.get("totalCallouts") or 0) < args.min_calls:
                continue
            ranked.append(score_row(it, mine))
        ranked.sort(key=lambda r: r["wilson"], reverse=True)
        print(f"      {len(ranked)} Solana profiles with ≥{args.min_calls} callouts "
              f"({skipped_evm} EVM skipped)", flush=True)

        n_deep = min(args.deep, len(ranked))
        print(f"\n[2/3] deep-diving top {n_deep} "
              f"(verifying their last {HISTORY_LIMIT} callouts)…", flush=True)
        for i, r in enumerate(ranked[:n_deep], 1):
            r.update(await deep_dive(client, r["wallet"]))
            print(f"      {i}/{n_deep} {r.get('name') or r['wallet'][:10] + '…'}", flush=True)

        print("\n[3/3] MOST CONSISTENT 2x CALLERS (pump.fun's own aggregates, "
              "ranked by confidence-adjusted 2x-rate)\n")
        hdr = (f"{'score':>6} {'2x%':>5} {'n':>5} {'avg':>6} {'med':>5} {'best':>7} "
               f"{'medMC':>8} {'last':>6}  wallet")
        print(hdr)
        print("-" * len(hdr))
        for r in ranked:
            best = f"{r['best']:.1f}x" if r.get("best") else "-"
            medmc = f"${r['med_mc']/1000:,.0f}k" if r.get("med_mc") else "?"
            last = (f"{r['v_last_call_h']:.0f}h"
                    if r.get("v_last_call_h") is not None else "-")
            tag = " (yours)" if r.get("followed") else ""
            print(f"{r['wilson']*100:5.0f}% {r['rate2x']*100:4.0f}% {r['n']:>5} "
                  f"{r['avg']:>5.1f}x {r['median']:>4.1f}x {best:>7} {medmc:>8} "
                  f"{last:>6}  {r['wallet'][:14]}…{tag}")

        verified = [r for r in ranked if r.get("v_rate2x") is not None]
        if verified:
            print(f"\nVERIFIED last-{HISTORY_LIMIT} (independent recount of the same "
                  f"profiles):")
            for r in verified:
                nm = f"@{r['name']}" if r.get("name") else r["wallet"][:10] + "…"
                fol = f" · {r['followers']:,} followers" if r.get("followers") else ""
                lc = r.get("v_last_call_h")
                lcs = f" · last call {lc:.0f}h ago" if lc is not None else ""

                print(f"  {r['v_rate2x']*100:4.0f}% 2x ({r['v_wins2x']}/{r['v_n']}) · "
                      f"avg {r['v_avg']:.1f}x · med MC ${r['v_med_mc']/1000:,.0f}k{fol}{lcs} · {nm}")

        in_window = [r for r in verified
                     if r.get("v_in_window_rate2x") is not None and (r.get("v_in_window_n") or 0) >= 5]
        in_window.sort(key=lambda r: (r["v_in_window_rate2x"], r["v_in_window_n"]), reverse=True)
        if in_window:
            print(f"\n>>> BEST INSIDE YOUR ${lo:,.0f}-{hi:,.0f} WINDOW "
                  f"(2x-rate on callouts in that range):")
            for r in in_window[:10]:
                nm = f"@{r['name']} " if r.get("name") else ""
                print(f"  {r['v_in_window_rate2x']*100:4.0f}% "
                      f"({r['v_in_window_wins']}/{r['v_in_window_n']} in window) · "
                      f"{r['v_n']} calls total · med MC ${r['v_med_mc']/1000:,.0f}k · "
                      f"{nm}/addcaller {r['wallet']}")
        else:
            print("\n(no profile had ≥5 callouts inside your window)")

        with open("caller_scan.json", "w") as f:
            json.dump(ranked, f, indent=1)
        print("\nsaved → caller_scan.json")
    finally:
        await client.close()


if __name__ == "__main__":
    t0 = time.time()
    try:
        asyncio.run(main())
    finally:
        print(f"done in {time.time()-t0:.0f}s", file=sys.stderr)
