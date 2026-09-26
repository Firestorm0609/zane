"""Find callers who hit 2x callouts BACK TO BACK — streaks, not averages.

A high aggregate 2x-rate doesn't mean consistency: 20 wins scattered across 100
callouts is a coin flip that eventually pays, while 3 wins in a row on one
afternoon is a caller riding a live narrative. This measures the second thing.

Per caller, from their real callout history (chronological):
  * longest run of consecutive callouts that each reached >= 2x
  * best single day: how many 2x callouts landed in one UTC day
  * how many separate streaks of 3+ they've had
Only callouts that actually reached 2x count as hits — a callout still at 0.6x
broke the run.

Notes and limits:
  * /callout/list ignores offset; limit=200 returns 100, and that's the cap,
    so "history" is the caller's most recent 100 callouts.
  * Timestamps are formatted in UTC for the day grouping.
  * Shares the bot's 60 req/min budget — keep --pacing high while it trades.

Usage:
    .venv/bin/python streak_scan.py                    # top 25 candidates
    .venv/bin/python streak_scan.py --limit 40 --pacing 6
"""
import argparse
import asyncio
import datetime as dt
import json
import os
import sys
import time
from collections import Counter

os.environ.setdefault("WALLET_ENC_KEY", "0" * 32)
import config  # noqa: E402
import callouts as callouts_mod  # noqa: E402
from callouts import CalloutClient  # noqa: E402
from scan_callers import (followed_wallets, is_solana, load_token,  # noqa: E402
                          wilson_lower)

WIN_X = 2.0                  # what counts as a hit
HISTORY = 100                # /callout/list hard cap
DAY = dt.timezone.utc


def analyse(callouts: list[dict]) -> dict:
    """Streak metrics for one caller's chronological callout history."""
    rows = []
    for c in callouts:
        mint = c.get("coinMint") or ""
        if not mint:
            continue
        rows.append({
            "mint": mint,
            "at": int(c.get("createdAt") or 0),
            "mult": float(c.get("maxMultiplier") or 0),
            "peak_mc": float(c.get("maxPriceSol") or 0),
            "mc": float(c.get("marketCap") or 0),
            "id": c.get("calloutId") or "",
        })
    rows.sort(key=lambda r: r["at"])
    # collapse repeat callouts of the SAME mint: calling one coin three times
    # that doubles once is one win, not three
    dedup, seen = [], set()
    for r in rows:
        if r["mint"] in seen:
            continue
        seen.add(r["mint"])
        dedup.append(r)

    streaks, cur = [], []
    for r in dedup:
        if r["mult"] >= WIN_X:
            cur.append(r)
        else:
            if cur:
                streaks.append(cur)
            cur = []
    if cur:
        streaks.append(cur)

    by_day: Counter = Counter()
    for r in dedup:
        if r["mult"] >= WIN_X:
            day = dt.datetime.fromtimestamp(r["at"] / 1000, DAY).date()
            by_day[day] += 1

    hits = sum(1 for r in dedup if r["mult"] >= WIN_X)
    return {
        "n": len(dedup),
        "hits": hits,
        "rate2x": hits / len(dedup) if dedup else 0.0,
        "wilson": wilson_lower(hits, len(dedup)),
        "max_streak": max((len(s) for s in streaks), default=0),
        "streaks3": sum(1 for s in streaks if len(s) >= 3),
        "best_day": max(by_day.values(), default=0),
        "days_3plus": sum(1 for v in by_day.values() if v >= 3),
        "best_day_date": by_day.most_common(1)[0][0].isoformat() if by_day else "",
        "first_at": dedup[0]["at"] if dedup else 0,
        "last_at": dedup[-1]["at"] if dedup else 0,
        "streaks": [{"len": len(s),
                     "date": dt.datetime.fromtimestamp(s[0]["at"] / 1000, DAY).date().isoformat(),
                     "mints": [x["mint"] for x in s],
                     "mults": [round(x["mult"], 2) for x in s]}
                    for s in sorted(streaks, key=lambda s: -len(s))[:4]],
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=25, help="candidates to analyse")
    ap.add_argument("--min-calls", type=int, default=50,
                    help="minimum total callouts to be considered")
    ap.add_argument("--pacing", type=float, default=6.0,
                    help="seconds between requests (shares the bot's budget)")
    ap.add_argument("--min-rate", type=float, default=0.10,
                    help="skip callers below this raw 2x rate (saves requests)")
    args = ap.parse_args()

    client = CalloutClient()
    callouts_mod.MIN_REQUEST_INTERVAL_S = args.pacing
    token = load_token()
    client.set_static_token(token)
    mine = followed_wallets()
    print(f"auth: {'token' if token else 'NONE'} · hit = peak >= {WIN_X:g}x · "
          f"history cap {HISTORY} · pacing {args.pacing:g}s\n", flush=True)

    try:
        st, board = await client._paced_get(
            f"{config.PUMP_CALLOUT_BASE}/callout/leaderboard", {"limit": "50"})
        if st != 200 or not isinstance(board, dict):
            print(f"leaderboard failed: HTTP {st} (token expired?)")
            return
        cands = []
        for it in board.get("callouts") or []:
            w = it.get("primaryWallet") or it.get("userId") or ""
            raw = float(it.get("pct2xOrMore") or 0)
            rate = raw / 100.0 if raw > 1.0 else raw     # fraction, not percent
            n = int(it.get("totalCallouts") or 0)
            if not is_solana(w) or n < args.min_calls or rate < args.min_rate:
                continue
            cands.append({"wallet": w, "n": n, "rate": rate})
        cands.sort(key=lambda c: c["rate"], reverse=True)
        cands = cands[:args.limit]
        print(f"analysing {len(cands)} Solana callers (of "
              f"{len(board.get('callouts') or [])} leaderboard profiles)\n", flush=True)

        results = []
        for i, c in enumerate(cands, 1):
            try:
                data = await client.list_callouts(c["wallet"], limit=HISTORY)
            except Exception as e:
                print(f"  {i}/{len(cands)} ! {c['wallet'][:10]}… {e}", flush=True)
                continue
            hist = data.get("callouts") or []
            if not hist:
                continue
            a = analyse(hist)
            prof = await client.caller_profile(c["wallet"])
            a["name"] = prof.get("username") or ""
            a["followers"] = prof.get("followers") or 0
            a["wallet"] = c["wallet"]
            a["followed"] = c["wallet"] in mine
            results.append(a)
            print(f"  {i}/{len(cands)} {a['name'] or c['wallet'][:10]:>18} "
                  f"streak={a['max_streak']} best_day={a['best_day']} "
                  f"({a['rate2x']*100:.0f}% of {a['n']})", flush=True)

        print("\n=== CONSECUTIVE 2x STREAKS (longest first) ===\n")
        hdr = (f"{'streak':>6} {'3+':>3} {'day':>3} {'d3+':>4} {'2x%':>5} {'n':>4} "
               f"{'last':>7}  caller")
        print(hdr)
        print("-" * len(hdr))
        for r in sorted(results, key=lambda r: (-r["max_streak"], -r["best_day"],
                                                -r["rate2x"])):
            last = (f"{(time.time()*1000 - r['last_at'])/3_600_000:.0f}h"
                    if r["last_at"] else "-")
            nm = f"@{r['name']}" if r["name"] else r["wallet"][:14] + "…"
            print(f"{r['max_streak']:>6} {r['streaks3']:>3} {r['best_day']:>3} "
                  f"{r['days_3plus']:>4} {r['rate2x']*100:4.0f}% {r['n']:>4} "
                  f"{last:>7}  {nm}{' (yours)' if r['followed'] else ''}")

        hit3 = [r for r in results if r["max_streak"] >= 3]
        print(f"\n{len(hit3)} of {len(results)} callers have had a run of 3+ "
              f"back-to-back 2x callouts.")
        for r in sorted(hit3, key=lambda r: -r["max_streak"])[:12]:
            nm = f"@{r['name']}" if r["name"] else r["wallet"][:12] + "…"
            print(f"\n  {nm} — longest {r['max_streak']} in a row, "
                  f"{r['streaks3']} separate 3+ runs, best day {r['best_day']} "
                  f"({r['best_day_date']})")
            for s in r["streaks"][:3]:
                mults = " · ".join(f"{m:g}x" for m in s["mults"])
                print(f"     {s['len']} in a row on {s['date']}: {mults}")

        with open("streak_scan.json", "w") as f:
            json.dump(results, f, indent=1)
        print("\nsaved → streak_scan.json")
    finally:
        await client.close()


if __name__ == "__main__":
    t0 = time.time()
    try:
        asyncio.run(main())
    finally:
        print(f"done in {time.time()-t0:.0f}s", file=sys.stderr)
