"""Rank callers by EXPECTED VALUE under the bot's own live exit rules.

The other scanners measure a caller in the abstract (2x-rate, streaks). This
one asks the only question that decides money: if we had taken every one of
their callouts with OUR exit settings, would we be up or down — and by how
much? Exit rules are read from bot.db, not hard-coded, so changing TP/trail/SL
moves the ranking with you.

NO market-cap window is applied by default. The mcap window is a per-caller
preference (MD's $5k-24k is MD's, not a house rule), and filtering every
candidate through one caller's window would misjudge them. Instead each
caller's own mcap profile is reported — median plus a 2x-rate breakdown by
market-cap band — so you can pick the window that actually suits them.

Simulation model (documented because it is an approximation):
  * `maxMultiplier` is the PEAK a coin reached after the call. Without the
    path, each callout is scored by the exit the rules would most likely have
    produced from that peak:
      - peak >= TP: sell tp_sell_pct at TP, remainder trails out at
        (1 - trail%) x peak
      - peak >= trail arm (1.2x): full trail exit at (1 - trail%) x peak
      - otherwise: full stop-loss exit at SL
  * Costs per leg: 1% venue + priority fee sized against the buy + a slippage
    haircut (--haircut).

Usage:
    .venv/bin/python ev_scan.py                     # all of streak_scan.json
    .venv/bin/python ev_scan.py --limit 25 --pacing 5 --haircut 0.03
    .venv/bin/python ev_scan.py --min-mcap 5000 --max-mcap 24000   # optional
"""
import argparse
import asyncio
import datetime as dt
import json
import os
import sqlite3
import sys
import time

os.environ.setdefault("WALLET_ENC_KEY", "0" * 32)
import config  # noqa: E402
import callouts as callouts_mod  # noqa: E402
from callouts import CalloutClient  # noqa: E402
from scan_callers import followed_wallets, is_solana, load_token  # noqa: E402

DAY = dt.timezone.utc
BANDS = [(0, 10_000), (10_000, 30_000), (30_000, 100_000), (100_000, 1e12)]


def our_rules(db_path: str = "bot.db") -> dict:
    """The live exit/execution settings we would trade with."""
    r = {"tp_x": 2.0, "tp_pct": 100.0, "sl_x": 0.5, "trail_pct": 0.0,
         "buy_sol": 0.1, "priority_fee": 0.001, "slip_pct": 0.0}
    try:
        c = sqlite3.connect(db_path)
        c.row_factory = sqlite3.Row
        row = c.execute("SELECT * FROM callers LIMIT 1").fetchone()
        if row:
            d = dict(row)
            r["tp_x"] = float(d.get("max_multiple") or 2.0)
            r["sl_x"] = float(d.get("stop_multiple") or 0.5)
            r["tp_pct"] = float(d.get("tp_sell_pct") or 100.0)
            r["trail_pct"] = float(d.get("trail_pct") or 0.0)
            r["buy_sol"] = float(d.get("buy_sol") or 0.1)
            pf = float(d.get("priority_fee") or 0)
            r["priority_fee"] = pf if pf > 0 else config.PRIORITY_FEE_SOL
            r["slip_pct"] = float(d.get("slippage") or 0)
    except Exception:
        pass
    return r


def simulate(peak: float, R: dict, haircut: float) -> float:
    """P&L in SOL for one callout that peaked at `peak`, under rules R."""
    basis = R["buy_sol"]
    tp_x, tp_pct = R["tp_x"], R["tp_pct"] / 100.0
    trail = R["trail_pct"] / 100.0
    sl_x = R["sl_x"]
    fee = 0.01 + (R["priority_fee"] / basis if basis else 0) + haircut
    net = max(0.0, 1.0 - fee)
    proceeds = 0.0
    if tp_x > 0 and peak >= tp_x:
        proceeds += basis * tp_pct * tp_x * net
        rest = 1.0 - tp_pct
        if rest > 0.001:
            exit_x = max(sl_x, (1.0 - trail) * peak) if trail > 0 else sl_x
            proceeds += basis * rest * exit_x * net
    elif trail > 0 and peak >= 1.2:
        proceeds += basis * (1.0 - trail) * peak * net
    else:
        proceeds += basis * sl_x * net
    return proceeds - basis


def pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="streak_scan.json", help="candidate file")
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--pacing", type=float, default=5.0)
    ap.add_argument("--haircut", type=float, default=0.03,
                    help="per-leg slippage/execution haircut (0.03 = 3%)")
    ap.add_argument("--min-calls", type=int, default=15,
                    help="minimum callouts in history to be ranked")
    ap.add_argument("--min-mcap", type=float, default=0.0)
    ap.add_argument("--max-mcap", type=float, default=0.0)
    args = ap.parse_args()

    R = our_rules()
    lo, hi = args.min_mcap, args.max_mcap
    print("OUR LIVE EXIT RULES")
    print(f"  buy {R['buy_sol']:g} SOL · TP {R['tp_x']:g}x selling {R['tp_pct']:g}% · "
          f"trail {R['trail_pct']:g}% (arms 1.2x) · SL {R['sl_x']:g}x · "
          f"priority {R['priority_fee']:g} SOL")
    print(f"  window filter: {'NONE (per-caller preference)' if not (lo or hi) else f'${lo:,.0f}-${hi:,.0f}'}")
    print(f"  cost/leg modelled: 1% venue + {R['priority_fee']/R['buy_sol']*100:.1f}% "
          f"priority + {args.haircut*100:.1f}% execution = "
          f"{(0.01 + R['priority_fee']/R['buy_sol'] + args.haircut)*100:.1f}%\n")

    try:
        cands = json.load(open(args.src))
    except Exception as e:
        print(f"cannot read {args.src}: {e}")
        return
    mine = followed_wallets()
    seen, pool = set(), []
    for c in cands:
        w = c.get("wallet") or ""
        if w and w not in seen and is_solana(w):
            seen.add(w)
            pool.append(c)
    pool.sort(key=lambda c: (-c.get("max_streak", 0), -c.get("rate2x", 0)))
    pool = pool[:args.limit]
    print(f"simulating {len(pool)} callers (pacing {args.pacing:g}s)…\n", flush=True)

    client = CalloutClient()
    callouts_mod.MIN_REQUEST_INTERVAL_S = args.pacing
    client.set_static_token(load_token())
    results, t0 = [], time.time()
    try:
        for i, c in enumerate(pool, 1):
            w = c["wallet"]
            try:
                data = await client.list_callouts(w, limit=100)
            except Exception as e:
                print(f"  {i}/{len(pool)} ! {w[:10]}… {e}", flush=True)
                continue
            rows = []
            for co in data.get("callouts") or []:
                mc = float(co.get("marketCap") or 0)
                peak = float(co.get("maxMultiplier") or 0)
                if peak <= 0:
                    continue
                if lo or hi:
                    if not mc or not (lo <= mc <= hi):
                        continue
                rows.append((mc, peak, int(co.get("createdAt") or 0)))
            if len(rows) < args.min_calls:
                print(f"  {i}/{len(pool)} {c.get('name') or w[:10]:>18} "
                      f"— only {len(rows)} callouts, skipped", flush=True)
                continue
            pnls = [simulate(p, R, args.haircut) for _, p, _ in rows]
            mcs = [mc for mc, _, _ in rows if mc]
            tss = [t for _, _, t in rows]
            span_d = (max(tss) - min(tss)) / 86_400_000 if len(tss) > 1 else 0.0
            ev = sum(pnls) / len(pnls)
            per_day = (len(rows) / span_d) if span_d > 0.5 else None
            bands = []
            for blo, bhi in BANDS:
                b = [p for mc, p, _ in rows if blo <= mc < bhi]
                if len(b) >= 5:
                    bands.append({
                        "band": f"${blo/1000:,.0f}-{bhi/1000:,.0f}k".replace("-1000000k", "+"),
                        "n": len(b),
                        "rate2x": sum(1 for p in b if p >= 2.0) / len(b),
                        "ev": sum(simulate(p, R, args.haircut) for p in b) / len(b),
                    })
            results.append({
                "name": c.get("name") or w[:12],
                "wallet": w,
                "n": len(rows),
                "rate2x": sum(1 for _, p, _ in rows if p >= 2.0) / len(rows),
                "rate25": sum(1 for _, p, _ in rows if p >= 2.5) / len(rows),
                "ev": ev,
                "ev_pct": ev / R["buy_sol"] * 100,
                "span_d": round(span_d, 1),
                "calls_per_day": round(per_day, 1) if per_day else None,
                "ev_per_day": round(ev * per_day, 4) if per_day else None,
                "win_pct": sum(1 for p in pnls if p > 0) / len(pnls),
                "worst": min(pnls),
                "med_mc": pct(mcs, 0.5),
                "p25_mc": pct(mcs, 0.25),
                "p75_mc": pct(mcs, 0.75),
                "bands": bands,
                "followed": w in mine,
            })
            print(f"  {i}/{len(pool)} {c.get('name') or w[:10]:>18} "
                  f"n={len(rows):>3} · EV {ev:+.4f} SOL ({ev/R['buy_sol']*100:+.0f}%) "
                  f"· 2x {results[-1]['rate2x']*100:.0f}%", flush=True)

        print("\n=== EXPECTED VALUE OF OUR RULES, PER CALLER "
              "(no mcap window applied)\n")
        hdr = (f"{'EV/trade':>9} {'EV/day':>8} {'2x%':>5} {'2.5x%':>6} {'win%':>5} "
               f"{'n':>4} {'c/day':>6} {'med MC':>8}  caller")
        print(hdr)
        print("-" * len(hdr))
        for r in sorted(results, key=lambda r: -r["ev"]):
            nm = f"@{r['name']}" if r["name"] else r["wallet"][:12] + "…"
            cpd = f"{r['calls_per_day']:.1f}" if r["calls_per_day"] else "-"
            epd = f"{r['ev_per_day']:+.3f}" if r["ev_per_day"] is not None else "-"
            print(f"{r['ev_pct']:+8.1f}% {epd:>8} {r['rate2x']*100:4.0f}% "
                  f"{r['rate25']*100:5.0f}% {r['win_pct']*100:4.0f}% {r['n']:>4} "
                  f"{cpd:>6} ${r['med_mc']/1000:>6,.0f}k  {nm}"
                  f"{' (yours)' if r['followed'] else ''}")

        print("\nWHERE EACH CALLER'S EDGE LIVES (2x-rate by market-cap band)")
        for r in sorted(results, key=lambda r: -r["ev"])[:8]:
            nm = f"@{r['name']}" if r["name"] else r["wallet"][:12]
            parts = [f"{b['band']} {b['rate2x']*100:>3.0f}% 2x n={b['n']:<3}"
                     for b in r["bands"]]
            print(f"  {nm:<20} " + " · ".join(parts))
            print(f"  {'':<20} mcap p25 ${r['p25_mc']/1000:,.0f}k · median "
                  f"${r['med_mc']/1000:,.0f}k · p75 ${r['p75_mc']/1000:,.0f}k")

        top = sorted(results, key=lambda r: -r["ev"])[:6]
        print("\nBEST UNDER OUR RULES:")
        for r in top:
            nm = f"@{r['name']}" if r["name"] else r["wallet"][:12]
            epd = f" · {r['ev_per_day']:+.3f} SOL/day" if r["ev_per_day"] else ""
            print(f"  {r['ev_pct']:+5.1f}% per trade{epd} · {r['rate2x']*100:.0f}% 2x · "
                  f"med MC ${r['med_mc']/1000:,.0f}k · {nm}")

        with open("ev_scan.json", "w") as f:
            json.dump(results, f, indent=1)
        print("\nsaved → ev_scan.json")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
