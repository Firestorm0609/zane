#!/usr/bin/env python3
"""Vet discovered caller wallets against pump.fun's own aggregate stats.

Companion to scan_callers.py. That script DISCOVERS callers from the
volume-ranked leaderboard — which structurally hides selective, low-volume
callers. This one scores a pool discovered another way (coin expansion over
winning coins) using /users/<wallet>/callout-stats, which is authoritative and
independent of how the wallet was found.

Why that matters: discovery surfaces are all biased in some direction (the
leaderboard favours volume; coin sampling favours whoever was active on the
coins you happened to pick). The SCORING endpoint is not. So any pool can be
ranked honestly as long as the final decision is made here.

Resumable: results append to the output file, so this can be run in batches
across several paused windows. Because the per-IP rate budget is shared with
the live poller, pause the bot first (`tmux send-keys -t sesh:1.0 C-c`) or the
two will trip the limit together.

Usage:
    .venv/bin/python hunt_callers.py --limit 150
    .venv/bin/python hunt_callers.py --limit 150 --pacing 3
    .venv/bin/python hunt_callers.py --summary        # just re-report so far
"""
import argparse
import json
import os
import time
import urllib.request

POOL = "hunt_pool.json"      # wallet -> {hits, ok, best} from the coin expansion
OUT = "hunt_vetted.json"     # accumulated callout-stats verdicts
MIN_ALLTIME_N = 600          # "selective" ceiling on lifetime callouts
PASS_2X = 45.0               # all-time 2x rate that counts as interesting


def base_url() -> str:
    try:
        import config
        return config.PUMP_CALLOUT_BASE
    except Exception:
        return "https://driftrace.tech/pump"


def load_token() -> str:
    try:
        with open(".pump-auth.json") as f:
            return (json.load(f) or {}).get("token", "") or ""
    except Exception:
        return ""


TOKEN = load_token()
BASE = base_url()


def get(path: str) -> dict:
    req = urllib.request.Request(
        BASE + path,
        headers={"User-Agent": "Mozilla/5.0", "Authorization": f"Bearer {TOKEN}"})
    with urllib.request.urlopen(req, timeout=25) as r:
        return json.load(r)


def fetch_stats(wallet: str) -> dict | None:
    s = get(f"/users/{wallet}/callout-stats") or {}
    a = s.get("allTime") or s.get("all_time") or {}
    mo = s.get("monthly") or {}
    n = a.get("totalCallouts")
    if not n:
        return None
    return {
        "w": wallet,
        "n": n,
        "2x": a.get("twoXPercent") or 0,
        "med": a.get("medianMultiple") or 0,
        "avg": a.get("averageMultiple") or 0,
        "m2x": mo.get("twoXPercent") or 0,
    }


def load_out() -> list[dict]:
    if os.path.exists(OUT):
        try:
            return json.load(open(OUT))
        except Exception:
            return []
    return []


def save_out(rows: list[dict]) -> None:
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rows, f, indent=1)
    os.replace(tmp, OUT)


def order_candidates(pool: dict, done: set[str]) -> list[str]:
    """Most-promising first, so a partial run is still informative.

    Wallets seen on >=2 winning coins carry real evidence, so they lead,
    ranked by in-pool hit-rate. Single-observation wallets follow by peak.
    """
    multi = [(w, e) for w, e in pool.items() if e.get("hits", 0) >= 2 and w not in done]
    single = [(w, e) for w, e in pool.items() if e.get("hits", 0) < 2 and w not in done]
    multi.sort(key=lambda kv: (kv[1]["ok"] / max(1, kv[1]["hits"]), kv[1]["best"]),
               reverse=True)
    single.sort(key=lambda kv: kv[1]["best"], reverse=True)
    return [w for w, _ in multi] + [w for w, _ in single]


def report(rows: list[dict]) -> None:
    keep = sorted([r for r in rows if r["n"] < MIN_ALLTIME_N and r["2x"] >= PASS_2X],
                  key=lambda r: -r["2x"])
    print(f"\n=== PASS (all-time n<{MIN_ALLTIME_N} AND 2x>={PASS_2X:g}%): "
          f"{len(keep)} of {len(rows)} vetted ===")
    print(f"    {'n':>6} {'2x%':>6} {'med':>5} {'mo2x':>6}  wallet")
    for r in keep:
        print(f"    {r['n']:>6} {r['2x']:>5.1f} {r['med']:>5.2f} {r['m2x']:>5.1f}  {r['w']}")

    nm = sorted([r for r in rows if r["n"] < MIN_ALLTIME_N and 35 <= r["2x"] < PASS_2X],
                key=lambda r: -r["2x"])
    print(f"\n=== near-miss (n<{MIN_ALLTIME_N}, 2x 35-{PASS_2X:g}%): {len(nm)} ===")
    for r in nm[:12]:
        print(f"    {r['n']:>6} {r['2x']:>5.1f} {r['med']:>5.2f} {r['m2x']:>5.1f}  {r['w']}")
    print("\n(Meme Detective reference: n=133  2x=75.9%  med=2.73x)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=150, help="wallets to vet this run")
    ap.add_argument("--pacing", type=float, default=3.0, help="seconds between requests")
    ap.add_argument("--summary", action="store_true", help="report only, no requests")
    args = ap.parse_args()

    rows = load_out()
    if args.summary:
        report(rows)
        return

    pool = json.load(open(POOL))
    done = {r["w"] for r in rows}
    todo = order_candidates(pool, done)[:args.limit]
    print(f"pool {len(pool)} · already vetted {len(done)} · vetting {len(todo)} "
          f"at {args.pacing:g}s pacing (~{len(todo)*args.pacing/60:.1f} min)")

    fails = 0
    for i, w in enumerate(todo, 1):
        try:
            r = fetch_stats(w)
        except Exception as e:
            fails += 1
            print(f"  ! {w[:12]}… {e}")
            time.sleep(args.pacing)
            continue
        if r:
            rows.append(r)
            save_out(rows)
        if i % 10 == 0:
            print(f"  {i}/{len(todo)}  (fails {fails})", flush=True)
        time.sleep(args.pacing)

    print(f"done: appended {len(rows)-len(done)} rows, {fails} failures")
    report(rows)


if __name__ == "__main__":
    main()
