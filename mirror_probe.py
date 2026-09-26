"""mirror_probe.py — how fast did an exec path actually get in?

The push stream tells us the moment a callout exists; the mirrorbot log tells
us when the CA left for the exec bot. What neither can show is the fill: that
only exists on-chain. This walks a mint's history back to genesis and prints
the entry ladder for the seconds after the callout — who bought, how much SOL,
and at what multiple of the callout's own price — so "is the mirror slow?"
gets answered with fills instead of vibes.

The best price anyone paid at or before t is the entry available to anyone
arriving at t, which is the number a slower path has to beat.

    .venv/bin/python mirror_probe.py --caller 6qudAN2kV8mtCcYJxb5QQ6Vr15itdHHdeVbYm99NKMhy --last 4
    .venv/bin/python mirror_probe.py --mint <mint> --created 1790427162851 --callout-price 1.3e-07
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Optional

import config

PAGE = 1000          # signatures per getSignaturesForAddress call
MAX_PAGES = 12       # 12k txs back; MD's callouts draw ~10k sigs in 25 min
WINDOW_S = 15.0      # how far past the callout we care about
MAX_TXS = 30         # parsed txs per mint
BATCH = 10           # getTransaction calls per HTTP request (Helius chokes >that)
LAMPORTS = 1_000_000_000
SUPPLY = 1_000_000_000           # pump.fun tokens are a fixed 1B supply


def _post(body: bytes, timeout: int) -> Any:
    """POST to the RPC, backing off on 429 — a throttled page must never look
    like 'this mint has no history'."""
    delay = 3.0
    for attempt in range(6):
        req = urllib.request.Request(config.SOLANA_RPC, data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 502, 503) or attempt == 5:
                raise
            print(f"    rpc {e.code} — backing off {delay:.0f}s")
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def rpc(method: str, params: list) -> Any:
    if method != "getSignaturesForAddress":     # only the cheap call gets parsed here
        return {}
    body = json.dumps({"jsonrpc": "2.0", "id": 1,
                       "method": method, "params": params}).encode()
    out = _post(body, 30)
    if isinstance(out, dict) and out.get("error"):
        print(f"    rpc error: {out['error']}")
        return {}
    return out


def rpc_batch(calls: list[tuple[str, list]]) -> list:
    """getTransaction in small batches; responses come back reordered by id."""
    out: list = []
    for i in range(0, len(calls), BATCH):
        chunk = calls[i:i + BATCH]
        body = json.dumps([{"jsonrpc": "2.0", "id": i + j, "method": m,
                            "params": p} for j, (m, p) in enumerate(chunk)]).encode()
        got = _post(body, 60)
        if isinstance(got, dict):
            got = [got]
        out += sorted(got, key=lambda r: r.get("id", 0))
        time.sleep(0.6)
    return out


def http_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0",
                                               "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.load(resp)


def callouts_for(caller: str, limit: int) -> list[dict]:
    """Recent callouts from the feed (same endpoint the bot polls)."""
    url = (f"{config.PUMP_CALLOUT_BASE}/callout/list/{caller}"
           f"?limit={limit}&sortBy=TIMESTAMP&sortOrder=desc")
    data = http_json(url)
    items = data.get("callouts") if isinstance(data, dict) else data
    return items or []


def early_signatures(mint: str, created: float) -> list[dict]:
    """Signatures from `created` to `created + WINDOW_S`, oldest first."""
    keep: list[dict] = []
    before: Optional[str] = None
    for n in range(MAX_PAGES):
        params: list = [mint, {"limit": PAGE}]
        if before:
            params[1]["before"] = before
        page = (rpc("getSignaturesForAddress", params).get("result") or [])
        page = [s for s in page if s.get("blockTime")]
        if not page:
            break
        keep = [s for s in page if s["blockTime"] >= created] + keep
        oldest = min(s["blockTime"] for s in page)
        before = page[-1]["signature"]
        print(f"    page {n+1}: {len(page)} sigs, oldest +{oldest - created:.0f}s")
        if oldest <= created:
            break
        time.sleep(0.4)
    hits = [s for s in keep if created <= s["blockTime"] <= created + WINDOW_S]
    return sorted(hits, key=lambda s: s["blockTime"])[:MAX_TXS]


def delivery(tx: dict, wallet: Optional[str] = None) -> Optional[dict]:
    """SOL paid and tokens received by the signer — i.e. what a buyer got.

    Returns None for anything that is not a fresh buy by the tx signer
    (sells, transfers, dev launches, routed trades we can't attribute) and
    for dust txs, which are MEV noise rather than real entries.
    Pass `wallet` to attribute a specific address instead of the signer.
    """
    meta = tx.get("meta") or {}
    if meta.get("err"):
        return None
    msg = (tx.get("transaction") or {}).get("message") or {}
    keys = msg.get("accountKeys") or []
    if not keys or len(meta.get("preBalances") or []) != len(keys):
        return None
    pubs = [k.get("pubkey") if isinstance(k, dict) else k for k in keys]
    who = wallet or pubs[0]
    if who not in pubs:
        return None
    idx = pubs.index(who)

    pre = {(b["accountIndex"]): b for b in (meta.get("preTokenBalances") or [])}
    deltas: dict[str, float] = {}
    for b in (meta.get("postTokenBalances") or []):
        if b.get("owner") != who:
            continue
        was = (pre.get(b["accountIndex"], {}).get("uiTokenAmount") or {}).get("uiAmount") or 0.0
        now = (b.get("uiTokenAmount") or {}).get("uiAmount") or 0.0
        deltas[b.get("mint")] = deltas.get(b.get("mint"), 0.0) + (now - was)
    if not deltas:
        return None
    mint, got = max(deltas.items(), key=lambda kv: kv[1])
    if got <= 0:
        return None                            # that's a sell, not an entry
    pre_lam = (meta.get("preBalances") or [0])[idx]
    post_lam = (meta.get("postBalances") or [0])[idx]
    spent = (pre_lam - post_lam - (meta.get("fee") or 0 if idx == 0 else 0)) / LAMPORTS
    if spent <= 0:
        return None
    price = spent / got
    if got < 1_000 and price > 1e-4:      # dust / ATA rent games, not entries
        return None
    return {"wallet": who, "mint": mint, "sol": spent, "tokens": got,
            "price": price}


def wallet_signatures(wallet: str, lo: float, hi: float) -> list[dict]:
    """The wallet's own txs inside [lo, hi] — one cheap history instead of
    walking every mint's full trade log."""
    keep: list[dict] = []
    before: Optional[str] = None
    for _ in range(8):
        params: list = [wallet, {"limit": 1000}]
        if before:
            params[1]["before"] = before
        page = (rpc("getSignaturesForAddress", params).get("result") or [])
        page = [s for s in page if s.get("blockTime")]
        if not page:
            break
        keep += [s for s in page if lo <= s["blockTime"] <= hi]
        if min(s["blockTime"] for s in page) <= lo:
            break
        before = page[-1]["signature"]
        time.sleep(0.3)
    return sorted(keep, key=lambda s: s["blockTime"])


def wallet_report(wallet: str, jobs: list[tuple[str, float, Optional[float], str]],
                  tail: float = 240.0) -> None:
    """What this wallet did with each callout: fill latency vs the callout
    second, size, and entry multiple against the callout's own price."""
    if not jobs:
        return
    lo = min(c for _, c, _, _ in jobs) - 5
    hi = max(c for _, c, _, _ in jobs) + tail
    sigs = wallet_signatures(wallet, lo, hi)
    if not sigs:
        print(f"{wallet[:8]}…: no txs at all in the callout windows "
              f"({len(jobs)} callouts) — it never traded them")
        return
    results = rpc_batch([("getTransaction", [s["signature"],
                                             {"encoding": "jsonParsed",
                                              "maxSupportedTransactionVersion": 0}])
                         for s in sigs])
    print(f"\nwallet {wallet} — {len(sigs)} txs in the callout windows")
    for job_mint, created, price, label in jobs:
        rows = []
        for s, r in zip(sigs, results):
            if not (created - 5 <= s["blockTime"] <= created + tail):
                continue
            d = delivery(r.get("result") or {}, wallet)
            if not d or d.get("mint") != job_mint:
                continue
            d["t"] = s["blockTime"] - created
            d["mult"] = (d["price"] / price) if price else None
            d["sig"] = s["signature"]
            rows.append(d)
        if not rows:
            print(f"  {label}: no fills")
            continue
        for r in rows:
            mult = f"{r['mult']:.2f}x" if r["mult"] else "?"
            print(f"  {label}  +{r['t']:6.1f}s  {r['sol']:8.3f} SOL  "
                  f"{r['tokens']:>14,.0f} tok  @ {mult}  {r['sig'][:12]}…")


def ladder(mint: str, created: float, call_price: Optional[float],
           label: str) -> Optional[dict]:
    sigs = early_signatures(mint, created)
    if not sigs:
        print(f"{label}: no early txs found in the first {WINDOW_S:.0f}s")
        return None
    calls = [("getTransaction", [s["signature"], {"encoding": "jsonParsed",
                                                  "maxSupportedTransactionVersion": 0}])
             for s in sigs]
    results = rpc_batch(calls)
    rows = []
    for s, r in zip(sigs, results):
        d = delivery(r.get("result") or {})
        if not d:
            continue
        d["t"] = s["blockTime"] - created
        d["mult"] = (d["price"] / call_price) if call_price else None
        rows.append(d)
    if not rows:
        print(f"{label}: {len(sigs)} early txs, none was an attributable buy")
        return None

    print(f"\n{label}  (created {dt.datetime.utcfromtimestamp(created):%H:%M:%S}Z, "
          f"{len(sigs)} txs in first {WINDOW_S:.0f}s)")
    for r in rows[:20]:
        mult = f"{r['mult']:.2f}x" if r["mult"] else "?"
        print(f"  +{r['t']:5.1f}s  {r['wallet'][:6]}…  {r['sol']:7.3f} SOL  "
              f"{r['tokens']:>14,.0f} tok  @ {mult}")

    marks = (1.0, 2.0, 3.0, 5.0, 10.0)
    avail = {}
    for m in marks:
        best = [r["mult"] for r in rows if r["mult"] and r["t"] <= m]
        if best:
            avail[m] = min(best)
    if avail:
        print("  entry available to anyone arriving at: " +
              "  ".join(f"+{m:.0f}s {v:.2f}x" for m, v in sorted(avail.items())))
    return {"mint": mint, "rows": rows, "avail": avail}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--caller", help="analyze this caller's recent callouts")
    ap.add_argument("--last", type=int, default=4, help="how many (default 4)")
    ap.add_argument("--mint")
    ap.add_argument("--created", type=float, help="callout createdAt, epoch seconds or ms")
    ap.add_argument("--callout-price", type=float, help="SOL per token at call")
    ap.add_argument("--wallet", help="attribute this wallet's fills instead of "
                                     "laddering every buyer")
    args = ap.parse_args()

    jobs: list[tuple[str, float, Optional[float], str]] = []
    if args.caller:
        for c in callouts_for(args.caller, args.last)[: args.last]:
            created = float(c.get("createdAt") or 0) / 1000.0
            jobs.append((c.get("coinMint", ""), created, c.get("calloutPrice"),
                         f"{c.get('coinMint','')[:8]}  {c.get('calloutId','')[:8]}"))
    elif args.mint and args.created:
        created = args.created / 1000.0 if args.created > 1e11 else args.created
        jobs.append((args.mint, created, args.callout_price, args.mint[:8]))
    else:
        ap.error("need --caller, or --mint with --created")

    if args.wallet:
        wallet_report(args.wallet, jobs)
        return

    wallets: dict[str, set] = {}
    out = []
    for mint, created, price, label in jobs:
        r = ladder(mint, created, price, label)
        if not r:
            continue
        out.append(r)
        for row in r["rows"]:
            wallets.setdefault(row["wallet"], set()).add(label)

    repeats = {w: s for w, s in wallets.items() if len(s) > 1}
    if repeats:
        print("\nwallets that bought in the early window of more than one "
              "callout (copy-bots / the exec path):")
        for w, labels in sorted(repeats.items(), key=lambda kv: -len(kv[1])):
            print(f"  {w}  in {len(labels)} of {len(out)} callouts")


if __name__ == "__main__":
    sys.exit(main())
