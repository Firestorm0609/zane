"""Background poller: watches callouts for all enabled callers and fires buys."""
import asyncio
import logging
import time
from typing import Optional

import callouts
import config
import jupiter
import trader
from db import DB
from solders.keypair import Keypair

log = logging.getLogger("poller")

# Cosmetic name/symbol lookup is capped so it can never hold up an alert.
META_TIMEOUT_S = 2.5

# ---- callout page sizing ----
# Every poll costs one request against a rate limit that is the scarce resource
# here, so the steady-state page stays small. After a blackout we pull the
# endpoint's hard cap instead — otherwise everything the caller posted while we
# were blind is lost, not just late (see _ingest).
FETCH_LIMIT = 20
CATCHUP_LIMIT = 100        # /callout/list hard cap
CATCHUP_AFTER_S = 90.0     # poll gap that counts as a blackout

# ---- 429 backoff ----
# Retry-After is a FLOOR, not a target. Coming back at the exact instant the
# penalty lapsed made every single request re-arm a fresh 600s ban: the log
# showed one poll attempt per 10m40s with only ~36s of working polling between
# bans (52 bans in a day => 3 alerts in 9 hours). Waiting a growing margin
# PAST the deadline is what earns a clean window back.
BACKOFF_BASE_S = 30.0
BACKOFF_MAX_S = 600.0
# After a penalty, come back GENTLY rather than at full speed: a burst straight
# out of a timeout is what refills the bucket. Every clean cycle buys a step of
# the fast interval back.
RECOVER_INTERVAL_S = 15.0
RECOVER_CYCLES = 6

# ---- browser relay ----
# This host's IP is throttled by pump.fun to a couple of requests a minute,
# which starves callout detection. The user's browser is already required to
# stay open (it forwards the auth token) and sits on an IP pump.fun trusts, so
# it can fetch callout pages and push them here instead. While a caller is
# being relayed we don't poll it from the VPS at all; if the tab goes away the
# relay goes quiet and normal polling resumes by itself.
RELAY_TRUST_S = 120.0


class Poller:
    def __init__(self, db: DB, client: callouts.CalloutClient, notify):
        """
        notify: async fn(tg_id:int, text:str) used to push alerts to users.
        """
        self.db = db
        self.client = client
        self.notify = notify
        self._wallet_cache: dict[int, Optional[Keypair]] = {}
        self._wallet_of: dict[str, str] = {}  # caller_id -> wallet (feed matching)
        self._fanout_tasks: set[asyncio.Task] = set()  # in-flight per-follower buys
        self._last_prune: float = 0.0
        self._last_ok_ts: dict[str, float] = {}  # caller_id -> last good poll
        self._backoff_s: float = 0.0  # extra wait added on top of Retry-After
        self._recover: int = 0  # clean cycles left before full-speed polling
        self._relay_ts: dict[str, float] = {}  # caller_id -> last browser push
        self._relay_logged: set[str] = set()  # callers we've announced as relayed
        self._learned_logged: float = 0.0  # last learned interval we logged

    async def _wallet(self, tg_id: int) -> Optional[Keypair]:
        if tg_id not in self._wallet_cache:
            w = await self.db.load_wallet(tg_id)
            self._wallet_cache[tg_id] = trader.keypair_from_secret(w[1]) if w else None
        return self._wallet_cache[tg_id]

    def invalidate_wallet(self, tg_id: int):
        self._wallet_cache.pop(tg_id, None)

    async def _resolve_wallet(self, caller_id: str) -> str:
        """Map caller_id -> wallet. Feed callouts carry userId = wallet.

        Wallets need no lookup at all (identity). Only UUID-stored callers
        cost one /users request, cached for the process lifetime.
        """
        if caller_id in self._wallet_of:
            return self._wallet_of[caller_id]
        wallet = caller_id
        # pump.fun user UUIDs have 4 dashes (e.g. 550e8400-e29b-41d4-a716-446655440000)
        if len(caller_id) == 36 and caller_id.count("-") == 4:
            try:
                info = await self.client.user_info(caller_id)
                wallet = info.get("address") or caller_id
            except Exception:
                pass
        self._wallet_of[caller_id] = wallet
        return wallet

    async def poll_once(self):
        """One pass: feed mode (1 request covers all callers) or per-caller fallback.

        A 429 is raised, never slept on here: the pause belongs in run(), which
        also owns the backoff state. Sleeping inside the pass is what made the
        poller resume exactly on the server's Retry-After boundary.
        """
        callers = await self.db.get_callers()  # all enabled across all users
        by_caller: dict[str, list[dict]] = {}
        for c in callers:
            by_caller.setdefault(c["caller_id"], []).append(c)

        self._last_caller_count = len(by_caller)
        # Feed mode is opt-in: /following-feed serves a stale 25-item sample and
        # silently missed most callouts, so per-caller is the default path.
        if config.FEED_MODE_ENABLED and self.client.bot_uuid:
            try:
                await self._poll_feed(by_caller)
                return
            except callouts.RateLimited:
                raise
            except Exception as e:
                log.warning("feed poll failed (%s) — falling back to per-caller", e)

        for caller_id, followers in by_caller.items():
            # Browser relay covering this caller? Let it do the fetching — it
            # has an IP pump.fun trusts and we don't, and this VPS budget is
            # the scarce resource. Going quiet falls back to polling here.
            # Covers both sources of pushed callouts: the browser relay and the
            # NATS calloutCreated stream (pump_callouts.py). Both feed
            # ingest_external, which stamps _relay_ts — so while push is
            # delivering, the HTTP poll for this caller is redundant; the moment
            # push goes quiet the stamp ages out and polling resumes by itself.
            relay_age = time.time() - self._relay_ts.get(caller_id, 0.0)
            if relay_age < RELAY_TRUST_S:
                if caller_id not in self._relay_logged:
                    self._relay_logged.add(caller_id)
                    log.info("poll %s: push relay active (%.0fs ago) — "
                             "skipping VPS poll", caller_id[:8], relay_age)
                continue
            if caller_id in self._relay_logged:
                self._relay_logged.discard(caller_id)
                log.warning("poll %s: push/relay quiet for %.0fs — resuming "
                            "VPS polling", caller_id[:8], relay_age)
            # A long gap (penalty pause, restart, freshly added caller) means
            # callouts happened while nobody was looking — fetch the deepest
            # page the endpoint allows so they are late rather than lost.
            gap = time.time() - self._last_ok_ts.get(caller_id, 0.0)
            catchup = gap > CATCHUP_AFTER_S or caller_id not in self._last_ok_ts
            limit = CATCHUP_LIMIT if catchup else FETCH_LIMIT
            try:
                data = await self.client.list_callouts(caller_id, limit=limit)
            except callouts.RateLimited:
                raise  # IP-wide penalty: stop the cycle, let run() back off
            except Exception as e:
                log.warning("poll %s failed: %s", caller_id[:8], e)
                continue
            self._last_ok_ts[caller_id] = time.time()
            if catchup and gap < 172800:  # don't log a fake gap on first sight
                log.info("poll %s: %.0fs since last good poll — catch-up fetch "
                         "(limit=%d)", caller_id[:8], gap, limit)
            await self._ingest(caller_id, followers, data)

    async def ingest_external(self, caller_id: str, data: dict) -> dict:
        """Accept a callout page pushed by the browser relay.

        Detection is deliberately identical to the polling path — same cursor,
        same dedupe, same fanout — so a relayed callout can't double-alert or be
        treated any differently from a polled one. Only the fetching moves (to
        an IP pump.fun isn't throttling), which is the whole point.

        Caller is validated against the tracked set so a stray page for some
        unrelated wallet can't inject callouts into someone's alerts.
        """
        if not isinstance(data, dict) or not caller_id:
            return {"ok": False, "error": "bad payload"}
        items = data.get("callouts")
        if not isinstance(items, list):
            return {"ok": False, "error": "no callouts list in payload"}
        followers = [f for f in await self.db.get_callers()
                     if f["caller_id"] == caller_id]
        if not followers:
            return {"ok": False, "error": "caller not tracked or disabled"}
        self._relay_ts[caller_id] = time.time()
        fanned = await self._ingest(caller_id, followers, data)
        return {"ok": True, "caller_id": caller_id[:12],
                "items": len(items), "new": fanned}

    async def _ingest(self, caller_id: str, followers: list[dict], data: dict) -> int:
        """Fan out every callout newer than each follower's stored cursor.

        Walks the page newest->oldest and STOPS at the cursor, per follower,
        instead of only testing equality with the newest id and then jumping the
        cursor straight to it. The old behaviour permanently dropped every
        callout past the newest 10 whenever a blackout outlasted one page —
        which, at one successful poll per ~10 minutes, was most of them.

        Returns the number of callouts fanned out.
        """
        items = data.get("callouts", []) or []
        if not items:
            return 0
        newest_id = items[0].get("calloutId", "")
        if not newest_id:
            return 0

        # First sight of this caller: seed the cursor rather than alerting on
        # every callout already sitting on the page.
        if not any(f.get("last_callout_id") for f in followers):
            for f in followers:
                await self.db.update_caller_cursor(f["tg_id"], caller_id, newest_id)
            log.info("poll %s: seeded cursor at %s (first sight)",
                     caller_id[:8], newest_id[:12])
            return 0

        done: set[int] = set()
        fresh: list[tuple[dict, list[dict]]] = []
        for c in items:
            cid = c.get("calloutId", "")
            if not cid:
                continue
            # a follower whose cursor is this callout has nothing newer left
            for f in followers:
                if f["tg_id"] not in done and f.get("last_callout_id") == cid:
                    done.add(f["tg_id"])
            targets = [f for f in followers if f["tg_id"] not in done]
            if targets:
                fresh.append((c, targets))
            if len(done) == len(followers):
                break

        fanned = 0
        for c, targets in reversed(fresh):  # oldest first: alerts read in order
            cid = c.get("calloutId", "")
            if await self.db.seen(cid):
                continue
            await self.db.mark_seen(cid)
            await self._fanout(c, targets)
            fanned += 1

        # advance cursor to newest
        for f in followers:
            if f.get("last_callout_id") != newest_id:
                await self.db.update_caller_cursor(f["tg_id"], caller_id, newest_id)
        return fanned

    async def _poll_feed(self, by_caller: dict[str, list[dict]]):
        """Single-request mode: one feed covers every followed caller.

        Feed callouts carry userId = caller wallet. Match against stored
        caller ids, resolving UUID-stored callers to wallets via /users.
        """
        data = await self.client.feed(limit=50)
        items = data.get("callouts", [])
        if not items:
            return
        # resolve wallet alias for every tracked caller (cached). Feed items
        # report the caller as a wallet (userId) OR as a pump.fun user uuid
        # depending on the endpoint — index BOTH so either shape matches.
        followers_by_key: dict[str, list[dict]] = {}
        for caller_id, followers in by_caller.items():
            wallet = await self._resolve_wallet(caller_id)
            for key in {wallet, caller_id}:
                followers_by_key.setdefault(key, []).extend(followers)

        for c in items:
            cid = c.get("calloutId", "")
            if not cid or await self.db.seen(cid):
                continue
            await self.db.mark_seen(cid)
            wallet = c.get("userId", "")
            followers = followers_by_key.get(wallet) \
                or followers_by_key.get(c.get("user_uuid", ""))
            if not followers:
                continue  # callout from someone we don't track
            await self._fanout(c, followers)

    async def _fanout(self, callout: dict, followers: list[dict]):
        """Notify every follower of a new callout; buy where enabled+eligible.

        Fresh callouts (<=60s) trigger auto-buy. Stale ones (60s-30min) are
        alerted as MISSED so the user can decide manually — e.g. after a
        rate-limit pause. Anything older is marked seen silently.

        Per-follower work runs as its own task: a slow tx confirm (up to ~20s)
        or a failing Telegram send (blocked user) must never stall the other
        followers' buys or the poll loop.
        """
        mint = callout.get("coinMint", "")
        if not mint:
            return
        is_sol = callouts.is_solana_mint(mint)
        cid = callout.get("calloutId", "")
        # The one number that says whether a slow entry was OUR latency or
        # pump.fun's: logged for every callout, so a bad one is never a mystery
        # (the 04:37 one could only be reasoned about after the fact).
        age_f = max(0.0, time.time() - (callout.get("createdAt", 0) or 0) / 1000)
        # `via` is the point of this line: push events land at ~0.05s and
        # polled ones at ~11s, so one number says which path found it and
        # whether the delay was ours or pump.fun's.
        log.info("callout %s detected — posted %.1fs ago (mint=%s via=%s)",
                 cid[:12], age_f, mint[:12], callout.get("source") or "http")
        age_s = int(age_f)
        if age_s > 1800:  # >30 min: too old to even mention
            # never drop silently — a blackout used to swallow these invisibly
            log.info("callout %s dropped: %dm old (>30m cap), mint=%s",
                     cid[:12], age_s // 60, mint[:12])
            return
        stale_note = ""
        if age_s > 60:
            # Truncated for the same reason as format_callout: the SIGNAL
            # above already carried the CA, and a full mint here would be
            # scraped as a second CA (i.e. a second buy of the same token).
            stale_note = (f"\n\n⏰ <b>MISSED</b> — callout is {age_s // 60}m old "
                          f"(bot was paused/down). Manual: /buy "
                          f"<code>{mint[:12]}…</code>")

        for f in followers:
            tg = f["tg_id"]
            # per-user dedupe (same callout could hit several callers)
            pos_key = f"{tg}:{cid}"
            if await self.db.seen(pos_key):
                continue
            await self.db.mark_seen(pos_key)
            # Bare-CA signal goes out FIRST, as its own task. It owns no feed
            # request, so it reaches an external exec bot well before the
            # enriched alert below (which waits ~1.2s on the shared feed lock
            # for the name lookup) — and it must never sit in front of our own
            # buy, which is why it is concurrent rather than awaited.
            sig = asyncio.create_task(
                self._signal(f, callout, mint, age_f, is_sol))
            self._fanout_tasks.add(sig)
            sig.add_done_callback(self._fanout_tasks.discard)
            task = asyncio.create_task(
                self._handle_follower(f, callout, stale_note, is_sol))
            self._fanout_tasks.add(task)
            task.add_done_callback(self._fanout_tasks.discard)

    async def _signal(self, f: dict, callout: dict, mint: str,
                      age_f: float, is_sol: bool) -> None:
        """Bare callout signal, sent the moment a callout is detected.

        Purpose-built for an external exec bot (alert -> mirror -> trade):
        the FULL mint sits alone on its own line and is the ONLY long
        base58-looking run in the message, so a CA scraper matches the mint
        and nothing else. No name/symbol lookup happens here on purpose —
        that shares the feed's rate-limit lock and costs ~1.2s, which is
        precisely the delay an exec bot must not pay.

        Fired for EVERY callout, including ones the local mcap/run-up filters
        reject and ones our own buy fails on: the filters decide our buys only,
        never whether the CA is announced. EVM callouts are announced too,
        tagged `evm` so they cannot be mistaken for a Solana CA.
        """
        tg = f["tg_id"]
        mc = float(callout.get("marketCap") or 0)
        if mc <= 0:
            mc = callouts.derive_mcap_usd(callout, self.client.sol_usd_last())
        bits = [f"MC ${mc:,.0f}" if mc > 0 else "MC unknown",
                f"posted {age_f:.1f}s ago",
                "sol" if is_sol else "evm"]
        try:
            await self.notify(tg, f"🔔 <b>SIGNAL</b>\n<code>{mint}</code>\n"
                                  + " · ".join(bits))
        except Exception:
            log.exception("signal failed tg=%s mint=%s", tg, mint)

    async def _summary(self, callout: dict, mint: str) -> str:
        """Alert text. Name/symbol are cosmetic and the lookup shares the
        feed's rate-limit lock, so it is capped — a slow lookup must never
        delay an alert or a buy."""
        try:
            meta = await asyncio.wait_for(self.client.coin_meta(mint),
                                          timeout=META_TIMEOUT_S)
        except Exception:
            meta = {"name": "?", "symbol": "?"}
        return callouts.format_callout(callout, meta)

    async def _handle_follower(self, f: dict, callout: dict,
                               stale_note: str, is_sol: bool):
        """Per-follower fanout work, fully contained: one failure (Telegram
        Forbidden, RPC down, wallet error) affects only this follower.

        Snipe path: once we know this follower will buy, the cosmetic metadata
        lookup is started in the background and the order is submitted
        immediately — nothing network-bound runs before the buy.
        """
        tg = f["tg_id"]
        mint = callout.get("coinMint", "")
        cid = callout.get("calloutId", "")

        async def say(text: str):
            try:
                await self.notify(tg, text)
            except Exception:
                log.exception("notify failed tg=%s", tg)

        try:
            if not is_sol:
                await say(await self._summary(callout, mint)
                          + "\n\n⚠️ Skipped: not a Solana mint.")
                return

            # venue routing: pump.fun mints via PumpPortal, other Solana
            # mints (Raydium/Meteora/other launchpads) via Jupiter
            via_jupiter = not callouts.is_pump_fun_mint(mint)

            # stale callouts: alert-only, never auto-buy
            if stale_note:
                await say(await self._summary(callout, mint) + stale_note)
                return

            buy_sol = float(f.get("buy_sol") or 0)
            if buy_sol <= 0:
                await say(await self._summary(callout, mint)
                          + "\n\nℹ️ Auto-buy disabled (0 SOL).")
                return

            # market-cap range filter (marketCap is USD at call time).
            # /following-feed items don't carry marketCap — when it's unknown
            # (0) we must NOT let a filter silently drop the callout.
            mc = float(callout.get("marketCap") or 0)
            if mc <= 0:
                # Feed-mode items carry no marketCap, which made the mcap filter
                # a silent no-op. Derive it from the call price with the cached
                # SOL/USD rate — no extra request on the buy path.
                mc = callouts.derive_mcap_usd(callout, self.client.sol_usd_last())
            min_mc = float(f.get("min_mcap") or 0)
            max_mc = float(f.get("max_mcap") or 0)
            if mc > 0 and ((min_mc and mc < min_mc) or (max_mc and mc > max_mc)):
                range_txt = (f"{min_mc:,.0f}–{max_mc:,.0f}" if max_mc
                             else f"{min_mc:,.0f}+")
                await say(
                    await self._summary(callout, mint)
                    + f"\n\nℹ️ Skipped by mcap filter "
                      f"(range {range_txt}: MC ${mc:,.0f}).")
                return

            # already-pumped filter: `multiple` is how far the coin has run
            # SINCE the callout. Chasing a call that's already 3x is how you
            # become the exit liquidity. Absent/0 (some feed shapes) = unknown,
            # and unknown must never silently drop a callout.
            run_up = float(callout.get("multiple") or 0)
            max_entry = float(f.get("max_entry_multiple") or 0)
            if run_up > 0 and max_entry > 0 and run_up > max_entry:
                await say(
                    await self._summary(callout, mint)
                    + f"\n\nℹ️ Skipped: already up <b>{run_up:.2f}x</b> since the "
                      f"call (your limit {max_entry:g}x).")
                return

            # ---- SNIPE PATH ----
            # metadata runs in the background; the wallet is cached; the buy
            # is submitted with wait_confirm=False so confirmation can't
            # delay the next callout's processing either.
            meta_task = asyncio.create_task(self._summary(callout, mint))
            try:
                kp = await self._wallet(tg)
            except Exception as e:
                log.exception("wallet load failed tg=%s", tg)
                await say(
                    await meta_task + f"\n\n❌ Wallet error: {e}\n"
                                      "Re-import with /wallet → Import.")
                return
            if not kp:
                await say(await meta_task
                          + "\n\n❌ No wallet imported — /wallet to add one.")
                return
            slip = float(f.get("slippage") or 0)
            # priority fee: caller override > global default; <0 = disabled
            pf = float(f.get("priority_fee") or 0)
            priority_fee = (None if pf < 0 else
                            pf if pf > 0 else (config.PRIORITY_FEE_SOL or None))
            if via_jupiter:
                sig = await jupiter.buy(
                    kp, mint, buy_sol,
                    slippage_bps=int(slip * 100) if slip > 0 else 100,
                    wait_confirm=False)
                venue = "Jupiter"
            else:
                sig = await trader.buy(
                    kp, mint, buy_sol,
                    slippage=f"{slip:g}" if slip > 0 else "10",
                    priority_fee=priority_fee,
                    wait_confirm=False)
                venue = "pump.fun"
            await self.db.open_position(tg, f["caller_id"], cid, mint, sig,
                                        buy_sol, tokens=0, entry_price=0)
            await say(await meta_task
                      + f"\n\n⚡️ <b>SNIP {buy_sol} SOL</b> via {venue} "
                        f"(submitted, confirming…)\n tx: <code>{sig}</code>")
        except Exception as e:
            log.exception("buy failed tg=%s mint=%s", tg, mint)
            # Truncated on purpose: the SIGNAL message already carried this
            # CA, and an external exec bot scrapes CAs from messages — putting
            # the full mint here too would buy the token a second time.
            await say(f"❌ <b>Buy failed</b> for <code>{mint[:12]}…</code>: {e}")

    def _next_pause(self, e: callouts.RateLimited) -> float:
        """Seconds to wait after a 429, and the backoff for the next one.

        Retry-After is a floor, not a target. Returning at the exact instant the
        penalty lapsed re-armed a fresh 600s ban every time, so the backoff is
        added ON TOP and doubles per consecutive ban (capped) — a clean cycle
        resets it. See BACKOFF_BASE_S.
        """
        self._backoff_s = (BACKOFF_BASE_S if self._backoff_s <= 0
                           else min(self._backoff_s * 2, BACKOFF_MAX_S))
        # whichever is later: the server's window or our own local cooldown
        wait = max(e.retry_after_s, self.client.cooldown_remaining())
        return wait + self._backoff_s

    async def run(self, interval_s: float):
        # per-caller mode = 1 request per caller per cycle, each spaced by the
        # pacing floor → the interval scales with the caller count so the
        # request rate stays ~40/min per base (ceiling 60/min). Recomputed every
        # cycle. With several bases the cycle shortens by that factor instead:
        # each base keeps the same cadence, the bot just checks more often.
        log.info("poller started (mode=%s, interval=%.1fs/caller%.1fs, feed=%.1fs, "
                 "bases=%d)",
                 "feed" if config.FEED_MODE_ENABLED else "per-caller",
                 interval_s, config.POLL_S_PER_CALLER, config.FEED_POLL_INTERVAL_S,
                 len(config.PUMP_CALLOUT_BASES))
        while True:
            pause: Optional[float] = None
            try:
                await self.poll_once()
                self._backoff_s = 0.0  # clean cycle — the penalty is behind us
                if self._recover:
                    self._recover -= 1
                    log.info("rate-limit recovery: %d clean cycles left before "
                             "full-speed polling", self._recover)
            except callouts.RateLimited as e:
                pause = self._next_pause(e)
                self._recover = RECOVER_CYCLES
                log.warning("rate limited — pausing poller %.0fs "
                            "(Retry-After %.0fs + %.0fs backoff)",
                            pause, e.retry_after_s, self._backoff_s)
            except Exception:
                log.exception("poll cycle error")
            # keep the callout dedupe table from growing forever
            try:
                if time.time() - self._last_prune > 3600:
                    self._last_prune = time.time()
                    await self.db.prune_dedupe()
            except Exception:
                log.exception("dedupe prune failed")
            # keep the SOL/USD rate warm for the derived-mcap filter — fire and
            # forget so a price fetch can never delay a poll cycle
            try:
                if self.client.sol_usd_stale():
                    asyncio.create_task(self.client.refresh_sol_usd())
            except Exception:
                log.exception("sol/usd refresh kick failed")
            if pause is not None:
                # a 429 penalty is IP-wide, so no other caller is pollable either
                await asyncio.sleep(pause)
                continue
            n_callers = max(1, getattr(self, "_last_caller_count", 1))
            # Each base is a separate egress with its own allowance, and the
            # rotation spreads one cycle's requests across them — so the bot may
            # poll n_bases times as often while every base keeps the SAME
            # cadence. POLL_INTERVAL_S and POLL_S_PER_CALLER describe ONE base.
            # The count is the bases actually usable right now: a base serving a
            # 429 penalty must NOT have its share absorbed by the others, or one
            # ban would push them past the cadence that keeps them alive.
            n_bases = max(1, len(self.client.healthy_bases()))
            if config.FEED_MODE_ENABLED and self.client.bot_uuid:
                interval = config.FEED_POLL_INTERVAL_S / n_bases
            else:
                interval = max(interval_s,
                               config.POLL_S_PER_CALLER * n_callers) / n_bases
            if self._recover:
                # ease back in instead of immediately re-filling the bucket
                interval = max(interval, RECOVER_INTERVAL_S)
            # The egress's own answer to "how fast may I poll?", already divided
            # across the bases (a base used once every N cycles only needs a
            # cycle of L/N). Bursting the whole allowance and then sitting blind
            # is worse than pacing to it: at a steady interval the alerts stay
            # inside the 60s freshness window instead of arriving minutes late.
            # 0.0 until a 429 teaches us.
            learned = self.client.sustainable_interval_s()
            if learned > interval:
                if abs(learned - self._learned_logged) >= 1.0:
                    self._learned_logged = learned
                    log.info("pacing to %.0fs (rate limit learned for this IP)",
                             learned)
                interval = learned
            await asyncio.sleep(interval)
