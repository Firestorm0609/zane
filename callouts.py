"""Client for pump.fun callout endpoints + mint validation."""
import asyncio
import html
import logging
import time
from collections import deque
from typing import Any, Optional

import aiohttp

import config

log = logging.getLogger("callouts")

# frontend-api-v3 allows ~60 req/min per IP (x-ratelimit-limit: 60), but
# Cloudflare 1015 penalties are aggressive (Retry-After up to 600s) — stay well under.
# 1.2s => ~50 req/min, leaving headroom while keeping feed-mode detection fast.
MIN_REQUEST_INTERVAL_S = 1.2

# Headroom applied to the request rate learned from a 429 (see BaseState.learn).
# 1.3 = aim for ~30% under the served limit so we stop at the edge of the
# allowance instead of tripping it every window.
RATE_SAFETY = 1.3

# Deriving a rate needs a rate-sized sample. A 429 that arrives when barely any
# requests were spent in the window says nothing about the allowance: the
# window was already mostly spent when we got there, or (behind a pooled or
# CDN-fronted egress) one upstream IP was hot while the rest kept serving.
# Dividing the window by that tiny count invents an interval ~30x too slow, and
# the poller then sits near-blind for hours. Below this floor the sample is
# discarded and the previous value stands.
RATE_LEARN_MIN_SAMPLE = 5

# A learned limit relaxes only on evidence: after each full clean window of
# successful requests the interval steps toward the configured one. One window
# (not three) keeps recovery inside a couple of hours instead of most of a day;
# the step being proportional is what keeps a genuinely throttled egress from
# being walked straight into a wall — it slows down and re-learns if it was
# right, rather than never speeding back up if it wasn't.
RATE_DECAY_WINDOWS = 1
RATE_DECAY_FACTOR = 0.85

# How much request history the rolling rate meter keeps. Longer than any
# Retry-After we accept (clamped to 900s), so the count inside a window is
# never missing requests the server could still be counting.
RATE_WINDOW_MAX_S = 1800.0

# A 429 on this egress is usually ONE pooled IP refusing while the rest keep
# serving. A 120-request probe at 3s ran at 92% success with the longest run of
# failures being 2, while the same traffic independently taught the poller "85
# req allowed per 600s" — so the wall is soft and per-IP, not a budget being
# spent. Believing every 429 turned one hot IP into a 600s blackout, which is
# the single slowest thing the bot does (worse than any interval: a blackout
# delays detection by ten minutes, not four seconds). So a 429 is retried
# before it is believed. If a retry is served the outlier teaches nothing (same
# reasoning as RATE_LEARN_MIN_SAMPLE); only if the retries ALSO fail do we cool
# down and learn.
RATE_RETRY_ATTEMPTS = 2
RATE_RETRY_DELAY_S = 1.5


class BaseState:
    """Rate state for ONE egress base — a pump.fun frontend we poll through.

    The server's allowance is per source IP, so everything that answers "how
    fast may I poll" lives here rather than on the client: the spacing floor,
    the 429 cooldown, and the ceiling learned from a 429. One shared state is
    what made a single base's ban stop every other base too.
    """

    __slots__ = ("url", "used", "last_req_ts", "cooldown_until", "req_ts",
                 "sustainable_interval_s", "last_ban_ts", "last_window_s")

    def __init__(self, url: str):
        self.url = url
        self.used = False                 # logged once, to prove rotation
        self.last_req_ts: float = 0.0
        self.cooldown_until: float = 0.0  # set when THIS base answers 429
        # Rolling meter of dispatched requests. The learner divides the
        # server's window by the requests spent INSIDE it, so the count must be
        # trailing: counting since the last ban spanned a two-day clean run and
        # derived a 0s interval, i.e. taught the poller nothing at all.
        self.req_ts: deque[float] = deque()
        self.sustainable_interval_s: float = 0.0
        self.last_ban_ts: float = 0.0
        self.last_window_s: float = 0.0

    def cooldown_remaining(self) -> float:
        return max(0.0, self.cooldown_until - time.monotonic())

    def note_request(self) -> None:
        """Record a request handed to this base (rolling-window rate meter)."""
        now = time.monotonic()
        self.req_ts.append(now)
        cutoff = now - RATE_WINDOW_MAX_S
        while self.req_ts and self.req_ts[0] < cutoff:
            self.req_ts.popleft()

    def spend_in_window(self, window_s: float) -> int:
        """Requests dispatched to this base in the trailing `window_s` seconds."""
        cutoff = time.monotonic() - window_s
        while self.req_ts and self.req_ts[0] < cutoff:
            self.req_ts.popleft()
        return len(self.req_ts)

    def learn(self, window_s: float) -> None:
        """Derive this base's sustainable request interval from a 429.

        The server hands us the window; the rolling meter says how many
        requests we spent inside it. Bursting the allowance is what leaves the
        bot blind for the next ten minutes, so the poller spreads them instead.

        A window we barely entered is not evidence, so a sample smaller than
        RATE_LEARN_MIN_SAMPLE is thrown away and the previous limit kept.
        """
        spent = self.spend_in_window(window_s)
        if spent < RATE_LEARN_MIN_SAMPLE:
            log.warning("%s: 429 after only %d request(s) in a %.0fs window — "
                        "too small a sample to be a rate; keeping learned "
                        "interval %.0fs", self.url, spent, window_s,
                        self.sustainable_interval_s)
            return
        self.sustainable_interval_s = (window_s / spent) * RATE_SAFETY
        self.last_window_s = window_s
        self.last_ban_ts = time.monotonic()
        log.warning("%s: learned rate limit — %d req spent in a trailing "
                    "%.0fs window, sustainable interval %.0fs ±%.0f%%",
                    self.url, spent, window_s, self.sustainable_interval_s,
                    (RATE_SAFETY - 1) * 100)

    def note_request_ok(self) -> None:
        """Forget a learned limit as a clean streak lengthens (per base).

        Without this a single bad patch would cap a base forever; the speedup
        only has to be slow, not non-existent, so a genuinely clean base works
        its way back to the configured interval on its own.
        """
        if not self.sustainable_interval_s or not self.last_ban_ts:
            return
        if (time.monotonic() - self.last_ban_ts
                > self.last_window_s * RATE_DECAY_WINDOWS):
            before = self.sustainable_interval_s
            self.sustainable_interval_s *= RATE_DECAY_FACTOR
            self.last_ban_ts = time.monotonic()  # next step another window out
            log.info("%s: learned rate limit easing: %.0fs -> %.0fs (clean "
                     "window)", self.url, before, self.sustainable_interval_s)


class RateLimited(Exception):
    """Raised when the API is in a 429 cooldown window."""
    def __init__(self, retry_after_s: float):
        self.retry_after_s = retry_after_s
        super().__init__(f"rate limited, retry after {retry_after_s:.0f}s")

BASE_HEADERS = {
    "Origin": "https://pump.fun",
    "Accept": "application/json",
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
    "Referer": "https://pump.fun/",
}


def _is_socks(proxy: str) -> bool:
    """SOCKS proxies can't go through aiohttp's per-request ``proxy=``.

    They need a ProxyConnector on the session itself, which is why the pump.fun
    session is built differently for them (see CalloutClient._http).
    """
    return proxy.lower().startswith(("socks4", "socks5"))


def _proxy_for(url: str) -> Optional[str]:
    """PUMP_PROXY applies ONLY to pump.fun requests.

    The trade path (PumpPortal/Jupiter), RPC, Telegram and the SOL/USD feed
    must not be dragged through it: an egress that fixes one rate limit can
    easily break an unrelated call, and the derived-mcap filter depends on
    that price feed.
    """
    if not config.PUMP_PROXY:
        return None
    return config.PUMP_PROXY if any(b in url for b in config.PUMP_CALLOUT_BASES) else None


def is_solana_mint(mint: str) -> bool:
    """Reject EVM (0x...) and other non-Solana addresses."""
    if not (32 <= len(mint) <= 44):
        return False
    alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    return all(c in alphabet for c in mint)


def is_evm_address(s: str) -> bool:
    """EVM-style caller wallet: 0x + 40 hex chars."""
    if not (s.startswith("0x") and len(s) == 42):
        return False
    try:
        int(s[2:], 16)
        return True
    except ValueError:
        return False


def is_valid_caller_id(s: str) -> bool:
    """Solana wallet, EVM wallet, or pump.fun user UUID."""
    if is_evm_address(s):
        return True
    if is_solana_mint(s):  # same base58 format as wallet addresses
        return True
    # UUID check
    if len(s) == 36 and s.count("-") == 4:
        try:
            int(s.replace("-", ""), 16)
            return True
        except ValueError:
            return False
    return False


def is_pump_fun_mint(mint: str) -> bool:
    """pump.fun launches use a vanity mint ending in 'pump'.

    Tokens called on other venues (Raydium, Meteora, other launchpads)
    don't have it and are not buyable via PumpPortal's pump pool.
    """
    return is_solana_mint(mint) and mint.endswith("pump")


class CalloutClient:
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._meta_cache: dict[str, dict[str, str]] = {}
        self._req_lock = asyncio.Lock()  # serialises dispatch across all bases
        # One rate state per egress base (see BaseState): the allowance is per
        # source IP, so a 429 on one base must only take THAT base out.
        self._bases = [BaseState(u) for u in config.PUMP_CALLOUT_BASES]
        self._rr = 0                     # round-robin cursor over the bases
        self.bot_uuid: Optional[str] = None  # set by init_feed() when JWT is present
        self._uuid_cache: dict[str, str] = {}  # caller_id -> user uuid
        # live auth (PumpAuth instance or a static token string)
        self._pump_auth = None            # set via attach_auth()
        self._static_token: str = config.PUMP_AUTH_TOKEN or ""
        self._current_token: str = self._static_token
        # SOL/USD for deriving mcap from a callout price (feed mode has no
        # marketCap). Refreshed in the background — never awaited on a buy path.
        self._sol_usd: float = 0.0
        self._sol_usd_ts: float = 0.0
        self._sol_usd_task: Optional[asyncio.Task] = None

    def attach_auth(self, pump_auth) -> None:
        """Give the client a PumpAuth so requests carry a fresh Bearer token
        and 401s trigger a forced refresh + retry."""
        self._pump_auth = pump_auth

    def set_static_token(self, token: str) -> None:
        """Use a raw Bearer token (e.g. one read from .pump-auth.json).

        Authenticated endpoints like /callout/leaderboard return 401 without
        this — anonymous-capable ones don't care.
        """
        self._static_token = token or ""

    async def _refresh_session_auth(self) -> bool:
        """Sync the session's Authorization header with the current token.

        Cheap (no HTTP in the Privy design — get_token is an in-memory check).
        Returns True if the session now carries a usable token. A rejected
        token is REMOVED from the headers so anonymous-capable endpoints
        (callout/list) keep working while we wait for a fresh one.
        """
        token = ""
        if self._pump_auth is not None:
            token = (await self._pump_auth.get_token()) or ""
        elif self._static_token:
            token = self._static_token
        if not token:
            if self._current_token and self._session and not self._session.closed:
                self._session.headers.pop("Authorization", None)
                self._current_token = ""
            return False
        if token != self._current_token and self._session and not self._session.closed:
            self._session.headers.update({"Authorization": f"Bearer {token}"})
            self._current_token = token
        return True

    async def _http(self) -> aiohttp.ClientSession:
        """The pump.fun session (this is the only thing PUMP_PROXY applies to)."""
        if self._session is None or self._session.closed:
            connector = None
            proxy = config.PUMP_PROXY
            if proxy and _is_socks(proxy):
                try:
                    from aiohttp_socks import ProxyConnector
                    connector = ProxyConnector.from_url(proxy)
                except ImportError:
                    # never take the bot down over an optional egress: fall back
                    # to direct polling rather than refusing to start
                    log.error("PUMP_PROXY=%s is a SOCKS proxy but aiohttp_socks "
                              "is not installed — polling DIRECT from this IP "
                              "(pip install aiohttp_socks)", proxy)
            elif proxy:
                log.info("pump.fun requests egress via %s",
                         proxy.rsplit("@", 1)[-1])
            self._session = aiohttp.ClientSession(
                headers=dict(BASE_HEADERS), timeout=aiohttp.ClientTimeout(total=10),
                connector=connector)
            if self._current_token:
                # re-attach the last known token after a session rebuild
                self._session.headers.update(
                    {"Authorization": f"Bearer {self._current_token}"})
        return self._session

    async def _paced_get(self, url: str, params: Optional[dict] = None,
                         _retried_auth: bool = False) -> tuple[int, Any]:
        """Rate-limited GET across every base.

        A 429 is a penalty on ONE egress, so the request is served by the next
        healthy base instead of the whole bot going blind. RateLimited escapes
        only once every base is cooling down.
        """
        left = len(self._bases)
        while True:
            try:
                return await self._get_once(url, params, _retried_auth)
            except RateLimited:
                if left <= 1 or not self.healthy_bases():
                    raise
                left -= 1
                log.warning("a base is cooling down — serving this request "
                            "from the next one")

    async def _get_once(self, url: str, params: Optional[dict] = None,
                        _retried_auth: bool = False) -> tuple[int, Any]:
        """One HTTP GET, on the base that is next in rotation.

        On 401: forces a token refresh and retries ONCE (covers the hourly
        Firebase expiry and revoked-token cases).
        """
        if not self.healthy_bases():
            raise RateLimited(self.cooldown_remaining())
        http = await self._http()
        await self._refresh_session_auth()
        async with self._req_lock:
            base, url = self._pick_base(url)
            if base.cooldown_until > time.monotonic():
                raise RateLimited(base.cooldown_remaining())
            # spacing between requests TO THIS BASE (the allowance is per IP)
            elapsed = time.monotonic() - base.last_req_ts
            if elapsed < MIN_REQUEST_INTERVAL_S:
                await asyncio.sleep(MIN_REQUEST_INTERVAL_S - elapsed)
            proxy = _proxy_for(url)
            base.note_request()  # spends part of THIS base's allowance
            async with http.get(url, params=params, proxy=proxy) as resp:
                base.last_req_ts = time.monotonic()
                if resp.status == 429:
                    # Don't believe a single 429 — see RATE_RETRY_ATTEMPTS. The
                    # sleep happens while _req_lock is held, deliberately: while
                    # we are waiting to find out whether the egress is hot, no
                    # other request should spend budget.
                    retry_after = self._retry_after_of(resp)
                    for attempt in range(1, RATE_RETRY_ATTEMPTS + 1):
                        log.info("429 from %s — retrying in %.1fs (%d/%d) before "
                                 "believing it", base.url, RATE_RETRY_DELAY_S,
                                 attempt, RATE_RETRY_ATTEMPTS)
                        await asyncio.sleep(RATE_RETRY_DELAY_S)
                        base.note_request()  # a retry is a request too
                        async with http.get(url, params=params,
                                            proxy=proxy) as retry:
                            base.last_req_ts = time.monotonic()
                            if retry.status == 200:
                                # outlier confirmed: it teaches nothing and the
                                # bot keeps its speed
                                base.note_request_ok()
                                try:
                                    return 200, await retry.json()
                                except Exception:
                                    return 200, None
                            if retry.status == 429:
                                retry_after = self._retry_after_of(retry)
                    log.warning("429 from %s — cooling down %.0fs (retries did "
                                "not clear it)", base.url, retry_after)
                    base.cooldown_until = time.monotonic() + retry_after
                    base.learn(retry_after)
                    raise RateLimited(retry_after)
                if resp.status == 401 and not _retried_auth:
                    log.info("401 from %s — forcing pump.fun token refresh + retry",
                             base.url)
                    if self._pump_auth is not None:
                        await self._pump_auth.notify_401()
                    if await self._refresh_session_auth():
                        # note: still holding _req_lock — _get_once recursion
                        # would deadlock, so do a single inline retry instead
                        base.note_request()  # the retry is a request too
                        async with http.get(url, params=params,
                                            proxy=proxy) as retry_resp:
                            if retry_resp.status == 200:
                                try:
                                    return 200, await retry_resp.json()
                                except Exception:
                                    return 200, None
                            return retry_resp.status, None
                if resp.status == 200:
                    base.note_request_ok()
                    try:
                        return 200, await resp.json()
                    except Exception:
                        return 200, None
                return resp.status, None

    @staticmethod
    def _retry_after_of(resp, default: float = 60.0) -> float:
        """Retry-After from a 429, clamped to the server's real ceiling."""
        try:
            value = float(resp.headers.get("Retry-After") or default)
        except (TypeError, ValueError):
            value = default
        return min(value, 900.0)

    def healthy_bases(self) -> list[BaseState]:
        """Bases that are not currently serving out a 429 penalty."""
        now = time.monotonic()
        return [b for b in self._bases if b.cooldown_until <= now]

    def _pick_base(self, url: str) -> tuple[BaseState, str]:
        """Next base in rotation, and this URL rewritten onto it.

        Callers build URLs from config.PUMP_CALLOUT_BASE; the rotation swaps in
        whichever base is due, so nothing else in the codebase has to know that
        more than one exists.
        """
        bases = self.healthy_bases() or self._bases
        base = bases[self._rr % len(bases)]
        self._rr = (self._rr + 1) % max(1, len(bases))
        if not base.used:
            base.used = True
            log.info("polling via base %s", base.url)
        path = url
        for known in config.PUMP_CALLOUT_BASES:
            if url.startswith(known):
                path = url[len(known):]
                break
        return base, base.url + path

    def sustainable_interval_s(self) -> float:
        """Cycle interval that keeps every base inside its learned ceiling.

        A base is hit once every N cycles (round-robin), so a base that learned
        `L` seconds between requests only needs a cycle of `L / N` — the other
        bases cover what would otherwise be idle waiting. 0.0 = nothing learned
        yet, so the configured interval applies.
        """
        bases = self.healthy_bases() or self._bases
        ceilings = [b.sustainable_interval_s for b in bases
                    if b.sustainable_interval_s > 0]
        if not ceilings:
            return 0.0
        return max(ceilings) / len(bases)

    def cooldown_remaining(self) -> float:
        """Seconds until the FIRST base is usable again (0 when one already is).

        Only reached when every base is cooling down: retrying as soon as one
        of them clears is what keeps the blind window short, and each base's
        own penalty is re-checked on the request that follows. Waiting for the
        LAST one instead was how a single 151s ban cost 10 minutes of polling.
        """
        if not self._bases:
            return 0.0
        return min(b.cooldown_remaining() for b in self._bases)

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def init_feed(self) -> str:
        """Resolve the bot account's uuid from the JWT. Raises if auth fails."""
        # make sure we're sending the freshest token we have
        await self._refresh_session_auth()
        profile = await self.my_profile()
        uuid = profile.get("userId") or profile.get("uuid")
        if not uuid:
            raise RuntimeError(f"no userId in profile: {str(profile)[:200]}")
        self.bot_uuid = uuid
        return uuid

    async def resolve_uuid(self, caller_id: str) -> str:
        """Map a stored caller id (wallet or uuid) to the pump.fun user uuid."""
        if caller_id in self._uuid_cache:
            return self._uuid_cache[caller_id]
        info = await self.user_info(caller_id)
        uuid = info.get("userId") or caller_id
        self._uuid_cache[caller_id] = uuid
        return uuid

    async def coin_meta(self, mint: str) -> dict[str, str]:
        """Fetch {name, symbol}: pump.fun first, Dexscreener fallback (off-platform tokens)."""
        if mint in self._meta_cache:
            return self._meta_cache[mint]
        url = f"{config.PUMP_CALLOUT_BASE}/coins/{mint}"
        status, data = await self._paced_get(url)
        if status == 200 and data:
            meta = {"name": data.get("name") or "?",
                    "symbol": data.get("symbol") or "?"}
            self._meta_cache[mint] = meta
            return meta
        # not on pump.fun — try Dexscreener (separate host, generous limits)
        try:
            http = await self._http()
            # dexscreener ignores our pump.fun headers; use a bare session
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=8)) as s:
                async with s.get(f"https://api.dexscreener.com/latest/dex/tokens/{mint}") as resp:
                    if resp.status == 200:
                        d = await resp.json()
                        pairs = d.get("pairs") or []
                        if pairs:
                            tok = pairs[0].get("baseToken", {})
                            meta = {"name": tok.get("name") or "?",
                                    "symbol": tok.get("symbol") or "?"}
                            self._meta_cache[mint] = meta
                            return meta
        except Exception:
            pass
        return {"name": "?", "symbol": "?"}

    async def list_callouts(self, caller_id: str, limit: int = 10,
                            page_token: str = "") -> dict[str, Any]:
        """GET /callout/list/{callerId} — newest callouts for one caller."""
        params: dict[str, str] = {
            "limit": str(limit),
            "sortBy": "TIMESTAMP",
            "sortOrder": "desc",
        }
        if page_token:
            params["pageToken"] = page_token
        url = f"{config.PUMP_CALLOUT_BASE}/callout/list/{caller_id}"
        status, data = await self._paced_get(url, params)
        if status == 200 and data is not None:
            return data
        raise RuntimeError(f"callout/list {status} (rate-limited or unavailable)")

    async def top_callouts(self, caller_id: str, limit: int = 10) -> dict[str, Any]:
        url = f"{config.PUMP_CALLOUT_BASE}/callout/top/{caller_id}"
        status, data = await self._paced_get(url, {"limit": str(limit)})
        if status == 200 and data is not None:
            return data
        raise RuntimeError(f"callout/top {status} (rate-limited or unavailable)")

    async def my_profile(self) -> dict[str, Any]:
        """GET /auth/my-profile — requires JWT. Returns bot account profile."""
        status, data = await self._paced_get(f"{config.PUMP_CALLOUT_BASE}/auth/my-profile")
        if status == 200 and data:
            return data
        raise RuntimeError(f"my-profile {status} — JWT missing or expired?")

    async def user_info(self, user_id: str) -> dict[str, Any]:
        """GET /users/{id} — accepts wallet or UUID; returns {address, userId, ...}."""
        status, data = await self._paced_get(f"{config.PUMP_CALLOUT_BASE}/users/{user_id}")
        if status == 200 and data:
            return data
        raise RuntimeError(f"users/{user_id[:10]}… {status}")

    def sol_usd_last(self) -> float:
        """Last known SOL/USD (0.0 if never fetched). Never blocks."""
        return self._sol_usd

    async def refresh_sol_usd(self) -> float:
        """Refresh the cached SOL/USD rate. Called off the buy path only."""
        if self._sol_usd_task and not self._sol_usd_task.done():
            return self._sol_usd
        self._sol_usd_task = asyncio.ensure_future(self._fetch_sol_usd())
        return self._sol_usd

    async def _fetch_sol_usd(self) -> float:
        try:
            url = ("https://api.coingecko.com/api/v3/simple/price"
                   "?ids=solana&vs_currencies=usd")
            # bare session: CoinGecko must not receive pump.fun's Origin/Referer
            # or bearer token, and it must not inherit PUMP_PROXY — only pump.fun
            # requests pay for that egress
            timeout = aiohttp.ClientTimeout(total=8)
            async with aiohttp.ClientSession(timeout=timeout) as s:
                async with s.get(url) as r:
                    data = await r.json()
            px = float((data.get("solana") or {}).get("usd") or 0)
            if px > 0:
                self._sol_usd, self._sol_usd_ts = px, time.time()
        except Exception as e:
            log.debug("sol/usd refresh failed: %s", e)
        return self._sol_usd

    def sol_usd_stale(self) -> bool:
        return time.time() - self._sol_usd_ts > SOL_USD_TTL_S

    async def caller_profile(self, caller_id: str) -> dict[str, Any]:
        """Public pump.fun profile of a caller, so we can name them for you.

        Returns {} for a plain wallet with no pump.fun account, and never
        raises — a missing name must not block adding a caller.
        """
        try:
            info = await self.user_info(caller_id)
        except Exception as e:
            log.debug("profile lookup failed for %s: %s", caller_id[:10], e)
            return {}
        return {
            "uuid": info.get("userId") or "",
            "username": (info.get("username") or "").strip(),
            "bio": (info.get("bio") or "").strip(),
            "x_username": (info.get("x_username") or "").strip(),
            "followers": int(info.get("followers") or 0),
            "avatar": info.get("profile_image") or "",
            "is_pump_user": bool(info.get("is_pump_user")),
        }

    async def follow(self, user_uuid: str) -> bool:
        """POST /following/v2/{uuid} — follow a user from the bot account (JWT).

        v2 is the current route; the old POST /following/{uuid} now 404s, so
        the bot account silently followed nobody and the feed stayed empty.
        """
        http = await self._http()
        await self._refresh_session_auth()
        url = f"{config.PUMP_CALLOUT_BASE}/following/v2/{user_uuid}"
        async with http.post(url, json={}, proxy=_proxy_for(url)) as resp:
            return resp.status in (200, 201, 204)

    async def unfollow(self, user_uuid: str) -> bool:
        http = await self._http()
        url = f"{config.PUMP_CALLOUT_BASE}/following/{user_uuid}"
        async with http.delete(url, proxy=_proxy_for(url)) as resp:
            return resp.status in (200, 201, 204)

    async def feed(self, limit: int = 50) -> dict[str, Any]:
        """GET /following-feed — callouts from everyone the bot account follows.

        ONE request covers all tracked callers (the old /callout/feed/{uuid}
        route is gone — it 404s now). Returns {'callouts': [...normalized...]},
        the same shape as list_callouts(), plus the raw trades the feed mixes in.
        """
        if not self.bot_uuid:
            raise RuntimeError("feed mode not initialized")
        status, data = await self._paced_get(
            f"{config.PUMP_CALLOUT_BASE}/following-feed", {"limit": str(limit)})
        if status != 200 or data is None:
            raise RuntimeError(f"following-feed {status}")
        items = data.get("items") or []
        # keep callout events only — the feed also carries trade events
        callouts = [normalize_feed_item(it) for it in items
                    if isinstance(it, dict)
                    and it.get("coinMint")
                    and it.get("eventType") in (None, "callout")]
        return {"callouts": callouts, "trades": data.get("trades") or []}

    async def caller_stats(self, caller_id: str) -> dict[str, Any]:
        """Aggregate stats across the caller's recent callouts."""
        data = await self.list_callouts(caller_id, limit=50)
        callouts = data.get("callouts", [])
        if not callouts:
            return {"count": 0}
        multis = [c.get("maxMultiplier") or 0 for c in callouts]
        wins = [m for m in multis if m >= 2.0]
        return {
            "count": len(callouts),
            "oldest_ts": min(c.get("createdAt", 0) for c in callouts),
            "newest_ts": max(c.get("createdAt", 0) for c in callouts),
            "win_rate_2x": round(len(wins) / len(multis), 3),
            "best_multiple": round(max(multis), 2),
            "avg_multiple": round(sum(multis) / len(multis), 2),
        }


# Every pump.fun token is minted with a fixed 1B supply, so a callout's market
# cap can be derived from its call price: mcap_SOL = price_SOL × 1e9. Verified
# against pump.fun's own `marketCap` on real callouts: +0.15% (the residual is
# just the SOL/USD rate moving between the call and now).
PUMP_TOKEN_SUPPLY = 1e9
SOL_USD_TTL_S = 60.0


def derive_mcap_usd(callout: dict[str, Any], sol_usd: float) -> float:
    """Market cap in USD from a callout's call price. 0 if it can't be derived.

    /following-feed items (feed mode) carry NO marketCap, so without this the
    user's mcap filter silently never fired. ``calloutPrice`` is the name in
    /callout/list, ``calloutPriceSol`` in the feed — accept either.
    """
    if sol_usd <= 0:
        return 0.0
    price = float(callout.get("calloutPriceSol") or callout.get("calloutPrice") or 0)
    if price <= 0:
        return 0.0
    return price * PUMP_TOKEN_SUPPLY * sol_usd


def normalize_feed_item(it: dict[str, Any]) -> dict[str, Any]:
    """Map a /following-feed item (callout event) onto the /callout/list
    callout shape the poller + formatter expect.

    Real item shape (2026-09):
      {id, userId (=caller WALLET), eventType:"callout", coinMint, timestamp,
       callout:{calloutPriceSol, multiple, maxPriceSol, thesis}, username}
    """
    c = it.get("callout") if isinstance(it.get("callout"), dict) else {}
    return {
        "calloutId": it.get("id") or "",
        "userId": it.get("userId") or "",           # wallet address
        "coinMint": it.get("coinMint") or "",
        "createdAt": it.get("timestamp") or 0,
        "marketCap": (it.get("marketCap") or it.get("coinMarketCap") or 0),
        # how far the coin has already run since the call — the poller's
        # "already pumped" filter compares this against the caller's limit
        "multiple": c.get("multiple") or 0,
        # the feed omits marketCap, so keep the call price to derive it
        "calloutPriceSol": c.get("calloutPriceSol") or 0,
        "maxMultiplier": c.get("multiple") or 0,
        "thesis": c.get("thesis") or "",
        "likes": it.get("likes") or 0,
        "viewCount": it.get("viewCount") or 0,
    }


def format_callout(c: dict[str, Any], meta: Optional[dict[str, str]] = None) -> str:
    """Human-readable summary of one callout."""
    mint = c.get("coinMint", "?")
    meta = meta or {"name": "?", "symbol": "?"}
    name = html.escape(meta.get("name") or "?")
    symbol = html.escape(meta.get("symbol") or "?")
    mc = c.get("marketCap") or 0
    mult = c.get("maxMultiplier") or 0
    thesis = html.escape((c.get("thesis") or "").strip())
    if len(thesis) > 200:
        thesis = thesis[:200] + "…"
    created = c.get("createdAt", 0) / 1000
    age_s = max(0, int(time.time() - created))
    age = f"{age_s}s ago" if age_s < 3600 else f"{age_s // 60}m ago"
    # Engagement is cosmetic and the realtime push payload doesn't carry it, so
    # omit the counters when absent instead of printing a fake "❤️ 0 · 👁 0".
    likes, views = c.get("likes"), c.get("viewCount")
    engagement = (f" · ❤️ {likes or 0} · 👁 {views or 0}"
                  if likes is not None or views is not None else "")
    evm = "" if is_solana_mint(mint) else " ⚠️ EVM (not tradable)"
    return (
        f"📣 <b>${symbol}</b> — {name}{evm}\n"
        f"• MC at call: <code>${mc:,.0f}</code>\n"
        f"• Best since: <b>{mult:.2f}x</b>\n"
        f"• {age}{engagement}\n"        + (f"• <i>{thesis}</i>\n" if thesis else "")
        # Truncated on purpose. The dedicated SIGNAL message is the ONE place
        # a full, scrapable CA is emitted; an external exec bot (mirrorbot ->
        # basedbot) reads CAs out of messages, so a full mint here would fire a
        # SECOND buy on the same token. Too short to match a base58 CA pattern.
        + f"• <code>{mint[:12]}…</code>"
    )


def dedupe_key(c: dict[str, Any]) -> str:
    return c.get("calloutId") or f"{c.get('userId')}:{c.get('coinMint')}:{c.get('createdAt')}"
