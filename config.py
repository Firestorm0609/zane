import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")

# ---- subscription gate ----
# Owner can mint unlock codes (/gencode) and bypasses the paywall.
BOT_OWNER_ID = int(os.environ.get("BOT_OWNER_ID", "727210504"))
# Price shown in the in-chat Stars payment (Telegram Stars units).
SUB_PRICE_STARS = int(os.environ.get("SUB_PRICE_STARS", "250"))

# Base for EVERY pump.fun frontend-api call — /callout/list/{caller},
# /callout/top, /coins/{mint}, /auth/my-profile, /following-feed,
# /callout/leaderboard. Overridable so the polling egress can be moved to a
# different IP without touching code: pump.fun counts its ~13 requests/600s
# PER EGRESS IP, so a second egress is the only way to poll faster than this
# host can on its own. Whatever you point this at must proxy the same paths
# and pass the query string through unchanged, e.g. a Vercel rewrite
#   { "src": "/pump/(.*)", "dest": "https://frontend-api-v3.pump.fun/$1" }
# Empty/unset keeps the default (the `or` matters: `PUMP_CALLOUT_BASE=` in .env
# must not turn every URL into a relative path).
PUMP_CALLOUT_BASE = (os.environ.get("PUMP_CALLOUT_BASE")
                     or "https://frontend-api-v3.pump.fun").strip().rstrip("/")
# Extra egress bases to ROTATE pump.fun requests across (comma-separated, same
# shape as PUMP_CALLOUT_BASE: a host that proxies /pump/* to the real API).
# Why: the allowance is per SOURCE IP, so one egress is stuck at roughly one
# request per 6s, while N independent egresses let the bot check N times as
# often with every base still at that same cadence. Measured from this box: a
# VPS IP and a phone IP both sit at ~13 req/600s, whereas a serverless edge
# (Vercel, Cloudflare Workers, Deno Deploy) sustains 10 req/min — one allowance
# per deployment, as long as their egress ranges differ. Verify a new base with
# callout_mirror.js's /whoami route before trusting it as a separate pool.
PUMP_CALLOUT_MIRRORS = [u.strip().rstrip("/") for u in
                        os.environ.get("PUMP_CALLOUT_MIRRORS", "").split(",")
                        if u.strip()]
# Rotation order: the primary first, then the mirrors.
PUMP_CALLOUT_BASES = [PUMP_CALLOUT_BASE] + [m for m in PUMP_CALLOUT_MIRRORS
                                            if m != PUMP_CALLOUT_BASE]
# Optional egress for pump.fun requests ONLY (http:// or https:// proxy URL,
# with credentials inline if needed: http://user:pass@host:port).
# Why: pump.fun throttles this host's IP to a couple of requests a minute, which
# starves callout detection — Retry-After 600s, ~36s of working polls per ban.
# Pointing this at an IP pump.fun hasn't penalised restores 3s polling. Nothing
# else follows it: RPC, PumpPortal/Jupiter, Telegram and the SOL/USD feed stay
# direct. SOCKS5 would need the aiohttp_socks package.
PUMP_PROXY = os.environ.get("PUMP_PROXY", "").strip()
PUMP_AUTH_TOKEN = os.environ.get("PUMP_AUTH_TOKEN", "")  # static fallback (expires ~1h)

# ---- pump.fun feed-mode auth (Privy session token) ----
# pump.fun uses Privy sign-in; the identity token is forwarded from a
# logged-in browser tab by pump_forwarder.user.js (no refresh endpoint exists,
# so there is no fully headless renewal).
PUMP_AUTH_FILE = os.environ.get("PUMP_AUTH_FILE", ".pump-auth.json")
# HTTP endpoint the userscript POSTs the token to (localhost-only; reached
# from a phone over an `ssh -L` tunnel).
PUMP_INGEST_HOST = os.environ.get("PUMP_INGEST_HOST", "127.0.0.1")
PUMP_INGEST_PORT = int(os.environ.get("PUMP_INGEST_PORT", "8766"))
# Shared secret for the ingest endpoints (/ingest, /callouts, /relay-status).
# Blank = no check, which is how it ships: the port is bound to 127.0.0.1 and
# only reachable through the tunnel. Set it and the userscript must send the
# same value in the X-Ingest-Key header.
PUMP_INGEST_KEY = os.environ.get("PUMP_INGEST_KEY", "")
# Browser profile holding the pump.fun login. Kept (auth cookies only) so a
# future browser-based token capture can resume without a fresh login; nothing
# in the bot reads it — Playwright/Chromium were removed from this box.
PUMP_PROFILE_DIR = os.environ.get("PUMP_PROFILE_DIR", ".pump-profile")

SOLANA_RPC = os.environ.get("SOLANA_RPC", "https://api.mainnet-beta.solana.com")
# Free RPC redundancy: every signed tx is broadcast to ALL of these at once and
# the first success wins (the rest still get a copy, so whichever endpoint is
# fastest/includes it first wins the race). No paid RPC or API key needed.
# Defaults: your configured RPC + PublicNode (free, keyless, ~100ms).
SOLANA_RPC_BROADCAST = [u.strip() for u in os.environ.get(
    "SOLANA_RPC_BROADCAST",
    f"{SOLANA_RPC},https://solana-rpc.publicnode.com").split(",") if u.strip()]
# Priority fee per tx, in SOL, to jump the queue ahead of the crowd. This is the
# biggest free landing lever on Solana. 0 = let the venue decide.
PRIORITY_FEE_SOL = float(os.environ.get("PRIORITY_FEE_SOL", "0.0005"))
WALLET_ENC_KEY = os.environ.get("WALLET_ENC_KEY", "")

PUMP_TRADE_URL = "https://pumpportal.fun/api/trade-local"

# Hard cap per buy in SOL. 0 = no cap (wallet balance + MIN_SOL_LEFT only).
MAX_BUY_SOL = float(os.environ.get("MAX_BUY_SOL", "0"))
MIN_SOL_LEFT = float(os.environ.get("MIN_SOL_LEFT", "0.01"))

# ---- real-time exits (pump.fun NATS trade stream) ----
# The exit engine polls every 20s, which is far too slow for a memecoin dump:
# the whole peak can be given back inside one cycle. pump.fun pushes every
# trade to a per-mint NATS subject and ships public read-only subscriber
# credentials to every visitor, so stops can fire in well under a second.
RT_EXITS_ENABLED = os.environ.get("RT_EXITS_ENABLED", "1") not in ("0", "false", "False")
RT_NATS_URL = os.environ.get("RT_NATS_URL",
                             "wss://unified-prod.nats.realtime.pump.fun/")
# Public viewer password; auto re-scraped from the site if it ever rotates.
RT_NATS_PASS = os.environ.get("RT_NATS_PASS", "OX745xvUbNQMuFqV")
# How often the held-mint subscription set + rule cache are refreshed.
RT_SYNC_INTERVAL_S = float(os.environ.get("RT_SYNC_INTERVAL_S", "3"))
# Ignore dust trades below this size — a wash print can spike the price and
# trip a trailing stop that a real trade would never have reached.
RT_MIN_TRADE_SOL = float(os.environ.get("RT_MIN_TRADE_SOL", "0.01"))
# Minimum gap between realtime triggers for the same position (the trigger is
# still re-confirmed against a real quote before anything is sold).
RT_TRIGGER_COOLDOWN_S = float(os.environ.get("RT_TRIGGER_COOLDOWN_S", "3"))
# How long a cached on-chain token balance is trusted by the realtime path.
RT_TOKEN_TTL_S = float(os.environ.get("RT_TOKEN_TTL_S", "60"))

# ---- real-time callout push (the app's own channel) ----
# The HTTP endpoint publishes a callout ~10s after it was created (p90 ~29s) —
# that delay is upstream and unfixable by polling faster. pump.fun's own app is
# instant because it SUBSCRIBES on the CORE NATS cluster instead of polling:
# calloutCreated.<mint>.<chainId> carries every callout platform-wide the moment
# it is written, with the at-call mcap and run-up included. Measured created->us:
# median 0.05s, worst 0.30s, and no HTTP requests at all — so it cannot 429.
# Anonymous subscriber credentials are re-scraped from the site on every connect.
PUMP_PUSH_ENABLED = os.environ.get("PUMP_PUSH_ENABLED", "1") not in ("0", "false", "False")
# Blank = discover the CORE wss url + password from pump.fun's runtime config.
PUMP_PUSH_NATS_URL = os.environ.get("PUMP_PUSH_NATS_URL", "").strip()
PUMP_PUSH_NATS_PASS = os.environ.get("PUMP_PUSH_NATS_PASS", "").strip()
PUMP_PUSH_SUBJECT = os.environ.get("PUMP_PUSH_SUBJECT", "calloutCreated.>").strip()

# Detection mode. Feed mode (one /following-feed request covers every caller)
# looked ideal, but that endpoint does NOT serve the newest callouts: it returns
# a fixed 25-item sample (measured: newest item hours stale while 9 of a
# caller's last 10 callouts were absent entirely). Detection therefore runs per
# caller against /callout/list/{id}?sortBy=TIMESTAMP&sortOrder=desc, which is
# complete and chronological. The loop raises the interval automatically so the
# request rate stays ~40/min (ceiling is 60/min) at any caller count.
# Set FEED_MODE_ENABLED=1 only to experiment with the broken feed path.
FEED_MODE_ENABLED = os.environ.get("FEED_MODE_ENABLED", "0") not in ("0", "false", "False")
POLL_INTERVAL_S = float(os.environ.get("POLL_INTERVAL_S", "3"))
# Seconds of rate budget one caller may consume per cycle (1.5s/caller ≈ 40
# requests/min for two callers, which leaves headroom for buys and metadata).
POLL_S_PER_CALLER = float(os.environ.get("POLL_S_PER_CALLER", "1.5"))
# Feed mode interval — only used when FEED_MODE_ENABLED=1 (1 request/cycle).
FEED_POLL_INTERVAL_S = float(os.environ.get("FEED_POLL_INTERVAL_S", "1.2"))
DB_PATH = os.environ.get("DB_PATH", "bot.db")

# pump.fun program — only mints launched there are buyable via PumpPortal
PUMP_PROGRAM = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJbfMAYA1Qaume"


