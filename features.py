"""Feature pipeline: registry + all individual feature functions."""
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional, TYPE_CHECKING

from .state import is_creator_blacklisted
from .utils import safe_float, safe_int

if TYPE_CHECKING:
    from .keywords import KeywordModel
    from .market import MarketContext

log = logging.getLogger(__name__)


@dataclass
class CoinContext:
    coin: dict
    market_ctx: "MarketContext"
    keyword_model: "KeywordModel"

    mc:                float = field(init=False)
    replies:           int   = field(init=False)
    desc:              str   = field(init=False)
    name:              str   = field(init=False)
    symbol:            str   = field(init=False)
    ts:                int   = field(init=False)
    coin_age_sec:      float = field(init=False)
    mc_momentum_pct:   float = field(init=False)
    bonding_curve_pct: float = field(init=False)

    def __post_init__(self):
        from .utils import now_ts
        self.mc      = safe_float(self.coin.get("usd_market_cap"))
        self.replies = safe_int(self.coin.get("reply_count"))
        self.desc    = (self.coin.get("description") or "").strip()
        self.name    = (self.coin.get("name")        or "").strip()
        self.symbol  = (self.coin.get("symbol")      or "").strip()
        self.ts      = now_ts()

        created_raw = self.coin.get("created_timestamp")
        if created_raw:
            created_ts = safe_int(created_raw)
            if created_ts > 1_000_000_000_000:
                created_ts //= 1000
            self.coin_age_sec = max(0.0, float(self.ts - created_ts))
        else:
            self.coin_age_sec = 0.0

        self.mc_momentum_pct = safe_float(self.coin.get("_mc_momentum_pct", 0.0))
        self.bonding_curve_pct = (
            min(1.0, max(0.0, self.mc / 69_000.0)) if self.mc > 0 else 0.0
        )


FeatureFn = Callable[[CoinContext], float]


class FeatureRegistry:
    def __init__(self):
        self._features: list[tuple[str, FeatureFn]] = []
        self._index: dict[str, int] = {}

    def register(self, name: str) -> Callable:
        def decorator(fn: FeatureFn):
            if name in self._index:
                raise ValueError(f"Feature '{name}' already registered")
            self._index[name] = len(self._features)
            self._features.append((name, fn))
            return fn
        return decorator

    def extract(self, ctx: CoinContext) -> list[float]:
        out = []
        for name, fn in self._features:
            try:
                v = fn(ctx)
                fv = float(v) if v is not None else 0.0
                out.append(fv if math.isfinite(fv) else 0.0)
            except Exception as e:
                log.warning("Feature %s FAILED: %s", name, e)
                out.append(0.0)
        return out

    def index_of(self, name: str) -> Optional[int]:
        return self._index.get(name)

    @property
    def names(self) -> list[str]:
        return [n for n, _ in self._features]

    def __len__(self) -> int:
        return len(self._features)


FEATURES = FeatureRegistry()


# ---- Market cap ----
@FEATURES.register("mc_log")
def _(ctx): return math.log1p(max(ctx.mc, 0))

@FEATURES.register("mc_zscore")
def _(ctx): return ctx.market_ctx.zscore_mc(ctx.mc)

@FEATURES.register("mc_percentile")
def _(ctx): return ctx.market_ctx.percentile_mc(ctx.mc)


# ---- Replies ----
@FEATURES.register("reply_log")
def _(ctx): return math.log1p(max(ctx.replies, 0))

@FEATURES.register("reply_percentile")
def _(ctx): return ctx.market_ctx.percentile_replies(ctx.replies)

@FEATURES.register("replies_per_kmc")
def _(ctx): return ctx.replies / max(ctx.mc / 1000, 1.0)


# ---- Socials ----
@FEATURES.register("has_twitter")
def _(ctx): return 1.0 if ctx.coin.get("twitter") else 0.0

@FEATURES.register("has_telegram")
def _(ctx): return 1.0 if ctx.coin.get("telegram") else 0.0

@FEATURES.register("has_website")
def _(ctx): return 1.0 if ctx.coin.get("website") else 0.0

@FEATURES.register("social_count")
def _(ctx):
    return float(sum([
        bool(ctx.coin.get("twitter")),
        bool(ctx.coin.get("telegram")),
        bool(ctx.coin.get("website")),
    ]))

@FEATURES.register("social_completeness")
def _(ctx):
    return float(sum([
        bool(ctx.coin.get("twitter")),
        bool(ctx.coin.get("telegram")),
        bool(ctx.coin.get("website")),
    ])) / 3.0


# ---- Description quality ----
@FEATURES.register("desc_len_log")
def _(ctx): return math.log1p(len(ctx.desc))

@FEATURES.register("desc_word_count_log")
def _(ctx): return math.log1p(len(ctx.desc.split()))

@FEATURES.register("desc_has_url")
def _(ctx): return 1.0 if re.search(r"https?://", ctx.desc, re.I) else 0.0

@FEATURES.register("desc_exclamation_density")
def _(ctx):
    words = len(ctx.desc.split()) or 1
    return ctx.desc.count("!") / words

@FEATURES.register("desc_unique_word_ratio")
def _(ctx):
    words = ctx.desc.lower().split()
    if not words: return 0.0
    return len(set(words)) / len(words)

@FEATURES.register("desc_uppercase_ratio")
def _(ctx):
    letters = [c for c in ctx.desc if c.isalpha()]
    if not letters: return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)


# ---- Name / symbol ----
@FEATURES.register("name_len")
def _(ctx): return float(len(ctx.name))

@FEATURES.register("symbol_len")
def _(ctx): return float(len(ctx.symbol))

@FEATURES.register("name_has_numbers")
def _(ctx): return 1.0 if any(c.isdigit() for c in ctx.name) else 0.0

@FEATURES.register("name_is_allcaps")
def _(ctx): return 1.0 if len(ctx.name) > 1 and ctx.name.isupper() else 0.0

@FEATURES.register("name_symbol_length_ratio")
def _(ctx):
    if not ctx.symbol: return 0.0
    return min(len(ctx.name) / max(len(ctx.symbol), 1), 10.0) / 10.0


# ---- Time ----
@FEATURES.register("hour_sin")
def _(ctx):
    h = datetime.fromtimestamp(ctx.ts, tz=timezone.utc).hour
    return math.sin(2 * math.pi * h / 24)

@FEATURES.register("hour_cos")
def _(ctx):
    h = datetime.fromtimestamp(ctx.ts, tz=timezone.utc).hour
    return math.cos(2 * math.pi * h / 24)

@FEATURES.register("dow_sin")
def _(ctx):
    d = datetime.fromtimestamp(ctx.ts, tz=timezone.utc).weekday()
    return math.sin(2 * math.pi * d / 7)


# ---- Keywords ----
@FEATURES.register("keyword_pump_score")
def _(ctx): return ctx.keyword_model.score(ctx.desc)

@FEATURES.register("keyword_positive_hits")
def _(ctx): return float(ctx.keyword_model.positive_hits(ctx.desc))

@FEATURES.register("keyword_negative_hits")
def _(ctx): return float(ctx.keyword_model.negative_hits(ctx.desc))


# ---- Coin lifecycle ----
@FEATURES.register("coin_age_log")
def _(ctx): return math.log1p(ctx.coin_age_sec / 3600.0)

@FEATURES.register("coin_age_capped")
def _(ctx): return min(1.0, ctx.coin_age_sec / (48 * 3600))

@FEATURES.register("reply_velocity")
def _(ctx):
    hours = max(ctx.coin_age_sec / 3600.0, 0.1)
    rph = ctx.replies / hours
    return min(rph, 100.0) / 100.0

@FEATURES.register("mc_momentum_pct")
def _(ctx):
    if ctx.mc_momentum_pct == 0.0 and ctx.coin_age_sec < 1800:
        return 0.5
    return max(0.0, min(1.0, (ctx.mc_momentum_pct + 100.0) / 300.0))

@FEATURES.register("bonding_curve_progress")
def _(ctx): return ctx.bonding_curve_pct


# ---- On-chain safety (RPC) ----
@FEATURES.register("mint_auth_safety")
def _(ctx):
    v = ctx.coin.get("_rpc_mint_auth_revoked")
    if v is None: return 0.5
    return 1.0 if v else 0.0

@FEATURES.register("freeze_auth_safety")
def _(ctx):
    v = ctx.coin.get("_rpc_freeze_auth_revoked")
    if v is None: return 0.5
    return 1.0 if v else 0.0

@FEATURES.register("holder_distribution")
def _(ctx):
    raw = ctx.coin.get("_rpc_top5_concentration")
    if raw is None: return 0.5
    return 1.0 - safe_float(raw, 0.5)

@FEATURES.register("bundle_clean")
def _(ctx):
    raw = ctx.coin.get("_rpc_bundle_score")
    if raw is None: return 0.5
    return 1.0 - safe_float(raw, 0.0)

@FEATURES.register("creator_freshness")
def _(ctx):
    count = ctx.coin.get("_rpc_creator_tx_count")
    if count is None: return 0.5
    count = safe_float(count, 0.0)
    return max(0.0, 1.0 - count / 200.0)

@FEATURES.register("creator_blacklisted")
def _(ctx):
    creator = (ctx.coin.get("creator") or ctx.coin.get("user")
               or ctx.coin.get("traderPublicKey") or "")
    return 0.0 if is_creator_blacklisted(creator) else 1.0


log.info("FeatureRegistry: %d features registered", len(FEATURES))
