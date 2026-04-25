"""Shared helpers: formatting, escaping, time, type-safe parsing."""
import math
import re
import time
from typing import Optional


def now_ts() -> int:
    return int(time.time())


def safe_float(v, default: float = 0.0) -> float:
    try:
        f = float(v if v is not None else 0)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def safe_int(v, default: int = 0) -> int:
    try:
        return int(v if v is not None else 0)
    except (TypeError, ValueError):
        return default


_MD2_ESCAPE_RE = re.compile(r"([_*\[\]()~`>#+\-=|{}.!\\])")


def esc(text) -> str:
    if text is None:
        return ""
    return _MD2_ESCAPE_RE.sub(r"\\\1", str(text))


def mdbold(text) -> str:
    s = esc(text)
    return f"*{s}*" if s else ""


def mditalic(text) -> str:
    s = esc(text)
    return f"_{s}_" if s else ""


def mdcode(text) -> str:
    if text is None or text == "":
        return "``"
    s = str(text).replace("\\", "\\\\").replace("`", "\\`")
    return f"`{s}`"


def fmt_pct(val, places: int = 1, signed: bool = False) -> str:
    f = safe_float(val, default=float("nan"))
    if not math.isfinite(f):
        return "—"
    fmt = f"{{:+.{places}f}}%" if signed else f"{{:.{places}f}}%"
    return fmt.format(f)


def fmt_prob(val, places: int = 1) -> str:
    f = safe_float(val, default=float("nan"))
    if not math.isfinite(f):
        return "—"
    return f"{f * 100:.{places}f}%"


def fmt_usd(val, places: int = 0, signed: bool = False) -> str:
    f = safe_float(val, default=float("nan"))
    if not math.isfinite(f):
        return "—"
    fmt = f"{{:+,.{places}f}}" if signed else f"{{:,.{places}f}}"
    return f"${fmt.format(f)}"


def fmt_duration(seconds: int) -> str:
    if seconds <= 0:
        return "0m"
    if seconds < 3600:
        return f"{seconds // 60}m"
    h = seconds // 3600
    m = (seconds % 3600) // 60
    return f"{h}h{m:02d}m" if m else f"{h}h"


def score_emoji(s) -> str:
    s = safe_int(s)
    if s >= 9: return "🔥"
    if s >= 8: return "⭐"
    if s >= 6: return "👍"
    if s >= 4: return "🤔"
    return "❌"


REC_EMOJI = {"PASS": "⛔", "WATCH": "👀", "BUY": "🚀"}


def validate_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    url = str(url).strip()
    return url if re.match(r"^https?://[^\s<>]+$", url, re.I) else None


def strip_md2(text: str) -> str:
    return re.sub(r"\\([_*\[\]()~`>#+\-=|{}.!\\])", r"\1", text)
