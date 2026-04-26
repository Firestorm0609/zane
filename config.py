"""All configuration loaded from environment."""
import os
from dotenv import load_dotenv

load_dotenv()


# ---------- Safe env-var helpers ----------
def _int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        raise SystemExit(
            f"CONFIG ERROR: {key}={raw!r} is not a valid integer. "
            f"Please fix your environment / .env file."
        )


def _float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        raise SystemExit(
            f"CONFIG ERROR: {key}={raw!r} is not a valid float. "
            f"Please fix your environment / .env file."
        )


def _bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# Telegram
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Endpoints
PUMP_PORTAL_URI = "wss://pumpportal.fun/api/data"
PUMP_FRONT      = "https://pump.fun/coin"

# Storage
DB_PATH     = os.getenv("DB_PATH", "monitor.db")
MODEL_PATH  = os.getenv("MODEL_PATH", "model.joblib")
SCALER_PATH = os.getenv("SCALER_PATH", "scaler.joblib")
LOG_PATH    = os.getenv("LOG_PATH", "pump_monitor.log")

MODEL_VERSION = "1.1"

# Scoring thresholds
DEFAULT_THRESHOLD       = _int("MONITOR_SCORE_THRESHOLD", 7)
MIN_MARKET_CAP          = _float("MONITOR_MIN_MARKET_CAP", 3000.0)
MAX_MARKET_CAP          = _float("MONITOR_MAX_MARKET_CAP", 150000.0)
SEEN_TTL_SEC            = _int("SEEN_TTL_SEC", 600)
BUY_THRESHOLD_DEFAULT   = _float("BUY_THRESHOLD", 0.65)
WATCH_THRESHOLD_DEFAULT = _float("WATCH_THRESHOLD", 0.45)

# ML
MAX_ML_WEIGHT     = _float("MAX_ML_WEIGHT", 0.85)
MIN_TRAIN_SAMPLES = _int("MIN_TRAIN_SAMPLES", 500)
RETRAIN_EVERY_SEC = _int("RETRAIN_EVERY_SEC", 24 * 3600)

# Paper trading
PAPER_ENABLED_DEFAULT      = _bool("PAPER_ENABLED_DEFAULT", False)
PAPER_ENTRY_SCORE          = _int("PAPER_ENTRY_SCORE", 8)
PAPER_STOP_LOSS_PCT        = _float("PAPER_STOP_LOSS_PCT", 20.0)
PAPER_TAKE_PROFIT_PCT      = _float("PAPER_TAKE_PROFIT_PCT", 35.0)
PAPER_TIME_STOP_SEC        = _int("PAPER_TIME_STOP_SEC", 4 * 60 * 60)
PAPER_MAX_CONCURRENT       = _int("PAPER_MAX_CONCURRENT", 3)
PAPER_MINT_COOLDOWN_SEC    = _int("PAPER_MINT_COOLDOWN_SEC", 30 * 60)
PAPER_POSITION_SIZE_USD    = _float("PAPER_POSITION_SIZE_USD", 100.0)
PAPER_POLL_INTERVAL_SEC    = _int("PAPER_POLL_INTERVAL_SEC", 60)
PAPER_STATS_LOOKBACK       = _int("PAPER_STATS_LOOKBACK", 1000)
PAPER_STARTING_BALANCE_USD = _float("PAPER_STARTING_BALANCE_USD", 1000.0)
PAPER_MAX_POSITION_PCT     = _float("PAPER_MAX_POSITION_PCT", 10.0)
PAPER_DAILY_LOSS_LIMIT_PCT = _float("PAPER_DAILY_LOSS_LIMIT_PCT", 20.0)
PAPER_LOSS_STREAK_PAUSE    = _int("PAPER_LOSS_STREAK_PAUSE", 3)
PAPER_FEE_PCT              = _float("PAPER_FEE_PCT", 1.0)
PAPER_SLIPPAGE_PCT         = _float("PAPER_SLIPPAGE_PCT", 2.0)

# Pinning
PIN_HIGH_CONVICTION     = _bool("PIN_HIGH_CONVICTION", True)
HIGH_CONVICTION_SCORE   = _int("HIGH_CONVICTION_SCORE", 9)
HIGH_CONVICTION_PROB    = _float("HIGH_CONVICTION_PROB", 0.8)
HIGH_CONVICTION_MAX_STD = _float("HIGH_CONVICTION_MAX_STD", 0.05)

# Lookback windows
LOOKBACK_WINDOWS = [
    ("15min", 15 * 60),
    ("1hr",   60 * 60),
    ("4hr",  4 * 60 * 60),
    ("24hr", 24 * 60 * 60),
    ("48hr", 48 * 60 * 60),
]
ML_LABEL_WINDOW    = os.getenv("ML_LABEL_WINDOW", "4hr")
PUMP_THRESHOLD_PCT = _float("PUMP_THRESHOLD_PCT", 50.0)
RUG_THRESHOLD_PCT  = _float("RUG_THRESHOLD_PCT", -50.0)

# Concurrency / runtime
MAX_CONCURRENT_PROCESS = _int("MAX_CONCURRENT_PROCESS", 50)
HTTP_TIMEOUT_SEC       = _int("HTTP_TIMEOUT_SEC", 10)
ENRICH_DELAY_SEC       = _int("ENRICH_DELAY_SEC", 4)
MAX_SEEN_ENTRIES       = _int("MAX_SEEN_ENTRIES", 10000)
MAX_GRADUATED_ENTRIES  = _int("MAX_GRADUATED_ENTRIES", 5000)
MAX_MARKET_CTX_ENTRIES = _int("MAX_MARKET_CTX_ENTRIES", 5000)
MARKET_CACHE_TTL_SEC   = _int("MARKET_CACHE_TTL_SEC", 30)
SNAPSHOT_COUNT         = _int("SNAPSHOT_COUNT", 5)

# Watchdog
STREAM_DEAD_ALERT_SEC    = _int("STREAM_DEAD_ALERT_SEC", 10 * 60)
STREAM_DEAD_COOLDOWN_SEC = _int("STREAM_DEAD_COOLDOWN_SEC", 30 * 60)

# Outcome notifications
OUTCOME_NOTIFY_ENABLED = _bool("OUTCOME_NOTIFY_ENABLED", False)
OUTCOME_NOTIFY_MIN_PCT = _float("OUTCOME_NOTIFY_MIN_PCT", 50.0)

# Backups
DB_BACKUP_INTERVAL_SEC = _int("DB_BACKUP_INTERVAL_SEC", 6 * 3600)
DB_BACKUP_PATH         = os.getenv("DB_BACKUP_PATH", "monitor_backup.db")

# Dead letters
DEAD_LETTER_RETRY_SEC          = _int("DEAD_LETTER_RETRY_SEC", 120)
DEAD_LETTER_MAX_RETRIES        = _int("DEAD_LETTER_MAX_RETRIES", 3)
DEAD_LETTER_FALLBACK           = os.getenv("DEAD_LETTER_FALLBACK", "dead_letters_fallback.jsonl")
DEAD_LETTER_FALLBACK_MAX_BYTES = _int("DEAD_LETTER_FALLBACK_MAX_BYTES", 10 * 1024 * 1024)

# Blacklist cache
BLACKLIST_CACHE_TTL_SEC = _int("BLACKLIST_CACHE_TTL_SEC", 60)

# Solana RPC
SOLANA_RPC_URL        = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
RPC_ENABLED           = _bool("RPC_ENABLED", True)
RPC_TIMEOUT_SEC       = _int("RPC_TIMEOUT_SEC", 8)
RPC_RATE_PER_SEC      = _float("RPC_RATE_PER_SEC", 3.0)
BUNDLE_SLOT_THRESHOLD = _int("BUNDLE_SLOT_THRESHOLD", 3)

# Confidence gating
CONFIDENCE_GATE_STD = _float("CONFIDENCE_GATE_STD", 0.08)

# ML availability
try:
    import numpy  # noqa: F401
    import joblib  # noqa: F401
    import sklearn  # noqa: F401
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

ALLOWED_CHAT_IDS = set(
    int(x) for x in os.getenv("ALLOWED_CHAT_IDS", "").split(",") if x.strip()
)


