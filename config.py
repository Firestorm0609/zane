"""All configuration loaded from environment."""
import os
from dotenv import load_dotenv

load_dotenv()

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
DEFAULT_THRESHOLD       = int(os.getenv("MONITOR_SCORE_THRESHOLD", "7"))
MIN_MARKET_CAP          = float(os.getenv("MONITOR_MIN_MARKET_CAP", "3000"))
MAX_MARKET_CAP          = float(os.getenv("MONITOR_MAX_MARKET_CAP", "150000"))
SEEN_TTL_SEC            = int(os.getenv("SEEN_TTL_SEC", "600"))
BUY_THRESHOLD_DEFAULT   = float(os.getenv("BUY_THRESHOLD", "0.65"))
WATCH_THRESHOLD_DEFAULT = float(os.getenv("WATCH_THRESHOLD", "0.45"))

# ML
MAX_ML_WEIGHT     = float(os.getenv("MAX_ML_WEIGHT", "0.85"))
MIN_TRAIN_SAMPLES = int(os.getenv("MIN_TRAIN_SAMPLES", "500"))
RETRAIN_EVERY_SEC = int(os.getenv("RETRAIN_EVERY_SEC", str(24 * 3600)))

# Paper trading
PAPER_ENABLED_DEFAULT      = os.getenv("PAPER_ENABLED_DEFAULT", "false").lower() == "true"
PAPER_ENTRY_SCORE          = int(os.getenv("PAPER_ENTRY_SCORE", "8"))
PAPER_STOP_LOSS_PCT        = float(os.getenv("PAPER_STOP_LOSS_PCT", "20"))
PAPER_TAKE_PROFIT_PCT      = float(os.getenv("PAPER_TAKE_PROFIT_PCT", "35"))
PAPER_TIME_STOP_SEC        = int(os.getenv("PAPER_TIME_STOP_SEC", str(4 * 60 * 60)))
PAPER_MAX_CONCURRENT       = int(os.getenv("PAPER_MAX_CONCURRENT", "3"))
PAPER_MINT_COOLDOWN_SEC    = int(os.getenv("PAPER_MINT_COOLDOWN_SEC", str(30 * 60)))
PAPER_POSITION_SIZE_USD    = float(os.getenv("PAPER_POSITION_SIZE_USD", "100"))
PAPER_POLL_INTERVAL_SEC    = int(os.getenv("PAPER_POLL_INTERVAL_SEC", "60"))
PAPER_STATS_LOOKBACK       = int(os.getenv("PAPER_STATS_LOOKBACK", "1000"))
PAPER_STARTING_BALANCE_USD = float(os.getenv("PAPER_STARTING_BALANCE_USD", "1000"))
PAPER_MAX_POSITION_PCT     = float(os.getenv("PAPER_MAX_POSITION_PCT", "10"))
PAPER_DAILY_LOSS_LIMIT_PCT = float(os.getenv("PAPER_DAILY_LOSS_LIMIT_PCT", "20"))
PAPER_LOSS_STREAK_PAUSE    = int(os.getenv("PAPER_LOSS_STREAK_PAUSE", "3"))
PAPER_FEE_PCT              = float(os.getenv("PAPER_FEE_PCT", "1.0"))
PAPER_SLIPPAGE_PCT         = float(os.getenv("PAPER_SLIPPAGE_PCT", "2.0"))

# Pinning
PIN_HIGH_CONVICTION     = os.getenv("PIN_HIGH_CONVICTION", "true").lower() == "true"
HIGH_CONVICTION_SCORE   = int(os.getenv("HIGH_CONVICTION_SCORE", "9"))
HIGH_CONVICTION_PROB    = float(os.getenv("HIGH_CONVICTION_PROB", "0.8"))
HIGH_CONVICTION_MAX_STD = float(os.getenv("HIGH_CONVICTION_MAX_STD", "0.05"))

# Lookback windows
LOOKBACK_WINDOWS = [
    ("15min", 15 * 60),
    ("1hr",   60 * 60),
    ("4hr",  4 * 60 * 60),
    ("24hr", 24 * 60 * 60),
    ("48hr", 48 * 60 * 60),
]
ML_LABEL_WINDOW    = os.getenv("ML_LABEL_WINDOW", "4hr")
PUMP_THRESHOLD_PCT = float(os.getenv("PUMP_THRESHOLD_PCT", "50"))
RUG_THRESHOLD_PCT  = float(os.getenv("RUG_THRESHOLD_PCT", "-50"))

# Concurrency / runtime
MAX_CONCURRENT_PROCESS = int(os.getenv("MAX_CONCURRENT_PROCESS", "50"))
HTTP_TIMEOUT_SEC       = int(os.getenv("HTTP_TIMEOUT_SEC", "10"))
ENRICH_DELAY_SEC       = int(os.getenv("ENRICH_DELAY_SEC", "4"))
MAX_SEEN_ENTRIES       = int(os.getenv("MAX_SEEN_ENTRIES", "10000"))
MAX_GRADUATED_ENTRIES  = int(os.getenv("MAX_GRADUATED_ENTRIES", "5000"))
MAX_MARKET_CTX_ENTRIES = int(os.getenv("MAX_MARKET_CTX_ENTRIES", "5000"))
MARKET_CACHE_TTL_SEC   = int(os.getenv("MARKET_CACHE_TTL_SEC", "30"))
SNAPSHOT_COUNT         = int(os.getenv("SNAPSHOT_COUNT", "5"))

# Watchdog
STREAM_DEAD_ALERT_SEC    = int(os.getenv("STREAM_DEAD_ALERT_SEC", str(10 * 60)))
STREAM_DEAD_COOLDOWN_SEC = int(os.getenv("STREAM_DEAD_COOLDOWN_SEC", str(30 * 60)))

# Outcome notifications
OUTCOME_NOTIFY_ENABLED = os.getenv("OUTCOME_NOTIFY_ENABLED", "false").lower() == "true"
OUTCOME_NOTIFY_MIN_PCT = float(os.getenv("OUTCOME_NOTIFY_MIN_PCT", "50"))

# Backups
DB_BACKUP_INTERVAL_SEC = int(os.getenv("DB_BACKUP_INTERVAL_SEC", str(6 * 3600)))
DB_BACKUP_PATH         = os.getenv("DB_BACKUP_PATH", "monitor_backup.db")

# Dead letters
DEAD_LETTER_RETRY_SEC          = int(os.getenv("DEAD_LETTER_RETRY_SEC", "120"))
DEAD_LETTER_MAX_RETRIES        = int(os.getenv("DEAD_LETTER_MAX_RETRIES", "3"))
DEAD_LETTER_FALLBACK           = os.getenv("DEAD_LETTER_FALLBACK", "dead_letters_fallback.jsonl")
DEAD_LETTER_FALLBACK_MAX_BYTES = int(os.getenv("DEAD_LETTER_FALLBACK_MAX_BYTES", str(10 * 1024 * 1024)))

# Blacklist cache
BLACKLIST_CACHE_TTL_SEC = int(os.getenv("BLACKLIST_CACHE_TTL_SEC", "60"))

# Solana RPC
SOLANA_RPC_URL        = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
RPC_ENABLED           = os.getenv("RPC_ENABLED", "true").lower() == "true"
RPC_TIMEOUT_SEC       = int(os.getenv("RPC_TIMEOUT_SEC", "8"))
RPC_RATE_PER_SEC      = float(os.getenv("RPC_RATE_PER_SEC", "3"))
BUNDLE_SLOT_THRESHOLD = int(os.getenv("BUNDLE_SLOT_THRESHOLD", "3"))

# Confidence gating
CONFIDENCE_GATE_STD = float(os.getenv("CONFIDENCE_GATE_STD", "0.08"))

# ML availability
try:
    import numpy  # noqa: F401
    import joblib  # noqa: F401
    import sklearn  # noqa: F401
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
ALLOWED_CHAT_IDS = set(int(x) for x in os.getenv("ALLOWED_CHAT_IDS", "").split(",") if x.strip())

