import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")
DATA_DIR = ROOT / "data" / "cache"
MODEL_DIR = ROOT / "models" / "registry"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_YEARS = 3
TEST_MONTHS = 6
REGIME_WINDOW = 20
N_REGIMES = 3
PURGE_GAP_DAYS = int(os.environ.get("STOCKER_PURGE_GAP_DAYS", "5"))

REGIME_BULL = "Bull"
REGIME_BEAR = "Bear"
REGIME_HIGHVOL = "HighVol"
REGIME_LABELS = [REGIME_BULL, REGIME_BEAR, REGIME_HIGHVOL]

PARADIGM_REGRESSION = "regression"
PARADIGM_CLASSIFICATION = "classification"

LAG_FEATURES = 5
ROLLING_WINDOW = 7
TRADING_DAYS_YEAR = 252

# --- Realism and risk controls ---
# Execution assumptions in basis points (1 bps = 0.01%)
TRANSACTION_COST_BPS = float(os.environ.get("STOCKER_TRANSACTION_COST_BPS", "10"))
SLIPPAGE_BPS = float(os.environ.get("STOCKER_SLIPPAGE_BPS", "5"))

# Position/risk controls
MAX_GROSS_LEVERAGE = float(os.environ.get("STOCKER_MAX_GROSS_LEVERAGE", "1.0"))
VOL_TARGET_ANNUAL = float(os.environ.get("STOCKER_VOL_TARGET_ANNUAL", "0.18"))
SIGNAL_DEADBAND = (TRANSACTION_COST_BPS + SLIPPAGE_BPS) / 10000.0 * 2 * 1.5
DRAWDOWN_THROTTLE_TRIGGER = float(os.environ.get("STOCKER_DRAWDOWN_THROTTLE_TRIGGER", "0.12"))
DRAWDOWN_THROTTLE_MULTIPLIER = float(os.environ.get("STOCKER_DRAWDOWN_THROTTLE_MULTIPLIER", "0.5"))

# Regime exposure caps to avoid full risk in stressed states.
REGIME_CAP_BULL = float(os.environ.get("STOCKER_REGIME_CAP_BULL", "1.0"))
REGIME_CAP_BEAR = float(os.environ.get("STOCKER_REGIME_CAP_BEAR", "0.7"))
REGIME_CAP_HIGHVOL = float(os.environ.get("STOCKER_REGIME_CAP_HIGHVOL", "0.4"))

# Data quality guardrails
MIN_DATA_ROWS = int(os.environ.get("STOCKER_MIN_DATA_ROWS", "252"))
MAX_STALE_DAYS = int(os.environ.get("STOCKER_MAX_STALE_DAYS", "12"))

# Live predictor refresh controls
PREDICTOR_REFRESH_MINUTES = int(os.environ.get("STOCKER_PREDICTOR_REFRESH_MINUTES", "60"))

# --- Data source config ---
DATA_SOURCE = os.environ.get("STOCKER_DATA_SOURCE", "yfinance")  # "yfinance" or "upstox"
UPSTOX_API_KEY = os.environ.get("UPSTOX_API_KEY", "")
UPSTOX_API_SECRET = os.environ.get("UPSTOX_API_SECRET", "")
UPSTOX_REDIRECT_URI = os.environ.get("UPSTOX_REDIRECT_URI", "https://127.0.0.1")
UPSTOX_ACCESS_TOKEN = os.environ.get("UPSTOX_ACCESS_TOKEN", "")

