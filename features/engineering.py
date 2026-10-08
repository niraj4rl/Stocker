import pandas as pd
import numpy as np
from utils.config import LAG_FEATURES, ROLLING_WINDOW


def build_features(df: pd.DataFrame, include_targets: bool = True) -> pd.DataFrame:
    """Build technical features, optionally including supervised targets."""
    df = df.copy()

    df = _add_rsi(df)
    df = _add_macd(df)
    df = _add_bollinger(df)
    df = _add_lags(df)
    df = _add_rolling_stats(df)
    df = _add_temporal(df)
    if include_targets:
        df = _add_target(df)
    df = df.dropna()
    return df


def _add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def _add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    # Use scale-free momentum features so a model cannot learn the stock's
    # nominal price level instead of its behavior.
    close = df["Close"].replace(0, np.nan)
    df["macd"] = (ema_fast - ema_slow) / close
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def _add_bollinger(df: pd.DataFrame, period: int = 20, std_mult: float = 2.0) -> pd.DataFrame:
    mid = df["Close"].rolling(period).mean()
    std = df["Close"].rolling(period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    df["bb_width"] = (upper - lower) / mid.replace(0, np.nan)
    df["bb_pct"] = (df["Close"] - lower) / (upper - lower).replace(0, np.nan)
    return df


def _add_lags(df: pd.DataFrame) -> pd.DataFrame:
    for i in range(1, LAG_FEATURES + 1):
        df[f"close_lag_{i}"] = df["Close"] / df["Close"].shift(i) - 1.0
    return df


def _add_rolling_stats(df: pd.DataFrame) -> pd.DataFrame:
    rolling_close = df["Close"].rolling(ROLLING_WINDOW)
    rolling_mean = rolling_close.mean()
    df[f"rolling_mean_{ROLLING_WINDOW}"] = df["Close"] / rolling_mean.replace(0, np.nan) - 1.0
    df[f"rolling_std_{ROLLING_WINDOW}"] = (
        rolling_close.std() / df["Close"].replace(0, np.nan)
    )
    df["volume_ma_10"] = df["Volume"].rolling(10).mean()
    df["volume_ratio"] = df["Volume"] / df["volume_ma_10"].replace(0, np.nan)
    return df


def _add_temporal(df: pd.DataFrame) -> pd.DataFrame:
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["week_of_year"] = df.index.isocalendar().week.astype(int)
    return df


def _add_target(df: pd.DataFrame) -> pd.DataFrame:
    df["target_pct_return"] = df["pct_return"].shift(-1)
    df["target_price"] = df["Close"].shift(-1)
    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    exclude = {
        "Open", "High", "Low", "Close", "Volume",
        "pct_return", "log_return",
        "target_pct_return", "target_price", "forecast_target",
        "classification_target", "regime",
    }
    return [c for c in df.columns if c not in exclude]


def build_classification_target(df: pd.DataFrame, window: int = 20) -> pd.Series:
    rolling_std = df["pct_return"].rolling(window).std()
    ret = df["target_pct_return"]
    conditions = [
        ret > rolling_std,
        ret < -rolling_std,
    ]
    labels = ["strong_up", "strong_down"]
    result = pd.Series("neutral", index=df.index)
    for cond, label in zip(conditions, labels):
        result = result.where(~cond, label)
    return result
