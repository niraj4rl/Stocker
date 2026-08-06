import pandas as pd
from utils.config import DATA_DIR, MIN_DATA_ROWS, MAX_STALE_DAYS
from data.robust_fetcher import fetch_ohlcv_robust


def _assert_data_quality(df: pd.DataFrame, ticker: str) -> None:
    """Raise if data is insufficient or too stale for model/backtest usage."""
    if df is None or df.empty:
        raise ValueError(f"No data returned for ticker: {ticker}")

    if len(df) < MIN_DATA_ROWS:
        raise ValueError(
            f"Insufficient history for {ticker}: {len(df)} rows < {MIN_DATA_ROWS}"
        )

    if "Close" not in df.columns or df["Close"].isna().all():
        raise ValueError(f"No valid Close prices for ticker: {ticker}")

    last_dt = pd.to_datetime(df.index.max())
    if getattr(last_dt, "tzinfo", None) is not None:
        last_dt = last_dt.tz_convert("UTC").tz_localize(None)
    now_utc = pd.Timestamp.now(tz="UTC").tz_localize(None)
    age_days = (now_utc - last_dt).days

    if age_days > MAX_STALE_DAYS:
        raise ValueError(
            f"Stale data for {ticker}: last point is {age_days} days old (max {MAX_STALE_DAYS})"
        )


def fetch_ohlcv(
    ticker: str,
    period: str = "5y",
    force_refresh: bool = False,
    data_source: str = "yfinance",
    access_token: str = "",
) -> pd.DataFrame:
    """
    Fetch OHLCV data using robust multi-source fetcher.
    Guarantees real data or raises error (no synthetic fallback).
    """
    return fetch_ohlcv_robust(
        ticker=ticker,
        period=period,
        force_refresh=force_refresh,
        data_source=data_source,
        access_token=access_token,
    )


def load_or_fetch(
    ticker: str,
    force_refresh: bool = False,
    data_source: str = "yfinance",
    access_token: str = "",
    period: str = "5y",
) -> pd.DataFrame:
    return fetch_ohlcv(
        ticker,
        period=period,
        force_refresh=force_refresh,
        data_source=data_source,
        access_token=access_token,
    )