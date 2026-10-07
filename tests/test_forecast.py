import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from forecast import MONTH_TRADING_DAYS, forecast_horizon


def make_ohlcv(n=420):
    rng = np.random.default_rng(42)
    close = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    index = pd.bdate_range("2024-01-01", periods=n)
    df = pd.DataFrame({
        "Open": close,
        "High": close * 1.01,
        "Low": close * 0.99,
        "Close": close,
        "Volume": np.full(n, 1_000_000),
    }, index=index)
    df["pct_return"] = df["Close"].pct_change()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    return df


def test_month_forecast_has_validated_schema():
    data = make_ohlcv()
    result = forecast_horizon(data, MONTH_TRADING_DAYS)
    assert result["horizon_trading_days"] == 21
    assert isinstance(result["predicted_return_pct"], float)
    assert result["validation_samples"] >= 30
    assert result["confidence"] in {"validated", "low"}
    assert result["as_of"] == str(data.index[-1].date())


def test_forecast_rejects_short_history():
    try:
        forecast_horizon(make_ohlcv(100), MONTH_TRADING_DAYS)
    except ValueError as exc:
        assert "Insufficient history" in str(exc)
    else:
        raise AssertionError("short history should be rejected")
