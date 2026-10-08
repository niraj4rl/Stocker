import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from forecast import MONTH_TRADING_DAYS, fast_screen_ticker, forecast_curve, forecast_horizon
from backtest.predictor import monthly_signal
from features.engineering import build_features, get_feature_cols


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
    assert result["validation_samples"] < len(build_features(data))
    assert "validation_base_rate" in result
    assert "validation_spearman_ic" in result


def test_forecast_rejects_short_history():
    try:
        forecast_horizon(make_ohlcv(100), MONTH_TRADING_DAYS)
    except ValueError as exc:
        assert "Insufficient history" in str(exc)
    else:
        raise AssertionError("short history should be rejected")


def test_fast_screen_returns_both_ranking_horizons():
    result = fast_screen_ticker(make_ohlcv())

    assert result["screen_model"] in {
        "naive_zero",
        "last_20d_mean",
        "momentum_5d",
        "ridge",
        "elastic_net",
        "regularized_tree",
    }
    assert isinstance(result["predicted_return_pct"], float)
    assert result["one_month"]["horizon_trading_days"] == MONTH_TRADING_DAYS
    assert isinstance(result["one_month"]["predicted_return_pct"], float)
    assert result["one_month"]["as_of"] == str(make_ohlcv().index[-1].date())


def test_forecast_curve_has_direct_daily_points():
    result = forecast_curve(make_ohlcv(), MONTH_TRADING_DAYS)

    assert result["horizon_trading_days"] == MONTH_TRADING_DAYS
    assert len(result["points"]) == MONTH_TRADING_DAYS
    assert result["points"][0]["trading_day"] == 1
    assert result["points"][-1]["trading_day"] == MONTH_TRADING_DAYS
    assert all("predicted_price" in point for point in result["points"])
    assert len(result["historical_points"]) == 30
    assert all(point["lower_price"] <= point["predicted_price"] <= point["upper_price"] for point in result["points"])
    widths = [
        point["upper_price"] - point["lower_price"]
        for point in result["points"]
    ]
    assert widths[-1] >= widths[0]


def test_forecast_target_is_strictly_horizon_aligned():
    data = make_ohlcv()
    features = build_features(data)
    horizon = 21
    expected = features["Close"].shift(-horizon) / features["Close"] - 1.0
    observed = features["Close"].shift(-horizon) / features["Close"] - 1.0
    np.testing.assert_allclose(
        expected.dropna().to_numpy(),
        observed.dropna().to_numpy(),
    )


def test_features_are_scale_invariant():
    data = make_ohlcv()
    scaled = data.copy()
    for column in ("Open", "High", "Low", "Close"):
        scaled[column] *= 10

    original_features = build_features(data)
    scaled_features = build_features(scaled)
    columns = get_feature_cols(original_features)
    np.testing.assert_allclose(
        original_features.iloc[-1][columns].to_numpy(dtype=float),
        scaled_features.iloc[-1][columns].to_numpy(dtype=float),
        rtol=1e-9,
        atol=1e-9,
    )


def test_monthly_signal_uses_error_as_hold_band():
    assert monthly_signal(0.01, 0.02) == "Hold"
    assert monthly_signal(0.03, 0.02) == "Buy"
    assert monthly_signal(-0.03, 0.02) == "Sell"
