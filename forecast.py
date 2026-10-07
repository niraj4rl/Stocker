"""Fast, directly supervised return forecasts with out-of-sample validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features.engineering import build_features, get_feature_cols


MONTH_TRADING_DAYS = 21
MIN_FORECAST_ROWS = 180


def forecast_horizon(
    df: pd.DataFrame,
    horizon: int = MONTH_TRADING_DAYS,
    min_rows: int = MIN_FORECAST_ROWS,
) -> dict:
    """Forecast a direct future return and report validation evidence.

    The model predicts the return from the latest feature date to exactly
    ``horizon`` future trading observations. Validation is chronological and
    compared with a zero-return baseline.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least one trading day")
    feature_frame = build_features(df, include_targets=False)
    frame = feature_frame.copy()
    frame["forecast_target"] = frame["Close"].shift(-horizon) / frame["Close"] - 1.0
    frame = frame.dropna(subset=["forecast_target"])
    feature_cols = get_feature_cols(frame)
    if len(frame) < min_rows:
        raise ValueError(
            f"Insufficient history for {horizon}-day forecast: "
            f"{len(frame)} rows < {min_rows}"
        )

    split = max(int(len(frame) * 0.8), len(frame) - 60)
    split = min(max(split, 1), len(frame) - 30)
    x_train = frame.iloc[:split][feature_cols]
    y_train = frame.iloc[:split]["forecast_target"].to_numpy()
    x_val = frame.iloc[split:][feature_cols]
    y_val = frame.iloc[split:]["forecast_target"].to_numpy()

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ])
    model.fit(x_train, y_train)
    val_pred = model.predict(x_val)
    model_mae = float(np.mean(np.abs(y_val - val_pred)))
    baseline_mae = float(np.mean(np.abs(y_val)))
    direction = float(np.mean(np.sign(y_val) == np.sign(val_pred)) * 100)

    latest = feature_frame.iloc[-1]
    latest_features = latest[feature_cols].to_frame().T
    predicted_return = float(model.predict(latest_features)[0])
    # Keep extreme extrapolations bounded by the observed training target range.
    lower, upper = np.quantile(y_train, [0.01, 0.99])
    predicted_return = float(np.clip(predicted_return, lower, upper))
    validation_passed = (
        len(y_val) >= 30
        and model_mae <= baseline_mae
        and direction >= 50.0
    )

    return {
        "horizon_trading_days": horizon,
        "predicted_return_pct": round(predicted_return * 100, 4),
        "predicted_price": round(float(latest["Close"]) * (1 + predicted_return), 2),
        "validation_mae": round(model_mae * 100, 4),
        "baseline_mae": round(baseline_mae * 100, 4),
        "validation_directional_accuracy": round(direction, 2),
        "validation_samples": int(len(y_val)),
        "validation_passed": bool(validation_passed),
        "confidence": "validated" if validation_passed else "low",
        "as_of": str(pd.Timestamp(latest.name).date()),
    }


def fast_screen_ticker(df: pd.DataFrame) -> dict:
    """Return one-day and one-month forecasts without HMM or candidate training."""
    one_day = forecast_horizon(df, horizon=1)
    one_month = forecast_horizon(df, horizon=MONTH_TRADING_DAYS)
    return {
        "predicted_return_pct": one_day["predicted_return_pct"],
        "one_month": one_month,
        "screen_model": "validated Ridge",
        "screen_validation_passed": one_day["validation_passed"],
    }
