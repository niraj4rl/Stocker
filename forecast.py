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


def _fit_direct_forecast(
    feature_frame: pd.DataFrame,
    horizon: int,
    min_rows: int,
) -> dict:
    """Fit one chronological direct-return model for a forecast horizon."""
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
    lower, upper = np.quantile(y_train, [0.01, 0.99])
    predicted_return = float(np.clip(predicted_return, lower, upper))
    validation_passed = (
        len(y_val) >= 30
        and model_mae <= baseline_mae
        and direction >= 50.0
    )

    return {
        "horizon_trading_days": horizon,
        "predicted_return": predicted_return,
        "predicted_price": float(latest["Close"]) * (1 + predicted_return),
        "validation_mae": model_mae,
        "baseline_mae": baseline_mae,
        "validation_directional_accuracy": direction,
        "validation_samples": int(len(y_val)),
        "validation_passed": bool(validation_passed),
        "as_of": str(pd.Timestamp(latest.name).date()),
    }


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
    fit = _fit_direct_forecast(feature_frame, horizon, min_rows)
    predicted_return = fit["predicted_return"]

    return {
        "horizon_trading_days": horizon,
        "predicted_return_pct": round(predicted_return * 100, 4),
        "predicted_price": round(fit["predicted_price"], 2),
        "validation_mae": round(fit["validation_mae"] * 100, 4),
        "baseline_mae": round(fit["baseline_mae"] * 100, 4),
        "validation_directional_accuracy": round(fit["validation_directional_accuracy"], 2),
        "validation_samples": fit["validation_samples"],
        "validation_passed": fit["validation_passed"],
        "confidence": "validated" if fit["validation_passed"] else "low",
        "as_of": fit["as_of"],
    }


def forecast_curve(
    df: pd.DataFrame,
    horizon: int = MONTH_TRADING_DAYS,
    min_rows: int = MIN_FORECAST_ROWS,
) -> dict:
    """Return direct model forecasts for every trading day in a future horizon."""
    if horizon < 1:
        raise ValueError("horizon must be at least one trading day")

    feature_frame = build_features(df, include_targets=False)
    latest = feature_frame.iloc[-1]
    current_price = float(latest["Close"])
    fits = [_fit_direct_forecast(feature_frame, day, min_rows) for day in range(1, horizon + 1)]
    future_dates = pd.bdate_range(
        start=pd.Timestamp(latest.name) + pd.offsets.BDay(1),
        periods=horizon,
    )
    points = [
        {
            "trading_day": day,
            "date": date.strftime("%Y-%m-%d"),
            "predicted_return_pct": round(fit["predicted_return"] * 100, 4),
            "predicted_price": round(fit["predicted_price"], 2),
            "validation_mae_pct": round(fit["validation_mae"] * 100, 4),
            "lower_price": round(
                max(0.0, fit["predicted_price"] - current_price * fit["validation_mae"]),
                2,
            ),
            "upper_price": round(
                fit["predicted_price"] + current_price * fit["validation_mae"],
                2,
            ),
            "validation_passed": fit["validation_passed"],
        }
        for day, (date, fit) in enumerate(zip(future_dates, fits), start=1)
    ]
    history = [
        {
            "date": pd.Timestamp(index).strftime("%Y-%m-%d"),
            "close": round(float(row["Close"]), 2),
        }
        for index, row in df.tail(30).iterrows()
    ]
    return {
        "horizon_trading_days": horizon,
        "current_price": round(current_price, 2),
        "historical_points": history,
        "points": points,
        "validated_points": sum(point["validation_passed"] for point in points),
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
