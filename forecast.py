"""Fast, directly supervised return forecasts with out-of-sample validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features.engineering import build_features, get_feature_cols
from utils.config import SIGNAL_DEADBAND, TRANSACTION_COST_BPS, SLIPPAGE_BPS


MONTH_TRADING_DAYS = 21
MIN_FORECAST_ROWS = 180
MIN_SELECTION_SAMPLES = 30


def _strategy_sharpe(predicted: np.ndarray, actual: np.ndarray) -> float:
    positions = np.where(predicted > SIGNAL_DEADBAND, 1.0, 0.0)
    turnover = np.abs(np.diff(np.r_[0.0, positions]))
    cost = (TRANSACTION_COST_BPS + SLIPPAGE_BPS) / 10000.0
    returns = positions * actual - turnover * cost
    if len(returns) < 2 or np.std(returns, ddof=1) == 0:
        return 0.0
    return float(np.sqrt(252) * np.mean(returns) / np.std(returns, ddof=1))


def _candidate_models() -> dict:
    return {
        "naive_zero": None,
        "last_20d_mean": None,
        "momentum_5d": None,
        "ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]),
        "elastic_net": Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=0.001, l1_ratio=0.2, max_iter=5000)),
        ]),
        "regularized_tree": RandomForestRegressor(
            n_estimators=100,
            max_depth=3,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=1,
        ),
    }


def _candidate_prediction(name: str, model, frame: pd.DataFrame, x: pd.DataFrame, horizon: int) -> np.ndarray:
    if name == "naive_zero":
        return np.zeros(len(x))
    if name == "last_20d_mean":
        value = float(frame["log_return"].tail(20).mean() * horizon)
        return np.full(len(x), value)
    if name == "momentum_5d":
        value = float(frame["log_return"].tail(5).mean() * horizon)
        return np.full(len(x), value)
    return model.predict(x)


def _fit_direct_forecast(
    feature_frame: pd.DataFrame,
    horizon: int,
    min_rows: int,
) -> dict:
    """Fit one chronological direct-return model for a forecast horizon."""
    frame = feature_frame.copy()
    frame["forecast_target"] = np.log(frame["Close"].shift(-horizon) / frame["Close"])
    frame = frame.dropna(subset=["forecast_target"])
    feature_cols = get_feature_cols(frame)
    if len(frame) < min_rows:
        raise ValueError(
            f"Insufficient history for {horizon}-day forecast: "
            f"{len(frame)} rows < {min_rows}"
        )

    split = max(int(len(frame) * 0.8), len(frame) - 60)
    val_start = split + horizon
    if val_start >= len(frame) - 30:
        val_start = len(frame) - 30
    split = min(max(split, 1), val_start - 1)
    x_train = frame.iloc[:split][feature_cols]
    y_train = frame.iloc[:split]["forecast_target"].to_numpy()
    x_val = frame.iloc[val_start:][feature_cols]
    y_val = frame.iloc[val_start:]["forecast_target"].to_numpy()

    candidates = _candidate_models()
    evaluations = []
    for name, candidate in candidates.items():
        fitted = candidate
        if fitted is not None:
            fitted.fit(x_train, y_train)
        val_pred = _candidate_prediction(name, fitted, frame.iloc[:split], x_val, horizon)
        sharpe = _strategy_sharpe(val_pred, y_val)
        se = float(np.std((val_pred > SIGNAL_DEADBAND) * y_val, ddof=1) / np.sqrt(len(y_val))) if len(y_val) > 1 else 0.0
        evaluations.append((sharpe - 1.96 * se, sharpe, name, fitted, val_pred, se))
    evaluations.sort(key=lambda item: item[1], reverse=True)
    _, best_sharpe, selected_name, model, val_pred, selected_se = evaluations[0]
    baseline = next(item for item in evaluations if item[2] == "naive_zero")
    if best_sharpe <= baseline[1] + max(selected_se, baseline[5]):
        selected_name, model, val_pred, best_sharpe, selected_se = (
            baseline[2], baseline[3], baseline[4], baseline[1], baseline[5]
        )
    model_mae = float(np.mean(np.abs(y_val - val_pred)))
    model_rmse = float(np.sqrt(np.mean((y_val - val_pred) ** 2)))
    baseline_mae = float(np.mean(np.abs(y_val)))
    direction = float(np.mean(np.sign(y_val) == np.sign(val_pred)) * 100)
    residuals = y_val - val_pred
    residual_std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else model_mae
    if np.unique(val_pred).size < 2 or np.unique(y_val).size < 2:
        rank_ic = 0.0
    else:
        rank_ic_value = pd.Series(val_pred).corr(pd.Series(y_val), method="spearman")
        rank_ic = float(rank_ic_value) if pd.notna(rank_ic_value) else 0.0

    latest = feature_frame.iloc[-1]
    latest_features = latest[feature_cols].to_frame().T
    predicted_log_return = float(
        _candidate_prediction(selected_name, model, frame, latest_features, horizon)[0]
    )
    lower, upper = np.quantile(y_train, [0.01, 0.99])
    predicted_log_return = float(np.clip(predicted_log_return, lower, upper))
    validation_passed = (
        selected_name != "naive_zero"
        and
        len(y_val) >= 30
        and model_mae <= baseline_mae
        and direction >= 50.0
    )

    return {
        "horizon_trading_days": horizon,
        "predicted_log_return": predicted_log_return,
        "predicted_return": float(np.exp(predicted_log_return) - 1.0),
        "predicted_price": float(latest["Close"]) * np.exp(predicted_log_return),
        "validation_mae": model_mae,
        "validation_rmse": model_rmse,
        "baseline_mae": baseline_mae,
        "validation_directional_accuracy": direction,
        "validation_base_rate": float(np.mean(y_val >= 0) * 100),
        "validation_spearman_ic": 0.0 if not np.isfinite(rank_ic) else rank_ic,
        "validation_residual_std": residual_std,
        "selected_model": selected_name,
        "selected_sharpe": best_sharpe,
        "baseline_sharpe": baseline[1],
        "selection_standard_error": selected_se,
        "selection_reliable": selected_name != "naive_zero",
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
        "predicted_return_pct": round(fit["predicted_return"] * 100, 4),
        "predicted_price": round(fit["predicted_price"], 2),
        "validation_mae": round(fit["validation_mae"] * 100, 4),
        "validation_rmse": round(fit["validation_rmse"] * 100, 4),
        "baseline_mae": round(fit["baseline_mae"] * 100, 4),
        "validation_directional_accuracy": round(fit["validation_directional_accuracy"], 2),
        "validation_base_rate": round(fit["validation_base_rate"], 2),
        "validation_spearman_ic": round(fit["validation_spearman_ic"], 4),
        "selected_model": fit["selected_model"],
        "selected_sharpe": round(fit["selected_sharpe"], 4),
        "baseline_sharpe": round(fit["baseline_sharpe"], 4),
        "selection_reliable": fit["selection_reliable"],
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
                max(
                    0.0,
                    fit["predicted_price"]
                    - current_price * fit["validation_residual_std"] * np.sqrt(day),
                ),
                2,
            ),
            "upper_price": round(
                fit["predicted_price"]
                + current_price * fit["validation_residual_std"] * np.sqrt(day),
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
        "screen_model": one_month["selected_model"],
        "screen_validation_passed": one_day["validation_passed"],
    }
