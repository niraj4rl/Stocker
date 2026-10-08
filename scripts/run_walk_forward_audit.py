"""Run a reproducible, purged walk-forward audit on liquid NSE names.

Usage:
    python scripts/run_walk_forward_audit.py

The script uses only cached/real OHLCV data, writes one row per ticker to
``backtest_results.csv``, and never substitutes missing data with estimates.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.nse_stocks import NIFTY_50
from data.robust_fetcher import fetch_ohlcv_robust
from features.engineering import build_features, get_feature_cols
from forecast import MONTH_TRADING_DAYS
from utils.config import TRANSACTION_COST_BPS, SLIPPAGE_BPS


def _rank_ic(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(pd.Series(actual).corr(pd.Series(predicted), method="spearman") or 0.0)


def audit_ticker(ticker: str) -> dict:
    raw = fetch_ohlcv_robust(ticker, period="5y")
    frame = build_features(raw, include_targets=False)
    cols = get_feature_cols(frame)
    horizon = MONTH_TRADING_DAYS
    rows = []
    first_test = max(252, len(frame) - 504)
    for endpoint in range(first_test, len(frame) - horizon, horizon):
        train_end = endpoint - horizon
        if train_end < 180:
            continue
        train = frame.iloc[:train_end]
        test = frame.iloc[endpoint : endpoint + 1]
        target = np.log(frame["Close"].shift(-horizon) / frame["Close"])
        y_train = target.iloc[:train_end].dropna()
        x_train = train.loc[y_train.index, cols]
        x_test = test[cols]
        model = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
        model.fit(x_train, y_train)
        pred = float(model.predict(x_test)[0])
        actual = float(target.iloc[endpoint])
        mean_pred = float(y_train.tail(20).mean())
        rows.append({"date": str(frame.index[endpoint].date()), "pred": pred, "actual": actual, "mean": mean_pred})

    if not rows:
        raise ValueError("not enough walk-forward rows")
    result = pd.DataFrame(rows)
    actual_return = np.expm1(result["actual"])
    predicted_return = np.expm1(result["pred"])
    residual = actual_return - predicted_return
    base_rate = float((actual_return >= 0).mean() * 100)
    accuracy = float((np.sign(actual_return) == np.sign(predicted_return)).mean() * 100)
    zero_mae = float(actual_return.abs().mean())
    model_mae = float(residual.abs().mean())
    cost = (TRANSACTION_COST_BPS + SLIPPAGE_BPS) / 10000
    position = np.where(predicted_return > cost, 1, np.where(predicted_return < -cost, -1, 0))
    strategy = position * actual_return - np.abs(np.diff(np.r_[0, position])) * cost
    return {
        "ticker": ticker,
        "selected_model": "ridge",
        "fallback_reason": "",
        "folds": len(result),
        "model_mae_pct": round(model_mae * 100, 4),
        "zero_return_mae_pct": round(zero_mae * 100, 4),
        "direction_accuracy_pct": round(accuracy, 2),
        "base_rate_pct": round(base_rate, 2),
        "spearman_ic": round(_rank_ic(result["actual"].to_numpy(), result["pred"].to_numpy()), 4),
        "mean_strategy_return_pct": round(float(strategy.mean()) * 100, 4),
        "mean_predicted_return_pct": round(float(predicted_return.mean()) * 100, 4),
        "mean_actual_return_pct": round(float(actual_return.mean()) * 100, 4),
        "_observations": result,
    }


def main() -> None:
    results = []
    observations = []
    for ticker in NIFTY_50:
        try:
            result = audit_ticker(ticker)
            observations.append(result.pop("_observations"))
            results.append(result)
        except Exception as exc:
            results.append({"ticker": ticker, "error": str(exc)})
    output = ROOT / "backtest_results.csv"
    pd.DataFrame(results).to_csv(output, index=False)
    pd.DataFrame(results)[
        ["ticker", "selected_model", "fallback_reason", "folds", "error"]
    ].to_csv(ROOT / "selection_log.csv", index=False)
    if observations:
        all_obs = pd.concat(observations, ignore_index=True)
        cost = (TRANSACTION_COST_BPS + SLIPPAGE_BPS) / 10000
        portfolio_rows = []
        for date, group in all_obs.groupby("date"):
            top_n = max(1, int(np.ceil(len(group) * 0.1)))
            top = group.nlargest(top_n, "pred")
            portfolio_rows.append({
                "date": date,
                "top_decile_return_pct": (np.expm1(top["actual"]).mean() - cost) * 100,
                "equal_weight_return_pct": (np.expm1(group["actual"]).mean() - cost) * 100,
            })
        portfolio = pd.DataFrame(portfolio_rows)
        portfolio.to_csv(ROOT / "backtest_portfolio_results.csv", index=False)
        portfolio.to_csv(ROOT / "portfolio_results.csv", index=False)
    print(f"Wrote {len(results)} ticker rows to {output}")


if __name__ == "__main__":
    main()
