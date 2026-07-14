import argparse
import uuid
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from data.ingestion import load_or_fetch
from features.engineering import build_features, get_feature_cols, build_classification_target
from regime.detector import RegimeDetector, label_regimes_expanding
from models.trainer import ModelTrainer, _cls_label_to_signal
from models.model_scorecard import ScorecardStore
from router.adaptive import AdaptiveRouter
from backtest.splitter import WalkForwardSplitter
from utils.metrics import compute_all_metrics, sharpe_ratio
from utils.config import (
    REGIME_WINDOW,
    PARADIGM_REGRESSION,
    PARADIGM_CLASSIFICATION,
    TRANSACTION_COST_BPS,
    SLIPPAGE_BPS,
    MAX_GROSS_LEVERAGE,
    VOL_TARGET_ANNUAL,
    SIGNAL_DEADBAND,
    TRADING_DAYS_YEAR,
    DRAWDOWN_THROTTLE_TRIGGER,
    DRAWDOWN_THROTTLE_MULTIPLIER,
    REGIME_CAP_BULL,
    REGIME_CAP_BEAR,
    REGIME_CAP_HIGHVOL,
    REGIME_BULL,
    REGIME_BEAR,
    REGIME_HIGHVOL,
)


def _get_model_name(model) -> str:
    if model is None:
        return "unknown"
    estimator = model
    if hasattr(model, "steps"):
        estimator = model.steps[-1][1]
    
    class_name = estimator.__class__.__name__.lower()
    if "xgb" in class_name:
        return "xgb"
    elif "randomforest" in class_name:
        return "rf"
    elif "ridge" in class_name:
        return "ridge"
    elif "svc" in class_name:
        return "svc"
    elif "svr" in class_name:
        return "svr"
    elif "kneighbors" in class_name:
        return "knn"
    elif "mlp" in class_name:
        return "mlp"
    return class_name


def run_backtest(
    ticker: str,
    verbose: bool = True,
    profile: str = "full",
    data_source: str = "yfinance",
    access_token: str = "",
) -> dict:
    print(f"\n=== Stocker Backtest: {ticker} ===\n")
    profile = (profile or "full").lower()
    if profile not in {"fast", "full"}:
        raise ValueError("profile must be 'fast' or 'full'")

    run_id = str(uuid.uuid4())[:8]
    store = ScorecardStore()
    model_io_records = []
    trainer = None

    raw = load_or_fetch(ticker, data_source=data_source, access_token=access_token)
    df = build_features(raw)

    if profile == "fast":
        cutoff = df.index.max() - relativedelta(years=2)
        df = df[df.index >= cutoff].copy()

    splitter = WalkForwardSplitter(
        train_years=1 if profile == "fast" else 3,
        test_months=12 if profile == "fast" else 6,
    )
    fold_descriptions = splitter.describe(df)
    if verbose:
        print(fold_descriptions.to_string(index=False))
        print()

    all_returns = {
        "adaptive": [],
        "buy_and_hold": [],
        "static_regression": [],
        "static_classification": [],
        "naive_regime_regression": [],
    }
    fold_results = []

    for fold, train_df, test_df, train_end, test_end in splitter.split(df):
        print(f"--- Fold {fold}: train to {train_end.date()}, test {test_df.index[0].date()}–{test_df.index[-1].date()} ---")

        detector = RegimeDetector()
        try:
            detector.fit(train_df)
        except Exception as e:
            print(f"[backtest] Regime detector fit failed: {e}, skipping fold")
            continue

        regime_labels_train = detector.predict(train_df)
        trainer = ModelTrainer(
            scorecard_store=store,
            xgb_estimators=50 if profile == "fast" else 200,
            include_svc=True,
            include_svr=True,
            include_knn=True,
            include_mlp=True,
            include_xgb=True,
            include_rf=True,
            include_ridge=True,
        )
        registry = trainer.train_for_fold(
            train_df, regime_labels_train,
            ticker=ticker, run_id=run_id, fold=fold,
        )

        router = AdaptiveRouter()
        router.build_from_registry(registry)

        static_reg_model, static_reg_cols = _best_static_model(registry, PARADIGM_REGRESSION)
        static_cls_model, static_cls_cols = _best_static_model(registry, PARADIGM_CLASSIFICATION)

        adaptive_rets, bh_rets = [], []
        static_reg_rets, static_cls_rets = [], []
        naive_reg_rets = []
        prev_adaptive_signal = 0.0
        prev_static_reg_signal = 0.0
        prev_static_cls_signal = 0.0
        prev_naive_signal = 0.0
        bh_position_open = False
        adaptive_equity = 1.0
        adaptive_peak = 1.0

        full_df = pd.concat([train_df.iloc[-REGIME_WINDOW * 2:], test_df])

        for i, (date, row) in enumerate(test_df.iterrows()):
            true_return = row["target_pct_return"]
            if pd.isna(true_return):
                continue

            lookback_end = full_df.index.get_loc(date)
            lookback_start = max(0, lookback_end - REGIME_WINDOW)
            recent = full_df.iloc[lookback_start:lookback_end + 1]

            try:
                regime = detector.predict_current(recent)
            except Exception:
                regime = "HighVol"

            recent_vol = float(recent["pct_return"].std() * np.sqrt(TRADING_DAYS_YEAR)) if len(recent) > 1 else 0.0
            if recent_vol > 0:
                position_size = min(MAX_GROSS_LEVERAGE, VOL_TARGET_ANNUAL / recent_vol)
            else:
                position_size = 1.0

            regime_cap = _regime_exposure_cap(regime)
            dd_throttle = _drawdown_throttle(adaptive_equity, adaptive_peak)
            risk_scalar = min(1.0, regime_cap) * dd_throttle
            position_size *= risk_scalar

            model, paradigm, feat_cols = router.get_model(registry, regime)
            adaptive_signal = _get_signal(model, paradigm, row, feat_cols)
            adaptive_signal *= position_size

            if static_reg_model is not None:
                reg_signal = _get_signal(static_reg_model, PARADIGM_REGRESSION, row, static_reg_cols)
                reg_signal *= position_size
                static_reg_rets.append(
                    _apply_execution_frictions(reg_signal, prev_static_reg_signal, true_return)
                )
                prev_static_reg_signal = reg_signal

            if static_cls_model is not None:
                cls_signal = _get_signal(static_cls_model, PARADIGM_CLASSIFICATION, row, static_cls_cols)
                cls_signal *= position_size
                static_cls_rets.append(
                    _apply_execution_frictions(cls_signal, prev_static_cls_signal, true_return)
                )
                prev_static_cls_signal = cls_signal

            naive_model, _, naive_cols = _get_regime_model(registry, regime, PARADIGM_REGRESSION)
            naive_signal = _get_signal(naive_model, PARADIGM_REGRESSION, row, naive_cols)
            naive_signal *= position_size
            naive_reg_rets.append(
                _apply_execution_frictions(naive_signal, prev_naive_signal, true_return)
            )
            prev_naive_signal = naive_signal

            adaptive_rets.append(
                _apply_execution_frictions(adaptive_signal, prev_adaptive_signal, true_return)
            )
            prev_adaptive_signal = adaptive_signal
            adaptive_equity *= (1 + adaptive_rets[-1])
            adaptive_peak = max(adaptive_peak, adaptive_equity)

            bh_return = true_return
            if not bh_position_open:
                bh_return -= _cost_rate(1.0)
                bh_position_open = True
            bh_rets.append(bh_return)

            # Record prediction details for this step of backtest
            if model is not None and feat_cols is not None and not np.isnan(row[feat_cols].values).any():
                try:
                    X = row[feat_cols].values.reshape(1, -1)
                    if paradigm == PARADIGM_REGRESSION:
                        pred_return = float(model.predict(X)[0])
                        bounds = None
                        if regime in registry and paradigm in registry[regime]:
                            bounds = registry[regime][paradigm].get("prediction_bounds")
                        if bounds and len(bounds) == 2:
                            pred_return = float(np.clip(pred_return, bounds[0], bounds[1]))
                        current_price = float(row["Close"])
                        pred_price = current_price * (1 + pred_return)
                        sig_str = "Buy" if pred_return > SIGNAL_DEADBAND else ("Sell" if pred_return < -SIGNAL_DEADBAND else "Hold")
                        output_payload = {
                            "predicted_return": float(pred_return),
                            "predicted_price": float(pred_price),
                            "signal": sig_str
                        }
                    else:
                        pred_label = model.predict(X)[0]
                        try:
                            proba = model.predict_proba(X)[0]
                            confidence = float(max(proba)) if len(proba) > 0 else None
                        except Exception:
                            confidence = None
                        signal_map = {"strong_up": "Buy", "neutral": "Hold", "strong_down": "Sell"}
                        sig_str = signal_map.get(pred_label, "Hold")
                        output_payload = {
                            "predicted_class": str(pred_label),
                            "confidence": confidence,
                            "signal": sig_str
                        }

                    raw_inputs = row[feat_cols].to_dict()
                    input_payload = {}
                    for k, v in raw_inputs.items():
                        if isinstance(v, (np.floating, float)):
                            input_payload[str(k)] = float(v)
                        elif isinstance(v, (np.integer, int)):
                            input_payload[str(k)] = int(v)
                        else:
                            input_payload[str(k)] = v

                    model_name = _get_model_name(model)

                    from models.model_io_store import ModelIORecord
                    record = ModelIORecord(
                        ticker=ticker,
                        run_id=run_id,
                        regime=regime,
                        paradigm=paradigm,
                        model_name=model_name,
                        context="backtest",
                        input_payload=input_payload,
                        output_payload=output_payload,
                        actual_price=None,
                        predicted_at=date.isoformat() if hasattr(date, "isoformat") else str(date)
                    )
                    model_io_records.append(record)
                except Exception:
                    pass

        fold_metrics = {
            "fold": fold,
            "test_start": str(test_df.index[0].date()),
            "test_end": str(test_df.index[-1].date()),
            "adaptive": compute_all_metrics(pd.Series(adaptive_rets)),
            "buy_and_hold": compute_all_metrics(pd.Series(bh_rets)),
            "static_regression": compute_all_metrics(pd.Series(static_reg_rets)) if static_reg_rets else {},
            "static_classification": compute_all_metrics(pd.Series(static_cls_rets)) if static_cls_rets else {},
            "naive_regime_regression": compute_all_metrics(pd.Series(naive_reg_rets)),
            "router_summary": router.summary(),
        }
        fold_results.append(fold_metrics)

        all_returns["adaptive"].extend(adaptive_rets)
        all_returns["buy_and_hold"].extend(bh_rets)
        all_returns["static_regression"].extend(static_reg_rets)
        all_returns["static_classification"].extend(static_cls_rets)
        all_returns["naive_regime_regression"].extend(naive_reg_rets)

        if verbose:
            print(f"  Adaptive Sharpe: {fold_metrics['adaptive']['sharpe']:.3f}  "
                  f"BH Sharpe: {fold_metrics['buy_and_hold']['sharpe']:.3f}")

    overall = {
        strategy: compute_all_metrics(pd.Series(rets))
        for strategy, rets in all_returns.items()
        if rets
    }

    # Log run metadata
    store.log_run(
        run_id=run_id,
        ticker=ticker,
        n_folds=len(fold_results),
        start_date=fold_results[0]["test_start"] if fold_results else "",
        end_date=fold_results[-1]["test_end"] if fold_results else "",
    )
    store.close()

    # Log model I/O records in bulk
    if model_io_records:
        try:
            from models.model_io_store import ModelIOStore
            io_store = ModelIOStore()
            io_store.log_io_batch(model_io_records)
            io_store.close()
            print(f"[backtest] Successfully logged {len(model_io_records)} model I/O records to database.")
        except Exception as exc:
            print(f"[backtest] Failed to log model I/O records: {exc}")

    print("\n=== Overall Results ===")
    for strategy, metrics in overall.items():
        print(f"  {strategy:30s} Sharpe={metrics['sharpe']:.3f}  "
              f"MaxDD={metrics['max_drawdown']:.3f}  "
              f"TotalReturn={metrics['total_return']:.3f}")

    return {
        "ticker": ticker,
        "run_id": run_id,
        "profile": profile,
        "fold_plan": fold_descriptions.to_dict(orient="records"),
        "fold_results": fold_results,
        "overall": overall,
        "all_returns": all_returns,
        "all_evaluations": trainer.all_evaluations if trainer else [],
        "execution_assumptions": {
            "transaction_cost_bps": TRANSACTION_COST_BPS,
            "slippage_bps": SLIPPAGE_BPS,
            "signal_deadband": SIGNAL_DEADBAND,
            "max_gross_leverage": MAX_GROSS_LEVERAGE,
            "vol_target_annual": VOL_TARGET_ANNUAL,
            "drawdown_throttle_trigger": DRAWDOWN_THROTTLE_TRIGGER,
            "drawdown_throttle_multiplier": DRAWDOWN_THROTTLE_MULTIPLIER,
            "regime_cap_bull": REGIME_CAP_BULL,
            "regime_cap_bear": REGIME_CAP_BEAR,
            "regime_cap_highvol": REGIME_CAP_HIGHVOL,
        },
    }


def _get_signal(model, paradigm: str, row: pd.Series, feat_cols: list) -> float:
    if model is None or feat_cols is None:
        return 0.0
    try:
        X = row[feat_cols].values.reshape(1, -1)
        if np.isnan(X).any():
            return 0.0
        if paradigm == PARADIGM_REGRESSION:
            pred = model.predict(X)[0]
            if pred > SIGNAL_DEADBAND:
                return 1.0
            if pred < -SIGNAL_DEADBAND:
                return -1.0
            return 0.0
        else:
            pred_label = model.predict(X)[0]
            return _cls_label_to_signal(np.array([pred_label]))[0]
    except Exception:
        return 0.0


def _best_static_model(registry: dict, paradigm: str):
    best_model, best_sharpe, best_cols = None, -np.inf, None
    for regime_data in registry.values():
        if paradigm in regime_data:
            s = regime_data[paradigm].get("sharpe", -np.inf)
            if s > best_sharpe:
                best_sharpe = s
                best_model = regime_data[paradigm]["model"]
                best_cols = regime_data[paradigm]["feature_cols"]
    return best_model, best_cols


def _get_regime_model(registry: dict, regime: str, paradigm: str):
    if regime in registry and paradigm in registry[regime]:
        entry = registry[regime][paradigm]
        return entry["model"], paradigm, entry["feature_cols"]
    return None, paradigm, None


def _cost_rate(turnover: float) -> float:
    if turnover <= 0:
        return 0.0
    return (TRANSACTION_COST_BPS + SLIPPAGE_BPS) / 10000.0 * float(turnover)


def _apply_execution_frictions(signal: float, prev_signal: float, true_return: float) -> float:
    gross = float(signal) * float(true_return)
    turnover = abs(float(signal) - float(prev_signal))
    return gross - _cost_rate(turnover)


def _regime_exposure_cap(regime: str) -> float:
    if regime == REGIME_BULL:
        return REGIME_CAP_BULL
    if regime == REGIME_BEAR:
        return REGIME_CAP_BEAR
    if regime == REGIME_HIGHVOL:
        return REGIME_CAP_HIGHVOL
    return REGIME_CAP_HIGHVOL


def _drawdown_throttle(current_equity: float, peak_equity: float = 1.0) -> float:
    if peak_equity <= 0:
        return 1.0
    drawdown = (peak_equity - current_equity) / peak_equity
    if drawdown >= DRAWDOWN_THROTTLE_TRIGGER:
        return DRAWDOWN_THROTTLE_MULTIPLIER
    return 1.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="RELIANCE.NS")
    args = parser.parse_args()
    run_backtest(args.ticker)
