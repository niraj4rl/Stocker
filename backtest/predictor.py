import numpy as np
import pandas as pd
import requests
from datetime import datetime, timezone
from data.ingestion import load_or_fetch
from features.engineering import build_features, get_feature_cols
from regime.detector import RegimeDetector
from models.trainer import ModelTrainer, _cls_label_to_signal
from router.adaptive import AdaptiveRouter
from utils.config import (
    REGIME_WINDOW,
    PARADIGM_REGRESSION,
    PARADIGM_CLASSIFICATION,
    TRAIN_YEARS,
    SIGNAL_DEADBAND,
    MAX_GROSS_LEVERAGE,
    VOL_TARGET_ANNUAL,
    TRADING_DAYS_YEAR,
    REGIME_CAP_BULL,
    REGIME_CAP_BEAR,
    REGIME_CAP_HIGHVOL,
    REGIME_BULL,
    REGIME_BEAR,
    REGIME_HIGHVOL,
)
from utils.metrics import compute_all_metrics
from dateutil.relativedelta import relativedelta


MODEL_DISPLAY_NAMES = {
    "xgb": "XGBoost",
    "rf": "Random Forest",
    "ridge": "Ridge Regression",
    "svc": "Support Vector (SVC)",
    "svr": "Support Vector (SVR)",
    "knn": "K-Nearest Neighbors",
    "mlp": "Neural Network (MLP)",
}


def _get_model_name(model) -> str:
    if model is None:
        return "Unknown"
    estimator = model
    if hasattr(model, "steps"):
        estimator = model.steps[-1][1]
    
    class_name = estimator.__class__.__name__.lower()
    if "xgb" in class_name:
        return "XGBoost"
    elif "randomforest" in class_name:
        return "Random Forest"
    elif "ridge" in class_name:
        return "Ridge Regression"
    elif "svc" in class_name:
        return "Support Vector (SVC)"
    elif "svr" in class_name:
        return "Support Vector (SVR)"
    elif "kneighbors" in class_name:
        return "K-Nearest Neighbors"
    elif "mlp" in class_name:
        return "Neural Network (MLP)"
    return class_name.title()


class StockerPredictor:
    def __init__(self, ticker: str, data_source: str = "yfinance", access_token: str = ""):
        self.ticker = ticker
        self.data_source = data_source
        self.access_token = access_token
        self.detector = None
        self.registry = None
        self.router = None
        self.df = None
        self.feat_cols = None
        self.is_ready = False

    def fit(self, force_refresh: bool = False):
        print(f"[predictor] Fitting Stocker for {self.ticker}...")
        raw = load_or_fetch(
            self.ticker,
            force_refresh=force_refresh,
            data_source=self.data_source,
            access_token=self.access_token,
            period="2y",
        )
        self.df = build_features(raw)

        cutoff = self.df.index[-1] - relativedelta(years=TRAIN_YEARS)
        train_df = self.df[self.df.index >= cutoff]

        self.detector = RegimeDetector()
        self.detector.fit(train_df)

        regime_labels = self.detector.predict(train_df)

        trainer = ModelTrainer()
        self.registry = trainer.train_for_fold(
            train_df, regime_labels,
            ticker=self.ticker,
        )

        self.router = AdaptiveRouter()
        self.router.build_from_registry(self.registry)

        self.feat_cols = get_feature_cols(self.df)
        self.is_ready = True
        print(f"[predictor] Ready.")

    def predict(self) -> dict:
        if not self.is_ready:
            raise RuntimeError("Call fit() before predict()")

        recent = self.df.iloc[-REGIME_WINDOW - 5:]
        regime = self.detector.predict_current(recent)

        model, paradigm, feat_cols = self.router.get_model(self.registry, regime)

        model_name = _get_model_name(model)
        if regime in self.registry and paradigm in self.registry[regime]:
            raw_name = self.registry[regime][paradigm].get("model_name")
            if raw_name:
                model_name = MODEL_DISPLAY_NAMES.get(raw_name, raw_name.title())

        regime_models = {}
        if self.registry:
            for r, winning_paradigm in self.router.lookup.items():
                if r in self.registry and winning_paradigm in self.registry[r]:
                    raw_m = self.registry[r][winning_paradigm].get("model_name")
                    m_obj = self.registry[r][winning_paradigm].get("model")
                    m_display = MODEL_DISPLAY_NAMES.get(raw_m, _get_model_name(m_obj))
                    regime_models[r] = m_display

        last_row = self.df.iloc[-1]

        current_price, price_source, price_is_fallback = self._fetch_live_price(float(last_row["Close"]))

        result = {
            "ticker": self.ticker,
            "current_price": round(current_price, 2),
            "current_price_source": price_source,
            "current_price_asof": datetime.now(timezone.utc).isoformat(),
            "current_price_is_fallback": price_is_fallback,
            "regime": regime,
            "paradigm": paradigm,
            "ml_model": model_name,
            "regime_models": regime_models,
            "routing_table": self.router.lookup.copy(),
            "prediction": None,
            "signal": "Hold",
            "confidence": None,
            "predicted_price": None,
            "predicted_return_pct": None,
            "model_validation_sharpe": None,
        }

        if model is None or feat_cols is None:
            result["error"] = "No model available for current regime"
            return result

        X = last_row[feat_cols].values.reshape(1, -1)
        if np.isnan(X).any():
            result["error"] = "Feature vector contains NaN"
            return result

        if paradigm == PARADIGM_REGRESSION:
            pred_return = float(model.predict(X)[0])
            bounds = None
            if regime in self.registry and paradigm in self.registry[regime]:
                bounds = self.registry[regime][paradigm].get("prediction_bounds")
            if bounds and len(bounds) == 2:
                pred_return = float(np.clip(pred_return, bounds[0], bounds[1]))
            pred_price = current_price * (1 + pred_return)
            signal = "Buy" if pred_return > SIGNAL_DEADBAND else ("Sell" if pred_return < -SIGNAL_DEADBAND else "Hold")
            result.update({
                "prediction": round(pred_return * 100, 4),
                "predicted_return_pct": round(pred_return * 100, 4),
                "predicted_price": round(pred_price, 2),
                "signal": signal,
            })

        else:
            pred_label = model.predict(X)[0]
            try:
                proba = model.predict_proba(X)[0]
                classes = list(model.classes_) if hasattr(model, "classes_") else []
                confidence = float(max(proba)) if len(proba) > 0 else None
            except Exception:
                confidence = None

            signal_map = {"strong_up": "Buy", "neutral": "Hold", "strong_down": "Sell"}
            signal = signal_map.get(pred_label, "Hold")
            result.update({
                "prediction": pred_label,
                "signal": signal,
                "confidence": round(confidence * 100, 1) if confidence else None,
            })

        if regime in self.registry and paradigm in self.registry[regime]:
            result["model_validation_sharpe"] = self.registry[regime][paradigm].get("sharpe")

        # Attach transparent risk-sizing context for front-end interpretation.
        recent_vol = float(self.df["pct_return"].iloc[-REGIME_WINDOW:].std() * np.sqrt(TRADING_DAYS_YEAR))
        if recent_vol > 0:
            position_size = min(MAX_GROSS_LEVERAGE, VOL_TARGET_ANNUAL / recent_vol)
        else:
            position_size = 1.0
        regime_cap = {
            REGIME_BULL: REGIME_CAP_BULL,
            REGIME_BEAR: REGIME_CAP_BEAR,
            REGIME_HIGHVOL: REGIME_CAP_HIGHVOL,
        }.get(regime, REGIME_CAP_HIGHVOL)
        position_size = min(position_size, regime_cap)
        result["position_size"] = round(float(position_size), 3)
        result["recent_vol_ann"] = round(recent_vol * 100, 2)

        if "error" not in result:
            try:
                from models.model_io_store import ModelIOStore, ModelIORecord
                # Extract input features as clean JSON dict
                raw_inputs = last_row[feat_cols].to_dict()
                input_payload = {}
                for k, v in raw_inputs.items():
                    if isinstance(v, (np.floating, float)):
                        input_payload[str(k)] = float(v)
                    elif isinstance(v, (np.integer, int)):
                        input_payload[str(k)] = int(v)
                    else:
                        input_payload[str(k)] = v

                # Extract output details
                output_payload = {}
                if paradigm == PARADIGM_REGRESSION:
                    output_payload = {
                        "predicted_return": float(pred_return),
                        "predicted_price": float(result.get("predicted_price") or 0.0),
                        "signal": result.get("signal", "Hold")
                    }
                else:
                    output_payload = {
                        "predicted_class": str(pred_label),
                        "confidence": float(result["confidence"]) if result.get("confidence") is not None else None,
                        "signal": result.get("signal", "Hold")
                    }

                model_name = _get_model_name(model)

                record = ModelIORecord(
                    ticker=self.ticker,
                    paradigm=paradigm,
                    model_name=model_name,
                    context="live_prediction",
                    input_payload=input_payload,
                    output_payload=output_payload,
                    regime=regime,
                    actual_price=None,
                    predicted_at=datetime.now().isoformat()
                )
                io_store = ModelIOStore()
                io_store.log_io(record)
                io_store.close()
            except Exception as exc:
                print(f"[predictor] Failed to log live prediction I/O: {exc}")

        return result

    def _fetch_live_price(self, fallback_price: float) -> tuple[float, str, bool]:
        """
        Fetch real-time stock quote from multiple live providers.
        Priority:
        1. Upstox LTP (if configured)
        2. yfinance fast_info / 1d intraday
        3. Yahoo Finance direct chart API
        4. NSE India direct equity quote API
        5. Historical dataset close (fallback)
        """
        if (self.data_source or "").lower() == "upstox" and self.access_token:
            try:
                from data.upstox_client import UpstoxDataClient
                client = UpstoxDataClient(self.access_token)
                px = float(client.get_live_price(self.ticker))
                if px > 0:
                    return px, "upstox_ltp", False
            except Exception as e:
                print(f"[predictor] Upstox live price failed: {e}")

        # Try yfinance fast_info / intraday
        try:
            yf_px = _fetch_latest_price_yfinance(self.ticker)
            if yf_px is not None and yf_px > 0:
                return float(yf_px), "yfinance_live", False
        except Exception as e:
            print(f"[predictor] yfinance live price failed: {e}")

        # Try Yahoo Chart API with browser headers
        try:
            yahoo_px = _fetch_latest_price_yahoo_direct(self.ticker)
            if yahoo_px is not None and yahoo_px > 0:
                return float(yahoo_px), "yahoo_chart_live", False
        except Exception as e:
            print(f"[predictor] Yahoo chart live price failed: {e}")

        # Try NSE India Quote API
        try:
            nse_px = _fetch_latest_price_nse(self.ticker)
            if nse_px is not None and nse_px > 0:
                return float(nse_px), "nse_quote", False
        except Exception as e:
            print(f"[predictor] NSE live price failed: {e}")

        # Final fallback: last available close from historical dataset
        return float(fallback_price), "historical_close", True

    def regime_history(self) -> pd.Series:
        if self.detector is None or self.df is None:
            return pd.Series()
        return self.detector.predict(self.df)

    def equity_curve_data(self) -> dict:
        if self.df is None:
            return {}
        returns = self.df["pct_return"].dropna()
        bh = (1 + returns).cumprod()
        return {"dates": bh.index.tolist(), "buy_and_hold": bh.tolist()}


def _fetch_latest_price_yfinance(ticker: str) -> float | None:
    """Fetch the latest market price using yfinance fast_info or 1d intraday data."""
    import yfinance as yf
    try:
        t = yf.Ticker(ticker)
        if hasattr(t, "fast_info") and t.fast_info is not None:
            price = getattr(t.fast_info, "last_price", None)
            if price is None and hasattr(t.fast_info, "get"):
                price = t.fast_info.get("last_price") or t.fast_info.get("regularMarketPrice")
            if price is not None and not np.isnan(price) and float(price) > 0:
                return float(price)
        
        # Try intraday 1-day bar
        df = t.history(period="1d", interval="1m", auto_adjust=True)
        if df is not None and len(df) > 0 and "Close" in df.columns:
            val = float(df["Close"].iloc[-1])
            if val > 0:
                return val
    except Exception:
        pass
    return None


def _fetch_latest_price_yahoo_direct(ticker: str) -> float | None:
    """Fetch live price directly from Yahoo Chart API with browser headers."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "application/json,text/plain,*/*",
    }
    
    # Try 1d intraday chart
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range=1d&interval=1m"
        resp = requests.get(url, headers=headers, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            result = data.get("chart", {}).get("result", [])
            if result:
                meta = result[0].get("meta", {})
                price = meta.get("regularMarketPrice")
                if price and float(price) > 0:
                    return float(price)
                indicators = result[0].get("indicators", {}).get("quote", [{}])[0]
                closes = [c for c in indicators.get("close", []) if c is not None]
                if closes and float(closes[-1]) > 0:
                    return float(closes[-1])
    except Exception:
        pass

    # Try 5d daily chart
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range=5d&interval=1d"
        resp = requests.get(url, headers=headers, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            result = data.get("chart", {}).get("result", [])
            if result:
                meta = result[0].get("meta", {})
                price = meta.get("regularMarketPrice")
                if price and float(price) > 0:
                    return float(price)
                indicators = result[0].get("indicators", {}).get("quote", [{}])[0]
                closes = [c for c in indicators.get("close", []) if c is not None]
                if closes and float(closes[-1]) > 0:
                    return float(closes[-1])
    except Exception:
        pass

    return None


def _fetch_latest_price_nse(ticker: str) -> float | None:
    """Fetch live equity price directly from NSE India API."""
    symbol = (ticker or "").upper().replace(".NS", "").replace(".BO", "")
    if not symbol:
        return None

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "application/json,text/plain,*/*",
        "Referer": "https://www.nseindia.com/",
    }
    session = requests.Session()
    try:
        session.get("https://www.nseindia.com", headers=headers, timeout=5)
        resp = session.get(
            f"https://www.nseindia.com/api/quote-equity?symbol={symbol}",
            headers=headers,
            timeout=6,
        )
        if resp.status_code == 200:
            data = resp.json()
            info = data.get("priceInfo", {})
            px = info.get("lastPrice") or info.get("close")
            if px and float(px) > 0:
                return float(px)
    except Exception:
        pass
    return None
