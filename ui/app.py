import sys
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta, timezone
import difflib
import requests

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.engine import run_backtest
from backtest.predictor import (
    StockerPredictor,
    _fetch_latest_price_yfinance,
    _fetch_latest_price_yahoo_direct,
    _fetch_latest_price_nse,
)
from data.nse_stocks import ALL_NSE_TICKERS
from data.source_validator import get_valid_tickers
from models.model_scorecard import ScorecardStore
from regime.detector import get_regime_stats
from utils.config import (
    DATA_SOURCE,
    PREDICTOR_REFRESH_MINUTES,
    REGIME_BEAR,
    REGIME_BULL,
    REGIME_HIGHVOL,
    UPSTOX_ACCESS_TOKEN,
)
from utils.export import (
    export_leaderboard_csv,
    export_model_comparison_csv,
    export_regime_analysis_csv,
    get_statistical_comparison,
)


REGIME_COLORS = {
    REGIME_BULL: "#7CC9B6",
    REGIME_BEAR: "#F2A9A1",
    REGIME_HIGHVOL: "#F5C99B",
}


class LivePredictionRequest(BaseModel):
    ticker: str = Field(..., examples=["RELIANCE.NS"])
    force_refresh: bool = Field(default=False, description="Force re-fetch of latest market data from online providers")


class BacktestRequest(BaseModel):
    ticker: str = Field(..., examples=["RELIANCE.NS"])
    mode: str = Field(default="fast", examples=["fast", "full"])


class ExportRequest(BaseModel):
    ticker: Optional[str] = None


app = FastAPI(title="stocker API", version="2.0.0")

# Lightweight in-process cache to prevent retraining on every live request.
_predictor_cache: dict[str, dict] = {}
_NSE_TICKER_SET = set(ALL_NSE_TICKERS)
_NSE_BASE_TO_TICKER = {t.replace(".NS", ""): t for t in ALL_NSE_TICKERS}

# Filter to only tickers with working data sources
print("[app] Validating data sources for NSE tickers...")
_validation = get_valid_tickers(ALL_NSE_TICKERS)
_VALID_TICKERS = _validation.get("valid", ALL_NSE_TICKERS)
_INVALID_TICKERS = _validation.get("invalid", [])
if _INVALID_TICKERS:
    print(f"[app] Warning: {len(_INVALID_TICKERS)} tickers have no reliable data source")
    print(f"[app] Valid tickers: {len(_VALID_TICKERS)} of {len(ALL_NSE_TICKERS)}")


def _safe_log_search(query_text: str, endpoint: str, ticker: str = "") -> dict:
    """Best-effort search logging; never let DB issues block API behavior."""
    try:
        store = ScorecardStore()
        meta = store.log_search(query_text=query_text, endpoint=endpoint, ticker=ticker)
        store.close()
        return meta
    except Exception as exc:
        print(f"[app] Search logging skipped ({endpoint}): {exc}")
        return {"force_refresh": False, "recent_search_count": 0}


def _open_scorecard_store() -> Optional[ScorecardStore]:
    """Open scorecard store if available, else return None."""
    try:
        return ScorecardStore()
    except Exception as exc:
        print(f"[app] Scorecard DB unavailable: {exc}")
        return None


def _normalize_ticker(raw: str) -> str:
    t = (raw or "").strip().upper().replace(" ", "").replace("-", "")
    if not t:
        return t

    if t in _NSE_TICKER_SET:
        return t

    if not t.endswith(".NS"):
        candidate = f"{t}.NS"
        if candidate in _NSE_TICKER_SET:
            return candidate

    base = t.replace(".NS", "")
    if base in _NSE_BASE_TO_TICKER:
        return _NSE_BASE_TO_TICKER[base]

    # Handle common user spelling variants, e.g. ASIANPAINTS -> ASIANPAINT.
    if base.endswith("S") and base[:-1] in _NSE_BASE_TO_TICKER:
        return _NSE_BASE_TO_TICKER[base[:-1]]

    close = difflib.get_close_matches(base, _NSE_BASE_TO_TICKER.keys(), n=1, cutoff=0.88)
    if close:
        return _NSE_BASE_TO_TICKER[close[0]]

    return f"{base}.NS"


def _suggest_tickers(raw: str, limit: int = 5) -> list[str]:
    """Suggest valid NSE symbols for user-typed ticker text."""
    base = (raw or "").strip().upper().replace(".NS", "")
    if not base:
        return []

    universe = list(_NSE_BASE_TO_TICKER.keys())
    prefix_like = [b for b in universe if base[:5] and b.startswith(base[:5])]
    contains_like = [b for b in universe if base[:4] and base[:4] in b]
    close = difflib.get_close_matches(base, universe, n=limit * 2, cutoff=0.55)

    ordered: list[str] = []
    for candidate in prefix_like + contains_like + close:
        if candidate not in ordered:
            ordered.append(candidate)

    return [_NSE_BASE_TO_TICKER[b] for b in ordered[:limit]]

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def landing() -> FileResponse:
    """Serve the landing page"""
    return FileResponse(static_dir / "landing.html")


@app.get("/app")
def dashboard() -> FileResponse:
    """Serve the main dashboard"""
    return FileResponse(static_dir / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/tickers")
def get_tickers() -> dict:
    # Only return tickers with known working data sources
    quick_picks = [t for t in ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "SBIN.NS", "ITC.NS"] if t in _VALID_TICKERS]
    return {
        "tickers": _VALID_TICKERS,
        "default": quick_picks[0] if quick_picks else (_VALID_TICKERS[0] if _VALID_TICKERS else "RELIANCE.NS"),
        "quick_picks": quick_picks,
        "note": f"Showing {len(_VALID_TICKERS)} reliable tickers out of {len(ALL_NSE_TICKERS)} total.",
    }


@app.get("/api/live-price/{ticker}")
def get_live_price(ticker: str) -> dict:
    """Fast live price quote endpoint without running model inference."""
    t = _normalize_ticker(ticker)
    now_iso = datetime.now(timezone.utc).isoformat()

    if (DATA_SOURCE or "").lower() == "upstox" and UPSTOX_ACCESS_TOKEN:
        try:
            from data.upstox_client import UpstoxDataClient
            client = UpstoxDataClient(UPSTOX_ACCESS_TOKEN)
            px = float(client.get_live_price(t))
            if px > 0:
                return {"ticker": t, "price": round(px, 2), "source": "upstox_ltp", "as_of": now_iso, "is_fallback": False}
        except Exception:
            pass

    try:
        yf_px = _fetch_latest_price_yfinance(t)
        if yf_px is not None and yf_px > 0:
            return {"ticker": t, "price": round(yf_px, 2), "source": "yfinance_live", "as_of": now_iso, "is_fallback": False}
    except Exception:
        pass

    try:
        yahoo_px = _fetch_latest_price_yahoo_direct(t)
        if yahoo_px is not None and yahoo_px > 0:
            return {"ticker": t, "price": round(yahoo_px, 2), "source": "yahoo_chart_live", "as_of": now_iso, "is_fallback": False}
    except Exception:
        pass

    try:
        nse_px = _fetch_latest_price_nse(t)
        if nse_px is not None and nse_px > 0:
            return {"ticker": t, "price": round(nse_px, 2), "source": "nse_quote", "as_of": now_iso, "is_fallback": False}
    except Exception:
        pass

    raise HTTPException(status_code=404, detail=f"Live price unavailable for {t}")


@app.post("/api/live-prediction")
def live_prediction(payload: LivePredictionRequest) -> dict:
    ticker = _normalize_ticker(payload.ticker)
    now = datetime.utcnow()
    cache_hit = False
    fitted_at = None
    force_refresh = payload.force_refresh

    try:
        search_meta = _safe_log_search(query_text=ticker, endpoint="live_prediction", ticker=ticker)
        if search_meta.get("force_refresh"):
            force_refresh = True

        cache_entry = _predictor_cache.get(ticker)
        if (
            not force_refresh
            and cache_entry
            and cache_entry.get("predictor") is not None
            and cache_entry.get("fitted_at") is not None
            and now - cache_entry["fitted_at"] < timedelta(minutes=PREDICTOR_REFRESH_MINUTES)
        ):
            predictor = cache_entry["predictor"]
            fitted_at = cache_entry["fitted_at"]
            cache_hit = True
        else:
            predictor = StockerPredictor(
                ticker,
                data_source=DATA_SOURCE,
                access_token=UPSTOX_ACCESS_TOKEN,
            )
            predictor.fit(force_refresh=force_refresh)
            fitted_at = now
            _predictor_cache[ticker] = {
                "predictor": predictor,
                "fitted_at": fitted_at,
            }

        result = predictor.predict()
    except Exception as exc:
        msg = str(exc)
        if ticker not in _VALID_TICKERS:
            suggestions = _suggest_tickers(payload.ticker)
            if suggestions:
                error = (
                    f"{ticker} is not a valid/available NSE symbol. "
                    f"Did you mean: {', '.join(suggestions)}"
                )
            else:
                error = (
                    f"{ticker} does not have reliable data available. "
                    f"Use one of: {', '.join(_VALID_TICKERS[:5])}..."
                )
            raise HTTPException(status_code=400, detail=error) from exc
        if "No data returned for ticker" in msg or "Insufficient history" in msg or "Stale data" in msg:
            raise HTTPException(status_code=400, detail=msg) from exc
        raise HTTPException(status_code=500, detail=f"Failed to load {ticker}: {exc}") from exc

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    df = predictor.df
    regime_history = predictor.regime_history()

    price_window = df.iloc[-252:]
    regime_window = regime_history.iloc[-252:]

    chart_points = []
    for dt, row in price_window.iterrows():
        chart_points.append({
            "date": dt.strftime("%Y-%m-%d"),
            "close": round(float(row["Close"]), 2),
            "regime": str(regime_window.loc[dt]) if dt in regime_window.index else None,
        })

    regime_stats = []
    try:
        stats_df = get_regime_stats(df.iloc[-756:], regime_history.iloc[-756:])
        regime_stats = stats_df.to_dict(orient="records")
    except Exception:
        regime_stats = []

    last_dt = pd.to_datetime(df.index.max())
    if getattr(last_dt, "tzinfo", None) is not None:
        last_dt = last_dt.tz_convert("UTC").tz_localize(None)
    now_utc = datetime.utcnow()
    data_age_days = int((now_utc - last_dt).days)
    data_as_of_date = last_dt.strftime("%Y-%m-%d")
    stale_data_warning = bool(data_age_days > 4)

    return {
        "result": result,
        "cache": {
            "hit": cache_hit,
            "fitted_at": fitted_at.isoformat() if fitted_at else None,
            "refresh_minutes": PREDICTOR_REFRESH_MINUTES,
            "force_refresh": force_refresh,
        },
        "chart": {
            "points": chart_points,
            "regime_colors": REGIME_COLORS,
        },
        "regime_stats": regime_stats,
        "data_as_of_date": data_as_of_date,
        "historical_data_date": data_as_of_date,
        "live_quote_time": datetime.now().strftime("%I:%M:%S %p"),
        "data_age_days": data_age_days,
        "stale_data_warning": stale_data_warning,
    }


@app.post("/api/backtest")
def full_backtest(payload: BacktestRequest) -> dict:
    ticker = _normalize_ticker(payload.ticker)
    mode = (payload.mode or "fast").strip().lower()
    try:
        _safe_log_search(query_text=f"{ticker}:{mode}", endpoint="backtest", ticker=ticker)
        results = run_backtest(
            ticker,
            verbose=False,
            profile=mode,
            data_source=DATA_SOURCE,
            access_token=UPSTOX_ACCESS_TOKEN,
        )
    except Exception as exc:
        msg = str(exc)
        if "No data returned for ticker" in msg or "Insufficient history" in msg or "Stale data" in msg:
            raise HTTPException(status_code=400, detail=msg) from exc
        if "profile must be" in msg:
            raise HTTPException(status_code=400, detail=msg) from exc
        raise HTTPException(status_code=500, detail=f"Backtest failed: {exc}") from exc
    return results


@app.get("/api/analysis")
def analysis(ticker: Optional[str] = None) -> dict:
    _safe_log_search(query_text=ticker or "ALL", endpoint="analysis", ticker=ticker or "")

    store = _open_scorecard_store()
    if store is None:
        return {
            "has_data": False,
            "scores": [],
            "leaderboard": [],
            "stats": {},
            "tickers": [],
            "recent_searches": [],
            "message": "Analysis storage is unavailable. Start PostgreSQL and configure DB_* variables to enable all-stocks analysis.",
        }

    try:
        all_scores = store.get_all_scores(ticker)
        leaderboard = store.get_leaderboard(ticker)
        recent_searches = store.get_recent_searches(limit=20)
        store.close()
    except Exception as exc:
        return {
            "has_data": False,
            "scores": [],
            "leaderboard": [],
            "stats": {},
            "tickers": [],
            "recent_searches": [],
            "message": (
                "Could not load model analysis data. Ensure PostgreSQL is running and DB_* "
                f"environment variables are configured. Error: {exc}"
            ),
        }

    if not all_scores:
        return {
            "has_data": False,
            "scores": [],
            "leaderboard": [],
            "stats": {},
            "message": "No analysis data yet. Run a full backtest first.",
        }

    scores_df = pd.DataFrame(all_scores)
    tickers = sorted(scores_df["ticker"].dropna().unique().tolist()) if "ticker" in scores_df.columns else []
    selected_ticker = ticker.strip().upper() if ticker else None
    stat_results = get_statistical_comparison(selected_ticker)

    return {
        "has_data": True,
        "scores": all_scores,
        "leaderboard": leaderboard,
        "stats": stat_results,
        "tickers": tickers,
        "recent_searches": recent_searches,
    }


@app.get("/api/recent-searches")
def recent_searches() -> dict:
    store = _open_scorecard_store()
    if store is None:
        return {
            "rows": [],
            "count": 0,
            "warning": "Search history is unavailable because analysis storage is not connected.",
        }

    try:
        rows = store.get_recent_searches(limit=20)
        store.close()
        return {"rows": rows, "count": len(rows)}
    except Exception as exc:
        return {"rows": [], "count": 0, "warning": f"Failed to load recent searches: {exc}"}


def _fetch_nse_quote_fallback(ticker: str) -> Optional[float]:
    symbol = (ticker or "").upper().replace(".NS", "")
    if not symbol:
        return None
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json,text/plain,*/*",
        "Referer": "https://www.nseindia.com/",
    }
    try:
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=8)
        resp = session.get(
            "https://www.nseindia.com/api/quote-equity",
            params={"symbol": symbol},
            headers=headers,
            timeout=8,
        )
        resp.raise_for_status()
        info = resp.json().get("priceInfo", {})
        px = info.get("lastPrice") or info.get("close")
        return float(px) if px is not None else None
    except Exception:
        return None


@app.post("/api/export/model-comparison")
def export_model(payload: ExportRequest) -> dict:
    path = export_model_comparison_csv(payload.ticker)
    if not path:
        raise HTTPException(status_code=404, detail="No data to export")
    return {"path": path}


@app.post("/api/export/leaderboard")
def export_leaderboard(payload: ExportRequest) -> dict:
    path = export_leaderboard_csv(payload.ticker)
    if not path:
        raise HTTPException(status_code=404, detail="No data to export")
    return {"path": path}


@app.post("/api/export/regime-analysis")
def export_regime(payload: ExportRequest) -> dict:
    path = export_regime_analysis_csv(payload.ticker)
    if not path:
        raise HTTPException(status_code=404, detail="No data to export")
    return {"path": path}
