"""
Robust multi-source data fetcher for NSE stocks.
Guarantees real data by using intelligent fallback through:
1. Local cache (bootstrap)
2. yfinance with retry & rate-limit handling
3. NSE direct API
4. Yahoo Finance direct chart API
5. Pre-computed bootstrap dataset
"""
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

from utils.config import DATA_DIR, MIN_DATA_ROWS, MAX_STALE_DAYS


class RobustNSEFetcher:
    """
    Multi-source fetcher that guarantees real OHLCV data for NSE stocks.
    Never returns synthetic data; always falls back through real sources.
    """
    
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds
    
    def __init__(self):
        self.cache_dir = DATA_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_ohlcv(
        self,
        ticker: str,
        period: str = "5y",
        force_refresh: bool = False,
        data_source: str = "yfinance",
        access_token: str = "",
    ) -> pd.DataFrame:
        """
        Get OHLCV data using intelligent multi-source fallback.
        Fetches fresh data if cache is missing, stale (>1 day old), or force_refresh is requested.
        Gracefully falls back to existing cache if offline.
        """
        cache_path = self.cache_dir / f"{ticker.replace('.', '_')}.parquet"
        cached_df: Optional[pd.DataFrame] = None
        is_cache_fresh = False

        if cache_path.exists():
            try:
                df_disk = pd.read_parquet(cache_path)
                if self._validate_quality(df_disk, ticker):
                    cached_df = df_disk
                    last_dt = pd.to_datetime(cached_df.index.max())
                    if getattr(last_dt, "tzinfo", None) is not None:
                        last_dt = last_dt.tz_convert("UTC").tz_localize(None)
                    now_utc = pd.Timestamp.now(tz="UTC").tz_localize(None)
                    age_days = (now_utc - last_dt).days
                    # Cache is considered fully fresh if it was updated for today or last business day
                    is_weekend = now_utc.dayofweek >= 5
                    fresh_threshold = 3 if is_weekend else 1
                    if age_days <= fresh_threshold and not force_refresh:
                        is_cache_fresh = True
                        print(f"[robust] Fresh cache hit for {ticker} ({len(cached_df)} rows, age: {age_days}d)")
                        return cached_df
            except Exception as e:
                print(f"[robust] Cache check failed for {ticker}: {e}")

        # Try real online sources in order of preference
        sources = []
        if (data_source or "").lower() == "upstox" and access_token:
            sources.append(("upstox", lambda t, p: self._fetch_upstox(t, p, access_token)))
        
        sources.extend([
            ("yfinance", self._fetch_yfinance),
            ("yahoo_chart", self._fetch_yahoo_chart),
            ("nse_quote_bootstrap", self._fetch_nse_bootstrap),
        ])
        
        for source_name, source_func in sources:
            try:
                print(f"[robust] Trying {source_name} for {ticker}...")
                df = source_func(ticker, period)
                if self._validate_quality(df, ticker):
                    try:
                        df.to_parquet(cache_path)
                        print(f"[robust] Saved updated cache for {ticker} ({len(df)} rows)")
                    except Exception as ce:
                        print(f"[robust] Warning: Could not write cache for {ticker}: {ce}")
                    print(f"[robust] Successfully fetched {ticker} via {source_name} ({len(df)} rows)")
                    return df
            except Exception as e:
                print(f"[robust] {source_name} failed for {ticker}: {e}")
                time.sleep(1)

        # If online fetching failed but we have a valid cached version, use it as fallback
        if cached_df is not None and len(cached_df) >= MIN_DATA_ROWS:
            print(f"[robust] Online sources unavailable; using existing cache for {ticker} ({len(cached_df)} rows)")
            return cached_df
        
        # ALL sources failed
        raise ValueError(
            f"Could not fetch real data for {ticker} from any source. "
            f"Ensure network connectivity and try again."
        )

    def _fetch_upstox(self, ticker: str, period: str, access_token: str) -> pd.DataFrame:
        """Fetch from Upstox API v2."""
        from data.upstox_client import UpstoxDataClient
        client = UpstoxDataClient(access_token)
        df = client.fetch_ohlcv(ticker, period=period)
        return self._add_returns(df)
    
    def _fetch_yfinance(self, ticker: str, period: str) -> pd.DataFrame:
        """Fetch from yfinance with retry logic and clean DataFrame formatting."""
        for attempt in range(self.MAX_RETRIES):
            try:
                # First try Ticker history which handles single ticker cleanly
                t = yf.Ticker(ticker)
                df = t.history(period=period, auto_adjust=True)
                
                # If history returned nothing, try yf.download as fallback
                if df is None or len(df) == 0:
                    df = yf.download(
                        ticker,
                        period=period,
                        auto_adjust=True,
                        progress=False,
                        threads=False,
                    )
                
                if df is not None and len(df) > 0:
                    # Clean MultiIndex or tuple columns if present
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [c[0] for c in df.columns]
                    else:
                        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                    
                    req_cols = ["Open", "High", "Low", "Close", "Volume"]
                    if all(col in df.columns for col in req_cols):
                        df = df[req_cols].copy()
                        df.index = pd.to_datetime(df.index)
                        if df.index.tz is not None:
                            df.index = df.index.tz_localize(None)
                        df = df.sort_index()
                        df = df.dropna(subset=["Close"])
                        df = df[df["Close"] > 0]
                        df = df[~df.index.duplicated(keep="last")]
                        if len(df) >= MIN_DATA_ROWS:
                            return self._add_returns(df)
            except Exception as e:
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY)
                else:
                    raise
        raise ValueError("yfinance retry exhausted or insufficient data")
    
    def _fetch_yahoo_chart(self, ticker: str, period: str) -> pd.DataFrame:
        """Fetch from Yahoo chart API directly with standard browser headers."""
        period_map = {"1y": "1y", "2y": "2y", "3y": "3y", "5y": "5y", "10y": "10y"}
        rng = period_map.get(period, "5y")
        
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "application/json,text/plain,*/*",
        }
        resp = requests.get(
            url,
            params={"range": rng, "interval": "1d"},
            timeout=15,
            headers=headers,
        )
        resp.raise_for_status()
        
        data = resp.json()
        result = data.get("chart", {}).get("result", [])
        if not result:
            raise ValueError("No data in Yahoo chart response")
        
        payload = result[0]
        ts = payload.get("timestamp", [])
        quote = payload.get("indicators", {}).get("quote", [{}])[0]
        
        if not ts or not quote:
            raise ValueError("Empty timestamp or quote in Yahoo chart response")
        
        opens = quote.get("open", [])
        highs = quote.get("high", [])
        lows = quote.get("low", [])
        closes = quote.get("close", [])
        volumes = quote.get("volume", [])
        
        idx = pd.to_datetime(ts, unit="s", utc=True)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)

        df = pd.DataFrame({
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": volumes,
        }, index=idx)
        
        df = df.dropna(subset=["Close"])
        df = df[df["Close"] > 0]
        df["Volume"] = df["Volume"].fillna(0).astype(int)
        df = df[~df.index.duplicated(keep="last")]
        return self._add_returns(df)
    
    def _fetch_nse_bootstrap(self, ticker: str, period: str) -> pd.DataFrame:
        """
        Fetch bootstrap data from NSE via static sources or pre-cached dataset.
        """
        symbol = ticker.replace(".NS", "").upper()
        
        bootstrap_file = Path(__file__).parent / "nse_bootstrap" / f"{symbol}.parquet"
        if bootstrap_file.exists():
            try:
                df = pd.read_parquet(bootstrap_file)
                return self._add_returns(df)
            except Exception:
                pass
        
        raise ValueError(f"No bootstrap dataset available for {ticker}")
    
    def _add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add percentage and log returns to OHLCV."""
        df = df.copy()
        df["pct_return"] = df["Close"].pct_change()
        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
        return df
    
    def _validate_quality(self, df: pd.DataFrame, ticker: str) -> bool:
        """Validate data quality."""
        if df is None or len(df) == 0:
            print(f"[robust] Empty DataFrame for {ticker}")
            return False

        if len(df) < MIN_DATA_ROWS:
            print(f"[robust] Insufficient rows: {len(df)} < {MIN_DATA_ROWS}")
            return False
        
        if "Close" not in df.columns or df["Close"].isna().all():
            print(f"[robust] All Close prices are NaN or missing")
            return False
        
        return True


# Global instance
_fetcher = None

def get_robust_fetcher() -> RobustNSEFetcher:
    """Singleton fetcher instance."""
    global _fetcher
    if _fetcher is None:
        _fetcher = RobustNSEFetcher()
    return _fetcher


def fetch_ohlcv_robust(
    ticker: str,
    period: str = "5y",
    force_refresh: bool = False,
    data_source: str = "yfinance",
    access_token: str = "",
) -> pd.DataFrame:
    """Convenience function to fetch OHLCV using robust fetcher."""
    fetcher = get_robust_fetcher()
    return fetcher.get_ohlcv(
        ticker=ticker,
        period=period,
        force_refresh=force_refresh,
        data_source=data_source,
        access_token=access_token,
    )
