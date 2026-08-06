"""
stocker - Unified CLI and Web App Runner

Usage:
  python run.py                          # Launch web app dashboard & open in browser
  python run.py --mode ui                # Launch web app dashboard explicitly
  python run.py --mode live --ticker INFY.NS   # Run CLI live prediction for a stock
  python run.py --mode fast --ticker RELIANCE.NS # Run fast backtest
  python run.py --mode full --ticker TCS.NS     # Run full 5-fold backtest
"""
import argparse
import sys
import os
import time
import webbrowser
import threading
from pathlib import Path

# Ensure root directory is on Python path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def open_browser_delayed(url: str, delay_seconds: float = 1.2):
    def _open():
        time.sleep(delay_seconds)
        try:
            webbrowser.open(url)
        except Exception:
            pass
    t = threading.Thread(target=_open, daemon=True)
    t.start()


def start_ui(host: str = "127.0.0.1", port: int = 8000, reload: bool = True, open_browser: bool = True):
    import uvicorn

    url = f"http://{host}:{port}/app"
    print("\n" + "=" * 60)
    print("  🚀 Stocker - Regime-Aware Stock Prediction System")
    print("=" * 60)
    print(f"  • Web UI Dashboard:  {url}")
    print(f"  • Landing Page:      http://{host}:{port}/")
    print(f"  • API Docs (Swagger): http://{host}:{port}/docs")
    print(f"  • Server running on: http://{host}:{port}")
    print("=" * 60 + "\n")

    if open_browser:
        open_browser_delayed(url, delay_seconds=1.2)

    uvicorn.run("ui.app:app", host=host, port=port, reload=reload)


def run_live_cli(ticker: str, force_refresh: bool = False):
    from backtest.predictor import StockerPredictor
    from utils.config import DATA_SOURCE, UPSTOX_ACCESS_TOKEN

    print(f"\n[stocker] Running live prediction for {ticker}...")
    predictor = StockerPredictor(
        ticker,
        data_source=DATA_SOURCE,
        access_token=UPSTOX_ACCESS_TOKEN,
    )
    predictor.fit(force_refresh=force_refresh)
    result = predictor.predict()

    print("\n" + "=" * 55)
    print(f"  PREDICTION REPORT: {ticker}")
    print("=" * 55)
    print(f"  Current Live Price: INR {result.get('current_price', 0):,.2f} ({result.get('current_price_source', 'N/A')})")
    print(f"  Detected Regime:    {result.get('regime', 'N/A')}")
    print(f"  Strategy Paradigm:  {result.get('paradigm', 'N/A')}")
    print(f"  ML Specialist:      {result.get('ml_model', 'N/A')}")
    print(f"  Signal:             {result.get('signal', 'N/A').upper()}")

    if result.get("paradigm") == "regression":
        ret = result.get("predicted_return_pct", 0)
        target = result.get("predicted_price", 0)
        print(f"  Expected Return:    {'+' if ret > 0 else ''}{ret:.3f}%")
        print(f"  Target Price:       INR {target:,.2f}")
    else:
        print(f"  Direction:          {result.get('prediction', 'N/A')}")
        print(f"  Model Confidence:   {result.get('confidence', 'N/A')}%")

    print("=" * 55 + "\n")


def run_backtest_cli(ticker: str, mode: str = "fast", verbose: bool = True):
    from backtest.engine import run_backtest
    from utils.config import DATA_SOURCE, UPSTOX_ACCESS_TOKEN

    print(f"\n[stocker] Running {mode} backtest for {ticker}...")
    run_backtest(
        ticker=ticker,
        verbose=verbose,
        profile=mode,
        data_source=DATA_SOURCE,
        access_token=UPSTOX_ACCESS_TOKEN,
    )


def main():
    parser = argparse.ArgumentParser(
        description="stocker — Regime-Aware Adaptive Stock Prediction",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["ui", "live", "fast", "full"],
        default="ui",
        help="Operating mode:\n"
             "  ui   - Launch Web UI Dashboard (default)\n"
             "  live - Run live prediction in terminal\n"
             "  fast - Run fast single-fold backtest\n"
             "  full - Run full 5-fold backtest",
    )
    parser.add_argument(
        "--ticker",
        default="RELIANCE.NS",
        help="NSE ticker symbol (e.g. RELIANCE.NS, TCS.NS, INFY.NS)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind UI server to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind UI server to (default: 8000)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open browser on startup",
    )
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="Disable uvicorn auto-reload",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force download fresh market data from online sources",
    )

    args = parser.parse_args()

    if args.mode == "ui":
        start_ui(
            host=args.host,
            port=args.port,
            reload=not args.no_reload,
            open_browser=not args.no_browser,
        )
    elif args.mode == "live":
        run_live_cli(ticker=args.ticker, force_refresh=args.refresh)
    elif args.mode in ("fast", "full"):
        run_backtest_cli(ticker=args.ticker, mode=args.mode)


if __name__ == "__main__":
    main()
