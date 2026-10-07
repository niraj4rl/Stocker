import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.nse_stocks import ALL_NSE_TICKERS, TICKER_ALIASES


def test_legacy_ticker_alias_is_not_scanned_twice():
    assert "ZOMATO.NS" not in ALL_NSE_TICKERS
    assert TICKER_ALIASES["ZOMATO.NS"] == "ETERNAL.NS"
    assert "ETERNAL.NS" in ALL_NSE_TICKERS
