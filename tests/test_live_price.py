import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from backtest.predictor import (
    StockerPredictor,
    _fetch_latest_price_yfinance,
    _fetch_latest_price_yahoo_direct,
    _fetch_latest_price_nse,
)
from models.trainer import EncodedXGBClassifier


def test_encoded_xgb_classifier_string_labels():
    """Test that EncodedXGBClassifier properly trains and predicts on string class labels."""
    X = np.random.randn(60, 5)
    y = np.random.choice(["strong_up", "neutral", "strong_down"], size=60)
    
    clf = EncodedXGBClassifier(n_estimators=10, max_depth=2, random_state=42)
    clf.fit(X, y)
    
    preds = clf.predict(X[:5])
    assert len(preds) == 5
    for p in preds:
        assert p in ["strong_up", "neutral", "strong_down"]
        
    probas = clf.predict_proba(X[:5])
    assert probas.shape == (5, 3)
    assert np.allclose(probas.sum(axis=1), 1.0)


def test_fetch_live_price_priority():
    """Test that predictor _fetch_live_price uses live quote and tracks source."""
    predictor = StockerPredictor("RELIANCE.NS")
    
    with patch("backtest.predictor._fetch_latest_price_yfinance", return_value=2950.50):
        px, source, is_fallback = predictor._fetch_live_price(2500.0)
        assert px == 2950.50
        assert source == "yfinance_live"
        assert is_fallback is False

    with patch("backtest.predictor._fetch_latest_price_yfinance", return_value=None), \
         patch("backtest.predictor._fetch_latest_price_yahoo_direct", return_value=2952.00):
        px, source, is_fallback = predictor._fetch_live_price(2500.0)
        assert px == 2952.00
        assert source == "yahoo_chart_live"
        assert is_fallback is False

    with patch("backtest.predictor._fetch_latest_price_yfinance", return_value=None), \
         patch("backtest.predictor._fetch_latest_price_yahoo_direct", return_value=None), \
         patch("backtest.predictor._fetch_latest_price_nse", return_value=2955.00):
        px, source, is_fallback = predictor._fetch_live_price(2500.0)
        assert px == 2955.00
        assert source == "nse_quote"
        assert is_fallback is False

    with patch("backtest.predictor._fetch_latest_price_yfinance", return_value=None), \
         patch("backtest.predictor._fetch_latest_price_yahoo_direct", return_value=None), \
         patch("backtest.predictor._fetch_latest_price_nse", return_value=None):
        px, source, is_fallback = predictor._fetch_live_price(2500.0)
        assert px == 2500.0
        assert source == "historical_close"
        assert is_fallback is True
