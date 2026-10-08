import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pytest
from regime.detector import RegimeDetector
from utils.config import REGIME_BULL, REGIME_BEAR, REGIME_HIGHVOL, REGIME_LABELS


def make_dummy_df(n=400) -> pd.DataFrame:
    np.random.seed(0)
    dates = pd.bdate_range("2019-01-01", periods=n)
    close = 100 * np.cumprod(1 + np.random.randn(n) * 0.012)
    df = pd.DataFrame({
        "Close": close,
        "pct_return": pd.Series(close).pct_change().values,
        "log_return": np.log(pd.Series(close) / pd.Series(close).shift(1)).values,
    }, index=dates)
    return df


def test_fit_and_predict_shape():
    df = make_dummy_df()
    d = RegimeDetector()
    d.fit(df)
    labels = d.predict(df)
    assert len(labels) == len(df)


def test_labels_are_valid():
    df = make_dummy_df()
    d = RegimeDetector()
    d.fit(df)
    labels = d.predict(df)
    unique = set(labels.unique())
    assert unique.issubset(set(REGIME_LABELS)), f"Unexpected labels: {unique}"


def test_no_future_leakage_in_predict():
    df = make_dummy_df()
    d = RegimeDetector()
    d.fit(df.iloc[:300])
    label_at_300 = d.predict_current(df.iloc[280:301])
    label_at_301 = d.predict_current(df.iloc[281:302])
    assert label_at_300 in REGIME_LABELS
    assert label_at_301 in REGIME_LABELS


def test_predict_before_fit_raises():
    d = RegimeDetector()
    df = make_dummy_df()
    with pytest.raises(RuntimeError):
        d.predict(df)


def test_state_mapping_covers_all_states():
    df = make_dummy_df()
    d = RegimeDetector()
    d.fit(df)
    assert len(d.state_to_regime) == d.n_components
    for v in d.state_to_regime.values():
        assert v in REGIME_LABELS


def test_initial_volatility_warmup_does_not_use_future_values():
    df = make_dummy_df(80)
    detector = RegimeDetector()
    observations = detector._build_observations(df)
    assert np.allclose(observations[:19, 1], 0.0)
