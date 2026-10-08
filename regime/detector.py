import numpy as np
import pandas as pd
from hmmlearn import hmm
from scipy import stats
from utils.config import (
    REGIME_WINDOW, TRADING_DAYS_YEAR,
    REGIME_BULL, REGIME_BEAR, REGIME_HIGHVOL,
)


class RegimeDetector:
    def __init__(self, n_components: int = 3):
        self.n_components = n_components
        self.model = None
        self.state_to_regime = {}
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> "RegimeDetector":
        obs = self._build_observations(df)
        obs_clean = obs[~np.isnan(obs).any(axis=1)]

        try:
            self.model = hmm.GaussianHMM(
                n_components=self.n_components,
                covariance_type="full",
                n_iter=200,
                min_covar=1e-6,
                random_state=42,
            )
            self.model.fit(obs_clean)
        except Exception:
            # Fallback for ill-conditioned covariance estimates on noisy windows.
            self.model = hmm.GaussianHMM(
                n_components=self.n_components,
                covariance_type="diag",
                n_iter=200,
                min_covar=1e-5,
                random_state=42,
            )
            self.model.fit(obs_clean)

        self.state_to_regime = self._map_states_to_regimes(obs_clean)
        self.is_fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if not self.is_fitted:
            raise RuntimeError("RegimeDetector must be fitted before predict()")
        # HMM ``predict`` uses a backward pass and therefore lets later
        # observations influence earlier labels. Use a filtered rule for the
        # live label instead: every threshold is derived from prior bars only.
        returns = df["log_return"].astype(float)
        trailing_return = returns.rolling(21, min_periods=5).mean()
        volatility = returns.rolling(REGIME_WINDOW, min_periods=10).std()
        prior_vol_median = volatility.shift(1).expanding(min_periods=20).median()
        high_vol = volatility > (prior_vol_median * 1.5)
        result = pd.Series(REGIME_HIGHVOL, index=df.index)
        result.loc[~high_vol & trailing_return.ge(0)] = REGIME_BULL
        result.loc[~high_vol & trailing_return.lt(0)] = REGIME_BEAR
        return result.fillna(REGIME_HIGHVOL)

    def predict_current(self, recent_df: pd.DataFrame) -> str:
        labels = self.predict(recent_df)
        return labels.iloc[-1]

    def _build_observations(self, df: pd.DataFrame) -> np.ndarray:
        log_ret = df["log_return"].values
        realized_vol = (
            pd.Series(log_ret)
            .rolling(REGIME_WINDOW)
            .std()
            .fillna(0)
            .values
            * np.sqrt(TRADING_DAYS_YEAR)
        )
        obs = np.column_stack([log_ret, realized_vol])
        obs = np.nan_to_num(obs, nan=0.0)
        return obs

    def _map_states_to_regimes(self, obs: np.ndarray) -> dict:
        states = self.model.predict(obs)
        vols = obs[:, 1]
        rets = obs[:, 0]

        state_stats = {}
        for s in range(self.n_components):
            mask = states == s
            if mask.sum() == 0:
                state_stats[s] = {"vol": 0, "ret": 0}
            else:
                state_stats[s] = {
                    "vol": float(np.mean(vols[mask])),
                    "ret": float(np.mean(rets[mask])),
                }

        vol_median = np.median([v["vol"] for v in state_stats.values()])
        mapping = {}
        high_vol_state = max(state_stats, key=lambda s: state_stats[s]["vol"])
        mapping[high_vol_state] = REGIME_HIGHVOL

        remaining = [s for s in state_stats if s != high_vol_state]
        for s in remaining:
            if state_stats[s]["ret"] >= 0:
                mapping[s] = REGIME_BULL
            else:
                mapping[s] = REGIME_BEAR

        for s in range(self.n_components):
            if s not in mapping:
                mapping[s] = REGIME_HIGHVOL
        return mapping


def label_regimes_expanding(df: pd.DataFrame, min_train: int = 252) -> pd.Series:
    labels = pd.Series(REGIME_HIGHVOL, index=df.index)
    detector = RegimeDetector()

    for i in range(min_train, len(df)):
        train_slice = df.iloc[:i]
        if len(train_slice) < min_train:
            continue
        if not detector.is_fitted or i % 21 == 0:
            try:
                detector.fit(train_slice)
            except Exception:
                continue
        try:
            labels.iloc[i] = detector.predict_current(df.iloc[max(0, i - REGIME_WINDOW):i + 1])
        except Exception:
            pass
    return labels


def get_regime_stats(df: pd.DataFrame, regime_labels: pd.Series) -> pd.DataFrame:
    df = df.copy()
    df["regime"] = regime_labels
    stats = []
    for regime in [REGIME_BULL, REGIME_BEAR, REGIME_HIGHVOL]:
        mask = df["regime"] == regime
        sub = df[mask]
        if len(sub) == 0:
            stats.append({
                "regime": regime,
                "count": 0,
                "pct_of_days": 0.0,
                "mean_daily_return": None,
                "mean_vol_ann": None,
                "status": "insufficient",
            })
            continue
        stats.append({
            "regime": regime,
            "count": int(mask.sum()),
            "pct_of_days": round(mask.mean() * 100, 1),
            "mean_daily_return": round(sub["pct_return"].mean() * 100, 3),
            "mean_vol_ann": round(sub["log_return"].std() * np.sqrt(TRADING_DAYS_YEAR) * 100, 1),
            "status": "ok",
        })
    return pd.DataFrame(stats)
