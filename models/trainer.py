"""
Enhanced model trainer that logs all candidate model evaluations
to the ScorecardStore, not just the winner.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import Ridge
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
from utils.config import (
    REGIME_BULL, REGIME_BEAR, REGIME_HIGHVOL, REGIME_LABELS,
    PARADIGM_REGRESSION, PARADIGM_CLASSIFICATION,
)
from utils.metrics import (
    sharpe_ratio, compute_regression_metrics, compute_classification_metrics,
    compute_trading_metrics,
)
from features.engineering import get_feature_cols, build_classification_target
from models.model_scorecard import ModelScorecard, ScorecardStore
from sklearn.base import BaseEstimator, ClassifierMixin
import uuid


class EncodedXGBClassifier(BaseEstimator, ClassifierMixin):
    """
    XGBClassifier wrapper that safely encodes arbitrary string class labels into integers
    during fit, and maps predictions back to original class strings.
    """
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        random_state: int = 42,
        verbosity: int = 0,
        n_jobs: int = -1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.verbosity = verbosity
        self.n_jobs = n_jobs
        self.classes_ = None
        self.model_ = None

    def fit(self, X, y):
        self.classes_, y_int = np.unique(y, return_inverse=True)
        self.model_ = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            verbosity=self.verbosity,
            n_jobs=self.n_jobs,
            eval_metric="mlogloss",
        )
        self.model_.fit(X, y_int)
        return self

    def predict(self, X):
        if self.model_ is None or self.classes_ is None:
            raise RuntimeError("EncodedXGBClassifier is not fitted")
        y_int = self.model_.predict(X)
        return self.classes_[y_int]

    def predict_proba(self, X):
        if self.model_ is None:
            raise RuntimeError("EncodedXGBClassifier is not fitted")
        return self.model_.predict_proba(X)


class ModelTrainer:
    def __init__(
        self,
        scorecard_store: ScorecardStore = None,
        xgb_estimators: int = 200,
        include_svc: bool = True,
        include_svr: bool = True,
        include_knn: bool = True,
        include_mlp: bool = True,
        include_xgb: bool = True,
        include_rf: bool = True,
        include_ridge: bool = True,
    ):
        self.registry = {}
        self.store = scorecard_store
        self.xgb_estimators = xgb_estimators
        self.include_svc = include_svc
        self.include_svr = include_svr
        self.include_knn = include_knn
        self.include_mlp = include_mlp
        self.include_xgb = include_xgb
        self.include_rf = include_rf
        self.include_ridge = include_ridge
        self.all_evaluations = []  # flat list of all model evaluations

    def train_for_fold(
        self,
        df_train: pd.DataFrame,
        regime_labels: pd.Series,
        ticker: str = "",
        run_id: str = "",
        fold: int = 0,
    ) -> dict:
        self.registry = {}
        df_train = df_train.copy()
        df_train["regime"] = regime_labels.reindex(df_train.index)

        if not run_id:
            run_id = str(uuid.uuid4())[:8]

        for regime in REGIME_LABELS:
            mask = df_train["regime"] == regime
            regime_df = df_train[mask].dropna()
            if len(regime_df) < 30:
                print(f"[models] Not enough data for regime {regime} ({len(regime_df)} rows), skipping")
                continue

            feat_cols = get_feature_cols(regime_df)
            X = regime_df[feat_cols].values
            y_reg = regime_df["target_pct_return"].values
            y_cls = build_classification_target(regime_df).values

            val_size = max(int(len(regime_df) * 0.2), 20)
            X_tr, X_val = X[:-val_size], X[-val_size:]
            y_reg_tr, y_reg_val = y_reg[:-val_size], y_reg[-val_size:]
            y_cls_tr, y_cls_val = y_cls[:-val_size], y_cls[-val_size:]
            val_returns = regime_df["target_pct_return"].iloc[-val_size:].values

            # Train & evaluate ALL regression models
            reg_results = self._train_regression_all(
                X_tr, y_reg_tr, X_val, y_reg_val,
                ticker=ticker, run_id=run_id, fold=fold, regime=regime,
                train_samples=len(X_tr), val_samples=val_size,
            )

            # Train & evaluate ALL classification models
            cls_results = self._train_classification_all(
                X_tr, y_cls_tr, X_val, y_cls_val, val_returns,
                ticker=ticker, run_id=run_id, fold=fold, regime=regime,
                train_samples=len(X_tr), val_samples=val_size,
            )

            # Pick best per paradigm
            best_reg = max(reg_results, key=lambda x: x["sharpe"]) if reg_results else None
            best_cls = max(cls_results, key=lambda x: x["sharpe"]) if cls_results else None

            reg_sharpe = best_reg["sharpe"] if best_reg else -999
            cls_sharpe = best_cls["sharpe"] if best_cls else -999

            # Mark winners
            if best_reg:
                best_reg["scorecard"].is_winner = True
            if best_cls:
                best_cls["scorecard"].is_winner = True

            # Log all scorecards to store
            scorecards_to_log = []
            for r in reg_results + cls_results:
                sc = r["scorecard"]
                self.all_evaluations.append(sc)
                scorecards_to_log.append(sc)
            if self.store and scorecards_to_log:
                self.store.log_scorecards(scorecards_to_log)

            self.registry[regime] = {
                PARADIGM_REGRESSION: {
                    "model": best_reg["model"] if best_reg else None,
                    "model_name": best_reg["name"] if best_reg else "unknown",
                    "sharpe": reg_sharpe,
                    "feature_cols": feat_cols,
                    "prediction_bounds": self._prediction_bounds(y_reg),
                },
                PARADIGM_CLASSIFICATION: {
                    "model": best_cls["model"] if best_cls else None,
                    "model_name": best_cls["name"] if best_cls else "unknown",
                    "sharpe": cls_sharpe,
                    "feature_cols": feat_cols,
                },
            }
            print(f"[models] {regime} — reg sharpe={reg_sharpe:.3f}, cls sharpe={cls_sharpe:.3f}")
        return self.registry

    def _train_regression_all(self, X_tr, y_tr, X_val, y_val, **meta):
        candidates = self._build_regression_candidates()

        results = []
        for name, model in candidates.items():
            try:
                model.fit(X_tr, y_tr)
                preds = model.predict(X_val)
                signals = np.sign(preds)
                strategy_returns = signals * y_val

                trading = compute_trading_metrics(strategy_returns)
                reg_metrics = compute_regression_metrics(y_val, preds)

                sc = ModelScorecard(
                    ticker=meta.get("ticker", ""),
                    run_id=meta.get("run_id", ""),
                    fold=meta.get("fold", 0),
                    regime=meta.get("regime", ""),
                    paradigm=PARADIGM_REGRESSION,
                    model_name=name,
                    unified_score=trading["sharpe"],
                    train_samples=meta.get("train_samples", 0),
                    val_samples=meta.get("val_samples", 0),
                    **trading,
                    **reg_metrics,
                )

                results.append({
                    "model": model,
                    "name": name,
                    "sharpe": trading["sharpe"],
                    "scorecard": sc,
                })
            except Exception as e:
                print(f"[models] regression {name} failed: {e}")
        return results

    def _train_classification_all(self, X_tr, y_cls_tr, X_val, y_cls_val,
                                   val_returns, **meta):
        candidates = self._build_classification_candidates()

        results = []
        for name, model in candidates.items():
            try:
                unique = np.unique(y_cls_tr)
                if len(unique) < 2:
                    continue
                model.fit(X_tr, y_cls_tr)
                preds = model.predict(X_val)
                signals = _cls_label_to_signal(preds)
                strategy_returns = signals * val_returns

                trading = compute_trading_metrics(strategy_returns)
                cls_metrics = compute_classification_metrics(y_cls_val, preds)

                sc = ModelScorecard(
                    ticker=meta.get("ticker", ""),
                    run_id=meta.get("run_id", ""),
                    fold=meta.get("fold", 0),
                    regime=meta.get("regime", ""),
                    paradigm=PARADIGM_CLASSIFICATION,
                    model_name=name,
                    unified_score=trading["sharpe"],
                    train_samples=meta.get("train_samples", 0),
                    val_samples=meta.get("val_samples", 0),
                    **trading,
                    **cls_metrics,
                )

                results.append({
                    "model": model,
                    "name": name,
                    "sharpe": trading["sharpe"],
                    "scorecard": sc,
                })
            except Exception as e:
                print(f"[models] classification {name} failed: {e}")
        return results

    def _build_regression_candidates(self) -> dict:
        candidates = {}
        if self.include_xgb:
            candidates["xgb"] = xgb.XGBRegressor(
                n_estimators=self.xgb_estimators,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0,
                n_jobs=-1,
            )
        if self.include_rf:
            candidates["rf"] = RandomForestRegressor(
                n_estimators=200,
                max_depth=6,
                random_state=42,
                n_jobs=-1,
            )
        if self.include_ridge:
            candidates["ridge"] = Pipeline([
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ])
        if self.include_svr:
            candidates["svr"] = Pipeline([
                ("scaler", StandardScaler()),
                ("model", SVR(kernel="rbf", C=1.0, epsilon=0.01)),
            ])
        if self.include_knn:
            candidates["knn"] = Pipeline([
                ("scaler", StandardScaler()),
                ("model", KNeighborsRegressor(n_neighbors=5, weights="distance", n_jobs=-1)),
            ])
        if self.include_mlp:
            candidates["mlp"] = Pipeline([
                ("scaler", StandardScaler()),
                ("model", MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    alpha=0.0005,
                    learning_rate_init=0.001,
                    max_iter=500,
                    early_stopping=True,
                    n_iter_no_change=20,
                    random_state=42,
                )),
            ])
        return candidates

    def _prediction_bounds(self, y: np.ndarray) -> tuple[float, float]:
        finite = np.asarray(y, dtype=float)
        finite = finite[np.isfinite(finite)]
        if len(finite) == 0:
            return -0.05, 0.05

        lower = float(np.nanquantile(finite, 0.01))
        upper = float(np.nanquantile(finite, 0.99))

        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            lower = float(np.nanmin(finite))
            upper = float(np.nanmax(finite))

        if lower >= upper:
            span = max(abs(lower), abs(upper), 0.05)
            lower, upper = -span, span

        return lower, upper

    def _build_classification_candidates(self) -> dict:
        candidates = {}
        if self.include_xgb:
            candidates["xgb"] = Pipeline([
                ("scaler", StandardScaler()),
                ("model", EncodedXGBClassifier(
                    n_estimators=self.xgb_estimators,
                    max_depth=4,
                    learning_rate=0.05,
                    random_state=42,
                    verbosity=0,
                    n_jobs=-1,
                )),
            ])
        if self.include_rf:
            candidates["rf"] = RandomForestClassifier(
                n_estimators=200,
                max_depth=6,
                random_state=42,
                n_jobs=-1,
            )
        if self.include_svc:
            candidates["svc"] = Pipeline([
                ("scaler", StandardScaler()),
                ("model", SVC(probability=True, kernel="rbf", C=1.0, random_state=42)),
            ])
        if self.include_knn:
            candidates["knn"] = Pipeline([
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1)),
            ])
        if self.include_mlp:
            candidates["mlp"] = Pipeline([
                ("scaler", StandardScaler()),
                ("model", MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    alpha=0.0005,
                    learning_rate_init=0.001,
                    max_iter=500,
                    early_stopping=True,
                    n_iter_no_change=20,
                    random_state=42,
                )),
            ])
        return candidates


def _cls_label_to_signal(labels: np.ndarray) -> np.ndarray:
    mapping = {"strong_up": 1.0, "neutral": 0.0, "strong_down": -1.0}
    return np.array([mapping.get(l, 0.0) for l in labels])
