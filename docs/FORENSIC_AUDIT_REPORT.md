# Stocker Forensic Technical Audit

**Audit date:** 2026-10-07  
**Scope:** Repository implementation, checked-in data/artifacts, documentation, tests, and executable behavior.  
**Evidence policy:** Conclusions below are based on repository code or artifacts. Missing evidence is explicitly labeled.

## Executive assessment

Stocker is implemented as a daily OHLCV, regime-aware machine-learning pipeline with:

- FastAPI backend and static JavaScript frontend
- Multi-source NSE data retrieval with Parquet caching
- Technical feature engineering
- Three-state Gaussian HMM regime detection
- Per-regime regression and classification model training
- Validation-Sharpe-based routing
- Walk-forward backtesting
- Optional PostgreSQL scorecard and model-I/O logging
- In-memory live prediction caching

The repository contains cached OHLCV input data and one model-comparison CSV, but no serialized model checkpoints, database dump, complete backtest output, or reproducible result package.

The test suite was executed with:

```text
.venv\Scripts\python.exe -m pytest stocker\tests -q
```

Observed result:

```text
38 passed, 1 warning in 3.27s
```

The warning came from joblib CPU-core detection on Windows.

## Repository map

| Area | Important files | Actual role |
|---|---|---|
| Data | `data/ingestion.py`, `data/robust_fetcher.py`, `data/nse_stocks.py`, `data/source_validator.py`, `data/upstox_client.py` | Fetch, validate, cache, normalize, and fall back between data sources |
| Features | `features/engineering.py` | Indicators, lags, rolling statistics, temporal features, and targets |
| Regimes | `regime/detector.py` | Gaussian HMM fitting and state-to-regime mapping |
| Models | `models/trainer.py` | Candidate construction, training, validation, and scorecards |
| Persistence | `models/model_scorecard.py`, `models/model_io_store.py` | PostgreSQL scorecards and prediction I/O |
| Routing | `router/adaptive.py` | Regression/classification selection and fallback |
| Backtesting | `backtest/splitter.py`, `backtest/engine.py` | Walk-forward splitting and trading simulation |
| Live inference | `backtest/predictor.py` | Historical fitting, live quote retrieval, prediction, and signals |
| Metrics | `utils/metrics.py` | Statistical and trading metrics |
| API | `ui/app.py` | FastAPI routes |
| Frontend | `ui/static/app.js`, `ui/static/index.html`, `ui/static/landing.html` | Dashboard and landing page |
| Tests | `tests/` | Unit and mocked behavior tests |
| Documentation | `README.md`, `docs/` | Design and methodology claims |

### Artifacts found

- 30 cached Parquet OHLCV files
- One cached NSE equity master CSV
- One model-comparison export: `data/exports/model_comparison_CYBERTECH_NS.csv`
- No notebooks
- No serialized models
- No Pickle, Joblib, Torch, HDF5, SQLite, or database dump artifacts
- `models/registry/` contains only `.gitkeep`
- No generated logs or complete backtest JSON output

## Actual execution pipeline

### Live prediction

1. `ui.app._normalize_ticker()` normalizes the user ticker and adds `.NS` when appropriate.
2. `StockerPredictor.fit()` calls `load_or_fetch()` with `period="2y"`.
3. `features.engineering.build_features()` builds indicators, lags, temporal fields, and next-period targets.
4. `RegimeDetector.fit()` fits a three-state Gaussian HMM.
5. `ModelTrainer.train_for_fold()` trains models independently within each detected regime.
6. `AdaptiveRouter.build_from_registry()` chooses regression or classification using validation Sharpe.
7. `StockerPredictor.predict()` detects the current regime and obtains the routed model.
8. A live quote is attempted through Upstox, yfinance, Yahoo chart API, and NSE, in that order.
9. The prediction becomes a Buy, Hold, or Sell signal.
10. Volatility targeting and regime exposure caps are applied.
11. Model input/output logging is attempted in PostgreSQL.
12. `ui.app.live_prediction()` returns prediction, regime, model, signal, confidence, price-source, freshness, regime statistics, and chart data.

### Backtest

`backtest.engine.run_backtest()` performs:

1. Data retrieval
2. Feature generation
3. Optional fast-profile truncation
4. Walk-forward splitting
5. Per-fold HMM fitting
6. Per-regime candidate-model training
7. Validation-Sharpe model selection
8. Adaptive and baseline strategy simulation
9. Fold and aggregate metric calculation
10. Optional PostgreSQL persistence

Compared strategies are adaptive routing, buy-and-hold, static regression, static classification, and naive regime regression.

## Data and ingestion audit

### Cached corpus

The checked-in cache contains:

- 30 Parquet files
- 19,357 total Parquet rows
- Seven columns per Parquet file
- 494 to 1,237 rows per ticker
- 60 total null cells
- No duplicate Parquet index values in the inspected files

The cached periods are heterogeneous. Examples:

- `CYBERTECH_NS.parquet`: 1,236 rows, 2021-07-14 through 2026-07-14
- `RELIANCE_NS.parquet`: 500 rows, 2024-08-07 through 2026-08-07
- `WIPRO_NS.parquet`: 1,237 rows, 2021-03-30 through 2026-03-30
- `^NSEI.parquet`: 497 rows, 2024-08-27 through 2026-08-27

The cache is stale relative to the audit date. Runtime freshness logic may reject it or attempt an online refresh.

### Source fallback order

`data.robust_fetcher.RobustNSEFetcher.get_ohlcv()` uses:

1. Existing Parquet cache
2. Upstox, if configured
3. yfinance
4. Yahoo chart API
5. NSE/bootstrap data
6. Existing cache fallback if online providers fail
7. An exception if no valid source remains

### Validation and cleaning

The yfinance path:

- Selects `Open`, `High`, `Low`, `Close`, and `Volume`
- Converts and sorts the datetime index
- Removes timezone information
- Drops rows with missing `Close`
- Removes non-positive `Close`
- Removes duplicate timestamps, keeping the last

The Yahoo chart path:

- Drops missing `Close`
- Removes non-positive `Close`
- Fills missing volume with zero
- Removes duplicate timestamps

There is no implemented OHLC consistency validation such as `High >= Low`, `High >= Close`, or `Low <= Close`. There is no explicit invalid-volume rejection.

The configured defaults are:

```text
MIN_DATA_ROWS = 252
MAX_STALE_DAYS = 12
DATA_SOURCE = "yfinance"
```

### Freshness inconsistency

`data.ingestion._assert_data_quality()` checks stale data, but normal `fetch_ohlcv()` execution delegates directly to `fetch_ohlcv_robust()` and does not call it.

The robust fetcher therefore:

- Uses freshness to decide whether a cache is immediately reusable
- Can still return an older cache as an offline fallback if it has enough rows
- Does not apply the configured stale-age threshold in `_validate_quality()`

This is implemented behavior.

## Feature engineering

Implemented in `features/engineering.py`:

| Feature | Exact implementation |
|---|---|
| RSI | Custom exponentially weighted RSI, period 14 |
| MACD | EMA fast 12, slow 26, signal 9 |
| MACD histogram | `macd - macd_signal` |
| Bollinger width | 20-period mean/std, multiplier 2.0 |
| Bollinger position | `(Close - lower) / (upper - lower)` |
| Close lags | `close_lag_1` through `close_lag_5` |
| Rolling mean/std | Seven-period rolling calculations |
| Volume MA | Ten-period rolling mean |
| Volume ratio | `Volume / volume_ma_10` |
| Temporal features | Day of week, month, ISO week |
| Returns | Existing percentage and log returns |

`build_features()` ends with `dropna()`, removing initial indicator rows and the final row with an unavailable next-period target.

Feature columns exclude OHLCV, current returns, target columns, and `regime`.

## Targets

Regression targets:

```text
target_pct_return = pct_return.shift(-1)
target_price      = Close.shift(-1)
```

Classification target:

- `strong_up` if next-period return is greater than the rolling 20-period return standard deviation
- `strong_down` if next-period return is less than the negative rolling standard deviation
- `neutral` otherwise

The classification threshold is volatility-dependent and is not fixed.

## Regime detection

`regime.detector.RegimeDetector` uses `hmmlearn.hmm.GaussianHMM` with:

```text
n_components = 3
covariance_type = "full"
n_iter = 200
min_covar = 1e-6
random_state = 42
```

If fitting fails, the fallback uses:

```text
covariance_type = "diag"
n_iter = 200
min_covar = 1e-5
random_state = 42
```

Observations are:

1. Daily log return
2. Annualized 20-observation rolling realized volatility

State mapping:

1. Highest average-volatility state becomes `HighVol`
2. Remaining state with nonnegative average return becomes `Bull`
3. Remaining state with negative average return becomes `Bear`

The code calculates a volatility median but does not use it.

The mapping is sample-dependent. Stable economic interpretation across securities or periods is:

**NOT DETERMINABLE FROM CODE**

No complete persisted repository-wide regime distribution is available.

## Model inventory

### Regression

- XGBoost Regressor
- Random Forest Regressor
- Ridge with StandardScaler
- SVR with StandardScaler
- KNN Regressor with StandardScaler
- MLP Regressor with StandardScaler

### Classification

- Encoded XGBoost Classifier with StandardScaler
- Random Forest Classifier
- SVC with StandardScaler
- KNN Classifier with StandardScaler
- MLP Classifier with StandardScaler

No Linear Regression, Decision Tree, ARIMA, Voting Regressor, or additional predictive model was found.

### Explicit hyperparameters

#### XGBoost

```text
n_estimators = 200 normally, 50 in fast mode
max_depth = 4
learning_rate = 0.05
subsample = 0.8 for regression
colsample_bytree = 0.8 for regression
random_state = 42
verbosity = 0
n_jobs = -1
classification eval_metric = "mlogloss"
```

#### Random Forest

```text
n_estimators = 200
max_depth = 6
random_state = 42
n_jobs = -1
```

#### Ridge

```text
alpha = 1.0
```

#### SVR/SVC

```text
kernel = "rbf"
C = 1.0
SVR epsilon = 0.01
SVC probability = True
random_state = 42 for SVC
```

#### KNN

```text
n_neighbors = 5
weights = "distance"
n_jobs = -1
```

#### MLP

```text
hidden_layer_sizes = (64, 32)
activation = "relu"
alpha = 0.0005
learning_rate_init = 0.001
max_iter = 500
early_stopping = True
n_iter_no_change = 20
random_state = 42
```

Parameters not listed above remain library defaults; this report does not substitute assumed defaults.

## Training and validation

For each regime:

- Regimes with fewer than 30 rows are skipped
- The last 20% of regime rows is validation
- Validation size is at least 20 rows
- Earlier rows are used for fitting
- No shuffle is used
- All enabled candidates are trained
- Best regression and classification models are selected independently by validation trading Sharpe

The walk-forward splitter:

- Sorts chronologically
- Uses expanding training windows
- Uses three years of training and six months of testing by default
- Uses a five-day purge gap by default
- Requires at least 200 training rows

Fast mode uses one year of training, twelve months of testing, and truncates to the most recent two years.

There is no random k-fold cross-validation, `TimeSeriesSplit`, hyperparameter search, or model checkpoint persistence.

## Leakage audit

Positive controls:

- Targets are shifted forward one period
- Rolling features use current and historical rows
- Walk-forward folds are chronological
- A purge gap is applied
- Candidate selection uses validation rather than test rows
- Model pipelines fit scaling within each model

Risks and limitations:

- Validation is formed after regime filtering, so it is the latest 20% of rows belonging to each regime rather than necessarily a contiguous global period.
- Live features use the last row with a known next-period target, while the displayed current price comes from a separate live quote.
- The exact temporal gap between those values is **NOT DETERMINABLE FROM CODE**.
- Stale cached data may be returned as an offline fallback.

## Metrics

Implemented metrics include:

### Regression

- MAE
- MSE
- RMSE
- R²
- MAPE
- Directional accuracy

R² is calculated on returns. When current prices are supplied, MAE/MSE/RMSE/MAPE use projected price values.

### Classification

- Accuracy
- Weighted F1
- Weighted precision
- Weighted recall
- Confusion matrix
- ROC-AUC when probabilities are available

### Trading

- Sharpe ratio
- Maximum drawdown
- Total return
- Calmar ratio
- Hit rate
- Profit factor
- Trade count
- Winning trades
- Losing trades

Sharpe uses a zero risk-free rate and annualization of 252 trading days.

## Empirical results available in the repository

`data/exports/model_comparison_CYBERTECH_NS.csv` contains:

- 420 rows
- Folds 0 and 1
- Bull and Bear regimes only
- No HighVol rows
- 252 regression rows
- 168 classification rows

Maximum stored validation Sharpe values:

| Regime | Paradigm | Maximum Sharpe | Model |
|---|---|---:|---|
| Bull | Regression | 2.5406 | Random Forest |
| Bull | Classification | 0.0000 | Random Forest |
| Bear | Regression | 2.1066 | SVM Regressor |
| Bear | Classification | 0.0000 | SVM Classifier |

Selected stored regression metrics:

| Regime | Model | MAE | RMSE | R² | Directional accuracy |
|---|---|---:|---:|---:|---:|
| Bull | Random Forest | 2.780658 | 8.710938 | -164,983.2800 | 55.56% |
| Bear | SVM Regressor | 0.033191 | 0.042419 | -0.2954 | 48.94% |

The stored file does not establish complete adaptive performance, HighVol performance, buy-and-hold comparison, or production profitability.

Therefore full-system claims remain:

**IMPLEMENTED BUT EMPIRICAL RESULT NOT AVAILABLE**

## Adaptive router

For every regime:

```text
if regression_sharpe >= classification_sharpe:
    choose regression
else:
    choose classification
```

Regression wins ties.

Fallback order:

1. Selected paradigm in the current regime
2. Other paradigm in the current regime
3. Any model from Bull, then Bear, then HighVol
4. No model

Fallback routing can use a model trained for a different regime.

Routing tables are generated at runtime and are not persisted as durable model artifacts.

## Backtest mechanics

Regression predictions become:

```text
prediction > SIGNAL_DEADBAND  => +1
prediction < -SIGNAL_DEADBAND => -1
otherwise                     => 0
```

Classification labels become:

```text
strong_up   => +1
neutral     => 0
strong_down => -1
```

Position sizing is based on:

```text
min(MAX_GROSS_LEVERAGE, VOL_TARGET_ANNUAL / recent_annualized_volatility)
```

Defaults:

```text
MAX_GROSS_LEVERAGE = 1.0
VOL_TARGET_ANNUAL = 0.18
Bull cap = 1.0
Bear cap = 0.7
HighVol cap = 0.4
Drawdown trigger = 0.12
Drawdown multiplier = 0.5
```

Execution costs:

```text
TRANSACTION_COST_BPS = 10
SLIPPAGE_BPS = 5
```

The cost is:

```text
(transaction_cost_bps + slippage_bps) / 10000 * turnover
```

Buy-and-hold pays the cost once on initial entry.

## Signal generation

Regression live predictions:

- Are clipped to the first and 99th percentile of training targets
- Produce a predicted price using `current_price * (1 + predicted_return)`
- Use the configured deadband for Buy/Hold/Sell

Classification live predictions:

- `strong_up` → Buy
- `neutral` → Hold
- `strong_down` → Sell
- Confidence is the maximum predicted class probability when available

The default deadband is:

```text
(10 + 5) / 10000 * 2 * 1.5 = 0.0045
```

Thus the default regression signal threshold is 0.45% predicted return.

## Freshness and API behavior

The API returns historical data date, data age, live quote timestamp, quote source, and stale warning.

The live API warning is:

```text
stale_data_warning = data_age_days > 4
```

This differs from:

- `MAX_STALE_DAYS = 12`
- Cache freshness threshold of one day on weekdays and three days on weekends

## Persistence

PostgreSQL is used through `psycopg2`.

### `stocker_records`

Stores scorecards, backtest-run metadata, and search records, including ticker, fold, regime, paradigm, model, winner flag, trading metrics, regression metrics, classification metrics, sample counts, dates, and timestamps.

### `stocker_model_io`

Stores run ID, ticker, regime, paradigm, model, context, JSON input payload, JSON output payload, actual price, prediction timestamp, and creation timestamp.

Both live and backtest logging are implemented, but database contents are:

**NOT DETERMINABLE FROM CODE**

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Landing page |
| GET | `/app` | Dashboard |
| GET | `/health` | Health status |
| GET | `/api/tickers` | Ticker universe |
| GET | `/api/live-price/{ticker}` | Fast live quote |
| POST | `/api/live-prediction` | Live model prediction |
| POST | `/api/backtest` | Fast/full backtest |
| GET | `/api/analysis` | Scorecard and leaderboard |
| GET | `/api/recent-searches` | Search history |
| POST | `/api/export/model-comparison` | Export model scores |
| POST | `/api/export/leaderboard` | Export leaderboard |
| POST | `/api/export/regime-analysis` | Export regime summary |

The API uses Pydantic request models, an in-process predictor cache, PostgreSQL where available, and HTTP 400/404/500 error responses for the corresponding failure classes.

## Frontend

The frontend:

- Loads ticker data
- Provides ticker autocomplete and quick picks
- Calls live prediction and backtest APIs
- Displays current price and source
- Displays regime, paradigm, and model
- Displays Buy/Hold/Sell and classification confidence
- Displays historical charts and regime statistics
- Displays data age and stale warnings
- Displays backtest equity curves
- Displays analysis and leaderboard data

No technical indicators are calculated in the frontend.

## Tests

Tests cover:

- Feature generation
- Ingestion validation
- Live-price fallback
- Metrics
- Model-I/O persistence with mocks
- NSE ticker lists
- Regime detection
- Routing
- Walk-forward splitting
- Upstox client behavior

They do not establish live provider availability, PostgreSQL contents, actual profitability, out-of-sample superiority, complete API integration behavior, HMM label stability, or full frontend behavior.

## Dependency audit

Declared dependencies in `requirements.txt`:

```text
yfinance==0.2.40
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.0
xgboost==2.0.3
hmmlearn==0.3.2
ta==0.11.0
fastapi==0.115.12
uvicorn==0.30.1
plotly==5.22.0
joblib==1.4.2
scipy==1.13.0
pyarrow==16.1.0
requests>=2.31.0
psycopg2-binary==2.9.9
python-dotenv==1.0.1
```

Python itself is not pinned in the dependency manifest.

The frontend has no package manifest; it is static HTML/CSS/JavaScript.

## Documentation contradictions

The following are **DOCUMENTED BUT NOT VERIFIED IN CODE**:

- Adaptive routing is superior to static models or buy-and-hold
- The system entirely eliminates look-ahead bias
- The project has completed robust empirical validation
- Every ticker produces a complete five-fold backtest
- Intraday accuracy is the primary historical modeling behavior
- Five years of data are used for live prediction

The README says five years are used for prediction, while live code explicitly requests two years.

The existing testing report states that tests could not previously run because of interpreter-path issues. That statement is stale in the current environment because the complete suite passed.

## Final conclusions

The repository verifies that Stocker implements:

1. Daily OHLCV ingestion and caching
2. The documented technical indicators with explicit parameters
3. A three-state Gaussian HMM
4. Six regression and five classification candidate models
5. Per-regime validation and Sharpe-based model selection
6. Adaptive regression/classification routing
7. Walk-forward backtesting
8. Transaction costs, slippage, volatility targeting, leverage limits, regime caps, and drawdown throttling
9. Live prediction, backtest, analysis, export, and ticker APIs
10. PostgreSQL scorecard and model-I/O persistence paths
11. A passing 38-test unit suite

The repository does not establish complete system-level numerical performance, HighVol performance, adaptive superiority, production profitability, or live calibration.

