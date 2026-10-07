# Stocker Technical Documentation

## 1. Project Overview and Objectives
Stocker is an end-to-end Machine Learning pipeline designed for forecasting NSE stock movements. Its primary objective is to predict future returns (regression) and directional classes (classification) under varying market regimes (Bull, Bear, HighVol) to aid algorithmic trading or decision support. The system focuses on transparent evaluation, rigorous temporal validation, and backtesting simulation to avoid data leakage and overfitting.

## 2. Problem Definition
Financial time series are notoriously noisy and non-stationary. Stocker addresses this by:
1. Partitioning the market into distinct volatility/trend regimes.
2. Treating prediction as dual tasks: estimating the precise percentage return (Regression) and the categorical direction (Classification).
3. Routing predictions to the best historically performing paradigm specific to the current market regime.

## 3. Complete System Architecture
- **Data Layer (`data/`)**: Modules for fetching historical and live market data (yfinance, Upstox, NSE APIs).
- **Feature Engineering (`features/`)**: Transforms raw OHLCV prices into stationary indicators.
- **Regime Detection (`regime/`)**: Unsupervised classification of rolling market conditions.
- **Model Training (`models/`)**: Generates candidates across algorithms (XGBoost, Random Forest, Ridge/SVC, KNN, MLP).
- **Routing Engine (`router/`)**: Selects the best performing model contextually.
- **Backtesting (`backtest/`)**: Simulates a walk-forward trading strategy considering transaction costs, slippage, and volatility sizing.
- **Backend API (`ui/app.py`)**: A FastAPI layer to orchestrate live requests and historical audits.
- **Frontend (`ui/static/`)**: A JS/HTML dashboard summarizing metrics, routing paths, and equity curves.
- **Database (`model_scorecard.py`)**: A PostgreSQL store persisting run logs and granular model metrics.

## 4. Technology Stack
- **Language:** Python 3.10+
- **Machine Learning:** scikit-learn, XGBoost
- **Data Manipulation:** pandas, numpy
- **Web Backend:** FastAPI, uvicorn
- **Frontend:** HTML, Vanilla JavaScript, Tailwind CSS, Chart.js
- **Database:** PostgreSQL (psycopg2)

## 5. Data Sources and Collection
Data is dynamically fetched via `data.ingestion.load_or_fetch`. The primary source is Yahoo Finance (`yfinance`), falling back to live NSE or Upstox data endpoints for intraday accuracy. Caching is employed locally to speed up iterations.

## 6. Data Preprocessing
- Missing values are dropped or forward-filled implicitly via `.dropna()` post feature-engineering.
- No look-ahead imputation is performed.

## 7. Feature Engineering
Calculated using only trailing data up to time $t$:
- **RSI (Relative Strength Index):** 14-period momentum oscillator.
- **MACD (Moving Average Convergence Divergence):** Difference between 12 and 26-period EMAs, along with a 9-period signal line.
- **Bollinger Bands:** 20-period moving average with ±2 standard deviation bands. Features include band width and percentage position.
- **Rolling Stats:** Rolling mean/std, and volume ratio relative to a 10-period average.
- **Lags:** Historical lagging closing prices.
- **Temporal:** Day of week, month, and week of year.

## 8. Target-Variable Construction
Targets are defined for the immediate next trading day ($t+1$):
- **Regression:** `target_pct_return = (Close[t+1] - Close[t]) / Close[t]`
- **Classification:** `strong_up`, `strong_down`, or `neutral` computed dynamically based on a trailing volatility window threshold.

## 9. Model Descriptions
For each regime, multiple candidates are fitted:
- **XGBoost:** Gradient boosted trees optimized for tabular structure.
- **Random Forest:** Ensemble bagging to reduce variance.
- **Ridge/SVC/SVR:** Linear and kernel-based methods with standard scaling.
- **KNN:** Nearest neighbor interpolation for highly non-linear local spaces.
- **MLP:** A 2-hidden-layer feed-forward neural network.

## 10. Training and Validation
A chronological split ensures no future data informs the present model:
- **Walk-Forward Splitting:** Simulates the progression of time. Training is expanded, and testing is performed on the subsequent block.
- **Validation:** Inside each fold, the latest 20% of the training block is used as a validation hold-out to select the best model without touching the test set.

## 11. Prediction Generation
At prediction time ($t$):
1. The live price is fetched.
2. The current regime is predicted using the most recent window.
3. The best model for that regime is invoked.
4. The prediction (return or class) is translated into a standardized signal (-1, 0, 1).

## 12. Model Selection
Handled by `AdaptiveRouter`. The model producing the highest validation Sharpe ratio (trading simulation on the validation set) is persisted in the routing table for out-of-sample execution.

## 13. Database Architecture
A single PostgreSQL table (`stocker_records`) stores:
- Evaluation parameters (Run ID, Fold, Ticker, Regime).
- Granular metrics (Price errors, classification metrics, trading metrics).
- A second table (`stocker_model_io`) stores inputs and outputs of every live prediction for strict model observability.

## 14. Backend/Frontend Interactions
- **Live Predict:** POST `/api/live-prediction` initiates feature generation, inference, and returns signal details and model routing rationale.
- **Analysis:** GET `/api/analysis` queries the database for historical validation results to construct a leaderboard.

## 15. Deployment and Execution Instructions
1. Install requirements: `pip install -r requirements.txt`
2. Start PostgreSQL locally or remote and export `DB_*` variables.
3. Start API: `python run.py` (or `uvicorn ui.app:app --host 0.0.0.0 --port 8000`).
4. Run full backtest for a ticker: `POST /api/backtest` via the frontend or curl.

## 16. Limitations and Future Improvements
- **Transaction Costs Constraints:** Theoretical slippage might not reflect true liquidity impacts on large volumes.
- **Options and Derivatives:** Current system operates purely on spot equity; derivatives would require theta/vega aware models.
- **Intraday Data:** The system uses daily bars; migrating to minute bars would require asynchronous streaming pipelines.
