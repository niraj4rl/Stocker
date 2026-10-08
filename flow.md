# Stocker prediction flow

This document freezes the existing product flow before backend internals are
changed. The public routes and existing response fields remain compatible.

## Runtime flow

```text
NSE ticker
  -> robust OHLCV fetcher (cache, yfinance, Yahoo chart, NSE fallback)
  -> adjusted OHLCV cleanup and freshness checks
  -> causal technical features
  -> causal filtered Bull/Bear/HighVol regime labels
  -> candidate models with naive fallbacks and cost-aware selection
  -> purged chronological validation and net-cost scoring
  -> selected stock/regime model (or naive fallback)
  -> direct 21-trading-day log-return forecast
  -> current-close target price and uncertainty range
  -> one-month Buy/Hold/Sell verdict
  -> API response consumed by stock page and rankings
```

## Current implementation trace

`ui/app.py` receives a ticker, constructs `StockerPredictor`, and calls
`fit()` followed by `predict()`. `fit()` loads real OHLCV through
`data/ingestion.py`, removes duplicate/zero-volume bars, checks history and
freshness, builds features in `features/engineering.py`, and fits the
regime/router path used by the existing next-day context. The live regime
labels use a filtered rolling/expanding rule; they do not use an HMM backward
pass. `forecast.py` builds direct forward-return forecasts for horizons 1
through 21. The monthly forecast evaluates naive zero-return, recent-mean,
momentum, Ridge, ElasticNet, and shallow-tree candidates on a chronological
validation window after the horizon purge. A candidate is retained only when
its net-of-cost Sharpe clears the naive baseline uncertainty; otherwise the
naive fallback is returned.

The rankings worker calls `fast_screen_ticker()` for each NSE symbol and
returns the same one-month forecast fields used by the stock page. Failures
are returned in the ranking job's `errors` list; no synthetic forecast is
created for an unavailable ticker.

## Public API contract

### `GET /api/health`

Returns `{ "status": string }`.

### `GET /api/tickers`

Returns:

```json
{
  "tickers": ["RELIANCE.NS"],
  "default": "RELIANCE.NS",
  "quick_picks": ["RELIANCE.NS"],
  "note": "string"
}
```

### `GET /api/live-price/{ticker}`

Returns `ticker`, numeric `price`, `source`, ISO `as_of`, and boolean
`is_fallback`.

### `POST /api/live-prediction`

Request: `{ "ticker": string, "force_refresh": boolean }`.

The response retains `result`, `cache`, `chart`, `regime_stats`,
`data_as_of_date`, `historical_data_date`, `live_quote_time`,
`data_age_days`, and `stale_data_warning`.

`result` retains the existing ticker, current-price, regime, model, paradigm,
signal, prediction, target-price, forecast, validation, routing, and
risk-sizing fields. New evidence fields may be added, but existing fields are
not removed or renamed.

### Rankings

`POST /api/rankings/start` returns a job id and scan totals.
`GET /api/rankings/{job_id}` returns progress, successful result rows, and
per-ticker errors. Existing result fields such as `ticker`,
`one_month_return_pct`, `one_month_predicted_price`, `current_price`,
`validation_passed`, and `forecast_as_of` remain available.

### Backtest and exports

`POST /api/backtest` retains the ticker/mode request and existing result
shape. Export and analysis routes remain unchanged.

## Horizon definitions

- Next-day signal: direct one-trading-day return.
- Next-month forecast: direct 21-trading-day forward log return.
- Target price: latest actual close multiplied by
  `exp(expected 21-day log return)`.

The one-month forecast is the only horizon used for the primary verdict.
Next-day output is explanatory context and is not used to label the monthly
verdict.
