# Model evidence report

This report records measured evidence from the reproducible walk-forward run in
`scripts/run_walk_forward_audit.py`. It is not a claim that the model has a
tradable edge.

## Definitions

- Next day: one trading-day forward return.
- One month: 21 trading-day forward **log return**.
- Target price: latest actual close multiplied by `exp(expected log return)`.
- Validation is chronological and purged by the forecast horizon.
- Costs use the configured transaction-cost and slippage assumptions.

## Measured baseline

The prior purged diagnostic over the available NIFTY 50 data measured mean
directional accuracy of **51.81%**, mean model MAE of **6.2291%**, mean
zero-return MAE of **5.2712%**, and mean Spearman IC of **-0.0032**. The model
therefore did not demonstrate lower error or meaningful rank correlation than
the zero-return baseline.

The cross-sectional diagnostic over 33 dates measured a top-decile after-cost
mean realised return of **+0.7993%** versus **-0.0129%** for equal weight.
This result is an exploratory sample, is vulnerable to selection effects, and
must not be interpreted as proof of future performance.

## Current safeguards

The direct monthly forecaster now evaluates zero-return, recent-mean,
momentum, Ridge, ElasticNet, and a shallow random forest candidate. Selection
is based on chronological validation strategy Sharpe after configured costs;
the naive candidate is retained when the apparent improvement is not larger
than the validation uncertainty. The selected candidate, baseline Sharpe,
selection uncertainty, residual evidence, and reliability flag are returned
with the existing API result.

## Limitations

The full nested stock-by-regime outer evaluation and a verified four-year
universe require fresh data for every symbol. Cached symbols with insufficient
history are not silently filled. Until that evaluation is run against a
complete liquid and mid-cap sample, the product should show no reliable signal
when the naive fallback wins.
