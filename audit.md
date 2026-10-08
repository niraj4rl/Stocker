# Prediction Logic Audit

Audit scope: data ingestion, feature construction, regime detection, model
training, validation, forecasts, verdicts, and rankings. This report was
written before implementation changes.

## Findings

| Severity | Location | Finding | Why it matters | Proposed fix |
|---|---|---|---|---|
| High | `forecast.py:31-70` | The 80/20 validation split has no forecast-horizon embargo. The validation targets overlap the end of training for 21-day forecasts. | A training row immediately before the split can share future realised prices with validation targets, making the validation score optimistic. | Purge at least `horizon` observations before validation and expose the split metadata. |
| High | `models/trainer.py:135-180` | Candidate models are selected on the same single validation block used to report performance. | Selecting the best of many models on that block overfits the reported score; it is not independent out-of-sample evidence. | Add chronological inner selection and a separate final holdout, or mark the current score as selection-biased. |
| High | `backtest/predictor.py:72-87` | The live predictor requests only `2y` of data while `TRAIN_YEARS` is `3`; the configured training window is silently unavailable. | The model does not train on the stated history and may have too few regime observations. | Request at least the configured training years plus warm-up, and reject insufficient history. |
| High | `backtest/predictor.py:159-218`, `ui/static/app.js` | The displayed verdict is the next-day model signal, while the prominent outlook and target price are one-month values. | A Sell beside a positive one-month return is internally ambiguous and can cause users to act on the wrong horizon. | Make the verdict explicitly one-month and derive it from the same one-month return/error evidence; label next-day movement separately. |
| High | `forecast.py:127-145` | The forecast interval is a constant one-step MAE-sized price band for every horizon. | It is not a prediction interval and understates uncertainty as horizon increases. | Use out-of-sample residual dispersion scaled by `sqrt(horizon)` and label it as an approximate range. |
| High | `regime/detector.py:18-43`, `backtest/predictor.py:82-91` | The HMM is fitted on the full training period and its smoothed state labels are then used to train/evaluate regime models on that same period. | Regime assignment can use information from later observations, contaminating regime-conditioned model evidence. | Use expanding/causal regime labels for training and keep current-state inference causal. |
| Medium | `regime/detector.py:69-75` | `bfill()` fills early volatility values from later observations. | This is direct look-ahead in the regime observation series, even if those rows are later discarded in some paths. | Use a causal expanding/rolling fallback and leave unavailable warm-up rows invalid. |
| Medium | `regime/detector.py:132-153` | Regime statistics are computed from labels produced by an in-sample HMM and can omit empty regimes; the UI then reconstructs missing rows with zeros. | The table can imply measured conditions where no observations existed, and its evidence is not out-of-sample. | Report counts explicitly, calculate percentages from observed labels, and mark absent regimes as unavailable. |
| Medium | `data/ingestion.py:5-26`, `data/robust_fetcher.py:150-190` | Quality checks do not reject zero-volume sessions, large unexplained price jumps, or insufficient two-year history; the source adjustment policy is implicit. | Splits, stale/suspended names, and bad bars can create false returns and unreliable rankings. | Validate adjusted prices, remove/flag zero-volume rows, detect extreme jumps, and require at least two years of observations for ranking/prediction. |
| Medium | `ui/app.py:220-255` | Rankings use a separate fast Ridge screen, rank raw predicted return, and have no liquidity, error, shrinkage, or prediction cap. | Ranking thousands of noisy forecasts creates winner's curse and promotes illiquid implausible predictions. | Add risk-adjusted/shrunk ranking fields, liquidity/history filters, and explicit exclusion reasons. |
| Medium | `forecast.py:159-169`, `ui/app.py:220-255` | Ranking and single-stock forecasts share the basic direct forecast but not the same live-price rebasing or verdict evidence. | The same ticker can show different price bases and confidence semantics across pages. | Return a common forecast schema and use the same current-close basis and validation fields. |
| Low | `data/robust_fetcher.py:35-45`, `ui/app.py:116-130` | Cache freshness and displayed dates use mixed UTC/local conversions and the live quote timestamp is not the last bar date. | Users may interpret “updated” as a fresh market observation when it is only a request time or cached bar. | Surface historical bar date and quote timestamp separately. |
| Low | `backtest/predictor.py:207-214` | Classification probability is exposed as confidence without calibration or baseline comparison. | A high class probability is not evidence of forecast accuracy. | Remove certainty wording and report calibrated/out-of-sample evidence only. |

## Required verification

The current implementation has no completed walk-forward comparison covering
zero-return, mean-return, momentum, MAE/RMSE, Spearman IC, confidence buckets,
and a top-decile portfolio after costs. Those metrics must be generated from a
real run before claiming improvement. Network/data availability may limit the
full 50-stock study; any such limitation must be reported rather than filled
with synthetic values.

## Measured verification after fixes

The new script `scripts/run_walk_forward_audit.py` was run against the cached
NIFTY 50 list on 2026-10-08. It produced 50 usable ticker rows and one
unavailable ticker (`TATAMOTORS.NS`), which was recorded in
`backtest_results.csv` rather than imputed.

| Metric | Previous single 20% split | New purged walk-forward |
|---|---:|---:|
| Usable tickers | 50 | 50 |
| Mean directional accuracy | 48.15% | 51.81% |
| Mean model MAE | 5.3754% | 6.2291% |
| Mean zero-return baseline MAE | not reported | 5.2712% |
| Mean Spearman IC | not reported | -0.0032 |
| Mean cost-aware strategy return per fold | not reported | 0.1787% |

These results do **not** demonstrate a reliable predictive edge: the model MAE
is worse than the zero-return baseline on average and IC is approximately zero.
The higher directional accuracy is therefore not presented as evidence of
useful skill. The old and new numbers use different evaluation designs, so the
comparison is diagnostic rather than a controlled A/B experiment.

The cross-sectional portfolio file is also generated by the audit script as
`backtest_portfolio_results.csv`. It ranks only the audited predictions at each
date, takes the top decile, subtracts the configured transaction/slippage
cost, and compares that with an equal-weight portfolio. It must be regenerated
after changing the model or data; it is not a guarantee of future returns.

The run produced 33 cross-sectional dates. Mean realised return after the
configured cost was **+0.7993%** for the predicted top decile versus **-0.0129%**
for equal weight. This is a small diagnostic sample, not a statistically
validated investment result; it is also subject to the data-cache and ticker
availability limitations above.
