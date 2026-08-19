# Frozen experiment contract

This document records the claims that can be reproduced from the repository.
It does not change or retrospectively reselect any frozen model.

## Authoritative snapshots and evaluation windows

The committed CSVs under `data/raw` are the authoritative input snapshots.
Normal model runs use `src.data` to read those local files; they do not download
fresh data. `src/dataimport.py` is a refresh utility and will deliberately change
the contract when called without explicit date bounds.

| Dataset | Raw snapshot end | First target period | Origins | Step | Horizons |
|---|---:|---:|---:|---:|---:|
| SPY | 2026-08-17 | 2019-01-02 | 380 | 5 trading days | 1, 5, 20 |
| Airline | 1960-12-01 | 1959-01-01 | 13 | 1 month | 1, 3, 12 |
| Electricity-CISO | 2026-08-18 00:00 UTC | 2025-01-01 00:00 UTC | 84 | 168 hours | 1, 24, 168 |

`audit_frozen_sources.py` checks every authoritative forecast target, origin
value, timestamp, training size, and origin-specific MASE scale against these
committed snapshots. It also verifies fixed SHA-256 hashes for the raw CSVs.
A fresh upstream download, especially Yahoo adjusted prices, is a new dataset
snapshot and must not silently replace this contract.

The standalone `results/forecasts/spy_naive.csv` and corresponding metric file
are legacy step-1 diagnostics. The authoritative SPY Naive results are the
Naive rows in `spy_classical_forecasts.csv`, which use the frozen step-5 design.

## Classical model selection

Selection used only observations before each evaluation cutoff.
`reproduce_classical_selection.py` makes the candidate searches checkable and
writes full-precision candidate tables under `results/model_selection`.

- SPY used a two-stage process. A no-drift ARIMA(p,1,q) grid with p and q in
  0..3 selected (0,1,2) within that grid. A separate four-model drift check
  compared (0,1,0) and (0,1,2), with and without drift. AIC preferred
  ARIMA(0,1,2)+drift; BIC preferred ARIMA(0,1,2) without drift. The frozen model
  is therefore AIC-preferred, not jointly AIC/BIC-preferred.
- Airline used the bounded p, q, P, Q in {0,1} search on log passengers with
  ordinary and seasonal differencing. AIC and BIC both selected
  ARIMA(0,1,1)(0,1,1)[12].
- Electricity used p, q in {0,1,2} on weekly log differences. AIC and BIC both
  selected ARIMA(2,0,2) with a constant.

These reproduction scripts document the historical choice; they are not
connected to the forecasting pipeline and do not update model configuration.

## Retry policy

The SPY retry perturbs its explicitly named drift parameter. The shared
transformed-ARIMA retry perturbs the first non-variance parameter: `ma.L1` for
Airline and `const` for Electricity. A retry is accepted only when it converges
and its optimizer objective is no worse than the initial fit.

The 25 accepted SPY retries changed aggregate RMSE/MAE by less than 0.001% at
all horizons. `analyze_spy_retry_effect.py` reproduces this directly from the
pre-retry forecast file in Git commit `e93958d` and the current frozen file;
parameter percentage changes are not a substitute for forecast-metric changes.

## PatchTST provenance

The frozen run treats the architecture in `PatchTSTTrainingConfig` as fixed and
performs no automated test-period tuning. The repository contains no tuning
loop. Because the PatchTST work entered Git in one squashed commit, Git history
alone cannot prove when every design choice was made; publication text should
not claim that it does. The defensible evidence is the fixed configuration,
deterministic seeds, origin-local validation, and frozen origin-level outputs.

