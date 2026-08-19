# Frozen forecasting experiment: key findings

## RMSE winners

- SPY: h=1: Naive; h=5: ARIMA(0,1,2)+drift; h=20: ARIMA(0,1,2)+drift.
- Airline: h=1: Log ARIMA(0,1,1)(0,1,1)[12]; h=3: Log ARIMA(0,1,1)(0,1,1)[12]; h=12: Log ARIMA(0,1,1)(0,1,1)[12].
- Electricity-CISO: h=1: Chronos-2; h=24: Chronos-2; h=168: Chronos-2.

## Dataset findings

- SPY: Naive has the lowest h=1 RMSE; ARIMA has the lowest h=5 and h=20 RMSE. Neither transformer improves RMSE on Naive or ARIMA at any frozen horizon.
- Airline: seasonal log-SARIMA wins all three horizons. Chronos-2 RMSE differences versus Naive range from -77.1% to -49.4%, but it is +22.6% to +38.1% worse than SARIMA. PatchTST performs worse than the seasonal baselines; with only 13 test origins, that comparison remains descriptive.
- Electricity-CISO: Chronos-2 wins all horizons (h=1 -46.5% vs Naive and -41.0% vs ARIMA; h=24 -1.3% vs Naive and -49.8% vs ARIMA; h=168 -14.2% vs Naive and -14.2% vs ARIMA). The transformed ARIMA is especially weak at h=24, where its RMSE exceeds Naive.
- PatchTST: scratch training adds no predictive value in this frozen experiment: it wins no RMSE, MAE, or MASE comparisons and is markedly worse than simpler alternatives despite origin-specific retraining.

## Statistical comparisons

- 24 formal SPY/Electricity comparisons were run; 19 have raw p < 0.05 and 14 remain below 0.05 after Holm adjustment.
- Airline comparisons are descriptive only because 13 origins are inadequate for reliable multi-step asymptotic inference.
- Statistical significance does not replace effect size: use the relative-performance table alongside the paired tests.

## Computational interpretation

- SPY: PatchTST scratch training totaled 3036.3 seconds across the frozen origins.
- Airline: PatchTST scratch training totaled 10.7 seconds across the frozen origins.
- Electricity-CISO: PatchTST scratch training totaled 160.7 seconds across the frozen origins.
- Chronos-2 required no experiment-specific training, but its recorded inference excludes one-time download/model-loading cost.
- ARIMA/SARIMA timings combine estimation and forecast generation; a separate inference time was not recorded.

## Reproducibility and limitations

- Every accuracy value was recomputed from frozen origin-level forecasts and matched the saved comparison within numerical precision.
- The standalone `spy_naive.csv` and `spy_naive_metrics.csv` are legacy step-1 artifacts (1,897 origins), not the authoritative 380-origin step-5 SPY contract, and were excluded.
- The cost comparison is hardware- and timing-scope-specific; no missing model-loading or separate ARIMA inference values were invented.
