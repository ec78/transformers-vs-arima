# Frozen forecast analysis methodology

All tables and figures are generated from the nine saved forecast CSVs. The
script imports no forecasting model, data-loader, or preprocessing module and
therefore cannot refit a model or run inference. Input SHA-256 hashes are saved
in `frozen_input_manifest.csv`.

## Accuracy and relative performance

RMSE and MAE are recomputed from saved origin-level actuals and forecasts. MASE
is the mean of the saved origin-specific scaled absolute errors. Directional
accuracy is the mean saved origin-relative sign match, including the deliberate
zero-change Naive forecasts. Percentage differences use
`100 * (model error / baseline error - 1)`, so negative means better.

## Forecast comparison tests

The paired Diebold-Mariano comparisons use squared-error loss. The long-run
variance uses a Bartlett/Newey-West HAC estimator. Its lag is the maximum of
the overlap lag, `ceil(horizon / origin_step) - 1`, and the automatic bandwidth
`floor(4 * (n / 100) ** (2 / 9))`. The statistic receives the
Harvey-Leybourne-Newbold finite-sample adjustment using the horizon expressed
in forecast-origin steps. Reported p-values are two-sided Student-t p-values
with `n - 1` degrees of freedom. Holm-adjusted values account for the family of
formal tests in the final analysis.

The DM test is asymptotic and remains sensitive to bandwidth choice. Airline
has only 13 origins, with severe overlap at longer horizons, so its comparisons
are descriptive and no formal p-values are reported.

## Computational cost

ARIMA/SARIMA `elapsed_seconds` combines origin-specific estimation and forecast
generation; separate inference timing was not saved. Chronos-2 timing excludes
one-time model download/loading. PatchTST has separate training and inference
timings. These scopes are retained rather than imputed. Hardware-specific
timings should not be interpreted as architecture-independent benchmarks.
