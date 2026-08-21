# Expected replication outputs

The scripts write to the repository's existing `results/` tree. They do not
create a second set of results under `replication/`.

## Forecast-level outputs

| Dataset | File suffix | Models | Expected rows |
|---|---|---:|---:|
| SPY | `spy_classical_forecasts.csv` | 2 | 2,280 |
| SPY | `spy_chronos_forecasts.csv` | 1 | 1,140 |
| SPY | `spy_patchtst_forecasts.csv` | 1 | 1,140 |
| Airline | `airline_classical_forecasts.csv` | 2 | 78 |
| Airline | `airline_chronos_forecasts.csv` | 1 | 39 |
| Airline | `airline_patchtst_forecasts.csv` | 1 | 39 |
| Electricity-CISO | `electricity_ciso_classical_forecasts.csv` | 2 | 504 |
| Electricity-CISO | `electricity_ciso_chronos_forecasts.csv` | 1 | 252 |
| Electricity-CISO | `electricity_ciso_patchtst_forecasts.csv` | 1 | 252 |

These counts follow directly from the frozen contract: 380 SPY origins, 13
Airline origins, and 84 Electricity-CISO origins, each evaluated at three
horizons. Classical files contain both Naive and ARIMA forecasts.

The authoritative SPY Naive results are the Naive rows in
`spy_classical_forecasts.csv`. The standalone `spy_naive.csv` is a legacy
diagnostic and is not an input to the final comparison.

## Metrics and diagnostics

`results/metrics/` should contain, for each dataset:

- classical, Chronos-2, and PatchTST headline metric files;
- final and initial ARIMA convergence summaries;
- one ARIMA fit-diagnostic row per forecast origin;
- PatchTST training diagnostics; and
- the consolidated classical/Chronos/PatchTST comparison.

ARIMA fit diagnostics should contain 380 SPY, 13 Airline, and 84
Electricity-CISO origins. PatchTST training diagnostics use the same respective
origin counts. Runtime fields are hardware-dependent and need not match the
committed values.

## Validation and final analysis

`audit_frozen_sources.py` must reconcile all 5,724 authoritative forecast rows
with the committed raw snapshots without an assertion failure. It writes:

- `results/analysis/raw_data_manifest.csv`;
- `results/analysis/source_reconciliation.csv`.

`run_final_analysis.py` writes the final comparison tables and methodology to
`results/analysis/` and four figures in both PNG and SVG format to
`results/figures/`. The analysis consumes saved forecast CSVs and performs no
fitting or inference.

The committed manifests are the authority for hashes and source reconciliation:

- `results/analysis/frozen_input_manifest.csv`;
- `results/analysis/raw_data_manifest.csv`.

## Success criteria

A replication is structurally successful when:

1. the test suite passes;
2. no runner exits with an error;
3. source reconciliation validates all 5,724 rows;
4. forecast files have the expected row counts and only frozen origins and
   horizons;
5. forecasts and required parameter diagnostics are finite;
6. non-converged ARIMA fits are retained or handled by the documented retry
   policy, never silently dropped; and
7. the final analysis completes from the regenerated forecast files.

Exact elapsed times are not a replication target. Small floating-point or
optimizer differences can occur across hardware and numerical-library builds;
any substantive forecast or metric difference requires investigation rather
than silently replacing the committed artifacts.
