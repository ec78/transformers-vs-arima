# transformers-vs-arima

## Frozen experiment

The dataset snapshots, evaluation windows, classical-selection evidence, retry
policy, and PatchTST provenance limitations are recorded in
`EXPERIMENT_CONTRACT.md`. Model and forecast outputs are frozen.

Historical integration diagnostics are named `smoke_*.py` or `diagnose_*.py`;
some intentionally run workflows or write diagnostic outputs. Automated tests
live under `tests/`, and safe default collection is restricted there by
`pytest.ini`.

## Frozen final analysis

The final tables, paired forecast-comparison tests, findings, and figures are
generated exclusively from the saved forecast CSVs. No fitting or inference is
performed.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-analysis.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe audit_frozen_sources.py
.\.venv\Scripts\python.exe analyze_spy_retry_effect.py
.\.venv\Scripts\python.exe run_final_analysis.py
```

Tables and documentation are written to `results/analysis`; PNG and SVG figures
are written to `results/figures`. The frozen input manifest records SHA-256
hashes for every consumed file. Statistical and computational-timing limitations
are documented in `results/analysis/methodology.md`.

Classical selection can be reproduced independently with:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-classical.txt
.\.venv\Scripts\python.exe reproduce_classical_selection.py --dataset all
```

This command does not alter the frozen forecasting configuration or forecasts.
