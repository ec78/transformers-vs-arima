# transformers-vs-arima

## Frozen final analysis

The final tables, paired forecast-comparison tests, findings, and figures are
generated exclusively from the saved forecast CSVs. No fitting or inference is
performed.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-analysis.txt
.\.venv\Scripts\python.exe -m pytest test_final_analysis.py -q
.\.venv\Scripts\python.exe run_final_analysis.py
```

Tables and documentation are written to `results/analysis`; PNG and SVG figures
are written to `results/figures`. The frozen input manifest records SHA-256
hashes for every consumed file. Statistical and computational-timing limitations
are documented in `results/analysis/methodology.md`.
