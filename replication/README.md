# Replication guide

This directory provides two replication paths for the frozen experiment:

1. **Frozen-result validation** checks the committed data, forecasts, metrics,
   and final analysis without fitting forecasting models.
2. **Full experiment rerun** refits or reruns every classical, Chronos-2, and
   PatchTST forecast and overwrites generated files under `results/`.

Read the repository-level `EXPERIMENT_CONTRACT.md` before running either path.
It defines the authoritative data snapshots, evaluation origins, horizons,
model specifications, MASE conventions, and retry policy.

## System requirements

- A full Git clone, not only a source archive, is recommended.
- Python 3.12. The frozen development environment used Python 3.12.0.
- PowerShell 5.1 or newer for the wrapper scripts.
- Sufficient disk space for Python packages and the downloaded
  `amazon/chronos-2` model.
- Network access to Hugging Face is required only for the first Chronos-2 model
  download when it is not already cached.

The committed experiment used CPU inference for Chronos-2 and CPU training for
PatchTST. CUDA is supported by the Chronos runner, but changing hardware makes
timing values non-comparable and may introduce small numerical differences.
PatchTST remains CPU-only in the frozen runner.

## Clean setup

From the repository root in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r replication\requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

The combined requirements file reuses the separately pinned classical,
analysis, Chronos-2, and PatchTST requirement files at the repository root.
Do not run `src/dataimport.py`: the CSV files committed under `data/raw/` are
the authoritative frozen inputs. Refreshing upstream data creates a different
experiment contract.

## Path A: validate committed results

This is the recommended first command and does not run model inference or
training:

```powershell
.\replication\run_validation.ps1
```

It runs the tests, reconciles every saved forecast with the raw snapshots,
regenerates the deterministic final analysis, and verifies the published SPY
retry-effect claim.

The retry-effect audit reads the pre-retry forecast from Git commit `e93958d`.
If that commit is unavailable in a source archive or shallow clone, obtain the
full history. `-SkipRetryEffect` permits the remaining checks to run, but then
that specific claim has not been replicated.

Classical model-selection searches are independent of frozen-result validation.
Include them with:

```powershell
.\replication\run_validation.ps1 -IncludeModelSelection
```

## Path B: rerun the full experiment

Run this only in a clean clone or dedicated branch. Existing generated
forecasts, metrics, diagnostics, tables, and figures under `results/` will be
replaced. Raw data files are not modified.

```powershell
.\replication\run_full_experiment.ps1 -ConfirmOverwrite
```

The fixed order is:

1. automated tests;
2. classical model-selection evidence;
3. SPY Naive and ARIMA forecasts;
4. Airline and Electricity-CISO Naive and ARIMA forecasts;
5. Chronos-2 zero-shot forecasts for all three datasets;
6. PatchTST forecasts for all three datasets;
7. raw-source and alignment reconciliation; and
8. final tables, tests, findings, and figures; and
9. the SPY retry-effect comparison against its pre-retry Git artifact.

The frozen command uses Chronos CPU inference, batch size 8, and six PatchTST
CPU threads. Those values can be stated explicitly:

```powershell
.\replication\run_full_experiment.ps1 `
    -ConfirmOverwrite `
    -ChronosDevice cpu `
    -ChronosBatchSize 8 `
    -PatchTSTThreads 6
```

Do not substitute another pretrained model if `amazon/chronos-2` cannot be
downloaded or loaded. Report the dependency, access, memory, or hardware
failure instead.

## Review after a full rerun

Inspect changes before accepting regenerated artifacts:

```powershell
git status --short
git diff --check
git diff -- results\analysis results\metrics
```

Runtime columns are expected to vary by machine. Differences in forecasts,
errors, convergence status, selected specifications, origin counts, or
alignment are scientifically meaningful and should be explained rather than
discarded as timing noise.

See `expected_outputs.md` for file counts, diagnostics, and success criteria.
