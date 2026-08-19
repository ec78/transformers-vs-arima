from pathlib import Path

import pandas as pd

from src.data import load_spy
from src.preprocessing import prepare_spy

from src.models_classical import (
    naive_forecast,
)

from src.evaluation import (
    walk_forward_evaluate,
    summarize_results,
)


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

FORECAST_DIR = (
    PROJECT_ROOT
    / "results"
    / "forecasts"
)

METRICS_DIR = (
    PROJECT_ROOT
    / "results"
    / "metrics"
)

FORECAST_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------
# Load and prepare SPY
# ---------------------------------------------------------

spy = load_spy()

spy = prepare_spy(
    spy
)


# ---------------------------------------------------------
# Define evaluation period
# ---------------------------------------------------------

test_start = pd.Timestamp(
    "2019-01-01"
)

initial_train_size = (
    spy.index.searchsorted(
        test_start
    )
)

print(
    "Initial training observations:",
    initial_train_size,
)

print(
    "First test observation:",
    spy.index[initial_train_size],
)


# ---------------------------------------------------------
# Run naive forecast evaluation
# ---------------------------------------------------------

results = walk_forward_evaluate(
    data=spy,
    target_col="y",
    model_func=naive_forecast,
    model_name="Naive",
    dataset_name="SPY",
    horizons=[1, 5, 20],
    initial_train_size=initial_train_size,
    step=1,
    window="expanding",
)


# ---------------------------------------------------------
# Inspect forecast results
# ---------------------------------------------------------

print()
print("=" * 60)
print("FORECAST RESULTS")
print("=" * 60)

print(
    results.head(10)
)

print()
print(
    "Number of forecast rows:",
    len(results),
)


# ---------------------------------------------------------
# Summarize accuracy
# ---------------------------------------------------------

initial_training_series = (
    spy["y"]
    .iloc[:initial_train_size]
)

summary = summarize_results(
    results=results,
    training_series=initial_training_series,
    mase_seasonality=1,
    include_direction=True,
)


print()
print("=" * 60)
print("ACCURACY SUMMARY")
print("=" * 60)

print(summary)


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

results.to_csv(
    FORECAST_DIR
    / "spy_naive.csv",
    index=False,
)

summary.to_csv(
    METRICS_DIR
    / "spy_naive_metrics.csv",
    index=False,
)

print()
print(
    "Results saved successfully."
)
