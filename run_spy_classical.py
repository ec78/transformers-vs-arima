from pathlib import Path

import pandas as pd

from src.data import load_spy
from src.preprocessing import prepare_spy

from src.models_classical import (
    naive_forecast,
    arima_log_price_forecast,
)

from src.evaluation import (
    walk_forward_evaluate,
    summarize_results,
    summarize_convergence_diagnostics,
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

spy = prepare_spy(
    load_spy()
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

horizons = [
    1,
    5,
    20,
]

step = 5


print(
    "Initial training observations:",
    initial_train_size,
)

print(
    "First test observation:",
    spy.index[
        initial_train_size
    ],
)


# ---------------------------------------------------------
# Initial training data for MASE
# ---------------------------------------------------------

initial_training_series = (
    spy["y"]
    .iloc[:initial_train_size]
)


# =========================================================
# Naive model
# =========================================================

print()
print("=" * 60)
print("RUNNING NAIVE MODEL")
print("=" * 60)

naive_results = walk_forward_evaluate(
    data=spy,
    target_col="y",
    model_func=naive_forecast,
    model_name="Naive",
    dataset_name="SPY",
    horizons=horizons,
    initial_train_size=initial_train_size,
    step=step,
    window="expanding",
)

naive_summary = summarize_results(
    results=naive_results,
    training_series=initial_training_series,
    mase_seasonality=1,
    include_direction=True,
)


# =========================================================
# ARIMA model
# =========================================================

print()
print("=" * 60)
print("RUNNING ARIMA MODEL")
print("=" * 60)

arima_results = walk_forward_evaluate(
    data=spy,
    target_col="y",
    model_func=arima_log_price_forecast,
    model_name="ARIMA(0,1,2)+drift",
    dataset_name="SPY",
    horizons=horizons,
    initial_train_size=initial_train_size,
    step=step,
    window="expanding",
    model_kwargs={
        "order": (0, 1, 2),
        "drift": True,
    },
)

arima_summary = summarize_results(
    results=arima_results,
    training_series=initial_training_series,
    mase_seasonality=1,
    include_direction=True,
)

arima_convergence_summary = (
    summarize_convergence_diagnostics(
        arima_results
    )
)

arima_initial_results = (
    arima_results.copy()
)

arima_initial_results[
    "converged"
] = arima_initial_results[
    "initial_converged"
]

arima_initial_results[
    "iterations"
] = arima_initial_results[
    "initial_iterations"
]

arima_initial_convergence_summary = (
    summarize_convergence_diagnostics(
        arima_initial_results
    )
)

arima_fit_diagnostics = (
    arima_results
    .groupby(
        "origin",
        as_index=False,
    )
    .first()
    [
        [
            "origin",
            "initial_converged",
            "initial_iterations",
            "initial_warnflag",
            "initial_objective_value",
            "initial_optimizer_message",
            "initial_optimizer_status",
            "initial_parameters",
            "retry_attempted",
            "retry_accepted",
            "retry_count",
            "retry_drift_offset",
            "retry_converged",
            "retry_iterations",
            "retry_warnflag",
            "retry_objective_value",
            "retry_optimizer_message",
            "retry_optimizer_status",
            "retry_history",
            "converged",
            "iterations",
            "warnflag",
            "objective_value",
            "optimizer_message",
            "optimizer_status",
            "parameters",
            "forecast_is_finite",
            "parameters_are_finite",
        ]
    ]
)


# =========================================================
# Combine results
# =========================================================

all_results = pd.concat(
    [
        naive_results,
        arima_results,
    ],
    ignore_index=True,
)

all_summary = pd.concat(
    [
        naive_summary,
        arima_summary,
    ],
    ignore_index=True,
)


# ---------------------------------------------------------
# Display summary
# ---------------------------------------------------------

print()
print("=" * 60)
print("SPY MODEL COMPARISON")
print("=" * 60)

print(
    all_summary.to_string(
        index=False
    )
)


# ---------------------------------------------------------
# ARIMA convergence diagnostics
# ---------------------------------------------------------

arima_convergence = (
    arima_results
    .groupby("origin")[
        "converged"
    ]
    .first()
)

print()
print("=" * 60)
print("ARIMA CONVERGENCE")
print("=" * 60)

print(
    "Forecast origins:",
    len(arima_convergence)
)

print(
    "Initial non-converged:",
    (
        ~arima_fit_diagnostics[
            "initial_converged"
        ].astype(bool)
    ).sum()
)

print(
    "Retries accepted:",
    arima_fit_diagnostics[
        "retry_accepted"
    ].sum()
)

print(
    "Converged:",
    arima_convergence.sum()
)

print(
    "Non-converged:",
    (
        ~arima_convergence
        .astype(bool)
    ).sum()
)

print(
    "Convergence rate:",
    arima_convergence.mean()
)

print()
print(
    arima_convergence_summary.to_string(
        index=False
    )
)

print()
print(
    "ACCURACY BY INITIAL CONVERGENCE STATUS"
)
print()
print(
    arima_initial_convergence_summary.to_string(
        index=False
    )
)


# ---------------------------------------------------------
# Save outputs
# ---------------------------------------------------------

all_results.to_csv(
    FORECAST_DIR
    / "spy_classical_forecasts.csv",
    index=False,
)

all_summary.to_csv(
    METRICS_DIR
    / "spy_classical_metrics.csv",
    index=False,
)

arima_convergence_summary.to_csv(
    METRICS_DIR
    / "spy_arima_convergence.csv",
    index=False,
)

arima_initial_convergence_summary.to_csv(
    METRICS_DIR
    / "spy_arima_initial_convergence.csv",
    index=False,
)

arima_fit_diagnostics.to_csv(
    METRICS_DIR
    / "spy_arima_fit_diagnostics.csv",
    index=False,
)

print()
print(
    "Results saved successfully."
)
