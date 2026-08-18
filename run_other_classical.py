from pathlib import Path

import pandas as pd

from src.data import (
    load_airline,
    load_electricity,
)

from src.preprocessing import (
    prepare_airline,
    prepare_electricity,
)

from src.models_classical import (
    naive_forecast,
    arima_transformed_forecast,
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
# Shared dataset workflow
# ---------------------------------------------------------

def run_classical_dataset(
    data,
    dataset_name,
    output_prefix,
    test_start,
    horizons,
    step,
    mase_seasonality,
    arima_name,
    arima_kwargs,
):
    """Run and save comparable Naive and ARIMA results."""

    initial_train_size = (
        data.index.searchsorted(
            test_start
        )
    )

    print()
    print("=" * 60)
    print(dataset_name)
    print("=" * 60)

    print(
        "Initial training observations:",
        initial_train_size,
    )

    print(
        "First test observation:",
        data.index[
            initial_train_size
        ],
    )

    naive_results = walk_forward_evaluate(
        data=data,
        target_col="y",
        model_func=naive_forecast,
        model_name="Naive",
        dataset_name=dataset_name,
        horizons=horizons,
        initial_train_size=initial_train_size,
        step=step,
        window="expanding",
        mase_seasonality=mase_seasonality,
    )

    arima_results = walk_forward_evaluate(
        data=data,
        target_col="y",
        model_func=arima_transformed_forecast,
        model_name=arima_name,
        dataset_name=dataset_name,
        horizons=horizons,
        initial_train_size=initial_train_size,
        step=step,
        window="expanding",
        model_kwargs=arima_kwargs,
        mase_seasonality=mase_seasonality,
    )

    initial_training_series = data[
        "y"
    ].iloc[
        :initial_train_size
    ]

    naive_summary = summarize_results(
        results=naive_results,
        training_series=initial_training_series,
        mase_seasonality=mase_seasonality,
        include_direction=True,
    )

    arima_summary = summarize_results(
        results=arima_results,
        training_series=initial_training_series,
        mase_seasonality=mase_seasonality,
        include_direction=True,
    )

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

    final_convergence_summary = (
        summarize_convergence_diagnostics(
            arima_results
        )
    )

    initial_results = arima_results.copy()

    initial_results[
        "converged"
    ] = initial_results[
        "initial_converged"
    ]

    initial_results[
        "iterations"
    ] = initial_results[
        "initial_iterations"
    ]

    initial_convergence_summary = (
        summarize_convergence_diagnostics(
            initial_results
        )
    )

    fit_diagnostics = (
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
                "retry_parameter",
                "retry_parameter_offset",
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

    print()
    print(
        all_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Initial non-converged:",
        (
            ~fit_diagnostics[
                "initial_converged"
            ].astype(bool)
        ).sum(),
    )

    print(
        "Retries accepted:",
        fit_diagnostics[
            "retry_accepted"
        ].sum(),
    )

    print(
        "Final non-converged:",
        (
            ~fit_diagnostics[
                "converged"
            ].astype(bool)
        ).sum(),
    )

    all_results.to_csv(
        FORECAST_DIR
        / f"{output_prefix}_classical_forecasts.csv",
        index=False,
    )

    all_summary.to_csv(
        METRICS_DIR
        / f"{output_prefix}_classical_metrics.csv",
        index=False,
    )

    final_convergence_summary.to_csv(
        METRICS_DIR
        / f"{output_prefix}_arima_convergence.csv",
        index=False,
    )

    initial_convergence_summary.to_csv(
        METRICS_DIR
        / f"{output_prefix}_arima_initial_convergence.csv",
        index=False,
    )

    fit_diagnostics.to_csv(
        METRICS_DIR
        / f"{output_prefix}_arima_fit_diagnostics.csv",
        index=False,
    )


def main():
    # -----------------------------------------------------
    # Airline passengers
    # -----------------------------------------------------

    airline = prepare_airline(
        load_airline()
    )

    # A bounded p, q, P, Q in {0, 1} search on observations
    # before 1959 selected the canonical airline model by
    # both AIC and BIC.
    run_classical_dataset(
        data=airline,
        dataset_name="Airline",
        output_prefix="airline",
        test_start=pd.Timestamp(
            "1959-01-01"
        ),
        horizons=[
            1,
            3,
            12,
        ],
        step=1,
        mase_seasonality=12,
        arima_name=(
            "Log ARIMA(0,1,1)"
            "(0,1,1)[12]"
        ),
        arima_kwargs={
            "order": (0, 1, 1),
            "seasonal_order": (0, 1, 1, 12),
            "trend": "n",
            "log_transform": True,
        },
    )

    # -----------------------------------------------------
    # CISO electricity demand
    # -----------------------------------------------------

    electricity = prepare_electricity(
        load_electricity(
            balancing_authority="CISO"
        )
    )

    # A bounded p, q in {0, 1, 2} search on weekly log
    # differences before 2025 selected ARIMA(2,0,2) by both
    # AIC and BIC. Weekly differencing retains the hourly
    # data while handling the dominant hour-of-week pattern.
    run_classical_dataset(
        data=electricity,
        dataset_name="Electricity-CISO",
        output_prefix="electricity_ciso",
        test_start=pd.Timestamp(
            "2025-01-01",
            tz="UTC",
        ),
        horizons=[
            1,
            24,
            168,
        ],
        step=168,
        mase_seasonality=168,
        arima_name=(
            "Log weekly-difference "
            "ARIMA(2,0,2)"
        ),
        arima_kwargs={
            "order": (2, 0, 2),
            "trend": "c",
            "log_transform": True,
            "seasonal_difference": 168,
            "low_memory": True,
        },
    )

    print()
    print(
        "Airline and electricity results saved "
        "successfully."
    )


if __name__ == "__main__":
    main()
