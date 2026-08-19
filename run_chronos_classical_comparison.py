import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import (
    load_spy,
    load_airline,
    load_electricity,
)

from src.preprocessing import (
    prepare_spy,
    prepare_airline,
    prepare_electricity,
)

from src.models_transformer import (
    CHRONOS_2_MODEL_ID,
    CHRONOS_2_REVISION,
    Chronos2ForecastAdapter,
    load_chronos_2_pipeline,
)

from src.evaluation import (
    generate_forecast_origins,
    summarize_results,
    walk_forward_evaluate,
)


PROJECT_ROOT = Path(__file__).resolve().parent
FORECAST_DIR = PROJECT_ROOT / "results" / "forecasts"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

FORECAST_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def _timestamp_key(value):
    return pd.Timestamp(
        value
    ).isoformat()


def _assert_frozen_origins(
    results,
    classical_path,
):
    classical = pd.read_csv(
        classical_path,
        parse_dates=["origin"],
    )

    expected = [
        _timestamp_key(value)
        for value in classical[
            "origin"
        ].drop_duplicates()
    ]

    observed = [
        _timestamp_key(value)
        for value in results[
            "origin"
        ].drop_duplicates()
    ]

    if observed != expected:
        raise AssertionError(
            "Chronos origins differ from the frozen "
            "classical origins."
        )


def audit_chronos_results(
    results,
    data,
    mase_seasonality,
    expected_origins,
    max_context_length,
):
    """Independently validate saved evaluation fields."""

    origins = results[
        "origin"
    ].drop_duplicates()

    if len(origins) != expected_origins:
        raise AssertionError(
            "Unexpected number of Chronos origins."
        )

    if not results[
        "forecast_is_finite"
    ].astype(bool).all():
        raise AssertionError(
            "Chronos produced a non-finite forecast."
        )

    if not np.isfinite(
        results["forecast"]
    ).all():
        raise AssertionError(
            "A saved Chronos forecast is not finite."
        )

    if results[
        "inference_exception"
    ].fillna("").ne("").any():
        raise AssertionError(
            "Chronos reported an inference exception."
        )

    for row in results.itertuples(
        index=False
    ):
        origin_position = data.index.get_loc(
            row.origin
        )

        forecast_position = data.index.get_loc(
            row.forecast_date
        )

        if forecast_position != (
            origin_position
            + row.horizon
        ):
            raise AssertionError(
                "Forecast timestamp is misaligned."
            )

        if row.actual != data.loc[
            row.forecast_date,
            "y",
        ]:
            raise AssertionError(
                "Saved actual does not match source data."
            )

        if row.origin_actual != data.loc[
            row.origin,
            "y",
        ]:
            raise AssertionError(
                "Saved origin value does not match source data."
            )

        train = data[
            "y"
        ].iloc[
            :origin_position + 1
        ].to_numpy(
            dtype=float
        )

        expected_scale = np.mean(
            np.abs(
                train[
                    mase_seasonality:
                ]
                - train[
                    :-mase_seasonality
                ]
            )
        )

        if not np.isclose(
            row.mase_scale,
            expected_scale,
        ):
            raise AssertionError(
                "Origin MASE scale is incorrect."
            )

        expected_scaled_error = (
            abs(
                row.actual
                - row.forecast
            )
            / expected_scale
        )

        if not np.isclose(
            row.scaled_absolute_error,
            expected_scaled_error,
        ):
            raise AssertionError(
                "Scaled absolute error is incorrect."
            )

        if row.context_length != min(
            row.train_size,
            max_context_length,
        ):
            raise AssertionError(
                "Chronos context length is incorrect."
            )

        if row.history_length != row.train_size:
            raise AssertionError(
                "Chronos history length is incorrect."
            )

        if _timestamp_key(
            row.context_end
        ) != _timestamp_key(
            row.origin
        ):
            raise AssertionError(
                "Chronos context includes data beyond origin."
            )


def _chronos_summary(results):
    summary = summarize_results(
        results=results,
        include_direction=True,
    )

    timing = (
        results
        .groupby(
            [
                "dataset",
                "model",
                "horizon",
            ],
            as_index=False,
        )[
            "inference_elapsed_seconds"
        ]
        .mean()
        .rename(
            columns={
                "inference_elapsed_seconds": (
                    "mean_inference_time"
                )
            }
        )
    )

    summary = summary.drop(
        columns=[
            "mean_elapsed_seconds",
        ]
    ).merge(
        timing,
        on=[
            "dataset",
            "model",
            "horizon",
        ],
        how="left",
        validate="one_to_one",
    )

    summary[
        "mean_elapsed_seconds"
    ] = summary[
        "mean_inference_time"
    ]

    return summary


def run_dataset(
    config,
    adapter,
    batch_size,
):
    data = config["data"]()
    horizons = config["horizons"]
    initial_train_size = (
        data.index.searchsorted(
            config["test_start"]
        )
    )

    origins = generate_forecast_origins(
        data=data,
        initial_train_size=initial_train_size,
        max_horizon=max(horizons),
        step=config["step"],
    )

    classical_path = (
        FORECAST_DIR
        / config[
            "classical_forecast_file"
        ]
    )

    preflight_origins = pd.DataFrame(
        {
            "origin": [
                data.index[
                    origin - 1
                ]
                for origin in origins
            ]
        }
    )

    _assert_frozen_origins(
        results=preflight_origins,
        classical_path=classical_path,
    )

    print()
    print("=" * 60)
    print(config["dataset_name"])
    print("=" * 60)
    print(
        "Forecast origins:",
        len(origins),
    )

    adapter.prepare_origins(
        series=data["y"],
        origins=origins,
        horizon=max(horizons),
        batch_size=batch_size,
    )

    results = walk_forward_evaluate(
        data=data,
        target_col="y",
        model_func=adapter,
        model_name="Chronos-2",
        dataset_name=config[
            "dataset_name"
        ],
        horizons=horizons,
        initial_train_size=initial_train_size,
        step=config["step"],
        window="expanding",
        mase_seasonality=config[
            "mase_seasonality"
        ],
    )

    _assert_frozen_origins(
        results=results,
        classical_path=classical_path,
    )

    audit_chronos_results(
        results=results,
        data=data,
        mase_seasonality=config[
            "mase_seasonality"
        ],
        expected_origins=config[
            "expected_origins"
        ],
        max_context_length=(
            adapter.max_context_length
        ),
    )

    summary = _chronos_summary(
        results
    )

    results.to_csv(
        FORECAST_DIR
        / config[
            "chronos_forecast_file"
        ],
        index=False,
    )

    summary.to_csv(
        METRICS_DIR
        / config[
            "chronos_metrics_file"
        ],
        index=False,
    )

    error_path = (
        METRICS_DIR
        / config[
            "chronos_metrics_file"
        ].replace(
            "_metrics.csv",
            "_errors.csv",
        )
    )

    if error_path.exists():
        error_path.unlink()

    print(
        summary.to_string(
            index=False
        )
    )

    return summary


def build_consolidated_comparison(
    chronos_summaries,
):
    classical_files = [
        "spy_classical_metrics.csv",
        "airline_classical_metrics.csv",
        "electricity_ciso_classical_metrics.csv",
    ]

    classical = pd.concat(
        [
            pd.read_csv(
                METRICS_DIR / filename
            )
            for filename in classical_files
        ],
        ignore_index=True,
    )

    classical = classical.rename(
        columns={
            "mean_elapsed_seconds": (
                "mean_inference_time"
            )
        }
    )

    chronos = pd.concat(
        chronos_summaries,
        ignore_index=True,
    )

    comparison = pd.concat(
        [
            classical,
            chronos,
        ],
        ignore_index=True,
    )

    comparison = comparison[
        [
            "dataset",
            "model",
            "horizon",
            "n_forecasts",
            "rmse",
            "mae",
            "mase",
            "directional_accuracy",
            "mean_inference_time",
        ]
    ].sort_values(
        [
            "dataset",
            "horizon",
            "model",
        ]
    ).reset_index(
        drop=True
    )

    comparison.to_csv(
        METRICS_DIR
        / "classical_chronos_comparison.csv",
        index=False,
    )

    return comparison


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        choices=[
            "cpu",
            "cuda",
        ],
        default=None,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    pipeline, device = (
        load_chronos_2_pipeline(
            model_id=CHRONOS_2_MODEL_ID,
            device=args.device,
        )
    )

    adapter = Chronos2ForecastAdapter(
        pipeline=pipeline,
        device=device,
    )

    configs = [
        {
            "dataset_name": "SPY",
            "data": lambda: prepare_spy(
                load_spy()
            ),
            "test_start": pd.Timestamp(
                "2019-01-01"
            ),
            "horizons": [1, 5, 20],
            "step": 5,
            "mase_seasonality": 1,
            "expected_origins": 380,
            "classical_forecast_file": (
                "spy_classical_forecasts.csv"
            ),
            "chronos_forecast_file": (
                "spy_chronos_forecasts.csv"
            ),
            "chronos_metrics_file": (
                "spy_chronos_metrics.csv"
            ),
        },
        {
            "dataset_name": "Airline",
            "data": lambda: prepare_airline(
                load_airline()
            ),
            "test_start": pd.Timestamp(
                "1959-01-01"
            ),
            "horizons": [1, 3, 12],
            "step": 1,
            "mase_seasonality": 12,
            "expected_origins": 13,
            "classical_forecast_file": (
                "airline_classical_forecasts.csv"
            ),
            "chronos_forecast_file": (
                "airline_chronos_forecasts.csv"
            ),
            "chronos_metrics_file": (
                "airline_chronos_metrics.csv"
            ),
        },
        {
            "dataset_name": "Electricity-CISO",
            "data": lambda: prepare_electricity(
                load_electricity(
                    balancing_authority="CISO"
                )
            ),
            "test_start": pd.Timestamp(
                "2025-01-01",
                tz="UTC",
            ),
            "horizons": [1, 24, 168],
            "step": 168,
            "mase_seasonality": 168,
            "expected_origins": 84,
            "classical_forecast_file": (
                "electricity_ciso_classical_forecasts.csv"
            ),
            "chronos_forecast_file": (
                "electricity_ciso_chronos_forecasts.csv"
            ),
            "chronos_metrics_file": (
                "electricity_ciso_chronos_metrics.csv"
            ),
        },
    ]

    summaries = []

    for config in configs:
        try:
            summary = run_dataset(
                config=config,
                adapter=adapter,
                batch_size=args.batch_size,
            )
        except Exception as error:
            pd.DataFrame(
                [
                    {
                        "dataset": config[
                            "dataset_name"
                        ],
                        "model": "Chronos-2",
                        "model_identifier": (
                            CHRONOS_2_MODEL_ID
                        ),
                        "model_revision": (
                            CHRONOS_2_REVISION
                        ),
                        "device": device,
                        "inference_exception": str(
                            error
                        ),
                    }
                ]
            ).to_csv(
                METRICS_DIR
                / (
                    config[
                        "chronos_metrics_file"
                    ].replace(
                        "_metrics.csv",
                        "_errors.csv",
                    )
                ),
                index=False,
            )

            raise

        summaries.append(
            summary
        )

    comparison = (
        build_consolidated_comparison(
            summaries
        )
    )

    print()
    print("=" * 60)
    print("CLASSICAL AND CHRONOS COMPARISON")
    print("=" * 60)
    print(
        comparison.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
