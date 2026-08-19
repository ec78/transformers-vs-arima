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

from src.models_patchtst import (
    PatchTSTTrainingConfig,
    patchtst_forecast,
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


def _origin_keys(values):
    return [
        _timestamp_key(value)
        for value in values
    ]


def assert_frozen_origins(
    observed_origins,
    reference_files,
):
    observed = _origin_keys(
        observed_origins
    )

    for reference_file in reference_files:
        reference = pd.read_csv(
            reference_file,
            usecols=["origin"],
            parse_dates=["origin"],
        )

        expected = _origin_keys(
            reference[
                "origin"
            ].drop_duplicates()
        )

        if observed != expected:
            raise AssertionError(
                "PatchTST origins differ from frozen file "
                f"{reference_file.name}."
            )


def audit_patchtst_results(
    results,
    data,
    config,
    expected_origins,
):
    if results[
        "origin"
    ].nunique() != expected_origins:
        raise AssertionError(
            "Unexpected number of PatchTST origins."
        )

    if not results[
        "forecast_is_finite"
    ].astype(bool).all():
        raise AssertionError(
            "PatchTST produced a non-finite forecast."
        )

    if not results[
        "parameters_are_finite"
    ].astype(bool).all():
        raise AssertionError(
            "PatchTST produced non-finite parameters."
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
                "PatchTST forecast timestamp is misaligned."
            )

        if row.actual != data.loc[
            row.forecast_date,
            "y",
        ]:
            raise AssertionError(
                "PatchTST actual does not match source data."
            )

        if row.origin_actual != data.loc[
            row.origin,
            "y",
        ]:
            raise AssertionError(
                "PatchTST origin value is incorrect."
            )

        train = data[
            "y"
        ].iloc[
            :origin_position + 1
        ].to_numpy(
            dtype=float
        )

        m = config[
            "mase_seasonality"
        ]

        expected_scale = np.mean(
            np.abs(
                train[m:]
                - train[:-m]
            )
        )

        if not np.isclose(
            row.mase_scale,
            expected_scale,
        ):
            raise AssertionError(
                "PatchTST MASE scale is incorrect."
            )

        if _timestamp_key(
            row.forecast_origin
        ) != _timestamp_key(
            row.origin
        ):
            raise AssertionError(
                "PatchTST training extends beyond origin."
            )

        if not (
            pd.Timestamp(
                row.training_data_end
            )
            < pd.Timestamp(
                row.validation_start
            )
            <= pd.Timestamp(
                row.validation_end
            )
            == pd.Timestamp(
                row.origin
            )
        ):
            raise AssertionError(
                "PatchTST validation chronology is invalid."
            )

        if pd.Timestamp(
            row.scaler_fit_end
        ) != pd.Timestamp(
            row.training_data_end
        ):
            raise AssertionError(
                "PatchTST scaler used validation data."
            )

        reconstructed = (
            row.standardized_forecast[
                row.horizon - 1
            ]
            * row.scaler_std
            + row.scaler_mean
        )

        if not np.isclose(
            row.forecast,
            reconstructed,
        ):
            raise AssertionError(
                "PatchTST forecast was not restored to "
                "the original scale."
            )

        if row.forecast_scale != "original":
            raise AssertionError(
                "PatchTST forecast scale is not original."
            )


def patchtst_summary(results):
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
            [
                "training_elapsed_seconds",
                "inference_elapsed_seconds",
            ]
        ]
        .mean()
        .rename(
            columns={
                "training_elapsed_seconds": (
                    "mean_training_seconds"
                ),
                "inference_elapsed_seconds": (
                    "mean_inference_seconds"
                ),
            }
        )
    )

    return summary.drop(
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


def run_dataset(config, device):
    data = config["data"]()
    training_config = config[
        "training_config"
    ]

    initial_train_size = (
        data.index.searchsorted(
            config["test_start"]
        )
    )

    origins = generate_forecast_origins(
        data=data,
        initial_train_size=initial_train_size,
        max_horizon=max(
            config["horizons"]
        ),
        step=config["step"],
    )

    origin_dates = [
        data.index[
            origin - 1
        ]
        for origin in origins
    ]

    assert_frozen_origins(
        observed_origins=origin_dates,
        reference_files=[
            FORECAST_DIR
            / config[
                "classical_forecast_file"
            ],
            FORECAST_DIR
            / config[
                "chronos_forecast_file"
            ],
        ],
    )

    print()
    print("=" * 60)
    print(config["dataset_name"])
    print("=" * 60)
    print(
        "Forecast origins:",
        len(origins),
    )
    print(
        "Configuration:",
        training_config.identifier,
    )

    results = walk_forward_evaluate(
        data=data,
        target_col="y",
        model_func=patchtst_forecast,
        model_name="PatchTST",
        dataset_name=config[
            "dataset_name"
        ],
        horizons=config["horizons"],
        initial_train_size=initial_train_size,
        step=config["step"],
        window="expanding",
        model_kwargs={
            "training_config": training_config,
            "device": device,
        },
        mase_seasonality=config[
            "mase_seasonality"
        ],
    )

    results[
        "absolute_error"
    ] = np.abs(
        results["actual"]
        - results["forecast"]
    )

    results[
        "squared_error"
    ] = (
        results["actual"]
        - results["forecast"]
    ) ** 2

    assert_frozen_origins(
        observed_origins=results[
            "origin"
        ].drop_duplicates(),
        reference_files=[
            FORECAST_DIR
            / config[
                "classical_forecast_file"
            ],
            FORECAST_DIR
            / config[
                "chronos_forecast_file"
            ],
        ],
    )

    audit_patchtst_results(
        results=results,
        data=data,
        config=config,
        expected_origins=config[
            "expected_origins"
        ],
    )

    results[
        "standardized_prediction"
    ] = (
        results["forecast"]
        - results["scaler_mean"]
    ) / results["scaler_std"]

    results = results.drop(
        columns=[
            "standardized_forecast",
        ]
    )

    summary = patchtst_summary(
        results
    )

    diagnostics = (
        results
        .groupby(
            "origin",
            as_index=False,
        )
        .first()
        [
            [
                "origin",
                "forecast_origin",
                "training_data_end",
                "validation_start",
                "validation_end",
                "scaler_fit_end",
                "scaler_mean",
                "scaler_std",
                "context_length",
                "patch_length",
                "patch_stride",
                "prediction_length",
                "window_stride",
                "num_training_windows",
                "num_validation_windows",
                "epochs_trained",
                "best_epoch",
                "early_stopped",
                "best_validation_loss",
                "training_elapsed_seconds",
                "inference_elapsed_seconds",
                "num_parameters",
                "forecast_is_finite",
                "parameters_are_finite",
                "training_warning",
                "training_exception",
                "model_identifier",
                "model_configuration",
                "transformers_version",
                "device",
            ]
        ]
    )

    results.to_csv(
        FORECAST_DIR
        / config[
            "patchtst_forecast_file"
        ],
        index=False,
    )

    summary.to_csv(
        METRICS_DIR
        / config[
            "patchtst_metrics_file"
        ],
        index=False,
    )

    diagnostics.to_csv(
        METRICS_DIR
        / config[
            "patchtst_diagnostics_file"
        ],
        index=False,
    )

    print(
        summary.to_string(
            index=False
        )
    )

    return summary


def build_comparison():
    benchmark = pd.read_csv(
        METRICS_DIR
        / "classical_chronos_comparison.csv"
    )

    benchmark[
        "mean_training_seconds"
    ] = np.where(
        benchmark["model"].str.contains(
            "ARIMA"
        ),
        benchmark[
            "mean_inference_time"
        ],
        0.0,
    )

    benchmark[
        "mean_inference_seconds"
    ] = np.where(
        benchmark["model"].str.contains(
            "ARIMA"
        ),
        np.nan,
        benchmark[
            "mean_inference_time"
        ],
    )

    patchtst = pd.concat(
        [
            pd.read_csv(path)
            for path in [
                METRICS_DIR
                / "spy_patchtst_metrics.csv",
                METRICS_DIR
                / "airline_patchtst_metrics.csv",
                METRICS_DIR
                / "electricity_ciso_patchtst_metrics.csv",
            ]
        ],
        ignore_index=True,
    )

    comparison = pd.concat(
        [
            benchmark,
            patchtst,
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
            "mean_training_seconds",
            "mean_inference_seconds",
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
        / (
            "classical_chronos_patchtst_"
            "comparison.csv"
        ),
        index=False,
    )

    return comparison


def dataset_configs():
    return {
        "spy": {
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
            "patchtst_forecast_file": (
                "spy_patchtst_forecasts.csv"
            ),
            "patchtst_metrics_file": (
                "spy_patchtst_metrics.csv"
            ),
            "patchtst_diagnostics_file": (
                "spy_patchtst_training_diagnostics.csv"
            ),
            "training_config": PatchTSTTrainingConfig(
                context_length=252,
                patch_length=21,
                patch_stride=7,
                prediction_length=20,
                validation_points=252,
                window_stride=5,
                batch_size=64,
            ),
        },
        "airline": {
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
            "patchtst_forecast_file": (
                "airline_patchtst_forecasts.csv"
            ),
            "patchtst_metrics_file": (
                "airline_patchtst_metrics.csv"
            ),
            "patchtst_diagnostics_file": (
                "airline_patchtst_training_diagnostics.csv"
            ),
            "training_config": PatchTSTTrainingConfig(
                context_length=36,
                patch_length=12,
                patch_stride=6,
                prediction_length=12,
                validation_points=24,
                window_stride=1,
                batch_size=16,
            ),
        },
        "electricity": {
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
            "patchtst_forecast_file": (
                "electricity_ciso_patchtst_forecasts.csv"
            ),
            "patchtst_metrics_file": (
                "electricity_ciso_patchtst_metrics.csv"
            ),
            "patchtst_diagnostics_file": (
                "electricity_ciso_patchtst_training_"
                "diagnostics.csv"
            ),
            "training_config": PatchTSTTrainingConfig(
                context_length=336,
                patch_length=24,
                patch_stride=12,
                prediction_length=168,
                validation_points=1344,
                window_stride=168,
                batch_size=32,
            ),
        },
    }


def main():
    import torch

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=[
            "all",
            "spy",
            "airline",
            "electricity",
        ],
        default="all",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=6,
    )

    args = parser.parse_args()

    torch.set_num_threads(
        args.threads
    )

    device = "cpu"
    configs = dataset_configs()

    selected = (
        list(configs)
        if args.dataset == "all"
        else [args.dataset]
    )

    for name in selected:
        config = configs[name]

        try:
            run_dataset(
                config=config,
                device=device,
            )
        except Exception as error:
            pd.DataFrame(
                [
                    {
                        "dataset": config[
                            "dataset_name"
                        ],
                        "model": "PatchTST",
                        "model_identifier": (
                            "PatchTST-HF-supervised-v1"
                        ),
                        "device": device,
                        "training_exception": str(
                            error
                        ),
                    }
                ]
            ).to_csv(
                METRICS_DIR
                / config[
                    "patchtst_metrics_file"
                ].replace(
                    "_metrics.csv",
                    "_errors.csv",
                ),
                index=False,
            )

            raise

    expected_metric_files = [
        METRICS_DIR
        / config[
            "patchtst_metrics_file"
        ]
        for config in configs.values()
    ]

    if all(
        path.exists()
        for path in expected_metric_files
    ):
        comparison = build_comparison()

        print()
        print("=" * 60)
        print("FULL BENCHMARK COMPARISON")
        print("=" * 60)
        print(
            comparison.to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()
