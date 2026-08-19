"""Analysis utilities for the frozen forecasting experiment.

This module reads saved forecast and metric CSVs only.  It deliberately has no
dependency on model adapters, data loaders, or preprocessing code so running the
analysis cannot fit a model or perform inference.
"""

from __future__ import annotations

import ast
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


DATASET_SPECS = {
    "SPY": {
        "file_stem": "spy",
        "horizons": [1, 5, 20],
        "expected_origins": 380,
        "origin_step": 5,
        "arima_model": "ARIMA(0,1,2)+drift",
    },
    "Airline": {
        "file_stem": "airline",
        "horizons": [1, 3, 12],
        "expected_origins": 13,
        "origin_step": 1,
        "arima_model": "Log ARIMA(0,1,1)(0,1,1)[12]",
    },
    "Electricity-CISO": {
        "file_stem": "electricity_ciso",
        "horizons": [1, 24, 168],
        "expected_origins": 84,
        "origin_step": 168,
        "arima_model": "Log weekly-difference ARIMA(2,0,2)",
    },
}

MODEL_FAMILIES = {
    "Naive": "Naive",
    "Chronos-2": "Chronos-2",
    "PatchTST": "PatchTST",
}

ACCURACY_COLUMNS = [
    "rmse",
    "mae",
    "mase",
    "directional_accuracy",
]


def model_family(model):
    """Return the comparable model family for a saved model label."""

    if model in MODEL_FAMILIES:
        return MODEL_FAMILIES[model]

    if "ARIMA" in model:
        return "ARIMA/SARIMA"

    raise ValueError(
        f"Unrecognized frozen model label: {model}"
    )


def frozen_input_paths(project_root):
    """Return every frozen file consumed or checked by the analysis."""

    project_root = Path(project_root)
    forecast_dir = project_root / "results" / "forecasts"
    metrics_dir = project_root / "results" / "metrics"

    paths = []
    for spec in DATASET_SPECS.values():
        stem = spec["file_stem"]
        paths.extend(
            [
                forecast_dir / f"{stem}_classical_forecasts.csv",
                forecast_dir / f"{stem}_chronos_forecasts.csv",
                forecast_dir / f"{stem}_patchtst_forecasts.csv",
            ]
        )

    paths.append(
        metrics_dir
        / "classical_chronos_patchtst_comparison.csv"
    )
    return paths


def build_input_manifest(project_root):
    """Hash the frozen inputs so the analysis source is unambiguous."""

    rows = []
    for path in frozen_input_paths(project_root):
        if not path.exists():
            raise FileNotFoundError(
                f"Required frozen input does not exist: {path}"
            )

        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(
            {
                "path": str(path.relative_to(project_root)),
                "size_bytes": path.stat().st_size,
                "sha256": digest,
            }
        )

    return pd.DataFrame(rows)


def _read_forecast(path):
    result = pd.read_csv(path)
    result["origin"] = pd.to_datetime(result["origin"], utc=True)
    result["forecast_date"] = pd.to_datetime(
        result["forecast_date"],
        utc=True,
    )
    result["source_file"] = path.name
    return result


def load_frozen_forecasts(project_root):
    """Load the nine authoritative frozen forecast files."""

    project_root = Path(project_root)
    forecast_dir = project_root / "results" / "forecasts"
    frames = []

    for dataset, spec in DATASET_SPECS.items():
        stem = spec["file_stem"]
        classical = _read_forecast(
            forecast_dir / f"{stem}_classical_forecasts.csv"
        )
        chronos = _read_forecast(
            forecast_dir / f"{stem}_chronos_forecasts.csv"
        )
        patchtst = _read_forecast(
            forecast_dir / f"{stem}_patchtst_forecasts.csv"
        )

        for frame in [classical, chronos, patchtst]:
            observed = set(frame["dataset"].unique())
            if observed != {dataset}:
                raise AssertionError(
                    f"Unexpected dataset labels in {frame.source_file.iloc[0]}: "
                    f"{observed}"
                )

        frames.extend([classical, chronos, patchtst])

    forecasts = pd.concat(frames, ignore_index=True, sort=False)
    forecasts["model_family"] = forecasts["model"].map(model_family)
    return forecasts


def validate_frozen_forecasts(forecasts):
    """Validate counts, alignment, and saved error arithmetic."""

    checks = []

    def record(check, passed, details):
        checks.append(
            {
                "check": check,
                "passed": bool(passed),
                "details": details,
            }
        )
        if not passed:
            raise AssertionError(f"{check}: {details}")

    key = ["dataset", "model", "origin", "horizon"]
    duplicates = forecasts.duplicated(key).sum()
    record(
        "unique forecast keys",
        duplicates == 0,
        f"duplicate rows={duplicates}",
    )

    finite = np.isfinite(
        forecasts[["actual", "forecast", "mase_scale"]]
    ).all().all()
    record(
        "finite saved forecasts and scales",
        finite,
        f"rows={len(forecasts)}",
    )

    positive_scales = (forecasts["mase_scale"] > 0).all()
    record(
        "positive MASE scales",
        positive_scales,
        "all origin-specific scales must be positive",
    )

    expected_scaled = (
        (forecasts["actual"] - forecasts["forecast"]).abs()
        / forecasts["mase_scale"]
    )
    scaled_matches = np.allclose(
        forecasts["scaled_absolute_error"],
        expected_scaled,
        rtol=1e-12,
        atol=1e-12,
    )
    record(
        "saved scaled errors",
        scaled_matches,
        "abs(actual - forecast) / mase_scale",
    )

    actual_direction = np.sign(
        forecasts["actual"] - forecasts["origin_actual"]
    )
    predicted_direction = np.sign(
        forecasts["forecast"] - forecasts["origin_actual"]
    )
    direction_matches = (
        np.array_equal(
            forecasts["actual_direction"].to_numpy(),
            actual_direction.to_numpy(),
        )
        and np.array_equal(
            forecasts["predicted_direction"].to_numpy(),
            predicted_direction.to_numpy(),
        )
        and np.array_equal(
            forecasts["direction_correct"].astype(bool).to_numpy(),
            (actual_direction == predicted_direction).to_numpy(),
        )
    )
    record(
        "origin-relative directions",
        direction_matches,
        "direction equals sign(value - origin_actual)",
    )

    for dataset, spec in DATASET_SPECS.items():
        data = forecasts[forecasts["dataset"] == dataset]
        expected_models = {
            "Naive",
            spec["arima_model"],
            "Chronos-2",
            "PatchTST",
        }
        record(
            f"{dataset} model set",
            set(data["model"].unique()) == expected_models,
            ", ".join(sorted(data["model"].unique())),
        )

        for model in sorted(expected_models):
            group = data[data["model"] == model]
            origin_count = group["origin"].nunique()
            horizons = sorted(group["horizon"].unique().tolist())
            expected_rows = (
                spec["expected_origins"]
                * len(spec["horizons"])
            )
            record(
                f"{dataset} / {model} frozen contract",
                origin_count == spec["expected_origins"]
                and horizons == spec["horizons"]
                and len(group) == expected_rows,
                (
                    f"origins={origin_count}; horizons={horizons}; "
                    f"rows={len(group)}"
                ),
            )

        reference = (
            data[data["model"] == "Naive"]
            .sort_values(["origin", "horizon"])
            .reset_index(drop=True)
        )
        alignment_columns = [
            "origin",
            "forecast_date",
            "horizon",
            "origin_actual",
            "actual",
            "mase_scale",
        ]
        for model in sorted(expected_models - {"Naive"}):
            candidate = (
                data[data["model"] == model]
                .sort_values(["origin", "horizon"])
                .reset_index(drop=True)
            )
            exact_columns = ["origin", "forecast_date", "horizon"]
            numeric_columns = ["origin_actual", "actual", "mase_scale"]
            aligned = all(
                reference[column].equals(candidate[column])
                for column in exact_columns
            ) and np.allclose(
                reference[numeric_columns],
                candidate[numeric_columns],
                rtol=1e-12,
                atol=1e-12,
            )
            record(
                f"{dataset} / {model} target alignment",
                aligned,
                ", ".join(alignment_columns),
            )

    return pd.DataFrame(checks)


def _origin_timing(group, column):
    if column not in group.columns:
        return np.nan, np.nan

    timing = (
        group[["origin", column]]
        .drop_duplicates("origin")
        [column]
        .astype(float)
    )
    return timing.mean(), timing.sum()


def build_authoritative_comparison(forecasts):
    """Aggregate authoritative metrics directly from origin-level rows."""

    rows = []
    grouped = forecasts.groupby(
        ["dataset", "model", "model_family", "horizon"],
        sort=False,
        dropna=False,
    )

    for (dataset, model, family, horizon), group in grouped:
        errors = group["actual"] - group["forecast"]

        if family == "ARIMA/SARIMA":
            mean_training, _ = _origin_timing(
                group,
                "elapsed_seconds",
            )
            mean_inference = np.nan
        elif family == "PatchTST":
            mean_training, _ = _origin_timing(
                group,
                "training_elapsed_seconds",
            )
            mean_inference, _ = _origin_timing(
                group,
                "inference_elapsed_seconds",
            )
        elif family == "Chronos-2":
            mean_training = 0.0
            mean_inference, _ = _origin_timing(
                group,
                "inference_elapsed_seconds",
            )
        else:
            mean_training = 0.0
            mean_inference, _ = _origin_timing(
                group,
                "elapsed_seconds",
            )

        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "model_family": family,
                "horizon": int(horizon),
                "n_forecasts": len(group),
                "rmse": np.sqrt(np.mean(np.square(errors))),
                "mae": np.mean(np.abs(errors)),
                "mase": group["scaled_absolute_error"].mean(),
                "directional_accuracy": (
                    group["direction_correct"].astype(bool).mean()
                ),
                "mean_training_time_per_origin": mean_training,
                "mean_inference_time_per_origin": mean_inference,
            }
        )

    result = pd.DataFrame(rows)
    return result.sort_values(
        ["dataset", "horizon", "model_family"]
    ).reset_index(drop=True)


def validate_saved_metrics(comparison, project_root):
    """Confirm recomputed metrics match the previously saved comparison."""

    saved_path = (
        Path(project_root)
        / "results"
        / "metrics"
        / "classical_chronos_patchtst_comparison.csv"
    )
    saved = pd.read_csv(saved_path)
    merged = comparison.merge(
        saved,
        on=["dataset", "model", "horizon"],
        suffixes=("_recomputed", "_saved"),
        validate="one_to_one",
    )
    if len(merged) != len(comparison) or len(merged) != len(saved):
        raise AssertionError(
            "Saved comparison and recomputed comparison have different rows."
        )

    checks = []
    for column in ["n_forecasts", *ACCURACY_COLUMNS]:
        left = merged[f"{column}_recomputed"].to_numpy(dtype=float)
        right = merged[f"{column}_saved"].to_numpy(dtype=float)
        matches = np.allclose(
            left,
            right,
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )
        maximum_difference = np.nanmax(np.abs(left - right))
        checks.append(
            {
                "check": f"recomputed {column} matches frozen metrics",
                "passed": bool(matches),
                "details": f"maximum absolute difference={maximum_difference}",
            }
        )
        if not matches:
            raise AssertionError(checks[-1]["check"])

    return pd.DataFrame(checks)


def percentage_difference(value, baseline):
    """Percentage difference where negative means lower error/better."""

    value = np.asarray(value, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    return 100.0 * (value / baseline - 1.0)


def build_relative_performance(comparison):
    """Calculate every requested frozen-baseline comparison."""

    rows = []
    for (dataset, horizon), group in comparison.groupby(
        ["dataset", "horizon"],
        sort=False,
    ):
        spec = DATASET_SPECS[dataset]
        indexed = group.set_index("model")
        baseline_models = {
            "naive": "Naive",
            "arima": spec["arima_model"],
            "chronos": "Chronos-2",
        }

        for row in group.itertuples(index=False):
            output = row._asdict()
            for metric in ["rmse", "mae", "mase"]:
                for baseline_name, baseline_model in baseline_models.items():
                    output[
                        f"{metric}_pct_difference_vs_{baseline_name}"
                    ] = float(
                        percentage_difference(
                            getattr(row, metric),
                            indexed.loc[baseline_model, metric],
                        )
                    )
            output["arima_baseline"] = spec["arima_model"]
            rows.append(output)

    return pd.DataFrame(rows).sort_values(
        ["dataset", "horizon", "model_family"]
    ).reset_index(drop=True)


def build_rankings(comparison):
    """Rank models independently for RMSE, MAE, and MASE."""

    records = []
    for (dataset, horizon), group in comparison.groupby(
        ["dataset", "horizon"],
        sort=False,
    ):
        for metric in ["rmse", "mae", "mase"]:
            ranked = group[["model", "model_family", metric]].copy()
            ranked["rank"] = ranked[metric].rank(
                method="min",
                ascending=True,
            )
            for row in ranked.itertuples(index=False):
                records.append(
                    {
                        "dataset": dataset,
                        "horizon": int(horizon),
                        "metric": metric,
                        "model": row.model,
                        "model_family": row.model_family,
                        "value": getattr(row, metric),
                        "rank": int(row.rank),
                    }
                )

    rankings = pd.DataFrame(records).sort_values(
        ["dataset", "horizon", "metric", "rank", "model"]
    ).reset_index(drop=True)
    win_counts = (
        rankings[rankings["rank"] == 1]
        .groupby(["model_family", "metric"], as_index=False)
        .size()
        .rename(columns={"size": "n_dataset_horizon_wins"})
    )
    complete_index = pd.MultiIndex.from_product(
        [
            ["Naive", "ARIMA/SARIMA", "Chronos-2", "PatchTST"],
            ["rmse", "mae", "mase"],
        ],
        names=["model_family", "metric"],
    )
    win_counts = (
        win_counts.set_index(["model_family", "metric"])
        .reindex(complete_index, fill_value=0)
        .reset_index()
    )
    return rankings, win_counts


def _newey_west_bandwidth(n_observations):
    return int(
        math.floor(4.0 * (n_observations / 100.0) ** (2.0 / 9.0))
    )


def diebold_mariano_test(
    errors_a,
    errors_b,
    horizon,
    origin_step,
):
    """Squared-loss DM test with Bartlett HAC and HLN adjustment.

    A negative statistic means model A has lower mean squared error.
    The HAC bandwidth is the larger of the forecast-overlap lag in origin
    units and the standard automatic Newey-West bandwidth.
    """

    errors_a = np.asarray(errors_a, dtype=float)
    errors_b = np.asarray(errors_b, dtype=float)
    if errors_a.shape != errors_b.shape:
        raise ValueError("Paired forecast errors must have the same shape.")
    if errors_a.ndim != 1 or len(errors_a) < 3:
        raise ValueError("At least three paired forecast errors are required.")

    differential = np.square(errors_a) - np.square(errors_b)
    n_observations = len(differential)
    effective_horizon = max(1, int(math.ceil(horizon / origin_step)))
    overlap_lag = max(0, effective_horizon - 1)
    automatic_lag = _newey_west_bandwidth(n_observations)
    # A lag cannot exceed the number of available sample
    # autocovariances. This cap is inert for the frozen tests.
    hac_lag = min(
        n_observations - 1,
        max(overlap_lag, automatic_lag),
    )

    mean_difference = differential.mean()
    centered = differential - mean_difference
    long_run_variance = np.dot(centered, centered) / n_observations
    for lag in range(1, hac_lag + 1):
        autocovariance = (
            np.dot(centered[lag:], centered[:-lag])
            / n_observations
        )
        bartlett_weight = 1.0 - lag / (hac_lag + 1.0)
        long_run_variance += 2.0 * bartlett_weight * autocovariance

    if not np.isfinite(long_run_variance) or long_run_variance <= 0:
        statistic = np.nan
        p_value = np.nan
    else:
        raw_statistic = mean_difference / np.sqrt(
            long_run_variance / n_observations
        )
        hln_term = (
            n_observations
            + 1
            - 2 * effective_horizon
            + effective_horizon
            * (effective_horizon - 1)
            / n_observations
        ) / n_observations
        hln_factor = np.sqrt(hln_term)
        statistic = raw_statistic * hln_factor
        p_value = 2.0 * stats.t.sf(
            abs(statistic),
            df=n_observations - 1,
        )

    lag1_autocorrelation = (
        pd.Series(differential).autocorr(lag=1)
        if n_observations > 2
        else np.nan
    )
    return {
        "loss_function": "squared error",
        "n_forecasts": n_observations,
        "effective_horizon_in_origin_steps": effective_horizon,
        "overlap_lag": overlap_lag,
        "automatic_hac_lag": automatic_lag,
        "hac_lag": hac_lag,
        "hac_kernel": "Bartlett",
        "horizon_adjustment": "Harvey-Leybourne-Newbold",
        "mean_loss_difference_a_minus_b": mean_difference,
        "mean_squared_error_a": np.mean(np.square(errors_a)),
        "mean_squared_error_b": np.mean(np.square(errors_b)),
        "loss_differential_lag1_autocorrelation": (
            lag1_autocorrelation
        ),
        "long_run_variance": long_run_variance,
        "test_statistic": statistic,
        "p_value": p_value,
    }


def _holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p_values), np.nan)
    valid = np.flatnonzero(np.isfinite(p_values))
    if len(valid) == 0:
        return adjusted

    order = valid[np.argsort(p_values[valid])]
    running_maximum = 0.0
    m = len(order)
    for rank, index in enumerate(order):
        candidate = (m - rank) * p_values[index]
        running_maximum = max(running_maximum, candidate)
        adjusted[index] = min(1.0, running_maximum)
    return adjusted


def build_statistical_comparisons(forecasts, comparison):
    """Run prespecified paired tests, leaving Airline descriptive."""

    records = []
    for dataset, spec in DATASET_SPECS.items():
        for horizon in spec["horizons"]:
            metric_group = comparison[
                (comparison["dataset"] == dataset)
                & (comparison["horizon"] == horizon)
            ]
            best_non_naive = (
                metric_group[metric_group["model"] != "Naive"]
                .sort_values("rmse")
                .iloc[0]["model"]
            )
            pairs = [
                (spec["arima_model"], "Chronos-2", "ARIMA vs Chronos-2"),
                (spec["arima_model"], "PatchTST", "ARIMA vs PatchTST"),
                ("Chronos-2", "PatchTST", "Chronos-2 vs PatchTST"),
                (
                    "Naive",
                    best_non_naive,
                    "Naive vs best non-Naive",
                ),
            ]

            for model_a, model_b, label in pairs:
                base = {
                    "dataset": dataset,
                    "horizon": horizon,
                    "comparison": label,
                    "model_a": model_a,
                    "model_b": model_b,
                    "best_non_naive_selected_by": (
                        "lowest frozen RMSE"
                        if label == "Naive vs best non-Naive"
                        else "not applicable"
                    ),
                }
                data = forecasts[
                    (forecasts["dataset"] == dataset)
                    & (forecasts["horizon"] == horizon)
                    & (forecasts["model"].isin([model_a, model_b]))
                ]
                paired = (
                    data.pivot(
                        index="origin",
                        columns="model",
                        values=["actual", "forecast"],
                    )
                    .sort_index()
                )
                actual_a = paired[("actual", model_a)].to_numpy()
                actual_b = paired[("actual", model_b)].to_numpy()
                if not np.allclose(actual_a, actual_b):
                    raise AssertionError(
                        f"Unpaired actuals for {dataset}, h={horizon}, {label}."
                    )
                errors_a = actual_a - paired[("forecast", model_a)].to_numpy()
                errors_b = actual_b - paired[("forecast", model_b)].to_numpy()

                if dataset == "Airline":
                    base.update(
                        {
                            "test_run": False,
                            "loss_function": "squared error (descriptive only)",
                            "n_forecasts": len(errors_a),
                            "effective_horizon_in_origin_steps": (
                                int(math.ceil(horizon / spec["origin_step"]))
                            ),
                            "overlap_lag": max(0, horizon - 1),
                            "automatic_hac_lag": np.nan,
                            "hac_lag": np.nan,
                            "hac_kernel": "not applied",
                            "horizon_adjustment": "not applied",
                            "mean_loss_difference_a_minus_b": (
                                np.mean(np.square(errors_a) - np.square(errors_b))
                            ),
                            "mean_squared_error_a": np.mean(np.square(errors_a)),
                            "mean_squared_error_b": np.mean(np.square(errors_b)),
                            "loss_differential_lag1_autocorrelation": (
                                pd.Series(
                                    np.square(errors_a) - np.square(errors_b)
                                ).autocorr(lag=1)
                            ),
                            "long_run_variance": np.nan,
                            "test_statistic": np.nan,
                            "p_value": np.nan,
                            "limitation": (
                                "Formal DM inference not run: only 13 origins; "
                                "multi-step overlap leaves too little information "
                                "for reliable asymptotic inference."
                            ),
                        }
                    )
                else:
                    base["test_run"] = True
                    base.update(
                        diebold_mariano_test(
                            errors_a=errors_a,
                            errors_b=errors_b,
                            horizon=horizon,
                            origin_step=spec["origin_step"],
                        )
                    )
                    base["limitation"] = (
                        "Asymptotic paired test; p-values are sensitive to HAC "
                        "bandwidth and the finite number of forecast origins."
                    )
                records.append(base)

    results = pd.DataFrame(records)
    results["holm_adjusted_p_value"] = _holm_adjust(results["p_value"])
    results["significant_at_5pct_raw"] = results["p_value"] < 0.05
    results["significant_at_5pct_holm"] = (
        results["holm_adjusted_p_value"] < 0.05
    )
    results["lower_mean_squared_error_model"] = np.where(
        results["mean_loss_difference_a_minus_b"] < 0,
        results["model_a"],
        results["model_b"],
    )
    return results


def _parameter_count(group, family):
    if family == "Naive":
        return 0, False, "no estimated parameters"
    if family == "Chronos-2":
        return 120_000_000, True, "published model size (approximately 120M)"
    if family == "PatchTST":
        counts = group["num_parameters"].dropna().astype(int).unique()
        if len(counts) != 1:
            raise AssertionError("PatchTST parameter count changed by origin.")
        return int(counts[0]), False, "saved num_parameters"

    counts = []
    for value in group["parameters"].dropna().drop_duplicates():
        parsed = ast.literal_eval(value)
        counts.append(len(parsed))
    if len(set(counts)) != 1:
        raise AssertionError("ARIMA parameter count changed by origin.")
    return counts[0], False, "saved parameter dictionaries"


def build_computational_cost(forecasts):
    """Summarize only timing information recorded by the frozen runs."""

    rows = []
    for (dataset, model, family), group in forecasts.groupby(
        ["dataset", "model", "model_family"],
        sort=False,
    ):
        origins = group.sort_values("horizon").drop_duplicates("origin")
        parameter_count, approximate, parameter_source = _parameter_count(
            origins,
            family,
        )

        if family == "ARIMA/SARIMA":
            training = origins["elapsed_seconds"].astype(float)
            inference = None
            training_requirement = "origin-specific maximum-likelihood estimation"
            pretrained = False
            retraining = True
            timing_scope = (
                "elapsed_seconds combines estimation and forecast generation; "
                "separate inference time was not saved"
            )
        elif family == "PatchTST":
            training = origins["training_elapsed_seconds"].astype(float)
            inference = origins["inference_elapsed_seconds"].astype(float)
            training_requirement = "supervised scratch training at every origin"
            pretrained = False
            retraining = True
            timing_scope = "separately saved training and inference times"
        elif family == "Chronos-2":
            training = pd.Series(np.zeros(len(origins)), index=origins.index)
            inference = origins["inference_elapsed_seconds"].astype(float)
            training_requirement = "none in experiment; pretrained zero-shot model"
            pretrained = True
            retraining = False
            timing_scope = (
                "inference only; one-time model download/load time was not saved"
            )
        else:
            training = pd.Series(np.zeros(len(origins)), index=origins.index)
            inference = origins["elapsed_seconds"].astype(float)
            training_requirement = "none"
            pretrained = False
            retraining = False
            timing_scope = "forecast-call elapsed time"

        total_training = float(training.sum())
        mean_training = float(training.mean())
        if inference is None:
            total_inference = np.nan
            mean_inference = np.nan
            total_recorded = total_training
            mean_recorded = mean_training
        else:
            total_inference = float(inference.sum())
            mean_inference = float(inference.mean())
            total_recorded = total_training + total_inference
            mean_recorded = mean_training + mean_inference

        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "model_family": family,
                "n_origins": len(origins),
                "training_requirement": training_requirement,
                "total_training_time_seconds": total_training,
                "mean_training_time_per_origin": mean_training,
                "total_inference_time_seconds": total_inference,
                "mean_inference_time_per_origin": mean_inference,
                "recorded_total_compute_time_seconds": total_recorded,
                "recorded_compute_time_per_origin": mean_recorded,
                "approximate_parameter_count": parameter_count,
                "parameter_count_is_approximate": approximate,
                "parameter_count_source": parameter_source,
                "pretrained": pretrained,
                "origin_specific_retraining_required": retraining,
                "timing_scope": timing_scope,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["dataset", "model_family"]
    ).reset_index(drop=True)


def _model_order(dataset, values):
    arima = DATASET_SPECS[dataset]["arima_model"]
    preferred = ["Naive", arima, "Chronos-2", "PatchTST"]
    return [model for model in preferred if model in set(values)]


def create_figures(comparison, relative, cost, figure_dir):
    """Create article-ready figures without changing source values."""

    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = (
        "transformers-vs-arima-frozen-analysis-v1"
    )
    import matplotlib.pyplot as plt

    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "Naive": "#7f8c8d",
        "ARIMA/SARIMA": "#2c7fb8",
        "Chronos-2": "#f28e2b",
        "PatchTST": "#59a14f",
    }
    created = []

    def grouped_bars(metric, title, ylabel, filename, source):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
        for axis, (dataset, spec) in zip(axes, DATASET_SPECS.items()):
            data = source[source["dataset"] == dataset]
            models = _model_order(dataset, data["model"].unique())
            horizons = spec["horizons"]
            x = np.arange(len(horizons), dtype=float)
            width = 0.19
            for index, model in enumerate(models):
                values = (
                    data[data["model"] == model]
                    .set_index("horizon")
                    .loc[horizons, metric]
                )
                family = model_family(model)
                axis.bar(
                    x + (index - 1.5) * width,
                    values,
                    width,
                    label=family,
                    color=colors[family],
                )
            axis.set_title(dataset)
            axis.set_xticks(x)
            axis.set_xticklabels([str(value) for value in horizons])
            axis.set_xlabel("Forecast horizon")
            axis.set_ylabel(ylabel)
            axis.grid(axis="y", alpha=0.25)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=4,
            frameon=False,
        )
        fig.suptitle(title, fontsize=15)
        fig.tight_layout(rect=(0, 0.1, 1, 0.94))
        for extension in ["png", "svg"]:
            path = figure_dir / f"{filename}.{extension}"
            fig.savefig(
                path,
                dpi=300 if extension == "png" else None,
                bbox_inches="tight",
                metadata=(
                    {"Date": "2026-08-19"}
                    if extension == "svg"
                    else None
                ),
            )
            if extension == "svg":
                content = path.read_text(encoding="utf-8")
                path.write_text(
                    "\n".join(
                        line.rstrip()
                        for line in content.splitlines()
                    )
                    + "\n",
                    encoding="utf-8",
                )
            created.append(path)
        plt.close(fig)

    grouped_bars(
        metric="rmse",
        title="Frozen walk-forward RMSE by dataset",
        ylabel="RMSE (dataset units)",
        filename="rmse_by_model_and_horizon",
        source=comparison,
    )
    grouped_bars(
        metric="mae",
        title="Frozen walk-forward MAE by dataset",
        ylabel="MAE (dataset units)",
        filename="mae_by_model_and_horizon",
        source=comparison,
    )
    grouped_bars(
        metric="rmse_pct_difference_vs_naive",
        title="Relative RMSE versus Naive (negative is better)",
        ylabel="RMSE difference versus Naive (%)",
        filename="relative_rmse_vs_naive",
        source=relative,
    )

    average_relative = (
        relative.groupby(
            ["dataset", "model", "model_family"],
            as_index=False,
        )["rmse_pct_difference_vs_naive"]
        .mean()
    )
    plot_data = average_relative.merge(
        cost[
            [
                "dataset",
                "model",
                "recorded_compute_time_per_origin",
            ]
        ],
        on=["dataset", "model"],
        validate="one_to_one",
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for axis, dataset in zip(axes, DATASET_SPECS):
        data = plot_data[plot_data["dataset"] == dataset]
        for row in data.itertuples(index=False):
            axis.scatter(
                row.recorded_compute_time_per_origin,
                row.rmse_pct_difference_vs_naive,
                s=75,
                color=colors[row.model_family],
                label=row.model_family,
                zorder=3,
            )
            axis.annotate(
                row.model_family,
                (
                    row.recorded_compute_time_per_origin,
                    row.rmse_pct_difference_vs_naive,
                ),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8,
            )
        axis.axhline(0.0, color="black", linewidth=0.9, linestyle="--")
        axis.set_xscale("log")
        axis.set_title(dataset)
        axis.set_xlabel("Recorded compute time/origin (seconds, log scale)")
        axis.set_ylabel("Mean RMSE difference vs Naive (%)")
        axis.grid(alpha=0.25)
    fig.suptitle(
        "Recorded computational cost versus average relative RMSE",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for extension in ["png", "svg"]:
        path = figure_dir / f"compute_cost_vs_accuracy.{extension}"
        fig.savefig(
            path,
            dpi=300 if extension == "png" else None,
            bbox_inches="tight",
            metadata=(
                {"Date": "2026-08-19"}
                if extension == "svg"
                else None
            ),
        )
        if extension == "svg":
            content = path.read_text(encoding="utf-8")
            path.write_text(
                "\n".join(
                    line.rstrip()
                    for line in content.splitlines()
                )
                + "\n",
                encoding="utf-8",
            )
        created.append(path)
    plt.close(fig)
    return created


def _format_percent(value):
    return f"{value:+.1f}%"


def write_methodology(path):
    text = """# Frozen forecast analysis methodology

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
`floor(4 * (n / 100) ** (2 / 9))`, capped at `n - 1` because only that many
sample autocovariances exist. The cap does not bind in the frozen results. The
statistic receives the Harvey-Leybourne-Newbold finite-sample adjustment using
the horizon expressed in forecast-origin steps. Reported p-values are two-sided
Student-t p-values with `n - 1` degrees of freedom. Holm-adjusted values account
for the family of formal tests in the final analysis.

The DM test is asymptotic and remains sensitive to bandwidth choice. Airline
has only 13 origins, with severe overlap at longer horizons, so its comparisons
are descriptive and no formal p-values are reported.

## Computational cost

ARIMA/SARIMA `elapsed_seconds` combines origin-specific estimation and forecast
generation; separate inference timing was not saved. Chronos-2 timing excludes
one-time model download/loading. PatchTST has separate training and inference
timings. These scopes are retained rather than imputed. Hardware-specific
timings should not be interpreted as architecture-independent benchmarks.
"""
    path.write_text(text, encoding="utf-8")


def write_key_findings(
    path,
    comparison,
    relative,
    rankings,
    tests,
    cost,
):
    winners = rankings[
        (rankings["metric"] == "rmse")
        & (rankings["rank"] == 1)
    ]
    lines = [
        "# Frozen forecasting experiment: key findings",
        "",
        "## RMSE winners",
        "",
    ]
    for dataset in DATASET_SPECS:
        selected = winners[winners["dataset"] == dataset]
        descriptions = [
            f"h={row.horizon}: {row.model}"
            for row in selected.itertuples(index=False)
        ]
        lines.append(f"- {dataset}: " + "; ".join(descriptions) + ".")

    lines.extend(["", "## Dataset findings", ""])
    spy = relative[relative["dataset"] == "SPY"]
    lines.append(
        "- SPY: Naive has the lowest h=1 RMSE; ARIMA has the lowest h=5 "
        "and h=20 RMSE. Neither transformer improves RMSE on Naive or ARIMA "
        "at any frozen horizon."
    )

    airline = relative[relative["dataset"] == "Airline"]
    chronos_airline = airline[airline["model"] == "Chronos-2"]
    airline_ranges = (
        chronos_airline["rmse_pct_difference_vs_naive"].min(),
        chronos_airline["rmse_pct_difference_vs_naive"].max(),
        chronos_airline["rmse_pct_difference_vs_arima"].min(),
        chronos_airline["rmse_pct_difference_vs_arima"].max(),
    )
    lines.append(
        "- Airline: seasonal log-SARIMA wins all three horizons. Chronos-2 "
        "RMSE differences versus Naive range from "
        f"{_format_percent(airline_ranges[0])} "
        f"to {_format_percent(airline_ranges[1])}, but it is "
        f"{_format_percent(airline_ranges[2])} to "
        f"{_format_percent(airline_ranges[3])} worse than SARIMA. PatchTST "
        "performs worse than the seasonal baselines; with only 13 test origins, "
        "that comparison remains descriptive."
    )

    electricity = relative[
        (relative["dataset"] == "Electricity-CISO")
        & (relative["model"] == "Chronos-2")
    ].set_index("horizon")
    electricity_parts = []
    for horizon in [1, 24, 168]:
        row = electricity.loc[horizon]
        electricity_parts.append(
            f"h={horizon} {_format_percent(row.rmse_pct_difference_vs_naive)} "
            "vs Naive and "
            f"{_format_percent(row.rmse_pct_difference_vs_arima)} vs ARIMA"
        )
    lines.append(
        "- Electricity-CISO: Chronos-2 wins all horizons ("
        + "; ".join(electricity_parts)
        + "). The transformed ARIMA is especially weak at h=24, where its "
        "RMSE exceeds Naive."
    )
    lines.append(
        "- PatchTST: scratch training adds no predictive value in this frozen "
        "experiment: it wins no RMSE, MAE, or MASE comparisons and is markedly "
        "worse than simpler alternatives despite origin-specific retraining."
    )

    formal = tests[tests["test_run"]].copy()
    raw_significant = int(formal["significant_at_5pct_raw"].sum())
    holm_significant = int(formal["significant_at_5pct_holm"].sum())
    lines.extend(
        [
            "",
            "## Statistical comparisons",
            "",
            (
                f"- {len(formal)} formal SPY/Electricity comparisons were run; "
                f"{raw_significant} have raw p < 0.05 and {holm_significant} "
                "remain below 0.05 after Holm adjustment."
            ),
            (
                "- Airline comparisons are descriptive only because 13 origins "
                "are inadequate for reliable multi-step asymptotic inference."
            ),
            (
                "- Statistical significance does not replace effect size: use "
                "the relative-performance table alongside the paired tests."
            ),
            "",
            "## Computational interpretation",
            "",
        ]
    )
    for dataset in DATASET_SPECS:
        selected = cost[cost["dataset"] == dataset].set_index("model_family")
        patch_seconds = selected.loc[
            "PatchTST", "total_training_time_seconds"
        ]
        lines.append(
            f"- {dataset}: PatchTST scratch training totaled "
            f"{patch_seconds:.1f} seconds across the frozen origins."
        )
    lines.extend(
        [
            "- Chronos-2 required no experiment-specific training, but its "
            "recorded inference excludes one-time download/model-loading cost.",
            "- ARIMA/SARIMA timings combine estimation and forecast generation; "
            "a separate inference time was not recorded.",
            "",
            "## Reproducibility and limitations",
            "",
            "- Every accuracy value was recomputed from frozen origin-level "
            "forecasts and matched the saved comparison within numerical "
            "precision.",
            "- The standalone `spy_naive.csv` and `spy_naive_metrics.csv` are "
            "legacy step-1 artifacts (1,897 origins), not the authoritative "
            "380-origin step-5 SPY contract, and were excluded.",
            "- The cost comparison is hardware- and timing-scope-specific; no "
            "missing model-loading or separate ARIMA inference values were "
            "invented.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_final_analysis(
    project_root,
    output_dir=None,
    figure_dir=None,
    create_plots=True,
):
    """Generate all final artifacts from frozen files and return them."""

    project_root = Path(project_root)
    output_dir = (
        Path(output_dir)
        if output_dir is not None
        else project_root / "results" / "analysis"
    )
    figure_dir = (
        Path(figure_dir)
        if figure_dir is not None
        else project_root / "results" / "figures"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_input_manifest(project_root)
    forecasts = load_frozen_forecasts(project_root)
    forecast_validation = validate_frozen_forecasts(forecasts)
    comparison = build_authoritative_comparison(forecasts)
    metric_validation = validate_saved_metrics(comparison, project_root)
    validation = pd.concat(
        [forecast_validation, metric_validation],
        ignore_index=True,
    )
    relative = build_relative_performance(comparison)
    rankings, win_counts = build_rankings(comparison)
    tests = build_statistical_comparisons(forecasts, comparison)
    cost = build_computational_cost(forecasts)

    tables = {
        "frozen_input_manifest.csv": manifest,
        "validation_summary.csv": validation,
        "authoritative_comparison.csv": comparison,
        "relative_performance.csv": relative,
        "rankings.csv": rankings,
        "win_counts.csv": win_counts,
        "forecast_comparison_tests.csv": tests,
        "computational_cost.csv": cost,
    }
    for filename, frame in tables.items():
        frame.to_csv(output_dir / filename, index=False)

    write_methodology(output_dir / "methodology.md")
    write_key_findings(
        output_dir / "key_findings.md",
        comparison=comparison,
        relative=relative,
        rankings=rankings,
        tests=tests,
        cost=cost,
    )

    figures = []
    if create_plots:
        figures = create_figures(
            comparison=comparison,
            relative=relative,
            cost=cost,
            figure_dir=figure_dir,
        )

    return {
        "manifest": manifest,
        "validation": validation,
        "comparison": comparison,
        "relative": relative,
        "rankings": rankings,
        "win_counts": win_counts,
        "tests": tests,
        "cost": cost,
        "figures": figures,
    }
