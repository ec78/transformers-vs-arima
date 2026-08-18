import time

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

def rmse(actual, forecast):
    """
    Root mean squared error.
    """

    actual = np.asarray(actual)
    forecast = np.asarray(forecast)

    return np.sqrt(
        np.mean(
            (actual - forecast) ** 2
        )
    )


def mae(actual, forecast):
    """
    Mean absolute error.
    """

    actual = np.asarray(actual)
    forecast = np.asarray(forecast)

    return np.mean(
        np.abs(actual - forecast)
    )


def mase(
    actual,
    forecast,
    training_series,
    seasonality=1,
):
    """
    Mean absolute scaled error.

    Parameters
    ----------
    actual : array-like
        Actual observed values.

    forecast : array-like
        Forecast values.

    training_series : array-like
        Training data used to construct the naive
        scaling denominator.

    seasonality : int, default=1
        Seasonal lag used for the naive benchmark.

        Examples
        --------
        1   : nonseasonal series
        12  : monthly annual seasonality
        24  : hourly daily seasonality
        168 : hourly weekly seasonality

    Returns
    -------
    float
        Mean absolute scaled error.
    """

    actual = np.asarray(
        actual,
        dtype=float,
    )

    forecast = np.asarray(
        forecast,
        dtype=float,
    )

    training_series = np.asarray(
        training_series,
        dtype=float,
    )

    if len(training_series) <= seasonality:
        raise ValueError(
            "Training series must contain more observations "
            "than the specified seasonality."
        )

    naive_errors = np.abs(
        training_series[seasonality:]
        - training_series[:-seasonality]
    )

    scale = np.mean(
        naive_errors
    )

    if scale == 0:
        raise ValueError(
            "MASE scale is zero. "
            "The training series may be constant."
        )

    return (
        mae(
            actual,
            forecast,
        )
        / scale
    )


def directional_accuracy(
    actual,
    forecast,
    origin_actual,
):
    """
    Calculate directional forecast accuracy.

    Direction is measured relative to the actual
    value at the forecast origin.

    Parameters
    ----------
    actual : array-like
        Actual future values.

    forecast : array-like
        Forecast future values.

    origin_actual : array-like or float
        Actual value at each forecast origin.

    Returns
    -------
    float
        Fraction of forecasts with the correct
        directional movement.
    """

    actual = np.asarray(
        actual,
        dtype=float,
    )

    forecast = np.asarray(
        forecast,
        dtype=float,
    )

    origin_actual = np.asarray(
        origin_actual,
        dtype=float,
    )

    actual_direction = np.sign(
        actual - origin_actual
    )

    forecast_direction = np.sign(
        forecast - origin_actual
    )

    return np.mean(
        actual_direction
        == forecast_direction
    )


# ---------------------------------------------------------
# Forecast origin generation
# ---------------------------------------------------------

def generate_forecast_origins(
    data,
    initial_train_size,
    max_horizon,
    step=1,
):
    """
    Generate valid walk-forward forecast origins.

    Parameters
    ----------
    data : pandas.DataFrame or pandas.Series
        Complete ordered time series.

    initial_train_size : int
        Number of observations available before
        the first forecast origin.

    max_horizon : int
        Largest forecast horizon required.

    step : int, default=1
        Number of observations to move forward
        between forecast origins.

    Returns
    -------
    list of int
        Integer positions representing
        forecast origins.
    """

    nobs = len(data)

    if initial_train_size <= 0:
        raise ValueError(
            "initial_train_size must be positive."
        )

    if initial_train_size >= nobs:
        raise ValueError(
            "initial_train_size must be smaller "
            "than the number of observations."
        )

    if max_horizon <= 0:
        raise ValueError(
            "max_horizon must be positive."
        )

    if step <= 0:
        raise ValueError(
            "step must be positive."
        )

    last_origin = (
        nobs - max_horizon
    )

    origins = list(
        range(
            initial_train_size,
            last_origin + 1,
            step,
        )
    )

    return origins


# ---------------------------------------------------------
# Results formatting
# ---------------------------------------------------------

def format_forecast_results(
    dataset_name,
    model_name,
    origin_date,
    origin_actual,
    forecast_dates,
    actual,
    forecast,
    metadata=None,
):
    """
    Format a single multi-step forecast into the
    project's standard results structure.

    Parameters
    ----------
    dataset_name : str
        Dataset identifier.

    model_name : str
        Model identifier.

    origin_date : datetime-like
        Forecast origin date.

    origin_actual : float
        Actual value at forecast origin.

    forecast_dates : array-like
        Dates corresponding to forecasts.

    actual : array-like
        Actual future values.

    forecast : array-like
        Forecast values.

    metadata : dict or None
        Optional model-specific diagnostics.

    Returns
    -------
    pandas.DataFrame
    """

    actual = np.asarray(
        actual
    )

    forecast = np.asarray(
        forecast
    )

    if len(actual) != len(forecast):
        raise ValueError(
            "actual and forecast must have the same length."
        )

    if len(forecast_dates) != len(actual):
        raise ValueError(
            "forecast_dates must have the same length "
            "as actual and forecast."
        )

    result = pd.DataFrame(
        {
            "dataset": dataset_name,
            "model": model_name,
            "origin": origin_date,
            "origin_actual": origin_actual,
            "forecast_date": forecast_dates,
            "horizon": np.arange(
                1,
                len(actual) + 1,
            ),
            "actual": actual,
            "forecast": forecast,
        }
    )

    if metadata is not None:
        for key, value in metadata.items():
            result[key] = value

    return result


# ---------------------------------------------------------
# Walk-forward evaluation
# ---------------------------------------------------------

def walk_forward_evaluate(
    data,
    target_col,
    model_func,
    model_name,
    dataset_name,
    horizons,
    initial_train_size,
    step=1,
    window="expanding",
    window_size=None,
    model_kwargs=None,
    mase_seasonality=1,
):
    """
    Run walk-forward forecasting evaluation.

    Parameters
    ----------
    data : pandas.DataFrame
        Time-series dataset.

    target_col : str
        Column to forecast.

    model_func : callable
        Forecasting function.

        A model may return either:

            numpy.ndarray

        or:

            {
                "forecast": array-like,
                "diagnostic_1": value,
                "diagnostic_2": value,
                ...
            }

    model_name : str
        Name used in saved results.

    dataset_name : str
        Dataset identifier.

    horizons : list of int
        Forecast horizons to evaluate.

    initial_train_size : int
        Number of observations in the initial
        training sample.

    step : int, default=1
        Number of observations between
        forecast origins.

    window : {"expanding", "rolling"}
        Training-window method.

    window_size : int or None
        Required for rolling windows.

    model_kwargs : dict or None
        Additional model-specific arguments.

    mase_seasonality : int, default=1
        Lag used to construct the origin-specific
        MASE scale from the training sample.

    Returns
    -------
    pandas.DataFrame
        Standardized forecast results.
    """

    if target_col not in data.columns:
        raise ValueError(
            f"Target column '{target_col}' not found."
        )

    if not isinstance(
        data.index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            "Data must use a DatetimeIndex."
        )

    if model_kwargs is None:
        model_kwargs = {}

    if mase_seasonality <= 0:
        raise ValueError(
            "mase_seasonality must be positive."
        )

    horizons = sorted(
        set(horizons)
    )

    if len(horizons) == 0:
        raise ValueError(
            "At least one forecast horizon is required."
        )

    if any(
        horizon <= 0
        for horizon in horizons
    ):
        raise ValueError(
            "All forecast horizons must be positive."
        )

    max_horizon = max(
        horizons
    )

    origins = generate_forecast_origins(
        data=data,
        initial_train_size=initial_train_size,
        max_horizon=max_horizon,
        step=step,
    )

    results = []

    for origin in origins:

        # -------------------------------------------------
        # Construct training sample
        # -------------------------------------------------

        if window == "expanding":

            train = data.iloc[
                :origin
            ][target_col]

        elif window == "rolling":

            if window_size is None:
                raise ValueError(
                    "window_size is required when "
                    "window='rolling'."
                )

            if window_size <= 0:
                raise ValueError(
                    "window_size must be positive."
                )

            train_start = max(
                0,
                origin - window_size,
            )

            train = data.iloc[
                train_start:origin
            ][target_col]

        else:
            raise ValueError(
                "window must be either "
                "'expanding' or 'rolling'."
            )

        # -------------------------------------------------
        # Future actual observations
        # -------------------------------------------------

        future = data.iloc[
            origin:origin + max_horizon
        ][target_col]

        origin_date = data.index[
            origin - 1
        ]

        origin_actual = data.iloc[
            origin - 1
        ][target_col]

        train_values = np.asarray(
            train,
            dtype=float,
        )

        if len(train_values) <= mase_seasonality:
            raise ValueError(
                "Training sample must contain more "
                "observations than mase_seasonality."
            )

        mase_scale = np.mean(
            np.abs(
                train_values[
                    mase_seasonality:
                ]
                - train_values[
                    :-mase_seasonality
                ]
            )
        )

        if not np.isfinite(mase_scale):
            raise ValueError(
                "MASE scale is not finite."
            )

        if mase_scale == 0:
            raise ValueError(
                "MASE scale is zero. "
                "The training series may be constant."
            )

        # -------------------------------------------------
        # Run forecasting model
        # -------------------------------------------------

        start_time = time.perf_counter()

        model_output = model_func(
            train,
            horizon=max_horizon,
            **model_kwargs,
        )

        elapsed_seconds = (
            time.perf_counter()
            - start_time
        )

        # -------------------------------------------------
        # Parse model output
        # -------------------------------------------------

        if isinstance(
            model_output,
            dict,
        ):

            if "forecast" not in model_output:
                raise ValueError(
                    f"{model_name} returned a dictionary "
                    "without a 'forecast' entry."
                )

            forecast = np.asarray(
                model_output["forecast"],
                dtype=float,
            )

            model_metadata = {
                key: value
                for key, value in model_output.items()
                if key != "forecast"
            }

        else:

            forecast = np.asarray(
                model_output,
                dtype=float,
            )

            model_metadata = {}

        # -------------------------------------------------
        # Validate forecast
        # -------------------------------------------------

        if len(forecast) != max_horizon:
            raise ValueError(
                f"{model_name} returned "
                f"{len(forecast)} forecasts; "
                f"{max_horizon} were expected."
            )

        if not np.all(
            np.isfinite(forecast)
        ):
            raise ValueError(
                f"{model_name} returned one or more "
                "non-finite forecasts."
            )

        # -------------------------------------------------
        # Store requested horizons
        # -------------------------------------------------

        for horizon in horizons:

            actual_value = future.iloc[
                horizon - 1
            ]

            forecast_value = forecast[
                horizon - 1
            ]

            forecast_date = future.index[
                horizon - 1
            ]

            actual_change = (
                actual_value
                - origin_actual
            )

            predicted_change = (
                forecast_value
                - origin_actual
            )

            actual_direction = np.sign(
                actual_change
            )

            predicted_direction = np.sign(
                predicted_change
            )

            results.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "origin": origin_date,
                    "origin_actual": origin_actual,
                    "forecast_date": forecast_date,
                    "horizon": horizon,
                    "actual": actual_value,
                    "forecast": forecast_value,
                    "actual_change": actual_change,
                    "predicted_change": predicted_change,
                    "actual_direction": actual_direction,
                    "predicted_direction": predicted_direction,
                    "direction_correct": (
                        actual_direction
                        == predicted_direction
                    ),
                    "mase_scale": mase_scale,
                    "scaled_absolute_error": (
                        abs(
                            actual_value
                            - forecast_value
                        )
                        / mase_scale
                    ),
                    "elapsed_seconds": elapsed_seconds,
                    "train_size": len(train),
                    "window": window,
                    **model_metadata,
                }
            )

    return pd.DataFrame(
        results
    )


# ---------------------------------------------------------
# Evaluation summary
# ---------------------------------------------------------

def summarize_results(
    results,
    training_series=None,
    mase_seasonality=1,
    include_direction=False,
):
    """
    Summarize forecast accuracy by dataset, model,
    and forecast horizon.

    Parameters
    ----------
    results : pandas.DataFrame
        Output from walk_forward_evaluate().

    training_series : array-like or None
        Training data used for MASE scaling.
        If None, MASE is omitted.

    mase_seasonality : int, default=1
        Seasonal lag for MASE.

    include_direction : bool, default=False
        If True, include directional accuracy.

    Returns
    -------
    pandas.DataFrame
        Accuracy summary.
    """

    required_columns = {
        "dataset",
        "model",
        "horizon",
        "actual",
        "forecast",
        "elapsed_seconds",
        "origin_actual",
    }

    missing_columns = (
        required_columns
        - set(results.columns)
    )

    if missing_columns:
        raise ValueError(
            "Results are missing required columns: "
            f"{missing_columns}"
        )

    summaries = []

    grouped = results.groupby(
        [
            "dataset",
            "model",
            "horizon",
        ]
    )

    for (
        dataset,
        model,
        horizon,
    ), group in grouped:

        row = {
            "dataset": dataset,
            "model": model,
            "horizon": horizon,
            "n_forecasts": len(group),

            "rmse": rmse(
                group["actual"],
                group["forecast"],
            ),

            "mae": mae(
                group["actual"],
                group["forecast"],
            ),

            "mean_elapsed_seconds": (
                group[
                    "elapsed_seconds"
                ].mean()
            ),
        }

        # -------------------------------------------------
        # MASE
        # -------------------------------------------------

        if "scaled_absolute_error" in group.columns:

            row["mase"] = group[
                "scaled_absolute_error"
            ].mean()

        elif training_series is not None:

            row["mase"] = mase(
                actual=group["actual"],
                forecast=group["forecast"],
                training_series=training_series,
                seasonality=mase_seasonality,
            )

        # -------------------------------------------------
        # Directional accuracy
        # -------------------------------------------------

        if include_direction:

            if "direction_correct" in group.columns:

                row[
                    "directional_accuracy"
                ] = group[
                    "direction_correct"
                ].mean()

            else:

                row[
                    "directional_accuracy"
                ] = directional_accuracy(
                    actual=group["actual"],
                    forecast=group["forecast"],
                    origin_actual=group[
                        "origin_actual"
                    ],
                )

        # -------------------------------------------------
        # Optional convergence diagnostics
        # -------------------------------------------------

        if "converged" in group.columns:

            convergence_by_origin = (
                group
                .groupby("origin")[
                    "converged"
                ]
                .first()
            )

            row[
                "convergence_rate"
            ] = convergence_by_origin.mean()

            row[
                "n_nonconverged"
            ] = (
                ~convergence_by_origin
                .astype(bool)
            ).sum()

        if "iterations" in group.columns:

            iterations_by_origin = (
                group
                .groupby("origin")[
                    "iterations"
                ]
                .first()
            )

            row[
                "mean_iterations"
            ] = (
                iterations_by_origin.mean()
            )

        summaries.append(
            row
        )

    summary = pd.DataFrame(
        summaries
    )

    return summary.sort_values(
        [
            "dataset",
            "model",
            "horizon",
        ]
    ).reset_index(
        drop=True
    )


# ---------------------------------------------------------
# Convergence diagnostic summary
# ---------------------------------------------------------

def summarize_convergence_diagnostics(
    results,
):
    """
    Summarize forecast accuracy by horizon and
    optimizer convergence status.
    """

    required_columns = {
        "horizon",
        "converged",
        "actual",
        "forecast",
        "iterations",
        "mase_scale",
    }

    missing_columns = (
        required_columns
        - set(results.columns)
    )

    if missing_columns:
        raise ValueError(
            "Results are missing required columns: "
            f"{missing_columns}"
        )

    summaries = []

    grouped = results.groupby(
        [
            "horizon",
            "converged",
        ],
        dropna=False,
    )

    for (
        horizon,
        converged,
    ), group in grouped:

        summaries.append(
            {
                "horizon": horizon,
                "converged": converged,
                "n_forecasts": len(group),
                "rmse": rmse(
                    group["actual"],
                    group["forecast"],
                ),
                "mae": mae(
                    group["actual"],
                    group["forecast"],
                ),
                "mean_iterations": group[
                    "iterations"
                ].mean(),
                "mean_mase_scale": group[
                    "mase_scale"
                ].mean(),
            }
        )

    return (
        pd.DataFrame(
            summaries
        )
        .sort_values(
            [
                "horizon",
                "converged",
            ]
        )
        .reset_index(
            drop=True
        )
    )
