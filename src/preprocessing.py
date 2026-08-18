import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

def validate_time_series(data):
    """
    Validate the common time-series structure used
    throughout the project.

    Expected structure
    ------------------
    - pandas DataFrame
    - DatetimeIndex
    - target column named 'y'

    Parameters
    ----------
    data : pandas.DataFrame
        Input time-series data.

    Returns
    -------
    pandas.DataFrame
        Cleaned and validated copy of the input data.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "Input data must be a pandas DataFrame."
        )

    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError(
            "Data must use a pandas DatetimeIndex."
        )

    if "y" not in data.columns:
        raise ValueError(
            "Data must contain a target column named 'y'."
        )

    data = data.copy()

    # Ensure chronological ordering
    data = data.sort_index()

    # Remove duplicate timestamps if present
    if data.index.has_duplicates:
        data = data[
            ~data.index.duplicated(keep="first")
        ]

    # Remove missing target values
    if data["y"].isna().any():
        data = data.dropna(
            subset=["y"]
        )

    return data


# ---------------------------------------------------------
# General transformations
# ---------------------------------------------------------

def add_log_transform(
    data,
    source_col="y",
    output_col="log_y",
):
    """
    Add a natural-log transformation of a column.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data.

    source_col : str, default="y"
        Column to transform.

    output_col : str, default="log_y"
        Name of the transformed column.

    Returns
    -------
    pandas.DataFrame
        Copy of the input data with the added log column.
    """

    data = data.copy()

    if source_col not in data.columns:
        raise ValueError(
            f"Column '{source_col}' was not found."
        )

    if (data[source_col] <= 0).any():
        raise ValueError(
            "Log transformation requires strictly "
            "positive values."
        )

    data[output_col] = np.log(
        data[source_col]
    )

    return data


def add_difference(
    data,
    source_col="y",
    periods=1,
    output_col="diff_y",
):
    """
    Add a differenced version of a column.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data.

    source_col : str, default="y"
        Column to difference.

    periods : int, default=1
        Number of periods to difference.

    output_col : str, default="diff_y"
        Name of the differenced column.

    Returns
    -------
    pandas.DataFrame
        Copy of the input data with the added
        differenced column.
    """

    data = data.copy()

    if source_col not in data.columns:
        raise ValueError(
            f"Column '{source_col}' was not found."
        )

    data[output_col] = (
        data[source_col]
        .diff(periods=periods)
    )

    return data


def add_log_difference(
    data,
    source_col="y",
    periods=1,
    output_col="log_diff_y",
):
    """
    Add log differences.

    For financial price series, a one-period log
    difference corresponds to a continuously
    compounded return.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data.

    source_col : str, default="y"
        Column to transform.

    periods : int, default=1
        Difference lag.

    output_col : str, default="log_diff_y"
        Name of the output column.

    Returns
    -------
    pandas.DataFrame
        Copy of the input data with log differences.
    """

    data = data.copy()

    if source_col not in data.columns:
        raise ValueError(
            f"Column '{source_col}' was not found."
        )

    if (data[source_col] <= 0).any():
        raise ValueError(
            "Log differences require strictly "
            "positive values."
        )

    data[output_col] = (
        np.log(data[source_col])
        .diff(periods=periods)
    )

    return data


# ---------------------------------------------------------
# Dataset-specific preprocessing
# ---------------------------------------------------------

def prepare_spy(data):
    """
    Prepare SPY data for forecasting experiments.

    Adds
    ----
    log_price
        Natural log of adjusted SPY price.

    return
        One-period log return.

    Notes
    -----
    The first value of 'return' will be NaN because
    there is no previous observation. It is intentionally
    retained because the underlying price observation
    remains valid.

    Parameters
    ----------
    data : pandas.DataFrame
        Raw SPY data with target column 'y'.

    Returns
    -------
    pandas.DataFrame
        Prepared SPY dataset.
    """

    data = validate_time_series(data)

    data = add_log_transform(
        data,
        source_col="y",
        output_col="log_price",
    )

    data = add_difference(
        data,
        source_col="log_price",
        periods=1,
        output_col="return",
    )

    return data


def prepare_airline(data):
    """
    Prepare the AirPassengers dataset.

    Adds
    ----
    log_y
        Natural log of monthly passenger counts.

    Notes
    -----
    Seasonal differencing is intentionally not applied
    here. It may be used later for specific model
    specifications.

    Parameters
    ----------
    data : pandas.DataFrame
        Raw airline passenger data.

    Returns
    -------
    pandas.DataFrame
        Prepared airline passenger dataset.
    """

    data = validate_time_series(data)

    data = add_log_transform(
        data,
        source_col="y",
        output_col="log_y",
    )

    return data


def prepare_electricity(data):
    """
    Prepare hourly electricity demand data.

    Processing steps
    ----------------
    1. Validate time-series structure.
    2. Restore a regular hourly time index.
    3. Interpolate isolated missing hourly observations.
    4. Flag interpolated observations.

    Parameters
    ----------
    data : pandas.DataFrame
        Raw hourly electricity demand data.

    Returns
    -------
    pandas.DataFrame
        Prepared electricity dataset containing:
        - y
        - interpolated
    """

    data = validate_time_series(data)

    data = fill_time_gaps(
        data,
        frequency="h",
        method="time",
    )

    return data


# ---------------------------------------------------------
# Time-based splitting
# ---------------------------------------------------------

def time_split(
    data,
    test_start,
):
    """
    Split a time series into training and test samples
    using a fixed date.

    Parameters
    ----------
    data : pandas.DataFrame
        Input time-series data.

    test_start : str or pandas.Timestamp
        First date included in the test sample.

    Returns
    -------
    train : pandas.DataFrame
        Training sample.

    test : pandas.DataFrame
        Test sample.
    """

    data = validate_time_series(data)

    test_start = pd.Timestamp(
        test_start
    )

    train = data.loc[
        data.index < test_start
    ].copy()

    test = data.loc[
        data.index >= test_start
    ].copy()

    if train.empty:
        raise ValueError(
            "Training sample is empty."
        )

    if test.empty:
        raise ValueError(
            "Test sample is empty."
        )

    return train, test


# ---------------------------------------------------------
# Time-series diagnostics
# ---------------------------------------------------------

def get_missing_timestamps(
    data,
    frequency,
):
    """
    Identify missing timestamps in a regularly
    spaced time series.

    Parameters
    ----------
    data : pandas.DataFrame
        Input time-series data.

    frequency : str
        Pandas frequency string, such as:
        - 'h' for hourly
        - 'D' for daily
        - 'MS' for month-start

    Returns
    -------
    pandas.DatetimeIndex
        Missing timestamps.
    """

    data = validate_time_series(data)

    expected_index = pd.date_range(
        start=data.index.min(),
        end=data.index.max(),
        freq=frequency,
    )

    missing_timestamps = (
        expected_index.difference(
            data.index
        )
    )

    return missing_timestamps


def fill_time_gaps(
    data,
    frequency,
    method="time",
):
    """
    Restore a regular time index and interpolate
    missing target values.

    Parameters
    ----------
    data : pandas.DataFrame
        Input time-series data.

    frequency : str
        Expected pandas frequency, such as 'h' or 'MS'.

    method : str, default="time"
        Pandas interpolation method.

    Returns
    -------
    pandas.DataFrame
        Time series with a regular index and interpolated
        target values.

    Notes
    -----
    This function creates observations only for timestamps
    missing from the expected regular time index. Missing
    target values are then interpolated using neighboring
    observations.
    """

    data = validate_time_series(data)

    full_index = pd.date_range(
        start=data.index.min(),
        end=data.index.max(),
        freq=frequency,
    )

    data = data.reindex(full_index)

    data.index.name = "date"

    # Identify observations introduced by reindexing
    data["interpolated"] = data["y"].isna()

    # Fill missing target values
    data["y"] = data["y"].interpolate(
        method=method,
    )

    return data

def summarize_time_series(
    data,
    frequency=None,
):
    """
    Create a basic diagnostic summary of a time series.

    Parameters
    ----------
    data : pandas.DataFrame
        Input time-series data.

    frequency : str or None
        Expected pandas frequency.
        If provided, missing timestamps are counted.

    Returns
    -------
    dict
        Summary statistics and structural diagnostics.
    """

    data = validate_time_series(data)

    summary = {
        "n_observations": len(data),
        "start_date": data.index.min(),
        "end_date": data.index.max(),
        "missing_y": int(
            data["y"].isna().sum()
        ),
        "duplicate_timestamps": int(
            data.index.duplicated().sum()
        ),
    }

    if frequency is not None:
        missing_timestamps = (
            get_missing_timestamps(
                data,
                frequency,
            )
        )

        summary["missing_timestamps"] = len(
            missing_timestamps
        )

    return summary