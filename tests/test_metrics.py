import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    directional_accuracy,
    mae,
    mase,
    rmse,
    summarize_results,
    walk_forward_evaluate,
)
from src.preprocessing import fill_time_gaps


def test_rmse_and_mae_known_answers():
    actual = np.array([1.0, 2.0])
    forecast = np.array([1.0, 4.0])

    assert np.isclose(rmse(actual, forecast), np.sqrt(2.0))
    assert np.isclose(mae(actual, forecast), 1.0)


def test_mase_known_answer_uses_requested_seasonal_lag():
    training = np.array([1.0, 3.0, 6.0, 10.0])
    actual = np.array([12.0, 16.0])
    forecast = np.array([10.0, 13.0])

    # Lag-1 scale = mean([2, 3, 4]) = 3; MAE = 2.5.
    assert np.isclose(
        mase(actual, forecast, training, seasonality=1),
        2.5 / 3.0,
    )
    # Lag-2 scale = mean([5, 7]) = 6.
    assert np.isclose(
        mase(actual, forecast, training, seasonality=2),
        2.5 / 6.0,
    )


def test_directional_accuracy_is_origin_relative_and_naive_is_zero_change():
    origin = np.array([10.0, 10.0, 10.0])
    actual = np.array([11.0, 9.0, 10.0])
    forecast = np.array([12.0, 8.0, 10.0])
    naive = origin.copy()

    assert directional_accuracy(actual, forecast, origin) == 1.0
    assert np.isclose(
        directional_accuracy(actual, naive, origin),
        1.0 / 3.0,
    )


def test_walk_forward_scale_is_origin_specific_and_training_only():
    data = pd.DataFrame(
        {"y": np.arange(1.0, 11.0)},
        index=pd.date_range("2020-01-01", periods=10, freq="D"),
    )
    observed_training_lengths = []

    def naive_with_audit(train, horizon):
        observed_training_lengths.append(len(train))
        return np.repeat(float(train.iloc[-1]), horizon)

    results = walk_forward_evaluate(
        data=data,
        target_col="y",
        model_func=naive_with_audit,
        model_name="Naive",
        dataset_name="Synthetic",
        horizons=[1, 2],
        initial_train_size=5,
        step=2,
        mase_seasonality=2,
    )

    assert observed_training_lengths == [5, 7]
    assert set(results["train_size"]) == {5, 7}
    assert np.allclose(results["mase_scale"], 2.0)
    assert np.allclose(
        results["scaled_absolute_error"],
        (results["actual"] - results["forecast"]).abs() / 2.0,
    )


def test_walk_forward_summary_rejects_global_mase_fallback():
    results = pd.DataFrame(
        {
            "dataset": ["Synthetic"],
            "model": ["Model"],
            "horizon": [1],
            "actual": [2.0],
            "forecast": [1.0],
            "elapsed_seconds": [0.0],
            "origin_actual": [1.0],
        }
    )

    with pytest.raises(ValueError, match="Global MASE fallback is disabled"):
        summarize_results(
            results,
            training_series=np.array([0.0, 1.0]),
        )


def test_fill_time_gaps_marks_only_inserted_timestamp():
    data = pd.DataFrame(
        {"y": [1.0, 3.0]},
        index=pd.to_datetime(
            ["2020-01-01T00:00:00Z", "2020-01-01T02:00:00Z"]
        ),
    )
    data.index.name = "date"

    filled = fill_time_gaps(data, frequency="h")

    assert len(filled) == 3
    assert filled["interpolated"].tolist() == [False, True, False]
    assert np.isclose(filled.iloc[1]["y"], 2.0)
