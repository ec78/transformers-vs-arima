import numpy as np
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
)


# ---------------------------------------------------------
# Dataset structure and preprocessing
# ---------------------------------------------------------

airline = prepare_airline(
    load_airline()
)

assert len(airline) == 144
assert pd.infer_freq(airline.index) == "MS"
assert airline.index.min() == pd.Timestamp(
    "1949-01-01"
)
assert airline.index.max() == pd.Timestamp(
    "1960-12-01"
)
assert np.allclose(
    airline["log_y"],
    np.log(airline["y"]),
)

electricity = prepare_electricity(
    load_electricity(
        balancing_authority="CISO"
    )
)

assert electricity.index.tz is not None
assert electricity.index.to_series().diff().dropna().eq(
    pd.Timedelta(hours=1)
).all()
assert not electricity["y"].isna().any()
assert electricity["interpolated"].sum() == 24
assert not electricity.loc[
    pd.Timestamp(
        "2025-01-01",
        tz="UTC",
    ):
]["interpolated"].any()


# ---------------------------------------------------------
# Seasonal MASE and alignment
# ---------------------------------------------------------

airline_results = walk_forward_evaluate(
    data=airline,
    target_col="y",
    model_func=naive_forecast,
    model_name="Naive",
    dataset_name="Airline",
    horizons=[1, 3, 12],
    initial_train_size=120,
    step=1,
    mase_seasonality=12,
)

first_origin = airline_results[
    airline_results["origin"]
    == pd.Timestamp("1958-12-01")
]

expected_scale = np.mean(
    np.abs(
        airline["y"].iloc[12:120].to_numpy()
        - airline["y"].iloc[:108].to_numpy()
    )
)

assert np.allclose(
    first_origin["mase_scale"],
    expected_scale,
)
assert first_origin["forecast_date"].tolist() == [
    pd.Timestamp("1959-01-01"),
    pd.Timestamp("1959-03-01"),
    pd.Timestamp("1959-12-01"),
]
assert np.allclose(
    first_origin["forecast"],
    airline["y"].iloc[119],
)
assert np.allclose(
    first_origin["predicted_change"],
    0.0,
)
assert np.allclose(
    first_origin["predicted_direction"],
    0.0,
)


# ---------------------------------------------------------
# Seasonal reconstruction and ARIMA diagnostics
# ---------------------------------------------------------

weekly_pattern = np.linspace(
    100.0,
    200.0,
    168,
)

synthetic_electricity = pd.Series(
    np.tile(
        weekly_pattern,
        4,
    ),
    index=pd.date_range(
        "2020-01-01",
        periods=168 * 4,
        freq="h",
        tz="UTC",
    ),
)

transformed_result = arima_transformed_forecast(
    train=synthetic_electricity,
    horizon=168,
    order=(0, 0, 0),
    trend="n",
    log_transform=True,
    seasonal_difference=168,
    low_memory=True,
)

assert np.allclose(
    transformed_result["forecast"],
    weekly_pattern,
)
assert transformed_result["forecast_is_finite"]
assert transformed_result["parameters_are_finite"]
assert set(
    transformed_result["parameters"]
) == {
    "sigma2",
}

airline_fit = arima_transformed_forecast(
    train=airline["y"].iloc[:120],
    horizon=12,
    order=(0, 1, 1),
    seasonal_order=(0, 1, 1, 12),
    trend="n",
    log_transform=True,
)

assert len(airline_fit["forecast"]) == 12
assert airline_fit["forecast_is_finite"]
assert airline_fit["parameters_are_finite"]
assert set(
    airline_fit["parameters"]
) == {
    "ma.L1",
    "ma.S.L12",
    "sigma2",
}

print(
    "Other classical workflow assertions passed."
)
