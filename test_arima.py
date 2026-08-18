import pandas as pd
import warnings

from statsmodels.tools.sm_exceptions import ConvergenceWarning

from src.data import load_spy
from src.preprocessing import prepare_spy

from src.models_classical import (
    arima_log_price_forecast,
)

from src.evaluation import (
    walk_forward_evaluate,
)


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

spy = prepare_spy(
    load_spy()
)


# ---------------------------------------------------------
# Locate start of out-of-sample period
# ---------------------------------------------------------

test_start = pd.Timestamp(
    "2019-01-01"
)

initial_train_size = spy.index.searchsorted(
    test_start
)


# ---------------------------------------------------------
# Construct a representative timing sample
# ---------------------------------------------------------

# We want:
# - full pre-2019 training history
# - 20 forecast origins
# - enough future observations for a 20-day horizon
#
# n = initial_train_size + n_origins + max_horizon - 1

n_origins = 20
max_horizon = 20

sample_end = (
    initial_train_size
    + n_origins
    + max_horizon
    - 1
)

spy_test = spy.iloc[
    :sample_end
].copy()


# ---------------------------------------------------------
# Run ARIMA evaluation
# ---------------------------------------------------------

results = walk_forward_evaluate(
    data=spy_test,
    target_col="y",
    model_func=arima_log_price_forecast,
    model_name="ARIMA(0,1,2)+drift",
    dataset_name="SPY",
    horizons=[1, 5, 20],
    initial_train_size=initial_train_size,
    step=1,
    window="expanding",
    model_kwargs={
        "order": (0, 1, 2),
        "drift": True,
    },
)


# ---------------------------------------------------------
# Inspect forecasts
# ---------------------------------------------------------

print()
print("=" * 60)
print("ARIMA TEST RESULTS")
print("=" * 60)

print(
    results.head(10)
)


# ---------------------------------------------------------
# Timing
# ---------------------------------------------------------

timing = (
    results
    .groupby("origin")["elapsed_seconds"]
    .first()
)

mean_seconds = timing.mean()

print()
print("=" * 60)
print("TIMING")
print("=" * 60)

print(
    "Forecast origins:",
    len(timing)
)

print(
    "Mean seconds per origin:",
    mean_seconds
)

print(
    "Estimated seconds for 1,897 origins:",
    mean_seconds * 1897
)

print(
    "Estimated minutes for 1,897 origins:",
    mean_seconds * 1897 / 60
)