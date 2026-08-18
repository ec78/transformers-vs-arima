import warnings

import pandas as pd

from statsmodels.tsa.arima.model import ARIMA

from src.data import load_spy
from src.preprocessing import prepare_spy


# ---------------------------------------------------------
# Load pre-2019 training sample
# ---------------------------------------------------------

spy = prepare_spy(
    load_spy()
)

training_data = spy.loc[
    spy.index < pd.Timestamp("2019-01-01"),
    "log_price",
].dropna()


# ---------------------------------------------------------
# Models to compare
# ---------------------------------------------------------

specifications = [
    {
        "name": "ARIMA(0,1,0)",
        "order": (0, 1, 0),
        "trend": "n",
    },
    {
        "name": "ARIMA(0,1,0) + drift",
        "order": (0, 1, 0),
        "trend": "t",
    },
    {
        "name": "ARIMA(0,1,2)",
        "order": (0, 1, 2),
        "trend": "n",
    },
    {
        "name": "ARIMA(0,1,2) + drift",
        "order": (0, 1, 2),
        "trend": "t",
    },
]


# ---------------------------------------------------------
# Estimate models
# ---------------------------------------------------------

results = []

for spec in specifications:

    print(
        f"Estimating {spec['name']}..."
    )

    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore"
        )

        model = ARIMA(
            training_data,
            order=spec["order"],
            trend=spec["trend"],
        )

        fitted = model.fit()

    results.append(
        {
            "model": spec["name"],
            "aic": fitted.aic,
            "bic": fitted.bic,
            "hqic": fitted.hqic,
            "loglikelihood": fitted.llf,
            "converged": fitted.mle_retvals.get(
                "converged",
                True,
            ),
        }
    )


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

results = pd.DataFrame(
    results
)

print()
print("=" * 70)
print("ARIMA DRIFT COMPARISON")
print("=" * 70)

print(
    results
    .sort_values("aic")
    .to_string(index=False)
)