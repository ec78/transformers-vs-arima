import warnings

import numpy as np
import pandas as pd

from statsmodels.tsa.arima.model import ARIMA

from src.data import load_spy
from src.preprocessing import prepare_spy


# ---------------------------------------------------------
# Load and prepare data
# ---------------------------------------------------------

spy = prepare_spy(
    load_spy()
)

test_start = pd.Timestamp(
    "2019-01-01"
)

training_data = spy.loc[
    spy.index < test_start,
    "log_price",
].dropna()


print("=" * 60)
print("ARIMA MODEL SELECTION: SPY LOG PRICES")
print("=" * 60)

print(
    "Training observations:",
    len(training_data),
)

print(
    "Training start:",
    training_data.index.min(),
)

print(
    "Training end:",
    training_data.index.max(),
)


# ---------------------------------------------------------
# Candidate specifications
# ---------------------------------------------------------

p_values = range(0, 4)
d_values = [1]
q_values = range(0, 4)


# ---------------------------------------------------------
# Estimate candidate models
# ---------------------------------------------------------

results = []

for p in p_values:

    for d in d_values:

        for q in q_values:

            order = (
                p,
                d,
                q,
            )

            print(
                f"Estimating ARIMA{order}..."
            )

            try:

                with warnings.catch_warnings():

                    warnings.simplefilter(
                        "ignore"
                    )

                    model = ARIMA(
                        training_data,
                        order=order,
                    )

                    fitted = model.fit()

                results.append(
                    {
                        "p": p,
                        "d": d,
                        "q": q,
                        "aic": fitted.aic,
                        "bic": fitted.bic,
                        "hqic": fitted.hqic,
                        "loglikelihood": fitted.llf,
                        "converged": (
                            fitted.mle_retvals.get(
                                "converged",
                                True,
                            )
                        ),
                    }
                )

            except Exception as error:

                print(
                    f"ARIMA{order} failed:"
                    f" {error}"
                )


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

results = pd.DataFrame(
    results
)

results = results.sort_values(
    "aic"
).reset_index(
    drop=True
)


print()
print("=" * 60)
print("TOP MODELS BY AIC")
print("=" * 60)

print(
    results.head(10).to_string(
        index=False
    )
)


print()
print("=" * 60)
print("TOP MODELS BY BIC")
print("=" * 60)

print(
    results
    .sort_values("bic")
    .head(10)
    .to_string(index=False)
)