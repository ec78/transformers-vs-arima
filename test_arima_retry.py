import numpy as np
import pandas as pd

from src.data import load_spy
from src.preprocessing import prepare_spy

from src.models_classical import (
    arima_log_price_forecast,
)


# ---------------------------------------------------------
# Known reproducible first-attempt failures
# ---------------------------------------------------------

spy = prepare_spy(
    load_spy()
)

for origin_date in [
    "2019-02-21",
    "2025-04-14",
]:
    origin = (
        spy.index.get_loc(
            pd.Timestamp(
                origin_date
            )
        )
        + 1
    )

    result = arima_log_price_forecast(
        train=spy["y"].iloc[:origin],
        horizon=20,
        order=(0, 1, 2),
        drift=True,
    )

    assert not bool(
        result[
            "initial_converged"
        ]
    )

    assert result[
        "initial_warnflag"
    ] == 2

    assert result[
        "retry_attempted"
    ]

    assert result[
        "retry_accepted"
    ]

    assert result[
        "retry_count"
    ] in {
        1,
        2,
    }

    assert bool(
        result["converged"]
    )

    assert result[
        "warnflag"
    ] == 0

    assert (
        result[
            "objective_value"
        ]
        <= result[
            "initial_objective_value"
        ]
    )

    assert np.all(
        np.isfinite(
            result["forecast"]
        )
    )

print(
    "ARIMA retry assertions passed."
)
