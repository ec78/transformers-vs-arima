from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import scipy.optimize

from statsmodels.tsa.arima.model import ARIMA

from src.data import load_spy
from src.preprocessing import prepare_spy

from src.evaluation import (
    generate_forecast_origins,
)


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

FORECAST_PATH = (
    PROJECT_ROOT
    / "results"
    / "forecasts"
    / "spy_classical_forecasts.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "spy_arima_optimizer_rerun.csv"
)


# ---------------------------------------------------------
# Existing workflow configuration
# ---------------------------------------------------------

spy = prepare_spy(
    load_spy()
)

test_start = pd.Timestamp(
    "2019-01-01"
)

initial_train_size = (
    spy.index.searchsorted(
        test_start
    )
)

origins = generate_forecast_origins(
    data=spy,
    initial_train_size=initial_train_size,
    max_horizon=20,
    step=5,
)


# ---------------------------------------------------------
# Previously saved convergence status
# ---------------------------------------------------------

saved_results = pd.read_csv(
    FORECAST_PATH,
    parse_dates=[
        "origin",
    ],
)

saved_arima = saved_results[
    saved_results["model"]
    == "ARIMA(0,1,2)+drift"
]

saved_status_column = (
    "initial_converged"
    if "initial_converged"
    in saved_arima.columns
    else "converged"
)

saved_convergence = (
    saved_arima
    .groupby("origin")[
        saved_status_column
    ]
    .first()
)


# ---------------------------------------------------------
# Diagnostic rerun
# ---------------------------------------------------------

diagnostics = []
original_optimizer = (
    scipy.optimize.fmin_l_bfgs_b
)

for origin in origins:

    train = spy["y"].iloc[
        :origin
    ]

    origin_date = spy.index[
        origin - 1
    ]

    log_prices = np.log(
        train.to_numpy(
            dtype=float
        )
    )

    log_returns = np.diff(
        log_prices
    )

    model = ARIMA(
        log_returns,
        order=(0, 0, 2),
        trend="c",
    )

    optimizer_info = {}

    def capture_optimizer_info(
        *args,
        **kwargs,
    ):
        output = original_optimizer(
            *args,
            **kwargs,
        )

        optimizer_info.update(
            output[2]
        )

        return output

    with patch.object(
        scipy.optimize,
        "fmin_l_bfgs_b",
        capture_optimizer_info,
    ):
        fitted = model.fit(
            method_kwargs={
                "method": "lbfgs",
                "maxiter": 1000,
                "disp": 0,
            }
        )

    mle_retvals = fitted.mle_retvals

    parameters = {
        name: float(value)
        for name, value in zip(
            fitted.param_names,
            fitted.params,
        )
    }

    ma_roots = np.roots(
        [
            parameters["ma.L2"],
            parameters["ma.L1"],
            1.0,
        ]
    )

    gradient = np.asarray(
        optimizer_info.get(
            "grad",
            np.nan,
        ),
        dtype=float,
    )

    row = {
        "origin": origin_date,
        "saved_converged": bool(
            saved_convergence.loc[
                origin_date
            ]
        ),
        "rerun_converged": bool(
            mle_retvals.get(
                "converged",
                False,
            )
        ),
        "warnflag": mle_retvals.get(
            "warnflag",
            np.nan,
        ),
        "task": optimizer_info.get(
            "task",
            None,
        ),
        "iterations": mle_retvals.get(
            "iterations",
            np.nan,
        ),
        "function_calls": optimizer_info.get(
            "funcalls",
            np.nan,
        ),
        "objective_value": mle_retvals.get(
            "fopt",
            np.nan,
        ),
        "gradient_l2_norm": np.linalg.norm(
            gradient
        ),
        "gradient_max_abs": np.max(
            np.abs(
                gradient
            )
        ),
        "drift": parameters["const"],
        "ma.L1": parameters["ma.L1"],
        "ma.L2": parameters["ma.L2"],
        "sigma2": parameters["sigma2"],
        "minimum_ma_root_modulus": min(
            np.abs(
                ma_roots
            )
        ),
        "training_return_volatility": np.std(
            log_returns,
            ddof=1,
        ),
        "trailing_20_return_volatility": np.std(
            log_returns[-20:],
            ddof=1,
        ),
        "trailing_60_return_volatility": np.std(
            log_returns[-60:],
            ddof=1,
        ),
    }

    for lag in [
        1,
        5,
        20,
    ]:
        row[
            f"absolute_{lag}_day_return"
        ] = abs(
            log_prices[-1]
            - log_prices[-lag - 1]
        )

    diagnostics.append(
        row
    )


# ---------------------------------------------------------
# Save and report reproducibility
# ---------------------------------------------------------

diagnostics = pd.DataFrame(
    diagnostics
)

diagnostics[
    "convergence_matches_saved"
] = (
    diagnostics["saved_converged"]
    == diagnostics["rerun_converged"]
)

diagnostics.to_csv(
    OUTPUT_PATH,
    index=False,
)

failed = diagnostics[
    ~diagnostics["rerun_converged"]
]

print(
    "Rerun forecast origins:",
    len(diagnostics),
)

print(
    "Rerun non-converged:",
    len(failed),
)

print(
    "All statuses match saved results:",
    diagnostics[
        "convergence_matches_saved"
    ].all(),
)

print()
print(
    failed[
        [
            "origin",
            "warnflag",
            "task",
            "iterations",
            "function_calls",
            "gradient_max_abs",
        ]
    ].to_string(
        index=False
    )
)

print()
print(
    "Diagnostics saved to:",
    OUTPUT_PATH,
)
