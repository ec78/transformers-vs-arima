"""Reproduce the documented pre-evaluation classical candidate searches.

This script is intentionally disconnected from the forecasting configuration.
It reports candidate information criteria and never updates frozen models or
forecast files. Convergence warnings remain visible.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from src.data import load_airline, load_electricity, load_spy
from src.preprocessing import (
    prepare_airline,
    prepare_electricity,
    prepare_spy,
)


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "model_selection"


def _fit_candidate(
    values,
    order,
    seasonal_order=(0, 0, 0, 0),
    trend="n",
    low_memory=False,
):
    try:
        result = ARIMA(
            values,
            order=order,
            seasonal_order=seasonal_order,
            trend=trend,
        ).fit(
            method_kwargs={
                "method": "lbfgs",
                "maxiter": 1000,
                "disp": 0,
            },
            low_memory=low_memory,
        )
        retvals = getattr(result, "mle_retvals", {})
        return {
            "aic": result.aic,
            "bic": result.bic,
            "hqic": result.hqic,
            "loglikelihood": result.llf,
            "converged": retvals.get("converged", np.nan),
            "iterations": retvals.get("iterations", retvals.get("nit", np.nan)),
            "warnflag": retvals.get("warnflag", retvals.get("status", np.nan)),
            "fit_exception": None,
        }
    except Exception as error:
        return {
            "aic": np.nan,
            "bic": np.nan,
            "hqic": np.nan,
            "loglikelihood": np.nan,
            "converged": False,
            "iterations": np.nan,
            "warnflag": np.nan,
            "fit_exception": str(error),
        }


def _rank_information_criteria(frame):
    frame = frame.copy()
    frame["aic_rank"] = frame["aic"].rank(method="min")
    frame["bic_rank"] = frame["bic"].rank(method="min")
    return frame.sort_values(["aic_rank", "bic_rank"]).reset_index(drop=True)


def reproduce_spy():
    data = prepare_spy(load_spy())
    values = data.loc[data.index < pd.Timestamp("2019-01-01"), "log_price"].dropna()

    grid_rows = []
    for p, q in itertools.product(range(4), repeat=2):
        row = {
            "dataset": "SPY",
            "selection_stage": "no-drift order grid",
            "training_end_exclusive": "2019-01-01",
            "n_training_observations": len(values),
            "p": p,
            "d": 1,
            "q": q,
            "trend": "n",
        }
        row.update(_fit_candidate(values, order=(p, 1, q), trend="n"))
        grid_rows.append(row)

    drift_rows = []
    for p, q, trend, label in [
        (0, 0, "n", "ARIMA(0,1,0)"),
        (0, 0, "t", "ARIMA(0,1,0)+drift"),
        (0, 2, "n", "ARIMA(0,1,2)"),
        (0, 2, "t", "ARIMA(0,1,2)+drift"),
    ]:
        row = {
            "dataset": "SPY",
            "selection_stage": "separate drift check",
            "training_end_exclusive": "2019-01-01",
            "n_training_observations": len(values),
            "model": label,
            "p": p,
            "d": 1,
            "q": q,
            "trend": trend,
        }
        row.update(_fit_candidate(values, order=(p, 1, q), trend=trend))
        drift_rows.append(row)

    return (
        _rank_information_criteria(pd.DataFrame(grid_rows)),
        _rank_information_criteria(pd.DataFrame(drift_rows)),
    )


def reproduce_airline():
    data = prepare_airline(load_airline())
    values = np.log(
        data.loc[data.index < pd.Timestamp("1959-01-01"), "y"].to_numpy(dtype=float)
    )
    rows = []
    for p, q, seasonal_p, seasonal_q in itertools.product([0, 1], repeat=4):
        row = {
            "dataset": "Airline",
            "training_end_exclusive": "1959-01-01",
            "n_training_observations": len(values),
            "p": p,
            "d": 1,
            "q": q,
            "P": seasonal_p,
            "D": 1,
            "Q": seasonal_q,
            "seasonal_period": 12,
            "trend": "n",
        }
        row.update(
            _fit_candidate(
                values,
                order=(p, 1, q),
                seasonal_order=(seasonal_p, 1, seasonal_q, 12),
                trend="n",
            )
        )
        rows.append(row)
    return _rank_information_criteria(pd.DataFrame(rows))


def reproduce_electricity():
    data = prepare_electricity(load_electricity("CISO"))
    values = np.log(
        data.loc[
            data.index < pd.Timestamp("2025-01-01", tz="UTC"),
            "y",
        ].to_numpy(dtype=float)
    )
    values = values[168:] - values[:-168]
    rows = []
    for p, q in itertools.product(range(3), repeat=2):
        row = {
            "dataset": "Electricity-CISO",
            "training_end_exclusive": "2025-01-01T00:00:00+00:00",
            "n_training_observations": len(values),
            "p": p,
            "d": 0,
            "q": q,
            "trend": "c",
            "source_transformation": "log weekly difference (lag 168)",
        }
        row.update(
            _fit_candidate(
                values,
                order=(p, 0, q),
                trend="c",
                low_memory=True,
            )
        )
        rows.append(row)
    return _rank_information_criteria(pd.DataFrame(rows))


def run(selected="all", output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    if selected in {"all", "spy"}:
        grid, drift = reproduce_spy()
        outputs["spy_order_grid_no_drift.csv"] = grid
        outputs["spy_drift_check.csv"] = drift
    if selected in {"all", "airline"}:
        outputs["airline_selection.csv"] = reproduce_airline()
    if selected in {"all", "electricity"}:
        outputs["electricity_selection.csv"] = reproduce_electricity()

    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["all", "spy", "airline", "electricity"],
        default="all",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    outputs = run(selected=args.dataset, output_dir=args.output_dir)
    for filename, frame in outputs.items():
        print(f"\n{filename}")
        print(frame.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
