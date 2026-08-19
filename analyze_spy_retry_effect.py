"""Compare the frozen SPY forecasts before and after the retry policy.

No model is fitted. The pre-retry forecast file is read from Git history and
compared with the current frozen forecast file.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PRE_RETRY_REF = "e93958d"
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
    / "spy_arima_retry_effect.csv"
)
MODEL_NAME = "ARIMA(0,1,2)+drift"


def load_pre_retry_forecasts(git_ref=DEFAULT_PRE_RETRY_REF):
    content = subprocess.check_output(
        [
            "git",
            "show",
            f"{git_ref}:results/forecasts/spy_classical_forecasts.csv",
        ],
        cwd=PROJECT_ROOT,
    )
    return pd.read_csv(io.BytesIO(content))


def summarize_retry_effect(before, after):
    before = before[before["model"] == MODEL_NAME].copy()
    after = after[after["model"] == MODEL_NAME].copy()
    merged = before.merge(
        after,
        on=["origin", "horizon"],
        suffixes=("_before", "_after"),
        validate="one_to_one",
    )

    if len(merged) != len(before) or len(merged) != len(after):
        raise AssertionError("Pre/post retry forecast keys do not match.")
    if not np.allclose(
        merged["actual_before"],
        merged["actual_after"],
        rtol=0,
        atol=1e-12,
    ):
        raise AssertionError("Pre/post retry actual observations differ.")

    changed = ~np.isclose(
        merged["forecast_before"],
        merged["forecast_after"],
        rtol=0,
        atol=1e-15,
    )
    changed_origins = merged.loc[changed, "origin"].nunique()
    rows = []
    for horizon, group in merged.groupby("horizon", sort=True):
        error_before = group["actual_before"] - group["forecast_before"]
        error_after = group["actual_after"] - group["forecast_after"]
        rmse_before = np.sqrt(np.mean(np.square(error_before)))
        rmse_after = np.sqrt(np.mean(np.square(error_after)))
        mae_before = np.mean(np.abs(error_before))
        mae_after = np.mean(np.abs(error_after))
        rows.append(
            {
                "horizon": int(horizon),
                "n_forecasts": len(group),
                "changed_origins": changed_origins,
                "rmse_before_retry": rmse_before,
                "rmse_after_retry": rmse_after,
                "rmse_pct_change": 100.0 * (rmse_after / rmse_before - 1.0),
                "mae_before_retry": mae_before,
                "mae_after_retry": mae_after,
                "mae_pct_change": 100.0 * (mae_after / mae_before - 1.0),
                "max_absolute_forecast_change": np.max(
                    np.abs(
                        group["forecast_after"]
                        - group["forecast_before"]
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def run(git_ref=DEFAULT_PRE_RETRY_REF, output_path=OUTPUT_PATH):
    before = load_pre_retry_forecasts(git_ref=git_ref)
    after = pd.read_csv(FORECAST_PATH)
    summary = summarize_retry_effect(before, after)
    if summary["changed_origins"].nunique() != 1 or (
        summary["changed_origins"].iloc[0] != 25
    ):
        raise AssertionError("Expected exactly 25 changed SPY origins.")
    maximum_metric_change = max(
        summary["rmse_pct_change"].abs().max(),
        summary["mae_pct_change"].abs().max(),
    )
    if maximum_metric_change >= 0.001:
        raise AssertionError("Retry metric effect is not below 0.001%.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-retry-ref", default=DEFAULT_PRE_RETRY_REF)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    summary = run(
        git_ref=args.pre_retry_ref,
        output_path=args.output,
    )
    print(summary.to_string(index=False))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
