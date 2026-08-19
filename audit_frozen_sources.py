"""Reconcile authoritative frozen forecasts with committed raw snapshots.

This is an integrity audit only. It does not fit or invoke any forecasting
model, and it does not modify raw data or frozen forecast files.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_airline, load_electricity, load_spy
from src.preprocessing import (
    prepare_airline,
    prepare_electricity,
    prepare_spy,
)


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
FORECAST_DIR = PROJECT_ROOT / "results" / "forecasts"
ANALYSIS_DIR = PROJECT_ROOT / "results" / "analysis"

RAW_SNAPSHOT_HASHES = {
    "spy.csv": "08e394938b0a1eaece531abb7f3e79aa7d3e6db1f01a6e6696f76ec6209e5834",
    "airline.csv": "7cb940ddaba95d867d1e414d11b262bd966d884be229eb112779d3e8f06f44d9",
    "electricity_ciso.csv": "c89e1c6f771a7c80fac0c7c9793417380f2d225d13dd7592a5786e256aa47601",
}

DATASET_SPECS = {
    "SPY": {
        "raw_file": "spy.csv",
        "metadata_file": "spy_metadata.json",
        "data": lambda: prepare_spy(load_spy()),
        "forecast_files": [
            "spy_classical_forecasts.csv",
            "spy_chronos_forecasts.csv",
            "spy_patchtst_forecasts.csv",
        ],
        "mase_seasonality": 1,
        "expected_origins": 380,
        "horizons": [1, 5, 20],
    },
    "Airline": {
        "raw_file": "airline.csv",
        "metadata_file": "airline_metadata.json",
        "data": lambda: prepare_airline(load_airline()),
        "forecast_files": [
            "airline_classical_forecasts.csv",
            "airline_chronos_forecasts.csv",
            "airline_patchtst_forecasts.csv",
        ],
        "mase_seasonality": 12,
        "expected_origins": 13,
        "horizons": [1, 3, 12],
    },
    "Electricity-CISO": {
        "raw_file": "electricity_ciso.csv",
        "metadata_file": "electricity_ciso_metadata.json",
        "data": lambda: prepare_electricity(load_electricity("CISO")),
        "forecast_files": [
            "electricity_ciso_classical_forecasts.csv",
            "electricity_ciso_chronos_forecasts.csv",
            "electricity_ciso_patchtst_forecasts.csv",
        ],
        "mase_seasonality": 168,
        "expected_origins": 84,
        "horizons": [1, 24, 168],
    },
}


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_raw_manifest():
    rows = []
    for dataset, spec in DATASET_SPECS.items():
        raw_path = RAW_DIR / spec["raw_file"]
        metadata_path = RAW_DIR / spec["metadata_file"]
        digest = _sha256(raw_path)
        expected_digest = RAW_SNAPSHOT_HASHES[spec["raw_file"]]
        if digest != expected_digest:
            raise AssertionError(
                f"Raw snapshot hash changed for {spec['raw_file']}. "
                "Treat refreshed data as a new experiment contract."
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        raw = pd.read_csv(raw_path)
        rows.append(
            {
                "dataset": dataset,
                "raw_file": str(raw_path.relative_to(PROJECT_ROOT)),
                "metadata_file": str(metadata_path.relative_to(PROJECT_ROOT)),
                "sha256": digest,
                "hash_matches_contract": True,
                "n_raw_rows": len(raw),
                "metadata_n_observations": metadata.get("n_observations"),
                "snapshot_start": metadata.get("start_date"),
                "snapshot_end": metadata.get("end_date"),
                "download_date": metadata.get("download_date"),
            }
        )
    return pd.DataFrame(rows)


def _to_utc_index(index):
    return pd.to_datetime(index, utc=True)


def reconcile_frozen_forecasts():
    summaries = []
    for dataset, spec in DATASET_SPECS.items():
        data = spec["data"]()
        values = data["y"].to_numpy(dtype=float)
        index = _to_utc_index(data.index)
        positions = {timestamp: position for position, timestamp in enumerate(index)}
        scale_cache = {}

        for filename in spec["forecast_files"]:
            path = FORECAST_DIR / filename
            forecasts = pd.read_csv(path)
            forecasts["origin"] = pd.to_datetime(forecasts["origin"], utc=True)
            forecasts["forecast_date"] = pd.to_datetime(
                forecasts["forecast_date"],
                utc=True,
            )
            if set(forecasts["dataset"].unique()) != {dataset}:
                raise AssertionError(f"Dataset label mismatch in {filename}.")

            for model, group in forecasts.groupby("model", sort=False):
                max_actual_difference = 0.0
                max_origin_difference = 0.0
                max_mase_scale_difference = 0.0
                max_train_size_difference = 0

                if group["origin"].nunique() != spec["expected_origins"]:
                    raise AssertionError(
                        f"Unexpected origin count for {dataset} / {model}."
                    )
                if sorted(group["horizon"].unique().tolist()) != spec["horizons"]:
                    raise AssertionError(
                        f"Unexpected horizons for {dataset} / {model}."
                    )

                for row in group.itertuples(index=False):
                    if row.origin not in positions or row.forecast_date not in positions:
                        raise AssertionError(
                            f"Timestamp missing from source for {dataset} / {model}."
                        )
                    origin_position = positions[row.origin]
                    forecast_position = positions[row.forecast_date]
                    if forecast_position != origin_position + int(row.horizon):
                        raise AssertionError(
                            f"Forecast alignment mismatch for {dataset} / {model}."
                        )

                    max_actual_difference = max(
                        max_actual_difference,
                        abs(float(row.actual) - values[forecast_position]),
                    )
                    max_origin_difference = max(
                        max_origin_difference,
                        abs(float(row.origin_actual) - values[origin_position]),
                    )
                    max_train_size_difference = max(
                        max_train_size_difference,
                        abs(int(row.train_size) - (origin_position + 1)),
                    )

                    if row.origin not in scale_cache:
                        train = values[: origin_position + 1]
                        lag = spec["mase_seasonality"]
                        scale_cache[row.origin] = np.mean(
                            np.abs(train[lag:] - train[:-lag])
                        )
                    max_mase_scale_difference = max(
                        max_mase_scale_difference,
                        abs(float(row.mase_scale) - scale_cache[row.origin]),
                    )

                tolerance = 1e-10
                passed = (
                    max_actual_difference <= tolerance
                    and max_origin_difference <= tolerance
                    and max_mase_scale_difference <= tolerance
                    and max_train_size_difference == 0
                )
                if not passed:
                    raise AssertionError(
                        f"Frozen/source reconciliation failed for {dataset} / {model}."
                    )
                summaries.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "source_file": filename,
                        "n_rows": len(group),
                        "n_origins": group["origin"].nunique(),
                        "max_actual_absolute_difference": max_actual_difference,
                        "max_origin_absolute_difference": max_origin_difference,
                        "max_mase_scale_absolute_difference": (
                            max_mase_scale_difference
                        ),
                        "max_train_size_difference": max_train_size_difference,
                        "timestamps_and_horizons_aligned": True,
                        "passed": True,
                    }
                )
    return pd.DataFrame(summaries)


def run(output_dir=ANALYSIS_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_raw_manifest()
    reconciliation = reconcile_frozen_forecasts()
    manifest.to_csv(output_dir / "raw_data_manifest.csv", index=False)
    reconciliation.to_csv(
        output_dir / "source_reconciliation.csv",
        index=False,
    )
    return manifest, reconciliation


def main():
    manifest, reconciliation = run()
    print("RAW SNAPSHOT MANIFEST")
    print(manifest.to_string(index=False))
    print("\nFROZEN FORECAST RECONCILIATION")
    print(reconciliation.to_string(index=False))


if __name__ == "__main__":
    main()
