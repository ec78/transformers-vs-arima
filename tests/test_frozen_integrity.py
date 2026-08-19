from pathlib import Path

import numpy as np
import pandas as pd

from analyze_spy_retry_effect import (
    FORECAST_PATH,
    load_pre_retry_forecasts,
    summarize_retry_effect,
)
from audit_frozen_sources import (
    build_raw_manifest,
    reconcile_frozen_forecasts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_raw_snapshot_hashes_match_contract():
    manifest = build_raw_manifest()

    assert len(manifest) == 3
    assert manifest["hash_matches_contract"].all()
    assert (
        manifest["n_raw_rows"]
        == manifest["metadata_n_observations"]
    ).all()


def test_all_authoritative_forecasts_reconcile_to_raw_snapshots():
    reconciliation = reconcile_frozen_forecasts()

    assert len(reconciliation) == 12
    assert reconciliation["passed"].all()
    assert reconciliation["timestamps_and_horizons_aligned"].all()
    assert reconciliation["n_origins"].sum() == 4 * (380 + 13 + 84)
    assert reconciliation["max_train_size_difference"].max() == 0
    assert reconciliation["max_actual_absolute_difference"].max() < 1e-10
    assert reconciliation["max_mase_scale_absolute_difference"].max() < 1e-10


def test_spy_retry_metric_effect_is_below_one_thousandth_percent():
    before = load_pre_retry_forecasts()
    after = pd.read_csv(FORECAST_PATH)
    summary = summarize_retry_effect(before, after)

    assert set(summary["changed_origins"]) == {25}
    assert np.max(
        np.abs(
            summary[["rmse_pct_change", "mae_pct_change"]].to_numpy()
        )
    ) < 0.001
    assert summary["max_absolute_forecast_change"].max() < 0.013
