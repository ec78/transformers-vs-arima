from pathlib import Path

import numpy as np

from src.final_analysis import (
    DATASET_SPECS,
    build_authoritative_comparison,
    build_rankings,
    build_relative_performance,
    diebold_mariano_test,
    load_frozen_forecasts,
    percentage_difference,
    validate_frozen_forecasts,
    validate_saved_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def test_percentage_difference_sign_convention():
    assert np.isclose(percentage_difference(90.0, 100.0), -10.0)
    assert np.isclose(percentage_difference(110.0, 100.0), 10.0)
    assert np.isclose(percentage_difference(100.0, 100.0), 0.0)


def test_dm_test_uses_pairing_hac_and_horizon_adjustment():
    errors_a = np.array([1.0, 1.2, 0.8, 1.1, 0.9, 1.3, 0.7, 1.0])
    errors_b = np.array([2.0, 2.1, 1.9, 2.2, 1.8, 2.3, 1.7, 2.0])

    result = diebold_mariano_test(
        errors_a=errors_a,
        errors_b=errors_b,
        horizon=20,
        origin_step=5,
    )

    assert result["n_forecasts"] == 8
    assert result["effective_horizon_in_origin_steps"] == 4
    assert result["overlap_lag"] == 3
    assert result["hac_lag"] >= result["overlap_lag"]
    assert result["mean_loss_difference_a_minus_b"] < 0
    assert result["test_statistic"] < 0
    assert 0 <= result["p_value"] <= 1


def test_frozen_contract_and_saved_metrics_match():
    forecasts = load_frozen_forecasts(PROJECT_ROOT)
    validation = validate_frozen_forecasts(forecasts)
    comparison = build_authoritative_comparison(forecasts)
    metric_validation = validate_saved_metrics(
        comparison,
        PROJECT_ROOT,
    )

    assert validation["passed"].all()
    assert metric_validation["passed"].all()
    assert len(comparison) == 36
    for dataset, spec in DATASET_SPECS.items():
        selected = comparison[comparison["dataset"] == dataset]
        assert sorted(selected["horizon"].unique()) == spec["horizons"]
        assert set(selected["n_forecasts"]) == {spec["expected_origins"]}


def test_relative_performance_and_rankings_are_metric_specific():
    forecasts = load_frozen_forecasts(PROJECT_ROOT)
    comparison = build_authoritative_comparison(forecasts)
    relative = build_relative_performance(comparison)
    rankings, win_counts = build_rankings(comparison)

    naive = relative[relative["model"] == "Naive"]
    assert np.allclose(naive["rmse_pct_difference_vs_naive"], 0.0)
    assert np.allclose(naive["mae_pct_difference_vs_naive"], 0.0)
    assert np.allclose(naive["mase_pct_difference_vs_naive"], 0.0)
    assert set(rankings["metric"]) == {"rmse", "mae", "mase"}
    assert "composite" not in rankings.columns
    assert set(win_counts["metric"]) == {"rmse", "mae", "mase"}
    assert len(win_counts) == 12
    patchtst_wins = win_counts[
        win_counts["model_family"] == "PatchTST"
    ]
    assert (patchtst_wins["n_dataset_horizon_wins"] == 0).all()
