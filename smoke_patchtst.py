import numpy as np
import pandas as pd

from src.models_patchtst import (
    PatchTSTTrainingConfig,
    build_patchtst_windows,
    patchtst_forecast,
)


config = PatchTSTTrainingConfig(
    context_length=12,
    patch_length=4,
    patch_stride=2,
    prediction_length=4,
    validation_points=16,
    window_stride=2,
    batch_size=8,
    num_hidden_layers=1,
    d_model=8,
    num_attention_heads=2,
    ffn_dim=16,
    dropout=0.0,
    max_epochs=2,
    patience=1,
)

series = pd.Series(
    np.linspace(
        10.0,
        30.0,
        80,
    ),
    index=pd.date_range(
        "2020-01-01",
        periods=80,
        freq="D",
    ),
)


# ---------------------------------------------------------
# Training-only scaling and chronological validation
# ---------------------------------------------------------

windows = build_patchtst_windows(
    train=series,
    config=config,
)

validation_start = (
    len(series)
    - config.validation_points
)

assert windows[
    "validation_start"
] == validation_start

assert np.isclose(
    windows["scaler_mean"],
    series.iloc[
        :validation_start
    ].mean(),
)

assert np.isclose(
    windows["scaler_std"],
    series.iloc[
        :validation_start
    ].to_numpy().std(),
)

for start in windows[
    "training_starts"
]:
    target_end = (
        start
        + config.context_length
        + config.prediction_length
    )

    assert target_end <= validation_start

for start in windows[
    "validation_starts"
]:
    target_start = (
        start
        + config.context_length
    )

    target_end = (
        target_start
        + config.prediction_length
    )

    assert target_start >= validation_start
    assert target_end <= len(series)

altered_validation = series.copy()
altered_validation.iloc[
    validation_start:
] += 1000.0

altered_windows = build_patchtst_windows(
    train=altered_validation,
    config=config,
)

assert windows[
    "scaler_mean"
] == altered_windows[
    "scaler_mean"
]

assert windows[
    "scaler_std"
] == altered_windows[
    "scaler_std"
]

assert np.array_equal(
    windows["train_past"],
    altered_windows["train_past"],
)

assert np.array_equal(
    windows["train_future"],
    altered_windows["train_future"],
)


# ---------------------------------------------------------
# Maintained Hugging Face model plumbing
# ---------------------------------------------------------

result = patchtst_forecast(
    train=series,
    horizon=4,
    training_config=config,
    device="cpu",
)

assert len(result["forecast"]) == 4
assert result["forecast_is_finite"]
assert result["parameters_are_finite"]
assert result[
    "forecast_origin"
] == series.index[-1]
assert result[
    "training_data_end"
] < result[
    "validation_start"
]
assert result[
    "validation_end"
] == series.index[-1]
assert result[
    "scaler_fit_end"
] < result[
    "validation_start"
]
assert result[
    "num_training_windows"
] == len(
    windows["training_starts"]
)
assert result[
    "num_validation_windows"
] == len(
    windows["validation_starts"]
)
assert result[
    "transformers_version"
] == "5.15.0"

print(
    "PatchTST leakage and plumbing assertions passed."
)
