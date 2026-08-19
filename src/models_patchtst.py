import json
import random
import time
import warnings
from dataclasses import (
    asdict,
    dataclass,
)
from importlib.metadata import version

import numpy as np


@dataclass(frozen=True)
class PatchTSTTrainingConfig:
    context_length: int
    patch_length: int
    patch_stride: int
    prediction_length: int
    validation_points: int
    window_stride: int
    batch_size: int
    num_hidden_layers: int = 2
    d_model: int = 32
    num_attention_heads: int = 4
    ffn_dim: int = 64
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 30
    patience: int = 5
    min_delta: float = 1e-4
    seed: int = 2026

    @property
    def identifier(self):
        return json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
        )


def _set_deterministic_seed(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_patchtst_windows(
    train,
    config,
):
    """
    Build chronological training and end-of-history validation
    windows using only data available at one forecast origin.
    """

    values = np.asarray(
        train,
        dtype=np.float32,
    )

    if not np.all(
        np.isfinite(values)
    ):
        raise ValueError(
            "PatchTST training data contains non-finite values."
        )

    validation_start = (
        len(values)
        - config.validation_points
    )

    minimum_training_points = (
        config.context_length
        + config.prediction_length
    )

    if validation_start < minimum_training_points:
        raise ValueError(
            "Not enough pre-validation observations for "
            "PatchTST supervised windows."
        )

    if config.validation_points < config.prediction_length:
        raise ValueError(
            "validation_points must be at least the "
            "prediction length."
        )

    scaler_values = values[
        :validation_start
    ]

    scaler_mean = float(
        scaler_values.mean()
    )

    scaler_std = float(
        scaler_values.std()
    )

    if (
        not np.isfinite(scaler_std)
        or scaler_std == 0
    ):
        raise ValueError(
            "PatchTST training-only scale is invalid."
        )

    scaled = (
        values
        - scaler_mean
    ) / scaler_std

    last_start = (
        len(values)
        - config.context_length
        - config.prediction_length
    )

    training_last_start = (
        validation_start
        - config.context_length
        - config.prediction_length
    )

    training_starts = list(
        range(
            0,
            training_last_start + 1,
            config.window_stride,
        )
    )

    validation_first_start = (
        validation_start
        - config.context_length
    )

    validation_starts = list(
        range(
            validation_first_start,
            last_start + 1,
            config.window_stride,
        )
    )

    if not training_starts:
        raise ValueError(
            "No PatchTST training windows were generated."
        )

    if not validation_starts:
        raise ValueError(
            "No PatchTST validation windows were generated."
        )

    def construct(starts):
        past_values = []
        future_values = []

        for start in starts:
            forecast_start = (
                start
                + config.context_length
            )

            forecast_end = (
                forecast_start
                + config.prediction_length
            )

            past_values.append(
                scaled[
                    start:forecast_start
                ]
            )

            future_values.append(
                scaled[
                    forecast_start:forecast_end
                ]
            )

        return (
            np.asarray(
                past_values,
                dtype=np.float32,
            )[..., None],
            np.asarray(
                future_values,
                dtype=np.float32,
            )[..., None],
        )

    train_past, train_future = construct(
        training_starts
    )

    validation_past, validation_future = construct(
        validation_starts
    )

    return {
        "train_past": train_past,
        "train_future": train_future,
        "validation_past": validation_past,
        "validation_future": validation_future,
        "training_starts": training_starts,
        "validation_starts": validation_starts,
        "validation_start": validation_start,
        "scaler_mean": scaler_mean,
        "scaler_std": scaler_std,
    }


def _forward_warnings(caught_warnings):
    for caught in caught_warnings:
        warnings.warn_explicit(
            message=caught.message,
            category=caught.category,
            filename=caught.filename,
            lineno=caught.lineno,
        )


def patchtst_forecast(
    train,
    horizon,
    training_config,
    device="cpu",
):
    """Fit a new PatchTST at one origin and forecast levels."""

    import torch
    from torch.optim import AdamW
    from torch.utils.data import (
        DataLoader,
        TensorDataset,
    )
    from transformers import (
        PatchTSTConfig,
        PatchTSTForPrediction,
    )

    if horizon != training_config.prediction_length:
        raise ValueError(
            "PatchTST horizon must equal the configured "
            "direct prediction length."
        )

    windows = build_patchtst_windows(
        train=train,
        config=training_config,
    )

    origin_seed = (
        training_config.seed
        + len(train)
    )

    _set_deterministic_seed(
        origin_seed
    )

    model_config = PatchTSTConfig(
        num_input_channels=1,
        context_length=(
            training_config.context_length
        ),
        patch_length=(
            training_config.patch_length
        ),
        patch_stride=(
            training_config.patch_stride
        ),
        prediction_length=(
            training_config.prediction_length
        ),
        num_hidden_layers=(
            training_config.num_hidden_layers
        ),
        d_model=training_config.d_model,
        num_attention_heads=(
            training_config.num_attention_heads
        ),
        ffn_dim=training_config.ffn_dim,
        attention_dropout=(
            training_config.dropout
        ),
        positional_dropout=(
            training_config.dropout
        ),
        ff_dropout=training_config.dropout,
        head_dropout=training_config.dropout,
        norm_type="layernorm",
        loss="mse",
        scaling=False,
    )

    model = PatchTSTForPrediction(
        model_config
    ).to(
        device
    )

    train_dataset = TensorDataset(
        torch.from_numpy(
            windows["train_past"]
        ),
        torch.from_numpy(
            windows["train_future"]
        ),
    )

    validation_dataset = TensorDataset(
        torch.from_numpy(
            windows["validation_past"]
        ),
        torch.from_numpy(
            windows["validation_future"]
        ),
    )

    generator = torch.Generator()
    generator.manual_seed(
        origin_seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=(
            training_config.weight_decay
        ),
    )

    best_validation_loss = np.inf
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    epochs_trained = 0

    training_start = time.perf_counter()

    with warnings.catch_warnings(
        record=True
    ) as caught_warnings:
        warnings.simplefilter(
            "always"
        )

        for epoch in range(
            1,
            training_config.max_epochs + 1,
        ):
            model.train()

            for past_values, future_values in train_loader:
                past_values = past_values.to(
                    device
                )
                future_values = future_values.to(
                    device
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                output = model(
                    past_values=past_values,
                    future_values=future_values,
                )

                if not torch.isfinite(
                    output.loss
                ):
                    raise FloatingPointError(
                        "PatchTST training loss is not finite."
                    )

                output.loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

            model.eval()
            validation_losses = []

            with torch.no_grad():
                for (
                    past_values,
                    future_values,
                ) in validation_loader:
                    output = model(
                        past_values=(
                            past_values.to(device)
                        ),
                        future_values=(
                            future_values.to(device)
                        ),
                    )

                    validation_losses.append(
                        float(
                            output.loss.detach().cpu()
                        )
                    )

            validation_loss = float(
                np.mean(
                    validation_losses
                )
            )

            if not np.isfinite(
                validation_loss
            ):
                raise FloatingPointError(
                    "PatchTST validation loss is not finite."
                )

            epochs_trained = epoch

            if validation_loss < (
                best_validation_loss
                - training_config.min_delta
            ):
                best_validation_loss = (
                    validation_loss
                )
                best_epoch = epoch
                best_state = {
                    name: parameter
                    .detach()
                    .cpu()
                    .clone()
                    for name, parameter in (
                        model.state_dict().items()
                    )
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= (
                training_config.patience
            ):
                break

    training_elapsed_seconds = (
        time.perf_counter()
        - training_start
    )

    _forward_warnings(
        caught_warnings
    )

    if best_state is None:
        raise RuntimeError(
            "PatchTST did not produce a valid checkpoint."
        )

    model.load_state_dict(
        best_state
    )
    model.eval()

    full_values = np.asarray(
        train,
        dtype=np.float32,
    )

    inference_context = (
        full_values[
            -training_config.context_length:
        ]
        - windows["scaler_mean"]
    ) / windows["scaler_std"]

    inference_tensor = torch.from_numpy(
        inference_context[
            None,
            :,
            None,
        ]
    ).to(
        device
    )

    inference_start = time.perf_counter()

    with torch.no_grad():
        prediction = model(
            past_values=inference_tensor
        ).prediction_outputs

    inference_elapsed_seconds = (
        time.perf_counter()
        - inference_start
    )

    standardized_forecast = (
        prediction[
            0,
            :,
            0,
        ]
        .detach()
        .cpu()
        .numpy()
    )

    forecast = (
        standardized_forecast
        * windows["scaler_std"]
        + windows["scaler_mean"]
    ).astype(
        float
    )

    parameters_are_finite = all(
        bool(
            torch.isfinite(parameter).all()
        )
        for parameter in model.parameters()
    )

    validation_start = windows[
        "validation_start"
    ]

    warning_text = " | ".join(
        str(caught.message)
        for caught in caught_warnings
    )

    return {
        "forecast": forecast,
        "standardized_forecast": (
            standardized_forecast.astype(float)
        ),
        "forecast_scale": "original",
        "model_identifier": "PatchTST-HF-supervised-v1",
        "transformers_version": version(
            "transformers"
        ),
        "device": device,
        "model_configuration": (
            training_config.identifier
        ),
        "context_length": (
            training_config.context_length
        ),
        "patch_length": (
            training_config.patch_length
        ),
        "patch_stride": (
            training_config.patch_stride
        ),
        "prediction_length": (
            training_config.prediction_length
        ),
        "window_stride": (
            training_config.window_stride
        ),
        "num_training_windows": len(
            windows["training_starts"]
        ),
        "num_validation_windows": len(
            windows["validation_starts"]
        ),
        "training_data_end": train.index[
            validation_start - 1
        ],
        "validation_start": train.index[
            validation_start
        ],
        "validation_end": train.index[-1],
        "forecast_origin": train.index[-1],
        "scaler_fit_end": train.index[
            validation_start - 1
        ],
        "scaler_mean": windows[
            "scaler_mean"
        ],
        "scaler_std": windows[
            "scaler_std"
        ],
        "epochs_trained": epochs_trained,
        "best_epoch": best_epoch,
        "early_stopped": (
            epochs_trained
            < training_config.max_epochs
        ),
        "best_validation_loss": (
            best_validation_loss
        ),
        "training_elapsed_seconds": (
            training_elapsed_seconds
        ),
        "inference_elapsed_seconds": (
            inference_elapsed_seconds
        ),
        "num_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
        ),
        "forecast_is_finite": bool(
            np.all(
                np.isfinite(forecast)
            )
        ),
        "parameters_are_finite": (
            parameters_are_finite
        ),
        "training_warning": warning_text,
        "training_exception": "",
    }
