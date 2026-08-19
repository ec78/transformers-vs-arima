import time
import warnings
from importlib.metadata import (
    PackageNotFoundError,
    version,
)

import numpy as np


CHRONOS_2_MODEL_ID = "amazon/chronos-2"
CHRONOS_2_REVISION = (
    "29ec3766d36d6f73f0696f85560a422f50e8498c"
)
CHRONOS_2_MAX_CONTEXT_LENGTH = 8192


def chronos_package_version():
    """Return the installed chronos-forecasting version."""

    try:
        return version(
            "chronos-forecasting"
        )
    except PackageNotFoundError:
        return "not-installed"


def load_chronos_2_pipeline(
    model_id=CHRONOS_2_MODEL_ID,
    revision=CHRONOS_2_REVISION,
    device=None,
):
    """Load the official Chronos-2 pipeline once."""

    try:
        import torch
        from chronos import Chronos2Pipeline
    except ImportError as error:
        raise ImportError(
            "Chronos-2 requires the official "
            "chronos-forecasting package."
        ) from error

    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    pipeline = (
        Chronos2Pipeline.from_pretrained(
            model_id,
            revision=revision,
            device_map=device,
        )
    )

    return pipeline, device


def _as_numpy_point_forecast(
    point_forecast,
):
    """Convert one Chronos point forecast to 1-D numpy."""

    if hasattr(
        point_forecast,
        "detach",
    ):
        point_forecast = (
            point_forecast
            .detach()
            .cpu()
            .numpy()
        )

    values = np.asarray(
        point_forecast,
        dtype=float,
    )

    if values.ndim == 2:
        if values.shape[0] != 1:
            raise ValueError(
                "Expected a univariate Chronos forecast."
            )

        values = values[0]

    if values.ndim != 1:
        raise ValueError(
            "Chronos point forecast must be one-dimensional."
        )

    return values


class Chronos2ForecastAdapter:
    """
    Batch Chronos-2 origins, then expose the existing model API.

    Cross-learning is disabled, so batching changes only execution
    efficiency and does not share information across forecast origins.
    """

    def __init__(
        self,
        pipeline,
        device,
        model_id=CHRONOS_2_MODEL_ID,
        model_revision=CHRONOS_2_REVISION,
        max_context_length=(
            CHRONOS_2_MAX_CONTEXT_LENGTH
        ),
    ):
        self.pipeline = pipeline
        self.device = device
        self.model_id = model_id
        self.model_revision = model_revision
        self.package_version = (
            chronos_package_version()
        )

        model_context_length = getattr(
            pipeline,
            "model_context_length",
            max_context_length,
        )

        self.max_context_length = min(
            int(model_context_length),
            int(max_context_length),
        )

        self._forecast_cache = {}

    @staticmethod
    def _forward_warnings(caught_warnings):
        for caught in caught_warnings:
            warnings.warn_explicit(
                message=caught.message,
                category=caught.category,
                filename=caught.filename,
                lineno=caught.lineno,
            )

    def _predict_batch(
        self,
        batch,
        horizon,
    ):
        inputs = [
            item["context"]
            for item in batch
        ]

        start_time = time.perf_counter()

        try:
            with warnings.catch_warnings(
                record=True
            ) as caught_warnings:
                warnings.simplefilter(
                    "always"
                )

                _, point_forecasts = (
                    self.pipeline.predict_quantiles(
                        inputs=inputs,
                        prediction_length=horizon,
                        quantile_levels=[0.5],
                        batch_size=len(inputs),
                        context_length=(
                            self.max_context_length
                        ),
                        cross_learning=False,
                    )
                )
        except Exception as error:
            origin_sizes = [
                item["train_size"]
                for item in batch
            ]

            raise RuntimeError(
                "Chronos-2 inference failed for training "
                f"sizes {origin_sizes}: {error}"
            ) from error

        elapsed_seconds = (
            time.perf_counter()
            - start_time
        )

        self._forward_warnings(
            caught_warnings
        )

        warning_text = " | ".join(
            str(caught.message)
            for caught in caught_warnings
        )

        if len(point_forecasts) != len(batch):
            raise ValueError(
                "Chronos-2 returned an unexpected number "
                "of point forecasts."
            )

        seconds_per_origin = (
            elapsed_seconds
            / len(batch)
        )

        for item, point_forecast in zip(
            batch,
            point_forecasts,
        ):
            forecast = (
                _as_numpy_point_forecast(
                    point_forecast
                )
            )

            if len(forecast) != horizon:
                raise ValueError(
                    "Chronos-2 returned an unexpected "
                    "forecast length."
                )

            self._forecast_cache[
                item["train_size"]
            ] = {
                "forecast": forecast,
                "model_identifier": self.model_id,
                "model_revision": self.model_revision,
                "chronos_package_version": (
                    self.package_version
                ),
                "device": self.device,
                "history_length": item[
                    "train_size"
                ],
                "context_length": len(
                    item["context"]
                ),
                "max_context_length": (
                    self.max_context_length
                ),
                "context_truncated": (
                    item["train_size"]
                    > self.max_context_length
                ),
                "context_end": item[
                    "context_end"
                ],
                "point_forecast_method": (
                    "standard predictions output "
                    "(trained 0.5 quantile)"
                ),
                "inference_elapsed_seconds": (
                    seconds_per_origin
                ),
                "inference_batch_size": len(batch),
                "forecast_is_finite": bool(
                    np.all(
                        np.isfinite(forecast)
                    )
                ),
                "inference_warning": warning_text,
                "inference_exception": "",
            }

    def prepare_origins(
        self,
        series,
        origins,
        horizon,
        batch_size=8,
    ):
        """Forecast all origins using only each origin's history."""

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be positive."
            )

        self._forecast_cache = {}
        batch = []

        for origin in origins:
            train = series.iloc[:origin]

            context = np.asarray(
                train.iloc[
                    -self.max_context_length:
                ],
                dtype=np.float32,
            )

            batch.append(
                {
                    "train_size": len(train),
                    "context": context,
                    "context_end": train.index[-1],
                }
            )

            if len(batch) == batch_size:
                self._predict_batch(
                    batch=batch,
                    horizon=horizon,
                )

                batch = []

        if batch:
            self._predict_batch(
                batch=batch,
                horizon=horizon,
            )

    def __call__(
        self,
        train,
        horizon=1,
    ):
        """Return a cached origin forecast to the evaluator."""

        train_size = len(train)

        if train_size not in self._forecast_cache:
            raise KeyError(
                "Chronos-2 forecast was not prepared for "
                f"training size {train_size}."
            )

        result = self._forecast_cache[
            train_size
        ]

        if len(result["forecast"]) != horizon:
            raise ValueError(
                "Cached Chronos-2 horizon does not match "
                "the evaluation horizon."
            )

        return result.copy()
