import warnings

import numpy as np
import pandas as pd

from src.evaluation import (
    generate_forecast_origins,
    walk_forward_evaluate,
)

from src.models_transformer import (
    Chronos2ForecastAdapter,
)

from run_chronos_classical_comparison import (
    audit_chronos_results,
)


class FakeChronos2Pipeline:
    model_context_length = 4

    def __init__(self):
        self.calls = []

    def predict_quantiles(
        self,
        inputs,
        prediction_length,
        quantile_levels,
        batch_size,
        context_length,
        cross_learning,
    ):
        copied_inputs = [
            np.asarray(item).copy()
            for item in inputs
        ]

        self.calls.append(
            {
                "inputs": copied_inputs,
                "prediction_length": prediction_length,
                "quantile_levels": quantile_levels,
                "batch_size": batch_size,
                "context_length": context_length,
                "cross_learning": cross_learning,
            }
        )

        warnings.warn(
            "mock inference warning",
            UserWarning,
        )

        quantiles = []
        points = []

        for item in copied_inputs:
            # Quantiles are intentionally different so the
            # test verifies use of the standard point output.
            quantiles.append(
                np.full(
                    (
                        1,
                        prediction_length,
                        1,
                    ),
                    -999.0,
                )
            )

            points.append(
                np.asarray(
                    [
                        item[-1]
                        + np.arange(
                            1,
                            prediction_length + 1,
                        )
                    ]
                )
            )

        return quantiles, points


data = pd.DataFrame(
    {
        "y": np.arange(
            1.0,
            11.0,
        )
    },
    index=pd.date_range(
        "2020-01-01",
        periods=10,
        freq="D",
    ),
)

origins = generate_forecast_origins(
    data=data,
    initial_train_size=6,
    max_horizon=2,
    step=1,
)

pipeline = FakeChronos2Pipeline()

adapter = Chronos2ForecastAdapter(
    pipeline=pipeline,
    device="cpu",
)

adapter.prepare_origins(
    series=data["y"],
    origins=origins,
    horizon=2,
    batch_size=2,
)


# ---------------------------------------------------------
# Efficient, leakage-free batching
# ---------------------------------------------------------

assert len(pipeline.calls) == 2

all_inputs = [
    item
    for call in pipeline.calls
    for item in call["inputs"]
]

assert len(all_inputs) == len(origins)

for call in pipeline.calls:
    assert call[
        "prediction_length"
    ] == 2
    assert call[
        "quantile_levels"
    ] == [0.5]
    assert call[
        "context_length"
    ] == 4
    assert not call[
        "cross_learning"
    ]

for origin, supplied in zip(
    origins,
    all_inputs,
):
    expected = data[
        "y"
    ].iloc[
        :origin
    ].tail(
        4
    ).to_numpy(
        dtype=np.float32
    )

    assert np.array_equal(
        supplied,
        expected,
    )


# ---------------------------------------------------------
# Existing walk-forward evaluation integration
# ---------------------------------------------------------

results = walk_forward_evaluate(
    data=data,
    target_col="y",
    model_func=adapter,
    model_name="Chronos-2",
    dataset_name="Mock",
    horizons=[1, 2],
    initial_train_size=6,
    step=1,
    mase_seasonality=1,
)

assert results.groupby(
    "horizon"
).size().eq(
    len(origins)
).all()

assert np.allclose(
    results["forecast"],
    results["actual"],
)

assert results[
    "context_length"
].eq(
    4
).all()

assert results[
    "context_truncated"
].astype(bool).all()

assert results[
    "model_identifier"
].eq(
    "amazon/chronos-2"
).all()

assert results[
    "model_revision"
].eq(
    "29ec3766d36d6f73f0696f85560a422f50e8498c"
).all()

assert results[
    "point_forecast_method"
].eq(
    "standard predictions output "
    "(trained 0.5 quantile)"
).all()

assert results[
    "inference_warning"
].str.contains(
    "mock inference warning"
).all()

assert results[
    "inference_exception"
].eq(
    ""
).all()

assert results[
    "forecast_is_finite"
].astype(bool).all()

audit_chronos_results(
    results=results,
    data=data,
    mase_seasonality=1,
    expected_origins=len(origins),
    max_context_length=4,
)

print(
    "Chronos adapter assertions passed."
)
