import numpy as np

from statsmodels.tsa.arima.model import ARIMA

# ---------------------------------------------------------
# ARIMA forecast
# ---------------------------------------------------------

def arima_log_price_forecast(
    train,
    horizon=1,
    order=(0, 1, 2),
    drift=True,
):
    """
    Forecast price levels using an ARIMA model
    for log prices.

    ARIMA(0,1,q) with drift is estimated through its
    equivalent stationary representation:

        MA(q) with a constant on log returns.

    Forecast log returns are accumulated to obtain
    forecast log prices and then transformed back
    to price levels.

    Parameters
    ----------
    train : pandas.Series
        Historical price levels.

    horizon : int, default=1
        Number of periods to forecast.

    order : tuple, default=(0, 1, 2)
        ARIMA specification for log prices.

        This implementation currently supports
        ARIMA(0,1,q) models.

    drift : bool, default=True
        If True, include a constant in the log-return
        model. This corresponds to drift in the
        integrated log-price model.

    Returns
    -------
    dict
        Dictionary containing:

        forecast : numpy.ndarray
            Forecast price levels.

        converged : bool
            Whether maximum-likelihood optimization
            satisfied the convergence criterion.

        iterations : int or float
            Number of optimizer iterations, when
            reported by statsmodels.
    """

    # -----------------------------------------------------
    # Basic validation
    # -----------------------------------------------------

    if len(train) < 2:
        raise ValueError(
            "Training series must contain at least "
            "two observations."
        )

    if (train <= 0).any():
        raise ValueError(
            "Log-price ARIMA requires strictly "
            "positive prices."
        )

    p, d, q = order

    if p != 0 or d != 1:
        raise ValueError(
            "This implementation currently supports "
            "ARIMA(0,1,q) specifications only."
        )

    # -----------------------------------------------------
    # Convert price levels to log returns
    # -----------------------------------------------------

    price_values = np.asarray(
        train,
        dtype=float,
    )

    log_prices = np.log(
        price_values
    )

    log_returns = np.diff(
        log_prices
    )

    # -----------------------------------------------------
    # Estimate stationary MA(q) representation
    # -----------------------------------------------------

    # A constant in the log-return model corresponds
    # to drift in the integrated log-price model.
    trend = "c" if drift else "n"

    model = ARIMA(
        log_returns,
        order=(0, 0, q),
        trend=trend,
    )

    initial_fitted = model.fit(
        method_kwargs={
            "method": "lbfgs",
            "maxiter": 1000,
            "disp": 0,
        }
    )

    def get_mle_retval(
        results,
        *names,
    ):
        """Read optimizer diagnostics across result variants."""

        mle_retvals = getattr(
            results,
            "mle_retvals",
            None,
        )

        for name in names:
            if hasattr(mle_retvals, "get"):
                value = mle_retvals.get(
                    name,
                    None,
                )
            else:
                value = getattr(
                    mle_retvals,
                    name,
                    None,
                )

            if value is not None:
                return value

        return np.nan

    initial_converged = get_mle_retval(
        initial_fitted,
        "converged",
    )

    initial_iterations = get_mle_retval(
        initial_fitted,
        "iterations",
        "nit",
    )

    initial_warnflag = get_mle_retval(
        initial_fitted,
        "warnflag",
        "status",
    )

    initial_objective_value = get_mle_retval(
        initial_fitted,
        "fopt",
        "fun",
        "objective",
    )

    initial_optimizer_message = get_mle_retval(
        initial_fitted,
        "message",
        "task",
    )

    initial_optimizer_status = get_mle_retval(
        initial_fitted,
        "status",
    )

    initial_parameters = {
        name: float(value)
        for name, value in zip(
            initial_fitted.param_names,
            initial_fitted.params,
        )
    }

    # A small deterministic drift perturbation avoids the
    # reproducible L-BFGS abnormal terminations observed in
    # the SPY walk-forward fits. A retry is selected only if
    # it converges at a non-worse objective value.
    fitted = initial_fitted
    retry_attempted = False
    retry_accepted = False
    retry_count = 0
    retry_drift_offset = np.nan
    retry_converged = np.nan
    retry_iterations = np.nan
    retry_warnflag = np.nan
    retry_objective_value = np.nan
    retry_optimizer_message = np.nan
    retry_optimizer_status = np.nan
    retry_history = []

    if (
        not bool(initial_converged)
        and drift
    ):
        retry_attempted = True

        drift_index = (
            initial_fitted.param_names.index(
                "const"
            )
        )

        for drift_offset in [
            1e-6,
            0.5e-6,
        ]:
            retry_count += 1
            retry_drift_offset = drift_offset

            retry_start_params = np.asarray(
                initial_fitted.params,
                dtype=float,
            ).copy()

            retry_start_params[
                drift_index
            ] += drift_offset

            retry_fitted = model.fit(
                start_params=retry_start_params,
                method_kwargs={
                    "method": "lbfgs",
                    "maxiter": 1000,
                    "disp": 0,
                }
            )

            retry_converged = get_mle_retval(
                retry_fitted,
                "converged",
            )

            retry_iterations = get_mle_retval(
                retry_fitted,
                "iterations",
                "nit",
            )

            retry_warnflag = get_mle_retval(
                retry_fitted,
                "warnflag",
                "status",
            )

            retry_objective_value = get_mle_retval(
                retry_fitted,
                "fopt",
                "fun",
                "objective",
            )

            retry_optimizer_message = get_mle_retval(
                retry_fitted,
                "message",
                "task",
            )

            retry_optimizer_status = get_mle_retval(
                retry_fitted,
                "status",
            )

            retry_history.append(
                {
                    "attempt": retry_count,
                    "drift_offset": drift_offset,
                    "converged": bool(
                        retry_converged
                    ),
                    "iterations": float(
                        retry_iterations
                    ),
                    "warnflag": float(
                        retry_warnflag
                    ),
                    "objective_value": float(
                        retry_objective_value
                    ),
                }
            )

            if (
                bool(retry_converged)
                and np.isfinite(
                    initial_objective_value
                )
                and np.isfinite(
                    retry_objective_value
                )
                and retry_objective_value
                <= initial_objective_value
            ):
                fitted = retry_fitted
                retry_accepted = True
                break

    # -----------------------------------------------------
    # Forecast future log returns
    # -----------------------------------------------------

    forecast_returns = np.asarray(
        fitted.forecast(
            steps=horizon
        ),
        dtype=float,
    )

    # -----------------------------------------------------
    # Reconstruct future log prices
    # -----------------------------------------------------

    last_log_price = log_prices[-1]

    forecast_log_prices = (
        last_log_price
        + np.cumsum(
            forecast_returns
        )
    )

    # -----------------------------------------------------
    # Convert back to price levels
    # -----------------------------------------------------

    price_forecast = np.exp(
        forecast_log_prices
    )

    # -----------------------------------------------------
    # Estimation diagnostics
    # -----------------------------------------------------

    converged = get_mle_retval(
        fitted,
        "converged",
    )

    iterations = get_mle_retval(
        fitted,
        "iterations",
        "nit",
    )

    warnflag = get_mle_retval(
        fitted,
        "warnflag",
        "status",
    )

    objective_value = get_mle_retval(
        fitted,
        "fopt",
        "fun",
        "objective",
    )

    optimizer_message = get_mle_retval(
        fitted,
        "message",
        "task",
    )

    optimizer_status = get_mle_retval(
        fitted,
        "status",
    )

    parameters = {
        name: float(value)
        for name, value in zip(
            fitted.param_names,
            fitted.params,
        )
    }

    # -----------------------------------------------------
    # Return forecasts and diagnostics
    # -----------------------------------------------------

    return {
        "forecast": price_forecast,
        "converged": converged,
        "iterations": iterations,
        "warnflag": warnflag,
        "objective_value": objective_value,
        "optimizer_message": optimizer_message,
        "optimizer_status": optimizer_status,
        "initial_converged": initial_converged,
        "initial_iterations": initial_iterations,
        "initial_warnflag": initial_warnflag,
        "initial_objective_value": (
            initial_objective_value
        ),
        "initial_optimizer_message": (
            initial_optimizer_message
        ),
        "initial_optimizer_status": (
            initial_optimizer_status
        ),
        "initial_parameters": initial_parameters,
        "retry_attempted": retry_attempted,
        "retry_accepted": retry_accepted,
        "retry_count": retry_count,
        "retry_drift_offset": retry_drift_offset,
        "retry_converged": retry_converged,
        "retry_iterations": retry_iterations,
        "retry_warnflag": retry_warnflag,
        "retry_objective_value": (
            retry_objective_value
        ),
        "retry_optimizer_message": (
            retry_optimizer_message
        ),
        "retry_optimizer_status": (
            retry_optimizer_status
        ),
        "retry_history": retry_history,
        "forecast_is_finite": bool(
            np.all(
                np.isfinite(price_forecast)
            )
        ),
        "parameters_are_finite": bool(
            np.all(
                np.isfinite(
                    list(parameters.values())
                )
            )
        ),
        "parameters": parameters,
    }

# ---------------------------------------------------------
# Naive forecast
# ---------------------------------------------------------

def naive_forecast(
    train,
    horizon=1,
):
    """
    Forecast all future periods using the most recent
    observed value.

    Parameters
    ----------
    train : pandas.Series or array-like
        Training series.

    horizon : int, default=1
        Number of periods to forecast.

    Returns
    -------
    numpy.ndarray
        Forecast values.
    """

    if len(train) == 0:
        raise ValueError(
            "Training series cannot be empty."
        )

    last_value = train.iloc[-1]

    return np.repeat(
        last_value,
        horizon,
    )


# ---------------------------------------------------------
# Seasonal naive forecast
# ---------------------------------------------------------

def seasonal_naive_forecast(
    train,
    horizon=1,
    season_length=12,
):
    """
    Forecast using observations from the most recent
    seasonal cycle.

    Parameters
    ----------
    train : pandas.Series
        Training series.

    horizon : int, default=1
        Number of periods to forecast.

    season_length : int, default=12
        Number of observations in one seasonal cycle.

        Examples
        --------
        12  : monthly data with annual seasonality
        24  : hourly data with daily seasonality
        168 : hourly data with weekly seasonality

    Returns
    -------
    numpy.ndarray
        Forecast values.
    """

    if len(train) < season_length:
        raise ValueError(
            "Training series must contain at least "
            "one full seasonal cycle."
        )

    seasonal_values = (
        train.iloc[-season_length:]
        .to_numpy()
    )

    forecast = np.array(
        [
            seasonal_values[
                i % season_length
            ]
            for i in range(horizon)
        ]
    )

    return forecast
