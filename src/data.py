from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def _load_csv(filename):
    """
    Load a time-series CSV from the raw data directory.

    The file must contain:
    - a date column named 'date'
    - a target column named 'y'
    """

    filepath = RAW_DATA_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"Data file not found: {filepath}\n"
            "Run src/dataimport.py to obtain the raw data."
        )

    data = pd.read_csv(
        filepath,
        parse_dates=["date"],
        index_col="date",
    )

    data.index.name = "date"

    return data


# ---------------------------------------------------------
# SPY
# ---------------------------------------------------------

def load_spy():
    """
    Load the locally stored SPY dataset.
    """

    return _load_csv("spy.csv")


# ---------------------------------------------------------
# Airline passengers
# ---------------------------------------------------------

def load_airline():
    """
    Load the locally stored AirPassengers dataset.
    """

    return _load_csv("airline.csv")


# ---------------------------------------------------------
# Electricity demand
# ---------------------------------------------------------

def load_electricity(
    balancing_authority="CISO",
):
    """
    Load locally stored hourly electricity demand.
    """

    filename = (
        f"electricity_{balancing_authority.lower()}.csv"
    )

    return _load_csv(filename)