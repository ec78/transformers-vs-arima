from pathlib import Path
import os

import pandas as pd
import requests
import yfinance as yf

from dotenv import load_dotenv
from statsmodels.datasets import get_rdataset


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def save_metadata(metadata, filename):
    """
    Save dataset metadata as a JSON file.
    """

    output_path = RAW_DATA_DIR / filename

    pd.Series(metadata).to_json(
        output_path,
        indent=2,
    )


# ---------------------------------------------------------
# SPY
# ---------------------------------------------------------

def load_spy(
    start=None,
    end=None,
    save_raw=True,
):
    """
    Download historical SPY adjusted closing prices.

    Parameters
    ----------
    start : str or None
        Starting date in YYYY-MM-DD format.
        If None, use the maximum available history.

    end : str or None
        Ending date in YYYY-MM-DD format.
        If None, download through the latest available date.

    save_raw : bool, default=True
        If True, save the dataset and metadata to data/raw.

    Returns
    -------
    pandas.DataFrame
        DataFrame with:
        - DatetimeIndex named 'date'
        - target column named 'y'
    """

    if start is None and end is None:
        data = yf.download(
            "SPY",
            period="max",
            auto_adjust=True,
            progress=False,
        )
    else:
        data = yf.download(
            "SPY",
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )

    # yfinance may return MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = (
        data[["Close"]]
        .rename(columns={"Close": "y"})
        .dropna()
        .copy()
    )
    
    data.columns.name = None
    data.index = pd.to_datetime(data.index)
    data.index.name = "date"

    if save_raw:
        data.to_csv(
            RAW_DATA_DIR / "spy.csv"
        )

        metadata = {
            "dataset": "SPY",
            "source": "Yahoo Finance",
            "ticker": "SPY",
            "frequency": "daily",
            "download_date": pd.Timestamp.now().isoformat(),
            "n_observations": len(data),
            "start_date": str(data.index.min().date()),
            "end_date": str(data.index.max().date()),
        }

        save_metadata(
            metadata,
            "spy_metadata.json",
        )

    return data


# ---------------------------------------------------------
# Airline passengers
# ---------------------------------------------------------

def load_airline(save_raw=True):
    """
    Load the classic monthly AirPassengers dataset.

    The dataset contains monthly totals of international
    airline passengers from January 1949 through December 1960.

    Parameters
    ----------
    save_raw : bool, default=True
        If True, save the dataset and metadata to data/raw.

    Returns
    -------
    pandas.DataFrame
        DataFrame with:
        - DatetimeIndex named 'date'
        - target column named 'y'
    """

    dataset = get_rdataset(
        "AirPassengers",
        package="datasets",
        cache=True,
    )

    raw = dataset.data.copy()

    dates = pd.date_range(
        start="1949-01-01",
        periods=len(raw),
        freq="MS",
    )

    data = pd.DataFrame(
        {
            "y": raw["value"].to_numpy()
        },
        index=dates,
    )

    data.index.name = "date"

    if save_raw:
        data.to_csv(
            RAW_DATA_DIR / "airline.csv"
        )

        metadata = {
            "dataset": "AirPassengers",
            "source": "R datasets package via statsmodels",
            "frequency": "monthly",
            "download_date": pd.Timestamp.now().isoformat(),
            "n_observations": len(data),
            "start_date": str(data.index.min().date()),
            "end_date": str(data.index.max().date()),
        }

        save_metadata(
            metadata,
            "airline_metadata.json",
        )

    return data


# ---------------------------------------------------------
# Electricity demand
# ---------------------------------------------------------

def load_electricity(
    balancing_authority="CISO",
    start_year=2019,
    end_year=None,
    save_raw=True,
):
    """
    Download hourly electricity demand from the U.S. EIA.

    Data are downloaded one year at a time and paginated
    within each year to avoid overly large API requests.

    Parameters
    ----------
    balancing_authority : str, default="CISO"
        EIA balancing authority code.
        CISO represents California ISO.

    start_year : int, default=2019
        First year to download.

    end_year : int or None
        Last year to download.
        If None, use the current year.

    save_raw : bool, default=True
        If True, save the combined dataset and metadata
        to data/raw.

    Returns
    -------
    pandas.DataFrame
        DataFrame with:
        - DatetimeIndex named 'date'
        - target column named 'y'
    """

    load_dotenv()

    api_key = os.getenv("EIA_API_KEY")

    if api_key is None:
        raise ValueError(
            "EIA_API_KEY was not found. "
            "Add it to the project .env file."
        )

    if end_year is None:
        end_year = pd.Timestamp.now().year

    base_url = (
        "https://api.eia.gov/v2/"
        "electricity/rto/region-data/data/"
    )

    all_rows = []

    # -----------------------------------------------------
    # Download one year at a time
    # -----------------------------------------------------

    for year in range(start_year, end_year + 1):

        print(f"Downloading electricity data for {year}...")

        offset = 0
        page_length = 5000

        while True:

            params = {
                "api_key": api_key,
                "frequency": "hourly",
                "data[0]": "value",
                "facets[respondent][]": balancing_authority,
                "facets[type][]": "D",
                "start": f"{year}-01-01",
                "end": f"{year}-12-31T23",
                "length": page_length,
                "offset": offset,
                "sort[0][column]": "period",
                "sort[0][direction]": "asc",
            }

            response = requests.get(
                base_url,
                params=params,
                timeout=60,
            )

            response.raise_for_status()

            payload = response.json()

            rows = payload["response"]["data"]

            if not rows:
                break

            all_rows.extend(rows)

            if len(rows) < page_length:
                break

            offset += page_length

    # -----------------------------------------------------
    # Convert API response to DataFrame
    # -----------------------------------------------------

    raw = pd.DataFrame(all_rows)

    if raw.empty:
        raise ValueError(
            "No electricity data were returned "
            "for the requested parameters."
        )

    data = (
        raw[["period", "value"]]
        .rename(
            columns={
                "period": "date",
                "value": "y",
            }
        )
        .copy()
    )

    # -----------------------------------------------------
    # Clean variables
    # -----------------------------------------------------

    data["date"] = pd.to_datetime(
        data["date"],
        utc=True,
    )

    data["y"] = pd.to_numeric(
        data["y"],
        errors="coerce",
    )

    data = (
        data
        .dropna()
        .drop_duplicates(subset="date")
        .sort_values("date")
        .set_index("date")
    )

    data.index.name = "date"

    # -----------------------------------------------------
    # Save raw data and metadata
    # -----------------------------------------------------

    if save_raw:

        filename = (
            f"electricity_{balancing_authority.lower()}.csv"
        )

        data.to_csv(
            RAW_DATA_DIR / filename
        )

        metadata = {
            "dataset": "EIA hourly electricity demand",
            "source": "U.S. Energy Information Administration",
            "balancing_authority": balancing_authority,
            "series_type": "demand",
            "frequency": "hourly",
            "start_year": start_year,
            "end_year": end_year,
            "download_date": pd.Timestamp.now().isoformat(),
            "n_observations": len(data),
            "start_date": str(data.index.min()),
            "end_date": str(data.index.max()),
        }

        save_metadata(
            metadata,
            (
                "electricity_"
                f"{balancing_authority.lower()}_metadata.json"
            ),
        )

    return data