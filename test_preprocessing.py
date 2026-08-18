from src.data import (
    load_spy,
    load_airline,
    load_electricity,
)

from src.preprocessing import (
    prepare_spy,
    prepare_airline,
    prepare_electricity,
    summarize_time_series,
)


# ---------------------------------------------------------
# Load local data
# ---------------------------------------------------------

spy = load_spy()
airline = load_airline()
electricity = load_electricity()


# ---------------------------------------------------------
# Preprocess
# ---------------------------------------------------------

spy_prepped = prepare_spy(spy)
airline_prepped = prepare_airline(airline)
electricity_prepped = prepare_electricity(electricity)


# ---------------------------------------------------------
# Inspect results
# ---------------------------------------------------------

print("=" * 60)
print("SPY")
print("=" * 60)
print(spy_prepped.head())
print(spy_prepped.tail())
print()

print("=" * 60)
print("AIRLINE")
print("=" * 60)
print(airline_prepped.head())
print(airline_prepped.tail())
print()

print("=" * 60)
print("ELECTRICITY")
print("=" * 60)
print(electricity_prepped.head())
print(electricity_prepped.tail())
print()


# ---------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------

print("=" * 60)
print("DIAGNOSTICS")
print("=" * 60)

print(
    "SPY:",
    summarize_time_series(spy_prepped)
)

print(
    "Airline:",
    summarize_time_series(
        airline_prepped,
        frequency="MS",
    )
)

print(
    "Electricity:",
    summarize_time_series(
        electricity_prepped,
        frequency="h",
    )
)

from src.preprocessing import get_missing_timestamps


missing_electricity = get_missing_timestamps(
    electricity_prepped,
    frequency="h",
)

print()
print("=" * 60)
print("MISSING ELECTRICITY TIMESTAMPS")
print("=" * 60)

print("Number missing:", len(missing_electricity))
print(missing_electricity)

print(
    "Interpolated electricity observations:",
    electricity_prepped["interpolated"].sum()
)