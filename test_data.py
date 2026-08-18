from src.data import (
    load_spy,
    load_airline,
    load_electricity,
)


# ---------------------------------------------------------
# SPY
# ---------------------------------------------------------

print("=" * 60)
print("SPY")
print("=" * 60)

spy = load_spy()

print(spy.head())
print()
print(spy.tail())
print()
print("Shape:", spy.shape)
print("Dtypes:")
print(spy.dtypes)
print("Start:", spy.index.min())
print("End:", spy.index.max())


# ---------------------------------------------------------
# Airline passengers
# ---------------------------------------------------------

print()
print("=" * 60)
print("AIRLINE PASSENGERS")
print("=" * 60)

airline = load_airline()

print(airline.head())
print()
print(airline.tail())
print()
print("Shape:", airline.shape)
print("Dtypes:")
print(airline.dtypes)
print("Start:", airline.index.min())
print("End:", airline.index.max())


# ---------------------------------------------------------
# Electricity demand
# ---------------------------------------------------------

print()
print("=" * 60)
print("ELECTRICITY DEMAND")
print("=" * 60)

electricity = load_electricity(
    balancing_authority="CISO",
    start_year=2019,
)

print(electricity.head())
print()
print(electricity.tail())
print()
print("Shape:", electricity.shape)
print("Dtypes:")
print(electricity.dtypes)
print("Start:", electricity.index.min())
print("End:", electricity.index.max())