import os

import requests
from dotenv import load_dotenv


load_dotenv()

api_key = os.getenv("EIA_API_KEY")

url = (
    "https://api.eia.gov/v2/"
    "electricity/rto/region-data/data/"
)

params = {
    "api_key": api_key,
    "frequency": "hourly",
    "data[0]": "value",
    "facets[respondent][]": "CISO",
    "facets[type][]": "D",
    "start": "2025-01-01",
    "end": "2025-01-03",
    "length": 100,
}

print("API key loaded:", api_key is not None)
print("Sending request...")

response = requests.get(
    url,
    params=params,
    timeout=30,
)

print("Status:", response.status_code)
print(response.url)
print(response.text[:1000])