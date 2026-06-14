"""Fetch real daily weather history for 8 Australian state capitals via Open-Meteo.

Writes one CSV: data/external/weather_au_capitals.csv with columns
    date, <city>_<variable>...

Wide format (one row per date, all cities × variables as columns) so the
existing `ExternalCsvSource(mode="daily")` picks it up unchanged.

Open-Meteo archive API (ERA5 reanalysis) - free, no key required.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]
OUT_CSV = REPO / "data" / "external" / "weather_au_capitals.csv"

START_DATE = "2021-01-01"
END_DATE = "2025-05-31"

# (city short-code, latitude, longitude)
CAPITALS: list[tuple[str, float, float]] = [
    ("per", -31.9523, 115.8613),   # Perth, WA
    ("adl", -34.9285, 138.6007),   # Adelaide, SA
    ("mel", -37.8136, 144.9631),   # Melbourne, VIC
    ("syd", -33.8688, 151.2093),   # Sydney, NSW
    ("bne", -27.4698, 153.0251),   # Brisbane, QLD
    ("hba", -42.8821, 147.3272),   # Hobart, TAS
    ("drw", -12.4634, 130.8456),   # Darwin, NT
    ("cbr", -35.2809, 149.1300),   # Canberra, ACT
]

DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "apparent_temperature_max",
    "precipitation_sum",
    "windspeed_10m_max",
    "windspeed_10m_mean",
    "shortwave_radiation_sum",
    "relative_humidity_2m_mean",
    "surface_pressure_mean",
]

API = "https://archive-api.open-meteo.com/v1/archive"


def fetch_city(city: str, lat: float, lon: float) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": ",".join(DAILY_VARS),
        "timezone": "Australia/Perth",
    }
    log.info("fetching %s (%.4f, %.4f)…", city, lat, lon)
    resp = requests.get(API, params=params, timeout=120)
    resp.raise_for_status()
    j = resp.json()
    if "daily" not in j:
        raise RuntimeError(f"{city}: response missing 'daily': {j}")
    daily = j["daily"]
    df = pd.DataFrame(daily)
    df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"])
    # prefix non-date columns with the city code
    rename = {c: f"{city}_{c}" for c in df.columns if c != "date"}
    return df.rename(columns=rename)


def main() -> None:
    out_dir = OUT_CSV.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for city, lat, lon in CAPITALS:
        for attempt in range(3):
            try:
                df = fetch_city(city, lat, lon)
                break
            except Exception as e:  # noqa: BLE001
                log.warning("  %s attempt %d failed: %s", city, attempt + 1, e)
                time.sleep(2 ** attempt)
        else:
            log.error("  %s: giving up after 3 attempts", city)
            continue
        log.info("  %s: %d rows, cols=%s", city, len(df), [c for c in df.columns if c != "date"])
        frames.append(df)
        time.sleep(0.5)  # be polite

    if not frames:
        raise RuntimeError("No data fetched.")

    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on="date", how="outer")
    merged = merged.sort_values("date").reset_index(drop=True)

    # Report missingness
    n_total = len(merged)
    miss = merged.isna().sum()
    bad = miss[miss > 0]
    if len(bad):
        log.warning("missing values (rows=%d):\n%s", n_total, bad.to_string())

    merged.to_csv(OUT_CSV, index=False)
    log.info("wrote %s  shape=%s", OUT_CSV, merged.shape)


if __name__ == "__main__":
    main()
