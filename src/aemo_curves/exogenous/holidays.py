"""Western Australia public-holiday flags.

STEM is the Western Australian wholesale electricity market, so the relevant
holiday calendar is WA public holidays. This module encodes fixed and
moveable holidays for 2021-2026; dates outside that range return False.

Accuracy: hand-checked against https://www.wa.gov.au/service/employment/
workplace-arrangements/public-holidays-western-australia for 2021-2026.
If you need later years, add them here.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

# Fixed public holidays in WA (month, day).
_FIXED_HOLIDAYS: dict[tuple[int, int], str] = {
    (1, 1): "new_year",
    (1, 26): "australia_day",
    (4, 25): "anzac_day",
    (12, 25): "christmas",
    (12, 26): "boxing_day",
}

# Moveable / observed holidays by year. Source: WA Gov.
# Each entry: date -> name.
_WA_HOLIDAYS_BY_YEAR: dict[int, dict[dt.date, str]] = {
    2021: {
        dt.date(2021, 3, 1):  "labour_day",
        dt.date(2021, 4, 2):  "good_friday",
        dt.date(2021, 4, 5):  "easter_monday",
        dt.date(2021, 6, 7):  "wa_day",
        dt.date(2021, 9, 27): "queens_birthday",
    },
    2022: {
        dt.date(2022, 3, 7):  "labour_day",
        dt.date(2022, 4, 15): "good_friday",
        dt.date(2022, 4, 18): "easter_monday",
        dt.date(2022, 6, 6):  "wa_day",
        dt.date(2022, 9, 26): "queens_birthday",
    },
    2023: {
        dt.date(2023, 3, 6):  "labour_day",
        dt.date(2023, 4, 7):  "good_friday",
        dt.date(2023, 4, 10): "easter_monday",
        dt.date(2023, 6, 5):  "wa_day",
        dt.date(2023, 9, 25): "kings_birthday",
    },
    2024: {
        dt.date(2024, 3, 4):  "labour_day",
        dt.date(2024, 3, 29): "good_friday",
        dt.date(2024, 4, 1):  "easter_monday",
        dt.date(2024, 6, 3):  "wa_day",
        dt.date(2024, 9, 23): "kings_birthday",
    },
    2025: {
        dt.date(2025, 3, 3):  "labour_day",
        dt.date(2025, 4, 18): "good_friday",
        dt.date(2025, 4, 21): "easter_monday",
        dt.date(2025, 6, 2):  "wa_day",
        dt.date(2025, 9, 29): "kings_birthday",
    },
    2026: {
        dt.date(2026, 3, 2):  "labour_day",
        dt.date(2026, 4, 3):  "good_friday",
        dt.date(2026, 4, 6):  "easter_monday",
        dt.date(2026, 6, 1):  "wa_day",
        dt.date(2026, 9, 28): "kings_birthday",
    },
}


def _to_date(d) -> dt.date:
    if isinstance(d, dt.date) and not isinstance(d, dt.datetime):
        return d
    return pd.Timestamp(d).date()


def is_wa_holiday(d) -> bool:
    """True if `d` is a WA public holiday."""
    d = _to_date(d)
    if (d.month, d.day) in _FIXED_HOLIDAYS:
        return True
    return d in _WA_HOLIDAYS_BY_YEAR.get(d.year, {})


def wa_holiday_flags(d) -> np.ndarray:
    """Per-date calendar features.

    Returns `[is_holiday, is_day_before_holiday, is_day_after_holiday,
    is_weekend, is_long_weekend_bridge]` as a 5-element float vector.
    """
    d = _to_date(d)
    next_day = d + dt.timedelta(days=1)
    prev_day = d - dt.timedelta(days=1)
    is_hol = is_wa_holiday(d)
    is_before = is_wa_holiday(next_day)
    is_after = is_wa_holiday(prev_day)
    is_weekend = d.weekday() >= 5
    # Long-weekend bridge: a non-holiday weekday that sits between a weekend
    # and a holiday (or vice-versa) - often behaves like a half-holiday.
    is_bridge = False
    if not is_hol and not is_weekend:
        if d.weekday() == 0 and is_wa_holiday(d + dt.timedelta(days=1)):
            is_bridge = True
        elif d.weekday() == 4 and is_wa_holiday(d - dt.timedelta(days=1)):
            is_bridge = True
    return np.array(
        [float(is_hol), float(is_before), float(is_after), float(is_weekend), float(is_bridge)],
        dtype=float,
    )


def wa_holiday_flag_names() -> list[str]:
    return ["is_holiday", "is_day_before_holiday", "is_day_after_holiday",
            "is_weekend", "is_bridge_day"]
