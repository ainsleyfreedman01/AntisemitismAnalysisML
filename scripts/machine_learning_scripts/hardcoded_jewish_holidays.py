"""Hard-coded Jewish holiday dates for 2023 and 2024.

This module provides a small helper to return canonical Jewish holiday dates
based on the ranges supplied by the user. It returns a list of dictionaries
with `date` (a datetime.date) and `name` fields and also optionally writes
out a CSV to `outputs/canonical_jewish_holidays.csv` when requested.
"""
from __future__ import annotations
import datetime
from typing import List, Dict
import os
import csv


def _daterange(start_date: datetime.date, end_date: datetime.date):
    cur = start_date
    while cur <= end_date:
        yield cur
        cur = cur + datetime.timedelta(days=1)


def get_hardcoded_jewish_holidays(write_csv: bool = True) -> List[Dict]:
    """Return a list of holiday dicts {'date': date, 'name': str}.

    The dates are taken from the project owner's supplied list for 2023 and 2024.
    """
    entries = [
        # 2023
        ("2023-02-05", "2023-02-05", "Tu BiShvat"),
        ("2023-03-06", "2023-03-07", "Purim"),
        ("2023-04-05", "2023-04-13", "Passover (Pesach)"),
        ("2023-04-17", "2023-04-18", "Yom HaShoah"),
        ("2023-04-24", "2023-04-25", "Yom HaZikaron"),
        ("2023-04-25", "2023-04-26", "Yom HaAtzmaut"),
        ("2023-05-08", "2023-05-09", "Lag BaOmer"),
        ("2023-05-25", "2023-05-27", "Shavuot"),
        ("2023-07-26", "2023-07-27", "Tisha B'Av"),
        ("2023-09-15", "2023-09-17", "Rosh Hashanah 5784"),
        ("2023-09-24", "2023-09-25", "Yom Kippur"),
        ("2023-09-29", "2023-10-06", "Sukkot"),
        ("2023-10-06", "2023-10-08", "Shemini Atzeret / Simchat Torah"),
        ("2023-12-07", "2023-12-15", "Hanukkah"),
        # 2024
        ("2024-01-24", "2024-01-25", "Tu BiShvat"),
        ("2024-03-23", "2024-03-24", "Purim"),
        ("2024-04-22", "2024-04-30", "Passover (Pesach)"),
        ("2024-05-05", "2024-05-06", "Yom HaShoah"),
        ("2024-05-12", "2024-05-13", "Yom HaZikaron"),
        ("2024-05-13", "2024-05-14", "Yom HaAtzmaut"),
        ("2024-05-25", "2024-05-26", "Lag BaOmer"),
        ("2024-06-11", "2024-06-13", "Shavuot"),
        ("2024-08-12", "2024-08-13", "Tisha B'Av"),
        ("2024-10-02", "2024-10-04", "Rosh Hashanah 5785"),
        ("2024-10-11", "2024-10-12", "Yom Kippur"),
        ("2024-10-16", "2024-10-23", "Sukkot"),
        ("2024-10-23", "2024-10-25", "Shemini Atzeret / Simchat Torah"),
        ("2024-12-25", "2025-01-02", "Hanukkah (2024-25)")
    ]

    holidays = []
    for start_s, end_s, name in entries:
        start = datetime.datetime.strptime(start_s, "%Y-%m-%d").date()
        end = datetime.datetime.strptime(end_s, "%Y-%m-%d").date()
        for d in _daterange(start, end):
            holidays.append({"date": d, "name": name})

    if write_csv:
        try:
            os.makedirs("outputs", exist_ok=True)
            out_path = os.path.join("outputs", "canonical_jewish_holidays.csv")
            with open(out_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["date", "holiday_name"])
                for h in holidays:
                    writer.writerow([h["date"].isoformat(), h["name"]])
        except Exception:
            # don't fail if writing the CSV is not possible in the current environment
            pass

    return holidays


if __name__ == "__main__":
    hs = get_hardcoded_jewish_holidays()
    print(f"Generated {len(hs)} holiday dates; sample:", hs[:5])
