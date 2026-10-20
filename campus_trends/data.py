from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = (
    "Date of Incident",
    "City of Incident",
    "State of Incident",
    "Incident Type",
    "Description",
    "College/University",
)
INCIDENT_TYPES = (
    "Protest/Action",
    "Harassment",
    "Vandalism",
    "BDS",
    "Physical Assault",
)


def load_reports(path: Path) -> pd.DataFrame:
    """Reject malformed inputs rather than silently changing the population."""
    raw = pd.read_csv(path, dtype=str)
    missing = set(REQUIRED_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    raw = raw.loc[:, list(REQUIRED_COLUMNS)].copy()
    for column in REQUIRED_COLUMNS:
        raw[column] = raw[column].str.strip()
        if raw[column].isna().any() or raw[column].eq("").any():
            raise ValueError(f"Missing or blank values in {column!r}")
    unknown = set(raw["Incident Type"]) - set(INCIDENT_TYPES)
    if unknown:
        raise ValueError(f"Unrecognized incident types: {sorted(unknown)}")
    raw["Date"] = pd.to_datetime(
        raw["Date of Incident"], format="%m/%d/%y", errors="raise"
    )
    if raw.empty:
        raise ValueError("The input contains no reports.")
    if raw.duplicated(subset=list(REQUIRED_COLUMNS)).any():
        raise ValueError("Exact duplicate reports found; review before analysis.")
    return raw.sort_values("Date", kind="stable").reset_index(drop=True)
