from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd


REQUIRED_COLUMNS = ("id", "job_title", "location", "connection")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def parse_connections(value: object) -> int | None:
    """
    Parse the project's `connection` field into an integer.

    Examples:
    - "500+" -> 500
    - "250" -> 250
    - 100 -> 100
    - None/"" -> None
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    s = str(value).strip()
    if not s:
        return None

    # common format: "500+"
    m = re.fullmatch(r"(\d+)\+", s)
    if m:
        return int(m.group(1))

    # plain integer in string form
    if s.isdigit():
        return int(s)

    # fallback: extract first integer token
    m = re.search(r"\d+", s)
    if m:
        return int(m.group(0))
    return None


@dataclass(frozen=True)
class IngestResult:
    df_raw: pd.DataFrame
    df: pd.DataFrame


def load_candidates(path: str | Path, *, source: Literal["xlsx", "csv"] | None = None) -> IngestResult:
    path = Path(path)
    if source is None:
        source = "xlsx" if path.suffix.lower() in {".xlsx", ".xls"} else "csv"

    if source == "xlsx":
        df_raw = pd.read_excel(path)
    elif source == "csv":
        df_raw = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported source={source!r}")

    df_raw = _normalize_columns(df_raw)

    missing = [c for c in REQUIRED_COLUMNS if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df_raw.columns)}")

    df = df_raw.copy()
    df["id"] = pd.to_numeric(df["id"], errors="raise").astype("int64")
    df["job_title"] = df["job_title"].astype(str).str.strip()
    df["location"] = df["location"].astype(str).str.strip()
    df["connection_raw"] = df["connection"]
    df["connections"] = df["connection"].map(parse_connections).astype("Int64")

    # Optional label column; keep if present.
    if "fit" in df.columns:
        df["fit"] = pd.to_numeric(df["fit"], errors="coerce")

    return IngestResult(df_raw=df_raw, df=df)

