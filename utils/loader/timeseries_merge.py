"""utils.loader.timeseries_merge

Robust helpers to normalize and merge time-series CSV files.

Design goals
------------
- Accept heterogeneous exports (TradingView/Yahoo/other) with minimal assumptions.
- Keep data in *epoch seconds* under a canonical `time` column.
- Merge incrementally (append + deduplicate) without losing extra columns.
- Be safe around common pitfalls: ms timestamps, strings, missing columns.

This module is intentionally dependency-light (pandas only) and is used by
`bin/data/sync_trading_view_to_market.py` and `bin/data/refresh_all_data.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


@dataclass(frozen=True)
class TimeSeriesMergeStats:
    rows_existing: int
    rows_incoming: int
    rows_merged: int
    rows_added_net: int
    n_dupes_dropped: int
    time_min: Optional[int]
    time_max: Optional[int]


def _ensure_time_seconds(time_like: pd.Series) -> pd.Series:
    """Coerce an epoch-like series to *seconds* (int64).

    Heuristic:
    - If values look like milliseconds (abs median >= 1e11), divide by 1000.
    - Accept negative epochs (pre-1970 historical series).
    """
    s = pd.to_numeric(time_like, errors="coerce")
    s = s.dropna()
    if s.empty:
        return pd.Series(dtype="int64")

    # Use median magnitude as robust estimator.
    median_abs = float(s.abs().median())
    scale = 1000.0 if median_abs >= 1e11 else 1.0
    out = pd.to_numeric(time_like, errors="coerce")
    out = (out / scale).round().astype("Int64")
    return out


def coerce_time_column(df: pd.DataFrame, *, time_col: str = "time") -> pd.DataFrame:
    """Return a copy with a clean integer `time` column (epoch seconds).

    If `time` is missing, attempts to parse a datetime column and build it.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=[time_col])

    out = df.copy()

    if time_col not in out.columns:
        # Try common datetime columns, else first column.
        candidates: Iterable[str] = (
            "datetime",
            "Datetime",
            "date",
            "Date",
            "timestamp",
            "Timestamp",
        )
        c = next((x for x in candidates if x in out.columns), None)
        if c is None:
            c = out.columns[0]
        dt = pd.to_datetime(out[c], errors="coerce", utc=True)
        out[time_col] = (dt.astype("int64") // 10**9).astype("Int64")

    out[time_col] = _ensure_time_seconds(out[time_col])
    out = out.dropna(subset=[time_col])
    out[time_col] = out[time_col].astype("int64")
    return out


def add_datetime_column(df: pd.DataFrame, *, time_col: str = "time", datetime_col: str = "datetime") -> pd.DataFrame:
    """Add a UTC ISO-8601 datetime column derived from epoch seconds."""
    if df is None or len(df) == 0:
        return df
    if time_col not in df.columns:
        return df
    out = df.copy()
    dt = pd.to_datetime(out[time_col], unit="s", errors="coerce", utc=True)
    # Store as ISO string to keep CSV stable across locales.
    iso = dt.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    if datetime_col not in out.columns:
        out[datetime_col] = iso
    else:
        mask = out[datetime_col].isna() | (out[datetime_col].astype(str).str.len() == 0)
        out.loc[mask, datetime_col] = iso.loc[mask]
    return out


def merge_timeseries_frames(
    existing: pd.DataFrame,
    incoming: pd.DataFrame,
    *,
    time_col: str = "time",
    prefer: str = "incoming",
) -> pd.DataFrame:
    """Merge two frames on `time` keeping the union of columns.

    Duplicate policy:
    - `prefer="incoming"`: keep the incoming row for duplicate timestamps.
    - `prefer="existing"`: keep the existing row for duplicate timestamps.
    """
    existing = pd.DataFrame() if existing is None else existing
    incoming = pd.DataFrame() if incoming is None else incoming

    if len(existing) == 0:
        merged = incoming.copy()
        merged = coerce_time_column(merged, time_col=time_col)
        return merged.sort_values(time_col).drop_duplicates(time_col, keep="last").reset_index(drop=True)
    if len(incoming) == 0:
        merged = existing.copy()
        merged = coerce_time_column(merged, time_col=time_col)
        return merged.sort_values(time_col).drop_duplicates(time_col, keep="last").reset_index(drop=True)

    a = coerce_time_column(existing, time_col=time_col)
    b = coerce_time_column(incoming, time_col=time_col)

    # Deduplicate within each frame first.
    a = a.sort_values(time_col).drop_duplicates(time_col, keep="last")
    b = b.sort_values(time_col).drop_duplicates(time_col, keep="last")

    # Union of columns, stable order: existing cols then new cols.
    cols = list(a.columns)
    for c in b.columns:
        if c not in cols:
            cols.append(c)

    a = a.reindex(columns=cols).set_index(time_col)
    b = b.reindex(columns=cols).set_index(time_col)

    if prefer not in {"incoming", "existing"}:
        raise ValueError("prefer must be 'incoming' or 'existing'")

    # Overlay strategy (robust): prefer side supplies values when non-null,
    # while the other side fills missing columns/values.
    if prefer == "incoming":
        merged = b.combine_first(a)
    else:
        merged = a.combine_first(b)

    merged = merged.reset_index().sort_values(time_col).reset_index(drop=True)
    return merged


def merge_timeseries_csv(
    dest_path: str | Path,
    incoming: pd.DataFrame,
    *,
    time_col: str = "time",
    prefer: str = "incoming",
    add_datetime: bool = False,
) -> tuple[pd.DataFrame, TimeSeriesMergeStats]:
    """Merge `incoming` into an on-disk CSV at `dest_path`.

    Returns (merged_df, stats). The caller decides whether/how to write.
    """
    dest = Path(dest_path)
    rows_existing = 0
    existing = pd.DataFrame()

    if dest.exists():
        try:
            existing = pd.read_csv(dest)
            rows_existing = len(existing)
        except Exception:
            # Corrupted / partial file: treat as empty and allow overwrite.
            existing = pd.DataFrame()
            rows_existing = 0

    rows_incoming = 0 if incoming is None else int(len(incoming))
    merged = merge_timeseries_frames(existing, incoming, time_col=time_col, prefer=prefer)
    if add_datetime:
        merged = add_datetime_column(merged, time_col=time_col)

    rows_merged = int(len(merged))
    rows_added_net = rows_merged - rows_existing

    # Count duplicates dropped compared to naive concat.
    naive = rows_existing + rows_incoming
    n_dupes_dropped = max(0, naive - rows_merged)

    tmin = int(merged[time_col].min()) if rows_merged and time_col in merged.columns else None
    tmax = int(merged[time_col].max()) if rows_merged and time_col in merged.columns else None

    stats = TimeSeriesMergeStats(
        rows_existing=rows_existing,
        rows_incoming=rows_incoming,
        rows_merged=rows_merged,
        rows_added_net=rows_added_net,
        n_dupes_dropped=n_dupes_dropped,
        time_min=tmin,
        time_max=tmax,
    )
    return merged, stats
