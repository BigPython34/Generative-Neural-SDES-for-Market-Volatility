"""
SOFR Rate Loader
================
Integrates real risk-free rates from NY Fed SOFR data.
Replaces hardcoded r=0.05 throughout the project.

Usage:
    from utils.sofr_loader import SOFRRateLoader
    sofr = SOFRRateLoader()
    r = sofr.get_rate()                         # latest overnight
    r = sofr.get_rate("2025-06-15")             # historical
    r = sofr.get_term_rate(maturity_days=30)     # term-adjusted
"""

import pandas as pd
import numpy as np
from pathlib import Path
from functools import lru_cache


_DEFAULT_PATH = "data/rates/sofr_daily_nyfed.csv"


class SOFRRateLoader:
    """Load and interpolate SOFR risk-free rates from NY Fed data."""

    def __init__(self, filepath: str = None):
        self._path = Path(filepath or _DEFAULT_PATH)
        self._df: pd.DataFrame = pd.DataFrame()
        self._load()

    def _load(self):
        if not self._path.exists():
            return
        raw = pd.read_csv(self._path)
        raw["date"] = pd.to_datetime(raw["Date"])
        raw["rate"] = pd.to_numeric(raw["Close"], errors="coerce") / 100.0
        self._df = (
            raw[["date", "rate"]]
            .dropna()
            .sort_values("date")
            .set_index("date")
        )

    @property
    def is_available(self) -> bool:
        return not self._df.empty

    def get_rate(self, as_of_date=None) -> float:
        """
        Overnight SOFR rate for a given date.
        Falls back to latest available if date not found,
        or to config default if no data at all.
        """
        if self._df.empty:
            return self._fallback_rate()

        if as_of_date is None:
            return float(self._df["rate"].iloc[-1])

        d = pd.to_datetime(as_of_date)
        avail = self._df[self._df.index <= d]
        if avail.empty:
            return float(self._df["rate"].iloc[0])
        return float(avail["rate"].iloc[-1])

    def get_term_rate(self, as_of_date=None, maturity_days: int = 30) -> float:
        """
        Simple term rate estimate via compound SOFR average.
        Uses trailing realized SOFR as proxy for forward term rate
        (consistent with CME Term SOFR methodology).
        """
        if self._df.empty:
            return self._fallback_rate()

        d = pd.to_datetime(as_of_date) if as_of_date else self._df.index[-1]
        window = self._df[self._df.index <= d].tail(maturity_days)

        if len(window) < 5:
            return self.get_rate(as_of_date)

        daily_factors = 1.0 + window["rate"] / 360.0
        compound = float(daily_factors.prod())
        n = len(window)
        annualized = (compound - 1.0) * (360.0 / n)
        return annualized

    def get_discount_factor(self, as_of_date=None, T_years: float = 0.08) -> float:
        """Discount factor exp(-r*T) using SOFR."""
        r = self.get_term_rate(as_of_date, maturity_days=max(1, int(T_years * 365)))
        return float(np.exp(-r * T_years))

    def get_rate_series(self, start_date=None, end_date=None) -> pd.Series:
        """Full SOFR time series for analytics."""
        if self._df.empty:
            return pd.Series(dtype=float)
        s = self._df["rate"]
        if start_date:
            s = s[s.index >= pd.to_datetime(start_date)]
        if end_date:
            s = s[s.index <= pd.to_datetime(end_date)]
        return s

    @staticmethod
    def _fallback_rate() -> float:
        from utils.config import load_config
        return load_config()["pricing"]["risk_free_rate"]


@lru_cache(maxsize=1)
def get_sofr() -> SOFRRateLoader:
    """Singleton accessor for SOFR loader."""
    return SOFRRateLoader()
