"""
VVIX-Based Calibration
======================
Estimates vol-of-vol parameter η from CBOE VVIX index data.
VVIX measures the expected volatility of the VIX index (30-day implied).

Theoretical link (rBergomi):
    Var[log VIX_{T}] ≈ η² · T^{2H}   for small T
    VVIX ≈ 100 · η · T^H             (annualized, T=1 ⟹ VVIX ≈ 100·η)

Usage:
    from quant.calibration.vvix_calibrator import VVIXCalibrator
    cal = VVIXCalibrator()
    result = cal.estimate_eta(H=0.1)
    print(result['eta_recommended'])
"""

import pandas as pd
import numpy as np
from pathlib import Path


_MARKET_PATH = "data/market/volatility/vvix_daily.csv"


class VVIXCalibrator:
    """Calibrate η (vol-of-vol) and detect vol-of-vol regimes from VVIX."""

    def __init__(self, filepath: str = None):
        if filepath:
            self._path = Path(filepath)
        elif Path(_MARKET_PATH).exists():
            self._path = Path(_MARKET_PATH)
        self._df: pd.DataFrame = pd.DataFrame()
        self._load()

    def _load(self):
        if not self._path.exists():
            return
        raw = pd.read_csv(self._path)
        cols_lower = [c.lower() for c in raw.columns]

        # --- Auto-detect column format (TradingView vs Yahoo) -----------
        if "time" in cols_lower and "close" in cols_lower:
            # TradingView format: time (unix), close (lowercase)
            tc = raw.columns[cols_lower.index("time")]
            cc = raw.columns[cols_lower.index("close")]
            hc = raw.columns[cols_lower.index("high")] if "high" in cols_lower else None
            lc = raw.columns[cols_lower.index("low")] if "low" in cols_lower else None
            raw["date"] = pd.to_datetime(
                pd.to_numeric(raw[tc], errors="coerce"), unit="s", utc=True
            )
            raw["vvix"] = pd.to_numeric(raw[cc], errors="coerce")
            raw["high_val"] = pd.to_numeric(raw[hc], errors="coerce") if hc else np.nan
            raw["low_val"] = pd.to_numeric(raw[lc], errors="coerce") if lc else np.nan
        elif "Date" in raw.columns and "Close" in raw.columns:
            # Yahoo format: Date (ISO), Close (capitalised)
            raw["date"] = pd.to_datetime(raw["Date"])
            raw["vvix"] = pd.to_numeric(raw["Close"], errors="coerce")
            raw["high_val"] = pd.to_numeric(raw.get("High"), errors="coerce")
            raw["low_val"] = pd.to_numeric(raw.get("Low"), errors="coerce")
        else:
            return  # unrecognised format

        self._df = (
            raw[["date", "vvix", "high_val", "low_val"]]
            .rename(columns={"high_val": "high", "low_val": "low"})
            .dropna(subset=["vvix"])
            .sort_values("date")
            .set_index("date")
        )

    @property
    def is_available(self) -> bool:
        return not self._df.empty

    # ------------------------------------------------------------------
    #  η estimation
    # ------------------------------------------------------------------
    def estimate_eta(self, H: float = 0.1, window_days: int = 252) -> dict:
        """
        Estimate rBergomi η from VVIX.

        For annualized VVIX and T=1 year:
            VVIX/100 ≈ η · 1^H = η     (when H is small, correction is modest)

        We also apply Gatheral-style correction:
            η ≈ (VVIX/100) · T^{-H}    with T = 30/365 (VIX looks at 30-day window)
        """
        if self._df.empty:
            return self._fallback_eta()

        recent = self._df.tail(window_days)
        vvix_mean = float(recent["vvix"].mean())
        vvix_current = float(recent["vvix"].iloc[-1])
        vvix_std = float(recent["vvix"].std())

        T_vix = 30.0 / 365.0
        correction = T_vix ** (-H)

        eta_raw_mean = vvix_mean / 100.0
        eta_raw_current = vvix_current / 100.0

        eta_corrected_mean = np.clip(eta_raw_mean * correction, 0.3, 6.0)
        eta_corrected_current = np.clip(eta_raw_current * correction, 0.3, 6.0)

        eta_recommended = float(np.clip(
            0.7 * eta_corrected_mean + 0.3 * eta_corrected_current, 0.5, 5.0
        ))

        return {
            "eta_recommended": eta_recommended,
            "eta_from_mean": float(eta_corrected_mean),
            "eta_from_current": float(eta_corrected_current),
            "vvix_mean": vvix_mean,
            "vvix_current": vvix_current,
            "vvix_std": vvix_std,
            "H_used": float(H),
            "T_correction": float(correction),
            "window_days": window_days,
            "source": "vvix_calibration",
            "date_range": (
                str(recent.index[0].date()),
                str(recent.index[-1].date()),
            ),
        }

    # ------------------------------------------------------------------
    #  Regime classification
    # ------------------------------------------------------------------
    def get_regime(self, as_of_date=None) -> dict:
        """
        Classify vol-of-vol regime from VVIX level.

        Historical VVIX percentiles (2013-2026):
            10th ≈ 75,  25th ≈ 82,  50th ≈ 92,
            75th ≈ 107,  90th ≈ 125,  99th ≈ 170
        """
        if self._df.empty:
            return {"regime": "unknown", "vvix": None, "percentile": None}

        if as_of_date is not None:
            d = pd.to_datetime(as_of_date)
            available = self._df[self._df.index <= d]
            if available.empty:
                return {"regime": "unknown", "vvix": None, "percentile": None}
            current = float(available["vvix"].iloc[-1])
            history = available["vvix"]
        else:
            current = float(self._df["vvix"].iloc[-1])
            history = self._df["vvix"]

        percentile = float((history <= current).mean() * 100)

        if current < 80:
            regime = "low_volofvol"
        elif current < 100:
            regime = "normal"
        elif current < 120:
            regime = "elevated"
        elif current < 145:
            regime = "high"
        else:
            regime = "panic"

        return {
            "regime": regime,
            "vvix": current,
            "percentile": percentile,
            "thresholds": {
                "low": 80, "normal": 100,
                "elevated": 120, "high": 145,
            },
        }

    def get_historical_regimes(self) -> pd.DataFrame:
        """Return full history with regime labels for analytics."""
        if self._df.empty:
            return pd.DataFrame()

        df = self._df.copy()
        df["regime"] = pd.cut(
            df["vvix"],
            bins=[0, 80, 100, 120, 145, 999],
            labels=["low_volofvol", "normal", "elevated", "high", "panic"],
        )
        return df

    @staticmethod
    def _fallback_eta() -> dict:
        from utils.config import load_config
        cfg = load_config()
        return {
            "eta_recommended": cfg["bergomi"]["eta"],
            "source": "config_fallback",
        }
