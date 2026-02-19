"""
Market Regime Detector
======================
Classifies the current volatility regime using multiple signals:
  - VIX level & term structure (contango / backwardation)
  - VVIX level (vol-of-vol regime)
  - Realized-Implied spread (VRP)
  - Hurst exponent dynamics

The detected regime can be used to:
  - Select model parameters (H, η, ρ) adaptively
  - Switch between P-measure and Q-measure models
  - Trigger stress-testing scenarios
  - Adjust hedging frequency

Usage:
    from quant.regime_detector import RegimeDetector
    detector = RegimeDetector()
    regime = detector.detect()
    print(regime['regime'], regime['confidence'])
"""

import numpy as np
import pandas as pd
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from typing import Optional

from utils.config import load_config


class MarketRegime(str, Enum):
    CALM = "calm"
    NORMAL = "normal"
    STRESSED = "stressed"
    CRISIS = "crisis"
    RECOVERY = "recovery"


@dataclass
class RegimeSignal:
    name: str
    value: float
    regime: MarketRegime
    weight: float
    detail: str


class RegimeDetector:
    """Multi-signal regime classifier for volatility markets."""

    def __init__(self):
        self.cfg = load_config()
        self._vix_data: Optional[pd.DataFrame] = None
        self._vvix_data: Optional[pd.DataFrame] = None
        self._futures_data: Optional[pd.DataFrame] = None
        self._spx_data: Optional[pd.DataFrame] = None
        self._load_data()

    def _load_data(self):
        vix_path = Path(self.cfg["data"]["source"])
        if vix_path.exists():
            df = pd.read_csv(vix_path)
            if "time" in df.columns:
                df["datetime"] = pd.to_datetime(df["time"], unit="s")
            elif "Date" in df.columns:
                df["datetime"] = pd.to_datetime(df["Date"])
            df = df.sort_values("datetime").set_index("datetime")
            self._vix_data = df

        vvix_path = Path("data/market/vvix/vvix_daily.csv")
        if vvix_path.exists():
            df = pd.read_csv(vvix_path)
            df["datetime"] = pd.to_datetime(df["Date"])
            df["vvix"] = pd.to_numeric(df["Close"], errors="coerce")
            self._vvix_data = df.set_index("datetime").sort_index()

        futures_path = Path("data/cboe_vix_futures_full/vix_futures_all.csv")
        if futures_path.exists():
            try:
                self._futures_data = pd.read_csv(futures_path)
            except Exception:
                pass

        spx_path = Path(self.cfg["data"].get("rv_source", "data/market/spx/spx_5m.csv"))
        if spx_path.exists():
            df = pd.read_csv(spx_path)
            if "time" in df.columns:
                df["datetime"] = pd.to_datetime(df["time"], unit="s")
            df = df.sort_values("datetime" if "datetime" in df.columns else df.columns[0])
            self._spx_data = df

    # ------------------------------------------------------------------
    #  Individual signals
    # ------------------------------------------------------------------
    def _signal_vix_level(self) -> Optional[RegimeSignal]:
        """VIX absolute level signal."""
        if self._vix_data is None or "close" not in self._vix_data.columns:
            return None
        vix = float(self._vix_data["close"].iloc[-1])
        if vix < 13:
            regime = MarketRegime.CALM
        elif vix < 20:
            regime = MarketRegime.NORMAL
        elif vix < 30:
            regime = MarketRegime.STRESSED
        else:
            regime = MarketRegime.CRISIS
        return RegimeSignal("vix_level", vix, regime, 0.30,
                            f"VIX={vix:.1f}")

    def _signal_vix_percentile(self) -> Optional[RegimeSignal]:
        """VIX percentile rank over trailing 252 days."""
        if self._vix_data is None or "close" not in self._vix_data.columns:
            return None
        closes = self._vix_data["close"].dropna()
        if len(closes) < 252:
            return None
        current = float(closes.iloc[-1])
        pct = float((closes.tail(252 * 26) <= current).mean() * 100)
        if pct < 20:
            regime = MarketRegime.CALM
        elif pct < 60:
            regime = MarketRegime.NORMAL
        elif pct < 85:
            regime = MarketRegime.STRESSED
        else:
            regime = MarketRegime.CRISIS
        return RegimeSignal("vix_percentile", pct, regime, 0.15,
                            f"VIX percentile={pct:.0f}%")

    def _signal_vvix(self) -> Optional[RegimeSignal]:
        """VVIX (vol-of-vol) signal."""
        if self._vvix_data is None:
            return None
        vvix = float(self._vvix_data["vvix"].iloc[-1])
        if vvix < 80:
            regime = MarketRegime.CALM
        elif vvix < 100:
            regime = MarketRegime.NORMAL
        elif vvix < 130:
            regime = MarketRegime.STRESSED
        else:
            regime = MarketRegime.CRISIS
        return RegimeSignal("vvix", vvix, regime, 0.20,
                            f"VVIX={vvix:.1f}")

    def _signal_term_structure(self) -> Optional[RegimeSignal]:
        """VIX futures term structure shape (contango vs backwardation)."""
        if self._futures_data is None:
            return None
        df = self._futures_data
        if "Trade Date" not in df.columns:
            return None

        last_date = df["Trade Date"].max()
        curve = df[df["Trade Date"] == last_date]
        price_col = next((c for c in ["Close", "close", "Settle"] if c in curve.columns), None)
        if price_col is None or len(curve) < 2:
            return None

        prices = curve[price_col].dropna().values[:4]
        if len(prices) < 2:
            return None

        spread = float(prices[-1] - prices[0])
        if spread > 3:
            regime = MarketRegime.CALM
        elif spread > 0:
            regime = MarketRegime.NORMAL
        elif spread > -3:
            regime = MarketRegime.STRESSED
        else:
            regime = MarketRegime.CRISIS
        shape = "contango" if spread > 0 else "backwardation"
        return RegimeSignal("term_structure", spread, regime, 0.20,
                            f"Spread={spread:+.1f} ({shape})")

    def _signal_vrp(self) -> Optional[RegimeSignal]:
        """
        Variance Risk Premium: VIX² - RV².
        Positive VRP = normal (vol sellers compensated).
        Negative VRP = stressed (realized exceeds implied).
        """
        if self._vix_data is None or self._spx_data is None:
            return None
        if "close" not in self._vix_data.columns or "close" not in self._spx_data.columns:
            return None

        vix_last = float(self._vix_data["close"].iloc[-1])
        implied_var = (vix_last / 100) ** 2

        spx_close = self._spx_data["close"].values
        if len(spx_close) < 78:
            return None
        log_ret = np.diff(np.log(spx_close[-78:]))
        rv = float(np.std(log_ret) * np.sqrt(252 * 6.5 * 60 / 5))
        realized_var = rv ** 2

        vrp = implied_var - realized_var

        if vrp > 0.005:
            regime = MarketRegime.CALM
        elif vrp > 0:
            regime = MarketRegime.NORMAL
        elif vrp > -0.005:
            regime = MarketRegime.STRESSED
        else:
            regime = MarketRegime.CRISIS
        return RegimeSignal("vrp", float(vrp), regime, 0.15,
                            f"VRP={vrp*100:.2f}% (IV={vix_last:.1f}, RV={rv*100:.1f}%)")

    # ------------------------------------------------------------------
    #  Composite detection
    # ------------------------------------------------------------------
    def detect(self) -> dict:
        """
        Run all signals and produce a weighted consensus regime.
        Returns dict with regime, confidence, signals breakdown,
        and recommended model parameters.
        """
        signals = [
            self._signal_vix_level(),
            self._signal_vix_percentile(),
            self._signal_vvix(),
            self._signal_term_structure(),
            self._signal_vrp(),
        ]
        signals = [s for s in signals if s is not None]

        if not signals:
            return {
                "regime": MarketRegime.NORMAL.value,
                "confidence": 0.0,
                "signals": [],
                "recommended_params": self._default_params(),
            }

        regime_scores = {r: 0.0 for r in MarketRegime}
        total_weight = sum(s.weight for s in signals)
        for s in signals:
            regime_scores[s.regime] += s.weight / total_weight

        best_regime = max(regime_scores, key=regime_scores.get)
        confidence = float(regime_scores[best_regime])

        return {
            "regime": best_regime.value,
            "confidence": confidence,
            "regime_scores": {r.value: round(v, 3) for r, v in regime_scores.items()},
            "signals": [
                {"name": s.name, "value": round(s.value, 4),
                 "regime": s.regime.value, "weight": s.weight, "detail": s.detail}
                for s in signals
            ],
            "recommended_params": self._regime_params(best_regime),
        }

    def _regime_params(self, regime: MarketRegime) -> dict:
        """Recommended model parameters per regime."""
        base = self.cfg["bergomi"]
        params = {
            MarketRegime.CALM: {
                "H": 0.10, "eta": 1.5, "rho": -0.65,
                "hedge_freq": "daily",
                "mc_paths": 5000,
                "measure_priority": "Q",
            },
            MarketRegime.NORMAL: {
                "H": base["hurst"], "eta": base["eta"], "rho": base["rho"],
                "hedge_freq": "daily",
                "mc_paths": 10000,
                "measure_priority": "Q",
            },
            MarketRegime.STRESSED: {
                "H": 0.06, "eta": 2.5, "rho": -0.80,
                "hedge_freq": "intraday_4h",
                "mc_paths": 20000,
                "measure_priority": "P",
            },
            MarketRegime.CRISIS: {
                "H": 0.03, "eta": 3.5, "rho": -0.90,
                "hedge_freq": "intraday_1h",
                "mc_paths": 50000,
                "measure_priority": "P",
            },
            MarketRegime.RECOVERY: {
                "H": 0.08, "eta": 2.0, "rho": -0.75,
                "hedge_freq": "daily",
                "mc_paths": 10000,
                "measure_priority": "Q",
            },
        }
        return params.get(regime, params[MarketRegime.NORMAL])

    def _default_params(self) -> dict:
        return self._regime_params(MarketRegime.NORMAL)
