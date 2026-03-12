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
    from quant.regimes.regime_detector import RegimeDetector
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
        self._skew_data: Optional[pd.DataFrame] = None
        self._pcr_data: Optional[pd.DataFrame] = None
        # VIX term structure indices from TradingView
        self._vix9d_data: Optional[pd.DataFrame] = None
        self._vix3m_data: Optional[pd.DataFrame] = None
        self._vix6m_data: Optional[pd.DataFrame] = None
        self._load_data()

    @staticmethod
    def _read_ts_csv(path: Path) -> Optional[pd.DataFrame]:
        """Read a CSV with 'time' (unix) or 'Date' column into datetime-indexed df."""
        if not path.exists():
            return None
        df = pd.read_csv(path)
        if "time" in df.columns:
            df["datetime"] = pd.to_datetime(df["time"], unit="s")
        elif "Date" in df.columns:
            df["datetime"] = pd.to_datetime(df["Date"])
        else:
            return None
        # Normalize column names to lowercase
        df.columns = [c.lower() for c in df.columns]
        df = df.sort_values("datetime").set_index("datetime")
        return df

    def _load_data(self):
        # VIX daily — prefer TradingView (36 years) over Yahoo (limited)
        for p in [
            Path("data/trading_view/volatility/vix_daily.csv"),
            Path(self.cfg["data"]["source"]),
            Path("data/market/vix/vix_daily.csv"),
        ]:
            self._vix_data = self._read_ts_csv(p)
            if self._vix_data is not None:
                break

        # VVIX daily — TradingView first (20 years)
        for p in [
            Path("data/trading_view/volatility/vvix_daily.csv"),
            Path("data/market/vvix/vvix_daily.csv"),
        ]:
            df = self._read_ts_csv(p)
            if df is not None:
                if "vvix" not in df.columns and "close" in df.columns:
                    df["vvix"] = df["close"]
                self._vvix_data = df
                break

        # VIX term structure — TradingView VIX9D / VIX3M / VIX6M daily
        self._vix9d_data = self._read_ts_csv(
            Path("data/trading_view/volatility/vix9d_daily.csv"))
        self._vix3m_data = self._read_ts_csv(
            Path("data/trading_view/volatility/vix3m_daily.csv"))
        self._vix6m_data = self._read_ts_csv(
            Path("data/trading_view/volatility/vix6m_daily.csv"))

        # CBOE VIX futures (fallback for term structure)
        futures_path = Path("data/cboe_vix_futures_full/vix_futures_all.csv")
        if futures_path.exists():
            try:
                self._futures_data = pd.read_csv(futures_path)
            except Exception:
                pass

        # SPX intraday for RV
        for p in [
            Path("data/trading_view/equity_indices/spx_5m.csv"),
            Path(self.cfg["data"].get("rv_source", "data/market/spx/spx_5m.csv")),
        ]:
            self._spx_data = self._read_ts_csv(p)
            if self._spx_data is not None:
                break

        # SKEW — TradingView (36 years)
        self._skew_data = self._read_ts_csv(
            Path("data/trading_view/sentiment/skew_daily.csv"))

        # Put-Call Ratio — TradingView SPX PCR (20 years)
        for p in [
            Path("data/trading_view/sentiment/pcspx_daily.csv"),
            Path("data/trading_view/sentiment/pc_daily.csv"),
        ]:
            self._pcr_data = self._read_ts_csv(p)
            if self._pcr_data is not None:
                break

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
        """VIX term structure shape using TradingView VIX9D/VIX3M/VIX6M.

        When VIX3M > VIX (contango) the market is calm.
        When VIX > VIX3M (backwardation) the market is stressed.
        The ratio VIX/VIX3M is more stable than raw spread.
        """
        # Prefer TradingView VIX term structure indices
        if (self._vix_data is not None and self._vix3m_data is not None
                and "close" in self._vix_data.columns
                and "close" in self._vix3m_data.columns):
            vix_now = float(self._vix_data["close"].iloc[-1])
            vix3m = float(self._vix3m_data["close"].iloc[-1])
            if vix3m > 0:
                ratio = vix_now / vix3m
                spread = float(vix3m - vix_now)
                if ratio < 0.90:
                    regime = MarketRegime.CALM
                elif ratio < 1.0:
                    regime = MarketRegime.NORMAL
                elif ratio < 1.15:
                    regime = MarketRegime.STRESSED
                else:
                    regime = MarketRegime.CRISIS
                shape = "contango" if spread > 0 else "backwardation"
                return RegimeSignal("term_structure", float(ratio), regime, 0.20,
                                    f"VIX/VIX3M={ratio:.2f} ({shape}, spread={spread:+.1f})")

        # Fallback: CBOE VIX futures
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

    def _signal_skew(self) -> Optional[RegimeSignal]:
        """CBOE SKEW index signal.

        SKEW measures tail-risk pricing (perceived probability of a 2+ sigma
        down move).  Normal range ~120-140.  Extremes > 150 signal
        strong demand for OTM puts (fear).  Low < 110 = complacency.
        """
        if self._skew_data is None or "close" not in self._skew_data.columns:
            return None
        skew = float(self._skew_data["close"].iloc[-1])
        if np.isnan(skew):
            return None
        if skew < 115:
            regime = MarketRegime.CALM
        elif skew < 135:
            regime = MarketRegime.NORMAL
        elif skew < 150:
            regime = MarketRegime.STRESSED
        else:
            regime = MarketRegime.CRISIS
        return RegimeSignal("skew", skew, regime, 0.10,
                            f"SKEW={skew:.0f}")

    def _signal_pcr(self) -> Optional[RegimeSignal]:
        """SPX Put-Call Ratio signal.

        High PCR (> 1.2) means heavy put buying = bearish sentiment.
        Low PCR (< 0.7) = complacency, potential reversal.
        """
        if self._pcr_data is None or "close" not in self._pcr_data.columns:
            return None
        pcr = float(self._pcr_data["close"].iloc[-1])
        if np.isnan(pcr) or pcr <= 0:
            return None
        if pcr < 0.7:
            regime = MarketRegime.CALM
        elif pcr < 1.0:
            regime = MarketRegime.NORMAL
        elif pcr < 1.3:
            regime = MarketRegime.STRESSED
        else:
            regime = MarketRegime.CRISIS
        return RegimeSignal("put_call_ratio", pcr, regime, 0.10,
                            f"PCR={pcr:.2f}")

    # ------------------------------------------------------------------
    #  Composite detection
    # ------------------------------------------------------------------
    def _collect_signals(self) -> list:
        """Gather all available signals."""
        raw = [
            self._signal_vix_level(),
            self._signal_vix_percentile(),
            self._signal_vvix(),
            self._signal_term_structure(),
            self._signal_vrp(),
            self._signal_skew(),
            self._signal_pcr(),
        ]
        return [s for s in raw if s is not None]

    def detect(self) -> dict:
        """
        Run all signals and produce a weighted consensus regime.
        Returns dict with regime, confidence, signals breakdown,
        and recommended model parameters.
        """
        signals = self._collect_signals()
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

    # ------------------------------------------------------------------
    #  Historical detection (for backtests)
    # ------------------------------------------------------------------
    def detect_from_values(
        self,
        vix: float,
        vvix: Optional[float] = None,
        vix3m: Optional[float] = None,
        skew: Optional[float] = None,
        pcr: Optional[float] = None,
        rv_annual: Optional[float] = None,
    ) -> dict:
        """Detect regime from explicit values (no data loading).

        Useful for historical backtests where signals are pre-computed
        per date.
        """
        signals: list[RegimeSignal] = []

        # VIX level
        if vix < 13:
            r = MarketRegime.CALM
        elif vix < 20:
            r = MarketRegime.NORMAL
        elif vix < 30:
            r = MarketRegime.STRESSED
        else:
            r = MarketRegime.CRISIS
        signals.append(RegimeSignal("vix_level", vix, r, 0.30, f"VIX={vix:.1f}"))

        # VVIX
        if vvix is not None and not np.isnan(vvix):
            if vvix < 80:
                r = MarketRegime.CALM
            elif vvix < 100:
                r = MarketRegime.NORMAL
            elif vvix < 130:
                r = MarketRegime.STRESSED
            else:
                r = MarketRegime.CRISIS
            signals.append(RegimeSignal("vvix", vvix, r, 0.20, f"VVIX={vvix:.1f}"))

        # Term structure
        if vix3m is not None and vix3m > 0 and not np.isnan(vix3m):
            ratio = vix / vix3m
            if ratio < 0.90:
                r = MarketRegime.CALM
            elif ratio < 1.0:
                r = MarketRegime.NORMAL
            elif ratio < 1.15:
                r = MarketRegime.STRESSED
            else:
                r = MarketRegime.CRISIS
            signals.append(RegimeSignal("term_structure", ratio, r, 0.20,
                                        f"VIX/VIX3M={ratio:.2f}"))

        # VRP
        if rv_annual is not None and not np.isnan(rv_annual):
            iv_var = (vix / 100) ** 2
            vrp = iv_var - rv_annual
            if vrp > 0.005:
                r = MarketRegime.CALM
            elif vrp > 0:
                r = MarketRegime.NORMAL
            elif vrp > -0.005:
                r = MarketRegime.STRESSED
            else:
                r = MarketRegime.CRISIS
            signals.append(RegimeSignal("vrp", vrp, r, 0.15, f"VRP={vrp*100:.2f}%"))

        # SKEW
        if skew is not None and not np.isnan(skew):
            if skew < 115:
                r = MarketRegime.CALM
            elif skew < 135:
                r = MarketRegime.NORMAL
            elif skew < 150:
                r = MarketRegime.STRESSED
            else:
                r = MarketRegime.CRISIS
            signals.append(RegimeSignal("skew", skew, r, 0.10, f"SKEW={skew:.0f}"))

        # PCR
        if pcr is not None and not np.isnan(pcr) and pcr > 0:
            if pcr < 0.7:
                r = MarketRegime.CALM
            elif pcr < 1.0:
                r = MarketRegime.NORMAL
            elif pcr < 1.3:
                r = MarketRegime.STRESSED
            else:
                r = MarketRegime.CRISIS
            signals.append(RegimeSignal("put_call_ratio", pcr, r, 0.10, f"PCR={pcr:.2f}"))

        if not signals:
            return {"regime": MarketRegime.NORMAL.value, "confidence": 0.0,
                    "signals": [], "recommended_params": self._default_params()}

        regime_scores = {rg: 0.0 for rg in MarketRegime}
        total_w = sum(s.weight for s in signals)
        for s in signals:
            regime_scores[s.regime] += s.weight / total_w
        best = max(regime_scores, key=regime_scores.get)
        return {
            "regime": best.value,
            "confidence": float(regime_scores[best]),
            "regime_scores": {rg.value: round(v, 3) for rg, v in regime_scores.items()},
            "signals": [{"name": s.name, "value": round(s.value, 4),
                         "regime": s.regime.value} for s in signals],
            "recommended_params": self._regime_params(best),
        }
