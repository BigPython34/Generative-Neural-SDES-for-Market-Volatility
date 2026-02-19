"""
Portfolio Risk Engine
=====================
Computes VaR, CVaR (ES), stressed VaR, and tail risk metrics
for single instruments or portfolios of options.

Uses the Neural SDE (P-measure) for realistic scenario generation,
with fallback to Rough Bergomi or parametric methods.

Supports:
  - Parametric VaR (delta-normal)
  - Historical simulation VaR
  - Monte Carlo VaR (Neural SDE / Bergomi / GBM)
  - Stressed VaR (conditional on regime)
  - Component VaR (contribution per position)
  - Tail risk metrics (Hill estimator, peak-over-threshold)

Usage:
    from quant.risk_engine import RiskEngine
    engine = RiskEngine(spot=100, r=0.045)
    engine.add_position('call', strike=100, T=0.25, quantity=10, iv=0.20)
    result = engine.compute_var(spot_paths, var_paths, confidence=0.99)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from utils.black_scholes import BlackScholes


@dataclass
class Position:
    """Single option or underlying position."""
    instrument: str     # 'call', 'put', 'stock'
    strike: float = 0.0
    T: float = 0.0
    quantity: float = 1.0
    iv: float = 0.20
    entry_price: float = 0.0


@dataclass
class RiskReport:
    """Comprehensive risk report."""
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    stressed_var_99: float
    max_loss: float
    expected_pnl: float
    pnl_std: float
    pnl_skew: float
    pnl_kurtosis: float
    n_scenarios: int
    tail_index: Optional[float] = None
    component_var: Optional[dict] = None


class RiskEngine:
    """Monte Carlo risk engine for options portfolios."""

    def __init__(self, spot: float, r: float = 0.045):
        self.spot = spot
        self.r = r
        self.positions: List[Position] = []

    def add_position(self, instrument: str, strike: float = 0.0,
                     T: float = 0.25, quantity: float = 1.0,
                     iv: float = 0.20, entry_price: float = None):
        """Add a position to the portfolio."""
        if entry_price is None and instrument in ('call', 'put'):
            entry_price = BlackScholes.price(self.spot, strike, T, self.r, iv, instrument)
        elif entry_price is None:
            entry_price = self.spot

        self.positions.append(Position(
            instrument=instrument, strike=strike, T=T,
            quantity=quantity, iv=iv, entry_price=entry_price
        ))

    def clear_positions(self):
        self.positions = []

    # ------------------------------------------------------------------
    #  Portfolio valuation
    # ------------------------------------------------------------------
    def _value_portfolio(self, spot_scenarios: np.ndarray,
                         vol_scenarios: np.ndarray = None,
                         dt_shift: float = 1/252) -> np.ndarray:
        """
        Value the portfolio under each scenario.

        Args:
            spot_scenarios: (n_scenarios,) terminal spot values
            vol_scenarios: (n_scenarios,) terminal vol values (optional)
            dt_shift: time decay per scenario (default 1 day)

        Returns:
            (n_scenarios,) portfolio P&L array
        """
        n = len(spot_scenarios)
        pnl = np.zeros(n)

        for pos in self.positions:
            if pos.instrument == 'stock':
                pos_pnl = pos.quantity * (spot_scenarios - self.spot)
            elif pos.instrument in ('call', 'put'):
                T_new = max(pos.T - dt_shift, 1e-6)
                vols = vol_scenarios if vol_scenarios is not None else np.full(n, pos.iv)
                new_prices = np.array([
                    BlackScholes.price(s, pos.strike, T_new, self.r, v, pos.instrument)
                    for s, v in zip(spot_scenarios, vols)
                ])
                pos_pnl = pos.quantity * (new_prices - pos.entry_price)
            else:
                pos_pnl = np.zeros(n)

            pnl += pos_pnl

        return pnl

    # ------------------------------------------------------------------
    #  VaR computation
    # ------------------------------------------------------------------
    def compute_var(self, spot_paths: np.ndarray,
                    var_paths: np.ndarray = None,
                    horizon_steps: int = None,
                    confidence_levels: tuple = (0.95, 0.99)) -> RiskReport:
        """
        Compute full risk report from MC paths.

        Args:
            spot_paths: (n_paths, n_steps+1) spot price paths
            var_paths: (n_paths, n_steps) variance paths (optional)
            horizon_steps: Steps ahead for VaR. If None, uses terminal values.
            confidence_levels: Tuple of confidence levels
        """
        if horizon_steps is not None:
            idx = min(horizon_steps, spot_paths.shape[1] - 1)
            s_terminal = spot_paths[:, idx]
        else:
            s_terminal = spot_paths[:, -1]

        if var_paths is not None:
            if horizon_steps is not None:
                idx_v = min(horizon_steps, var_paths.shape[1]) - 1
                vol_terminal = np.sqrt(np.clip(var_paths[:, idx_v], 1e-8, None))
            else:
                vol_terminal = np.sqrt(np.clip(var_paths[:, -1], 1e-8, None))
        else:
            vol_terminal = None

        pnl = self._value_portfolio(s_terminal, vol_terminal)

        var_95 = float(-np.percentile(pnl, (1 - 0.95) * 100))
        var_99 = float(-np.percentile(pnl, (1 - 0.99) * 100))

        cvar_95 = float(-np.mean(pnl[pnl <= -var_95])) if (pnl <= -var_95).any() else var_95
        cvar_99 = float(-np.mean(pnl[pnl <= -var_99])) if (pnl <= -var_99).any() else var_99

        # Stressed VaR: conditional on spot drop > 5%
        stress_mask = s_terminal < self.spot * 0.95
        if stress_mask.sum() > 10:
            stress_pnl = pnl[stress_mask]
            stressed_var = float(-np.percentile(stress_pnl, 1))
        else:
            stressed_var = var_99 * 1.5

        # Tail index (Hill estimator)
        tail_index = self._hill_estimator(pnl)

        # Component VaR
        component_var = self._component_var(s_terminal, vol_terminal, var_99)

        return RiskReport(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            stressed_var_99=stressed_var,
            max_loss=float(-np.min(pnl)),
            expected_pnl=float(np.mean(pnl)),
            pnl_std=float(np.std(pnl)),
            pnl_skew=float(self._skewness(pnl)),
            pnl_kurtosis=float(self._kurtosis(pnl)),
            n_scenarios=len(pnl),
            tail_index=tail_index,
            component_var=component_var,
        )

    # ------------------------------------------------------------------
    #  Parametric VaR (delta-normal)
    # ------------------------------------------------------------------
    def parametric_var(self, vol: float = 0.20, horizon_days: int = 1,
                       confidence: float = 0.99) -> dict:
        """
        Delta-normal parametric VaR.
        Quick approximation using portfolio Greeks.
        """
        from scipy.stats import norm

        z = norm.ppf(confidence)
        T_h = horizon_days / 252.0
        sigma_daily = vol * np.sqrt(T_h)

        total_delta = 0.0
        total_gamma = 0.0
        total_vega = 0.0
        total_value = 0.0

        for pos in self.positions:
            if pos.instrument == 'stock':
                total_delta += pos.quantity
                total_value += pos.quantity * self.spot
            elif pos.instrument in ('call', 'put'):
                d = BlackScholes.delta(self.spot, pos.strike, pos.T, self.r, pos.iv, pos.instrument)
                g = BlackScholes.gamma(self.spot, pos.strike, pos.T, self.r, pos.iv)
                v = BlackScholes.vega(self.spot, pos.strike, pos.T, self.r, pos.iv)
                total_delta += pos.quantity * d
                total_gamma += pos.quantity * g
                total_vega += pos.quantity * v
                total_value += pos.quantity * pos.entry_price

        delta_var = abs(total_delta) * self.spot * sigma_daily * z
        gamma_adj = 0.5 * abs(total_gamma) * (self.spot * sigma_daily) ** 2

        return {
            "delta_var": float(delta_var),
            "delta_gamma_var": float(delta_var + gamma_adj),
            "portfolio_delta": float(total_delta),
            "portfolio_gamma": float(total_gamma),
            "portfolio_vega": float(total_vega),
            "portfolio_value": float(total_value),
        }

    # ------------------------------------------------------------------
    #  Stress testing
    # ------------------------------------------------------------------
    def stress_test(self, scenarios: dict = None) -> dict:
        """
        Run predefined stress scenarios.

        Default scenarios based on historical events:
          - Black Monday 1987: spot -22%, vol +150%
          - Lehman 2008: spot -8%, vol +80%
          - COVID 2020: spot -12%, vol +200%
          - Flash Crash 2010: spot -9%, vol +50%
          - Volmageddon 2018: spot -4%, vol +100%
        """
        if scenarios is None:
            scenarios = {
                "Black Monday": {"spot_shock": -0.22, "vol_shock": 1.50},
                "Lehman": {"spot_shock": -0.08, "vol_shock": 0.80},
                "COVID March": {"spot_shock": -0.12, "vol_shock": 2.00},
                "Flash Crash": {"spot_shock": -0.09, "vol_shock": 0.50},
                "Volmageddon": {"spot_shock": -0.04, "vol_shock": 1.00},
                "Rate Shock +200bp": {"spot_shock": -0.05, "vol_shock": 0.30, "rate_shock": 0.02},
                "Melt-up": {"spot_shock": 0.10, "vol_shock": -0.30},
            }

        results = {}
        for name, shocks in scenarios.items():
            s_new = self.spot * (1 + shocks["spot_shock"])
            rate_new = self.r + shocks.get("rate_shock", 0.0)

            portfolio_pnl = 0.0
            for pos in self.positions:
                if pos.instrument == 'stock':
                    portfolio_pnl += pos.quantity * (s_new - self.spot)
                elif pos.instrument in ('call', 'put'):
                    vol_new = pos.iv * (1 + shocks["vol_shock"])
                    vol_new = np.clip(vol_new, 0.01, 3.0)
                    new_price = BlackScholes.price(
                        s_new, pos.strike, pos.T, rate_new, vol_new, pos.instrument
                    )
                    portfolio_pnl += pos.quantity * (new_price - pos.entry_price)

            results[name] = {
                "pnl": float(portfolio_pnl),
                "spot_shocked": float(s_new),
                **shocks,
            }

        return results

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------
    def _component_var(self, s_terminal, vol_terminal, total_var_99):
        """Compute each position's marginal contribution to VaR."""
        if len(self.positions) <= 1:
            return None
        components = {}
        for i, pos in enumerate(self.positions):
            # Marginal: VaR(full) - VaR(full \ pos_i)
            excluded = [p for j, p in enumerate(self.positions) if j != i]
            original = self.positions
            self.positions = excluded
            pnl_excl = self._value_portfolio(s_terminal, vol_terminal)
            var_excl = float(-np.percentile(pnl_excl, 1))
            self.positions = original

            components[f"pos_{i}_{pos.instrument}_{pos.strike}"] = float(total_var_99 - var_excl)

        return components

    @staticmethod
    def _hill_estimator(losses: np.ndarray, k_fraction: float = 0.05) -> Optional[float]:
        """Hill tail index estimator for extreme losses."""
        sorted_losses = np.sort(-losses)[::-1]
        k = max(10, int(len(sorted_losses) * k_fraction))
        if k >= len(sorted_losses) or sorted_losses[k] <= 0:
            return None
        top_k = sorted_losses[:k]
        threshold = sorted_losses[k]
        if threshold <= 0:
            return None
        return float(1.0 / np.mean(np.log(top_k / threshold)))

    @staticmethod
    def _skewness(x):
        m = np.mean(x)
        s = np.std(x)
        if s < 1e-12:
            return 0.0
        return np.mean(((x - m) / s) ** 3)

    @staticmethod
    def _kurtosis(x):
        m = np.mean(x)
        s = np.std(x)
        if s < 1e-12:
            return 0.0
        return np.mean(((x - m) / s) ** 4) - 3.0
