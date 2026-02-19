"""
P&L Attribution Engine
======================
Decomposes option P&L into Greeks contributions using
Taylor expansion of the BS pricing function.

Full second-order expansion:
    ΔC ≈ Δ·δS + ½Γ·(δS)² + ν·δσ + Θ·δt
        + Vanna·δS·δσ + ½Volga·(δσ)² + ρ̃·δr
        + higher-order residual

Supports:
  - Single option attribution
  - Portfolio-level attribution
  - Time series attribution (day-by-day)
  - Model comparison (BS vs Neural SDE implied Greeks)

Usage:
    from quant.pnl_attribution import PnLAttributor
    attr = PnLAttributor(spot=100, strike=100, T=0.25, r=0.045, sigma=0.20)
    result = attr.attribute(spot_new=99, sigma_new=0.22, dt=1/252)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from utils.black_scholes import BlackScholes


@dataclass
class Attribution:
    """P&L attribution breakdown for a single period."""
    total_pnl: float
    delta_pnl: float
    gamma_pnl: float
    vega_pnl: float
    theta_pnl: float
    vanna_pnl: float
    volga_pnl: float
    rho_pnl: float
    residual: float      # unexplained = total - sum(greeks)
    explained_pct: float  # % of total explained by Greeks


class PnLAttributor:
    """Greeks-based P&L decomposition engine."""

    def __init__(self, spot: float, strike: float, T: float,
                 r: float = 0.045, sigma: float = 0.20,
                 opt_type: str = 'call', quantity: float = 1.0):
        self.spot = spot
        self.strike = strike
        self.T = T
        self.r = r
        self.sigma = sigma
        self.opt_type = opt_type
        self.quantity = quantity

        self.initial_price = BlackScholes.price(spot, strike, T, r, sigma, opt_type)

    def attribute(self, spot_new: float, sigma_new: float = None,
                  dt: float = 1/252, r_new: float = None) -> Attribution:
        """
        Full second-order P&L attribution for one period.

        Args:
            spot_new: New spot price
            sigma_new: New implied vol (if None, assumes unchanged)
            dt: Time elapsed (in years, default = 1 trading day)
            r_new: New risk-free rate (if None, unchanged)
        """
        if sigma_new is None:
            sigma_new = self.sigma
        if r_new is None:
            r_new = self.r

        dS = spot_new - self.spot
        d_sigma = sigma_new - self.sigma
        d_r = r_new - self.r

        # Greeks at initial point
        delta = BlackScholes.delta(self.spot, self.strike, self.T, self.r, self.sigma, self.opt_type)
        gamma = BlackScholes.gamma(self.spot, self.strike, self.T, self.r, self.sigma)
        vega_100 = BlackScholes.vega(self.spot, self.strike, self.T, self.r, self.sigma)
        vega = vega_100 * 100  # convert from per-1% to per-unit
        theta_daily = BlackScholes.theta(self.spot, self.strike, self.T, self.r, self.sigma, self.opt_type)
        theta = theta_daily * 365  # per year
        vanna = BlackScholes.vanna(self.spot, self.strike, self.T, self.r, self.sigma)
        volga = BlackScholes.volga(self.spot, self.strike, self.T, self.r, self.sigma)
        rho_100 = BlackScholes.rho(self.spot, self.strike, self.T, self.r, self.sigma, self.opt_type)
        rho_val = rho_100 * 100

        # Taylor expansion
        delta_pnl = delta * dS
        gamma_pnl = 0.5 * gamma * dS ** 2
        vega_pnl = vega * d_sigma
        theta_pnl = theta * (-dt)
        vanna_pnl = vanna * dS * d_sigma
        volga_pnl = 0.5 * volga * d_sigma ** 2
        rho_pnl = rho_val * d_r

        explained = delta_pnl + gamma_pnl + vega_pnl + theta_pnl + vanna_pnl + volga_pnl + rho_pnl

        # Actual P&L
        T_new = max(self.T - dt, 1e-8)
        new_price = BlackScholes.price(spot_new, self.strike, T_new, r_new, sigma_new, self.opt_type)
        total_pnl = new_price - self.initial_price

        residual = total_pnl - explained
        explained_pct = (explained / total_pnl * 100) if abs(total_pnl) > 1e-10 else 100.0

        # Scale by quantity
        q = self.quantity
        return Attribution(
            total_pnl=float(total_pnl * q),
            delta_pnl=float(delta_pnl * q),
            gamma_pnl=float(gamma_pnl * q),
            vega_pnl=float(vega_pnl * q),
            theta_pnl=float(theta_pnl * q),
            vanna_pnl=float(vanna_pnl * q),
            volga_pnl=float(volga_pnl * q),
            rho_pnl=float(rho_pnl * q),
            residual=float(residual * q),
            explained_pct=float(np.clip(explained_pct, -999, 999)),
        )

    def attribute_series(self, spot_series: np.ndarray,
                         sigma_series: np.ndarray = None,
                         dt: float = 1/252,
                         r_series: np.ndarray = None) -> List[Attribution]:
        """
        Day-by-day P&L attribution over a time series.
        Updates the attributor state after each period.
        """
        n = len(spot_series)
        if sigma_series is None:
            sigma_series = np.full(n, self.sigma)
        if r_series is None:
            r_series = np.full(n, self.r)

        results = []
        for i in range(n):
            attr = self.attribute(
                spot_new=float(spot_series[i]),
                sigma_new=float(sigma_series[i]),
                dt=dt,
                r_new=float(r_series[i]),
            )
            results.append(attr)

            # Update state for next period
            self.initial_price = BlackScholes.price(
                float(spot_series[i]), self.strike,
                max(self.T - (i + 1) * dt, 1e-8),
                float(r_series[i]), float(sigma_series[i]), self.opt_type
            )
            self.spot = float(spot_series[i])
            self.sigma = float(sigma_series[i])
            self.r = float(r_series[i])
            self.T = max(self.T - dt, 1e-8)

        return results

    @staticmethod
    def summarize(attributions: List[Attribution]) -> dict:
        """Aggregate P&L attribution over multiple periods."""
        fields = ['total_pnl', 'delta_pnl', 'gamma_pnl', 'vega_pnl',
                  'theta_pnl', 'vanna_pnl', 'volga_pnl', 'rho_pnl', 'residual']
        summary = {}
        for f in fields:
            values = [getattr(a, f) for a in attributions]
            summary[f] = {
                'total': float(sum(values)),
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
            }
        total_explained = sum(summary[f]['total'] for f in fields if f != 'total_pnl' and f != 'residual')
        total_pnl = summary['total_pnl']['total']
        summary['explained_pct'] = float(
            total_explained / total_pnl * 100 if abs(total_pnl) > 1e-10 else 100.0
        )
        return summary


class PortfolioPnLAttributor:
    """P&L attribution for a portfolio of options."""

    def __init__(self, spot: float, r: float = 0.045):
        self.spot = spot
        self.r = r
        self.attributors: List[PnLAttributor] = []
        self.labels: List[str] = []

    def add_position(self, strike: float, T: float, sigma: float,
                     opt_type: str = 'call', quantity: float = 1.0,
                     label: str = None):
        attr = PnLAttributor(self.spot, strike, T, self.r, sigma, opt_type, quantity)
        self.attributors.append(attr)
        self.labels.append(label or f"{opt_type}_{strike}_{T:.2f}")

    def attribute(self, spot_new: float, sigmas_new: dict = None,
                  dt: float = 1/252, r_new: float = None) -> dict:
        """
        Attribute P&L for entire portfolio.

        Args:
            sigmas_new: dict mapping label -> new IV. If None, uses initial IVs.
        """
        results = {}
        portfolio_total = Attribution(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        for i, (attr, label) in enumerate(zip(self.attributors, self.labels)):
            sig_new = sigmas_new.get(label, attr.sigma) if sigmas_new else None
            res = attr.attribute(spot_new, sig_new, dt, r_new)
            results[label] = res

            for field in ['total_pnl', 'delta_pnl', 'gamma_pnl', 'vega_pnl',
                          'theta_pnl', 'vanna_pnl', 'volga_pnl', 'rho_pnl', 'residual']:
                curr = getattr(portfolio_total, field)
                setattr(portfolio_total, field, curr + getattr(res, field))

        if abs(portfolio_total.total_pnl) > 1e-10:
            explained = (portfolio_total.total_pnl - portfolio_total.residual)
            portfolio_total.explained_pct = float(explained / portfolio_total.total_pnl * 100)

        results['__portfolio__'] = portfolio_total
        return results
