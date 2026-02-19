"""
Delta Hedging Simulator
=======================
Simulates discrete delta-hedging strategies under different models
and measures the hedging P&L (tracking error).

Compares hedging performance of:
  - Black-Scholes delta (flat vol)
  - Bartlett's delta (minimum-variance, vanna/volga correction)

Fully vectorized over paths for fast Monte Carlo execution.

Usage:
    from quant.hedging_simulator import HedgingSimulator
    sim = HedgingSimulator(spot=100, strike=100, T=0.25, r=0.045, iv=0.20)
    result = sim.run(spot_paths, var_paths, hedge_freq='daily')
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from scipy.stats import norm


@dataclass
class HedgingResult:
    """Results from a hedging simulation."""
    model_name: str
    mean_pnl: float
    std_pnl: float
    mean_abs_pnl: float
    max_loss: float
    sharpe: float
    tracking_error: float
    hedge_cost: float
    n_rebalances: int
    pnl_distribution: np.ndarray


def _bs_delta_vec(S, K, T_rem, r, sigma, opt_type):
    """Vectorized BS delta across all paths for a single time step."""
    T_safe = np.maximum(T_rem, 1e-8)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T_safe) / (sigma * np.sqrt(T_safe))
    if opt_type == 'call':
        return norm.cdf(d1)
    return norm.cdf(d1) - 1.0


def _bs_vanna_vec(S, K, T_rem, r, sigma):
    """Vectorized BS vanna across all paths for a single time step."""
    T_safe = np.maximum(T_rem, 1e-8)
    sqrt_T = np.sqrt(T_safe)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T_safe) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return -norm.pdf(d1) * d2 / sigma


class HedgingSimulator:
    """Simulates delta-hedging and compares model performance."""

    def __init__(self, spot: float, strike: float, T: float,
                 r: float = 0.045, iv: float = 0.20,
                 opt_type: str = 'call', transaction_cost_bps: float = 2.0):
        self.spot = spot
        self.strike = strike
        self.T = T
        self.r = r
        self.iv = iv
        self.opt_type = opt_type
        self.tc_bps = transaction_cost_bps / 10000.0

        from utils.black_scholes import BlackScholes
        self.option_price = BlackScholes.price(spot, strike, T, r, iv, opt_type)

    def run(self, spot_paths: np.ndarray, var_paths: np.ndarray = None,
            hedge_freq: str = 'daily', rebalance_every: int = None) -> dict:
        n_paths, n_total = spot_paths.shape
        n_steps = n_total - 1
        dt = self.T / n_steps

        if rebalance_every is None:
            freq_map = {'daily': 1, 'intraday_4h': 4, 'intraday_1h': 8, 'weekly': 5}
            rebalance_every = freq_map.get(hedge_freq, 1)

        rebal_set = set(range(0, n_steps, rebalance_every))

        results = {}
        results['Black-Scholes'] = self._hedge_bs_vec(spot_paths, dt, rebal_set)
        results['Bartlett'] = self._hedge_bartlett_vec(spot_paths, var_paths, dt, rebal_set)
        return results

    def _hedge_bs_vec(self, spot_paths, dt, rebal_set) -> HedgingResult:
        """Vectorized BS delta hedge across all paths simultaneously."""
        n_paths, n_total = spot_paths.shape
        n_steps = n_total - 1

        cash = np.full(n_paths, self.option_price)
        delta_prev = np.zeros(n_paths)

        for step in range(n_steps):
            S = spot_paths[:, step]
            T_rem = max(self.T - step * dt, 1e-8)

            if step in rebal_set:
                delta_new = _bs_delta_vec(S, self.strike, T_rem, self.r, self.iv, self.opt_type)
                trade = delta_new - delta_prev
                cash -= trade * S
                cash -= np.abs(trade) * S * self.tc_bps
                delta_prev = delta_new

            cash *= np.exp(self.r * dt)

        S_T = spot_paths[:, -1]
        stock_value = delta_prev * S_T
        if self.opt_type == 'call':
            payoff = np.maximum(S_T - self.strike, 0)
        else:
            payoff = np.maximum(self.strike - S_T, 0)

        pnl = cash + stock_value - payoff
        return self._build_result('Black-Scholes', pnl, len(rebal_set))

    def _hedge_bartlett_vec(self, spot_paths, var_paths, dt, rebal_set) -> HedgingResult:
        """Vectorized Bartlett (minimum-variance) delta hedge."""
        n_paths, n_total = spot_paths.shape
        n_steps = n_total - 1

        cash = np.full(n_paths, self.option_price)
        delta_prev = np.zeros(n_paths)
        rho_sv = -0.7

        for step in range(n_steps):
            S = spot_paths[:, step]
            T_rem = max(self.T - step * dt, 1e-8)

            if var_paths is not None and step < var_paths.shape[1]:
                local_vol = np.sqrt(np.clip(var_paths[:, step], 1e-8, None))
            else:
                local_vol = self.iv

            if step in rebal_set:
                bs_delta = _bs_delta_vec(S, self.strike, T_rem, self.r, local_vol, self.opt_type)
                vanna = _bs_vanna_vec(S, self.strike, T_rem, self.r, local_vol)
                d_sigma_dS = rho_sv * local_vol / S
                delta_new = np.clip(bs_delta + vanna * d_sigma_dS, -2.0, 2.0)

                trade = delta_new - delta_prev
                cash -= trade * S
                cash -= np.abs(trade) * S * self.tc_bps
                delta_prev = delta_new

            cash *= np.exp(self.r * dt)

        S_T = spot_paths[:, -1]
        stock_value = delta_prev * S_T
        if self.opt_type == 'call':
            payoff = np.maximum(S_T - self.strike, 0)
        else:
            payoff = np.maximum(self.strike - S_T, 0)

        pnl = cash + stock_value - payoff
        return self._build_result('Bartlett', pnl, len(rebal_set))

    def _build_result(self, name, pnl_arr, n_rebal) -> HedgingResult:
        mean_pnl = float(np.mean(pnl_arr))
        std_pnl = float(np.std(pnl_arr))
        sharpe = mean_pnl / std_pnl if std_pnl > 1e-10 else 0.0

        return HedgingResult(
            model_name=name,
            mean_pnl=mean_pnl,
            std_pnl=std_pnl,
            mean_abs_pnl=float(np.mean(np.abs(pnl_arr))),
            max_loss=float(np.min(pnl_arr)),
            sharpe=float(sharpe),
            tracking_error=std_pnl * np.sqrt(252 / self.T) if self.T > 0 else 0.0,
            hedge_cost=float(self.tc_bps * self.spot * n_rebal),
            n_rebalances=n_rebal,
            pnl_distribution=pnl_arr,
        )

    @staticmethod
    def compare_results(results: dict) -> str:
        """Pretty-print comparison table."""
        lines = [
            f"{'Model':<20} {'Mean P&L':>10} {'Std P&L':>10} "
            f"{'Track Err':>10} {'Max Loss':>10} {'Sharpe':>8}",
            "-" * 72,
        ]
        for name, res in results.items():
            lines.append(
                f"{name:<20} {res.mean_pnl:>10.4f} {res.std_pnl:>10.4f} "
                f"{res.tracking_error:>10.2f} {res.max_loss:>10.4f} {res.sharpe:>8.4f}"
            )
        return "\n".join(lines)
