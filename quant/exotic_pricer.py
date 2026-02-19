"""
Exotic Option Pricer
====================
Monte Carlo pricing for path-dependent exotic options using
Neural SDE, Rough Bergomi, or Black-Scholes spot paths.

Supported products:
  - Asian (arithmetic & geometric, fixed/floating strike)
  - Lookback (fixed/floating strike)
  - Autocallable (multi-barrier with coupon)
  - Cliquet (locally capped/floored)
  - Variance swap
  - Volatility swap

Usage:
    from quant.exotic_pricer import ExoticPricer
    pricer = ExoticPricer(spot=100, r=0.045, T=0.25)
    price = pricer.asian_call(spot_paths, strike=100)
"""

import numpy as np
from utils.black_scholes import BlackScholes


class ExoticPricer:
    """Monte Carlo exotic option pricer from simulated spot paths."""

    def __init__(self, spot: float, r: float = 0.045, T: float = 0.25):
        self.spot = spot
        self.r = r
        self.T = T

    def _discount(self, T: float = None) -> float:
        return np.exp(-self.r * (T or self.T))

    # ------------------------------------------------------------------
    #  Asian options
    # ------------------------------------------------------------------
    def asian_call(self, paths: np.ndarray, strike: float = None,
                   averaging: str = "arithmetic") -> dict:
        """
        Asian call: payoff = max(A(S) - K, 0)
        where A(S) is arithmetic or geometric average of path.

        Args:
            paths: (n_paths, n_steps+1) spot paths including S0
            strike: Fixed strike. If None, uses floating strike (A(S) vs S_T).
            averaging: 'arithmetic' or 'geometric'
        """
        if averaging == "geometric":
            avg = np.exp(np.mean(np.log(paths[:, 1:]), axis=1))
        else:
            avg = np.mean(paths[:, 1:], axis=1)

        if strike is None:
            payoff = np.maximum(paths[:, -1] - avg, 0)
        else:
            payoff = np.maximum(avg - strike, 0)

        price = self._discount() * np.mean(payoff)
        std_err = self._discount() * np.std(payoff) / np.sqrt(len(payoff))

        return {"price": float(price), "std_error": float(std_err),
                "type": f"asian_call_{averaging}"}

    def asian_put(self, paths: np.ndarray, strike: float = None,
                  averaging: str = "arithmetic") -> dict:
        """Asian put: payoff = max(K - A(S), 0)."""
        if averaging == "geometric":
            avg = np.exp(np.mean(np.log(paths[:, 1:]), axis=1))
        else:
            avg = np.mean(paths[:, 1:], axis=1)

        if strike is None:
            payoff = np.maximum(avg - paths[:, -1], 0)
        else:
            payoff = np.maximum(strike - avg, 0)

        price = self._discount() * np.mean(payoff)
        std_err = self._discount() * np.std(payoff) / np.sqrt(len(payoff))
        return {"price": float(price), "std_error": float(std_err),
                "type": f"asian_put_{averaging}"}

    # ------------------------------------------------------------------
    #  Lookback options
    # ------------------------------------------------------------------
    def lookback_call(self, paths: np.ndarray, strike: float = None) -> dict:
        """
        Lookback call:
          Fixed strike:  payoff = max(max(S) - K, 0)
          Floating strike: payoff = S_T - min(S)
        """
        if strike is not None:
            payoff = np.maximum(np.max(paths, axis=1) - strike, 0)
        else:
            payoff = paths[:, -1] - np.min(paths, axis=1)

        price = self._discount() * np.mean(payoff)
        std_err = self._discount() * np.std(payoff) / np.sqrt(len(payoff))
        return {"price": float(price), "std_error": float(std_err),
                "type": "lookback_call"}

    def lookback_put(self, paths: np.ndarray, strike: float = None) -> dict:
        """
        Lookback put:
          Fixed strike:  payoff = max(K - min(S), 0)
          Floating strike: payoff = max(S) - S_T
        """
        if strike is not None:
            payoff = np.maximum(strike - np.min(paths, axis=1), 0)
        else:
            payoff = np.max(paths, axis=1) - paths[:, -1]

        price = self._discount() * np.mean(payoff)
        std_err = self._discount() * np.std(payoff) / np.sqrt(len(payoff))
        return {"price": float(price), "std_error": float(std_err),
                "type": "lookback_put"}

    # ------------------------------------------------------------------
    #  Barrier options
    # ------------------------------------------------------------------
    def down_and_out_call(self, paths: np.ndarray, strike: float,
                          barrier: float) -> dict:
        """Down-and-out call: dies if S ever touches barrier from above."""
        alive = (np.min(paths, axis=1) > barrier).astype(float)
        payoff = np.maximum(paths[:, -1] - strike, 0) * alive

        price = self._discount() * np.mean(payoff)
        std_err = self._discount() * np.std(payoff) / np.sqrt(len(payoff))
        ko_prob = 1.0 - float(np.mean(alive))
        return {"price": float(price), "std_error": float(std_err),
                "knockout_prob": ko_prob, "type": "down_and_out_call"}

    def up_and_out_put(self, paths: np.ndarray, strike: float,
                       barrier: float) -> dict:
        """Up-and-out put: dies if S ever touches barrier from below."""
        alive = (np.max(paths, axis=1) < barrier).astype(float)
        payoff = np.maximum(strike - paths[:, -1], 0) * alive

        price = self._discount() * np.mean(payoff)
        std_err = self._discount() * np.std(payoff) / np.sqrt(len(payoff))
        ko_prob = 1.0 - float(np.mean(alive))
        return {"price": float(price), "std_error": float(std_err),
                "knockout_prob": ko_prob, "type": "up_and_out_put"}

    def down_and_in_call(self, paths: np.ndarray, strike: float,
                         barrier: float) -> dict:
        """Down-and-in call: activates only if S touches barrier."""
        touched = (np.min(paths, axis=1) <= barrier).astype(float)
        payoff = np.maximum(paths[:, -1] - strike, 0) * touched

        price = self._discount() * np.mean(payoff)
        std_err = self._discount() * np.std(payoff) / np.sqrt(len(payoff))
        return {"price": float(price), "std_error": float(std_err),
                "type": "down_and_in_call"}

    # ------------------------------------------------------------------
    #  Autocallable
    # ------------------------------------------------------------------
    def autocallable(self, paths: np.ndarray, coupon_rate: float = 0.08,
                     autocall_barrier: float = 1.0,
                     ki_barrier: float = 0.6,
                     observation_freq: int = None) -> dict:
        """
        Autocallable structured product (Phoenix-style).

        At each observation date:
          - If S >= autocall_barrier * S0: early redemption, pay 100 + accrued coupon
          - At maturity: if S >= ki_barrier * S0: pay 100
                         else: pay 100 * (S_T / S0)  (capital loss)

        Args:
            paths: (n_paths, n_steps+1) spot paths
            coupon_rate: Annual coupon rate
            autocall_barrier: % of S0 for early call (1.0 = 100%)
            ki_barrier: % of S0 for knock-in put (0.6 = 60%)
            observation_freq: Steps between observations. If None, quarterly.
        """
        n_paths, n_steps_plus_1 = paths.shape
        n_steps = n_steps_plus_1 - 1
        s0 = paths[:, 0]

        if observation_freq is None:
            observation_freq = max(1, n_steps // 4)

        obs_indices = list(range(observation_freq, n_steps + 1, observation_freq))
        if n_steps not in obs_indices:
            obs_indices.append(n_steps)

        dt_per_obs = self.T / len(obs_indices)
        coupon_per_obs = coupon_rate * dt_per_obs

        payoffs = np.zeros(n_paths)
        redeemed = np.zeros(n_paths, dtype=bool)

        for i, idx in enumerate(obs_indices):
            s_obs = paths[:, idx]
            ratio = s_obs / s0

            callable_mask = (~redeemed) & (ratio >= autocall_barrier)
            accrued_coupon = coupon_per_obs * (i + 1)
            payoffs[callable_mask] = 1.0 + accrued_coupon
            time_to_obs = dt_per_obs * (i + 1)
            payoffs[callable_mask] *= np.exp(-self.r * time_to_obs)
            redeemed[callable_mask] = True

        remaining = ~redeemed
        if remaining.any():
            s_final = paths[remaining, -1]
            ratio_final = s_final / s0[remaining]
            full_coupon = coupon_rate * self.T
            mat_payoff = np.where(
                ratio_final >= ki_barrier,
                1.0 + full_coupon,
                ratio_final,
            )
            payoffs[remaining] = mat_payoff * self._discount()

        price_pct = float(np.mean(payoffs)) * 100
        std_err_pct = float(np.std(payoffs) / np.sqrt(n_paths)) * 100
        early_redemption_rate = float(redeemed.mean())

        return {
            "price_pct": price_pct,
            "std_error_pct": std_err_pct,
            "early_redemption_rate": early_redemption_rate,
            "n_observations": len(obs_indices),
            "type": "autocallable",
        }

    # ------------------------------------------------------------------
    #  Cliquet (Ratchet)
    # ------------------------------------------------------------------
    def cliquet(self, paths: np.ndarray,
                local_cap: float = 0.05,
                local_floor: float = -0.03,
                global_cap: float = None,
                global_floor: float = 0.0) -> dict:
        """
        Cliquet / Ratchet option.
        Sums locally capped/floored periodic returns.

        payoff = max(global_floor, min(global_cap, Σ capped_returns))

        Highly sensitive to vol dynamics and forward skew — ideal showcase
        for rough vol models vs Black-Scholes.
        """
        log_returns = np.diff(np.log(paths), axis=1)
        returns = np.exp(log_returns) - 1.0

        capped = np.clip(returns, local_floor, local_cap)
        total_return = np.sum(capped, axis=1)

        if global_floor is not None:
            total_return = np.maximum(total_return, global_floor)
        if global_cap is not None:
            total_return = np.minimum(total_return, global_cap)

        payoff = self.spot * total_return
        price = self._discount() * np.mean(payoff)
        std_err = self._discount() * np.std(payoff) / np.sqrt(len(payoff))

        return {"price": float(price), "std_error": float(std_err),
                "mean_capped_return": float(np.mean(total_return)),
                "type": "cliquet"}

    # ------------------------------------------------------------------
    #  Variance / Volatility swaps
    # ------------------------------------------------------------------
    def variance_swap(self, paths: np.ndarray, strike_var: float) -> dict:
        """
        Variance swap: payoff = (RV² - K_var) * notional
        RV² = (252/n) * Σ (log S_{i+1}/S_i)²
        """
        log_ret = np.diff(np.log(paths), axis=1)
        n_steps = log_ret.shape[1]
        annualization = 252.0 * (self.T * 252 / n_steps)

        realized_var = np.sum(log_ret ** 2, axis=1) * (252.0 / n_steps)
        payoff = realized_var - strike_var

        price = self._discount() * np.mean(payoff)
        std_err = self._discount() * np.std(payoff) / np.sqrt(len(payoff))
        fair_strike = float(np.mean(realized_var))

        return {"price": float(price), "std_error": float(std_err),
                "fair_strike_var": fair_strike,
                "fair_strike_vol": float(np.sqrt(fair_strike)) * 100,
                "type": "variance_swap"}

    def volatility_swap(self, paths: np.ndarray, strike_vol: float) -> dict:
        """
        Volatility swap: payoff = (RV - K_vol)
        RV = sqrt((252/n) * Σ (log S_{i+1}/S_i)²)

        Note: convexity adjustment means K_vol < sqrt(K_var).
        """
        log_ret = np.diff(np.log(paths), axis=1)
        n_steps = log_ret.shape[1]
        realized_vol = np.sqrt(np.sum(log_ret ** 2, axis=1) * (252.0 / n_steps))
        payoff = realized_vol - strike_vol

        price = self._discount() * np.mean(payoff)
        std_err = self._discount() * np.std(payoff) / np.sqrt(len(payoff))
        fair_strike = float(np.mean(realized_vol))

        return {"price": float(price), "std_error": float(std_err),
                "fair_strike_vol": fair_strike,
                "convexity_adj": float(np.mean(realized_vol ** 2) - fair_strike ** 2),
                "type": "volatility_swap"}

    # ------------------------------------------------------------------
    #  Multi-product summary
    # ------------------------------------------------------------------
    def price_all(self, paths: np.ndarray, strike: float = None) -> dict:
        """Price all exotic products for comparison across models."""
        if strike is None:
            strike = self.spot

        return {
            "asian_arith_call": self.asian_call(paths, strike, "arithmetic"),
            "asian_geom_call": self.asian_call(paths, strike, "geometric"),
            "lookback_call_fixed": self.lookback_call(paths, strike),
            "lookback_call_floating": self.lookback_call(paths),
            "barrier_doc": self.down_and_out_call(paths, strike, strike * 0.9),
            "autocallable": self.autocallable(paths),
            "cliquet": self.cliquet(paths),
            "var_swap": self.variance_swap(paths, (0.20) ** 2),
            "vol_swap": self.volatility_swap(paths, 0.20),
        }
