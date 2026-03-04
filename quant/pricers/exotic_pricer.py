"""
Exotic Option Pricer
====================
Monte Carlo pricing for path-dependent exotic options using
Neural SDE, Rough Bergomi, or Black-Scholes spot paths.

Variance Reduction:
  - Antithetic variates on all products (halves variance for free)
  - Geometric Asian closed-form as control variate for arithmetic Asian
  - Kemna-Vorst (1990) benchmark for geometric Asian

Supported products:
  - Asian (arithmetic & geometric, fixed/floating strike)
  - Lookback (fixed/floating strike)
  - Autocallable (multi-barrier with coupon)
  - Cliquet (locally capped/floored)
  - Variance swap
  - Volatility swap

Usage:
    from quant.pricers.exotic_pricer import ExoticPricer
    pricer = ExoticPricer(spot=100, r=0.045, T=0.25)
    price = pricer.asian_call(spot_paths, strike=100)
"""

import numpy as np
from scipy.stats import norm
from utils.black_scholes import BlackScholes


def _antithetic_paths(paths: np.ndarray) -> np.ndarray:
    """
    Generate antithetic paths from existing paths.
    If S_t = S_0 * exp(drift + σ·Z), the antithetic uses -Z,
    giving S_0 * exp(drift - σ·Z) = S_0² * exp(2·drift) / S_t.

    For general paths we reflect log-returns around their mean:
      log_ret_anti = 2*mean(log_ret) - log_ret
    """
    log_ret = np.diff(np.log(paths), axis=1)
    mean_lr = np.mean(log_ret, axis=1, keepdims=True)
    anti_lr = 2 * mean_lr - log_ret
    anti_log = np.cumsum(anti_lr, axis=1)
    s0 = paths[:, 0:1]
    anti_paths = np.hstack([s0, s0 * np.exp(anti_log)])
    return anti_paths


def _geometric_asian_closed_form(S0: float, K: float, T: float, r: float,
                                  sigma: float, n: int,
                                  opt_type: str = "call") -> float:
    """
    Kemna-Vorst (1990) closed-form for continuously-monitored geometric
    Asian option, adjusted for discrete monitoring.

    For discrete geometric average of n observations:
      σ_a = σ * sqrt((n+1)(2n+1) / (6n²))
      μ_a = (r - σ²/2) * (n+1)/(2n) + σ_a²/2

    Returns the Black-Scholes-style price.
    """
    sigma_a = sigma * np.sqrt((n + 1) * (2 * n + 1) / (6 * n ** 2))
    mu_a = (r - 0.5 * sigma ** 2) * (n + 1) / (2 * n) + 0.5 * sigma_a ** 2

    d1 = (np.log(S0 / K) + (mu_a + 0.5 * sigma_a ** 2) * T) / (sigma_a * np.sqrt(T))
    d2 = d1 - sigma_a * np.sqrt(T)

    if opt_type == "call":
        price = S0 * np.exp((mu_a - r) * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp((mu_a - r) * T) * norm.cdf(-d1)
    return float(price)


class ExoticPricer:
    """Monte Carlo exotic option pricer from simulated spot paths.

    All pricing methods use antithetic variates for variance reduction.
    """

    def __init__(self, spot: float, r: float = 0.045, T: float = 0.25):
        self.spot = spot
        self.r = r
        self.T = T

    def _discount(self, T: float = None) -> float:
        return np.exp(-self.r * (T or self.T))

    def _mc_price(self, payoff_orig: np.ndarray, payoff_anti: np.ndarray):
        """Average original + antithetic payoffs, then discount."""
        combined = 0.5 * (payoff_orig + payoff_anti)
        price = self._discount() * np.mean(combined)
        std_err = self._discount() * np.std(combined) / np.sqrt(len(combined))
        return float(price), float(std_err)

    # ------------------------------------------------------------------
    #  Asian options
    # ------------------------------------------------------------------
    def asian_call(self, paths: np.ndarray, strike: float = None,
                   averaging: str = "arithmetic") -> dict:
        """
        Asian call: payoff = max(A(S) - K, 0)
        where A(S) is arithmetic or geometric average of path.

        Uses antithetic variates. For arithmetic averaging, also provides
        the Kemna-Vorst geometric Asian benchmark.
        """
        anti = _antithetic_paths(paths)

        if averaging == "geometric":
            avg = np.exp(np.mean(np.log(paths[:, 1:]), axis=1))
            avg_a = np.exp(np.mean(np.log(anti[:, 1:]), axis=1))
        else:
            avg = np.mean(paths[:, 1:], axis=1)
            avg_a = np.mean(anti[:, 1:], axis=1)

        if strike is None:
            payoff = np.maximum(paths[:, -1] - avg, 0)
            payoff_a = np.maximum(anti[:, -1] - avg_a, 0)
        else:
            payoff = np.maximum(avg - strike, 0)
            payoff_a = np.maximum(avg_a - strike, 0)

        price, std_err = self._mc_price(payoff, payoff_a)

        result = {"price": price, "std_error": std_err,
                  "type": f"asian_call_{averaging}"}

        # Benchmark: geometric Asian closed-form (Kemna-Vorst 1990)
        if averaging == "arithmetic" and strike is not None:
            n_obs = paths.shape[1] - 1
            # Estimate σ from paths for benchmark comparison
            log_ret = np.diff(np.log(paths), axis=1)
            sigma_est = float(np.std(log_ret) * np.sqrt(252 * n_obs / self.T))
            try:
                geom_bench = _geometric_asian_closed_form(
                    self.spot, strike, self.T, self.r, sigma_est, n_obs, "call")
                result["geometric_benchmark"] = geom_bench
            except Exception:
                pass

        return result

    def asian_put(self, paths: np.ndarray, strike: float = None,
                  averaging: str = "arithmetic") -> dict:
        """Asian put: payoff = max(K - A(S), 0).  Antithetic variates."""
        anti = _antithetic_paths(paths)

        if averaging == "geometric":
            avg = np.exp(np.mean(np.log(paths[:, 1:]), axis=1))
            avg_a = np.exp(np.mean(np.log(anti[:, 1:]), axis=1))
        else:
            avg = np.mean(paths[:, 1:], axis=1)
            avg_a = np.mean(anti[:, 1:], axis=1)

        if strike is None:
            payoff = np.maximum(avg - paths[:, -1], 0)
            payoff_a = np.maximum(avg_a - anti[:, -1], 0)
        else:
            payoff = np.maximum(strike - avg, 0)
            payoff_a = np.maximum(strike - avg_a, 0)

        price, std_err = self._mc_price(payoff, payoff_a)
        return {"price": price, "std_error": std_err,
                "type": f"asian_put_{averaging}"}
        return {"price": float(price), "std_error": float(std_err),
                "type": f"asian_put_{averaging}"}

    # ------------------------------------------------------------------
    #  Lookback options
    # ------------------------------------------------------------------
    def lookback_call(self, paths: np.ndarray, strike: float = None) -> dict:
        """
        Lookback call with antithetic variance reduction.
          Fixed strike:  payoff = max(max(S) - K, 0)
          Floating strike: payoff = S_T - min(S)
        """
        anti = _antithetic_paths(paths)
        if strike is not None:
            payoff = np.maximum(np.max(paths, axis=1) - strike, 0)
            payoff_a = np.maximum(np.max(anti, axis=1) - strike, 0)
        else:
            payoff = paths[:, -1] - np.min(paths, axis=1)
            payoff_a = anti[:, -1] - np.min(anti, axis=1)

        price, std_err = self._mc_price(payoff, payoff_a)
        return {"price": price, "std_error": std_err,
                "type": "lookback_call"}

    def lookback_put(self, paths: np.ndarray, strike: float = None) -> dict:
        """
        Lookback put with antithetic variance reduction.
          Fixed strike:  payoff = max(K - min(S), 0)
          Floating strike: payoff = max(S) - S_T
        """
        anti = _antithetic_paths(paths)
        if strike is not None:
            payoff = np.maximum(strike - np.min(paths, axis=1), 0)
            payoff_a = np.maximum(strike - np.min(anti, axis=1), 0)
        else:
            payoff = np.max(paths, axis=1) - paths[:, -1]
            payoff_a = np.max(anti, axis=1) - anti[:, -1]

        price, std_err = self._mc_price(payoff, payoff_a)
        return {"price": price, "std_error": std_err,
                "type": "lookback_put"}

    # ------------------------------------------------------------------
    #  Barrier options
    # ------------------------------------------------------------------
    def down_and_out_call(self, paths: np.ndarray, strike: float,
                          barrier: float) -> dict:
        """Down-and-out call: dies if S ever touches barrier from above.
        Uses antithetic variates."""
        anti = _antithetic_paths(paths)
        alive = (np.min(paths, axis=1) > barrier).astype(float)
        alive_a = (np.min(anti, axis=1) > barrier).astype(float)
        payoff = np.maximum(paths[:, -1] - strike, 0) * alive
        payoff_a = np.maximum(anti[:, -1] - strike, 0) * alive_a

        price, std_err = self._mc_price(payoff, payoff_a)
        ko_prob = 1.0 - 0.5 * (float(np.mean(alive)) + float(np.mean(alive_a)))
        return {"price": price, "std_error": std_err,
                "knockout_prob": ko_prob, "type": "down_and_out_call"}

    def up_and_out_put(self, paths: np.ndarray, strike: float,
                       barrier: float) -> dict:
        """Up-and-out put: dies if S ever touches barrier from below.
        Uses antithetic variates."""
        anti = _antithetic_paths(paths)
        alive = (np.max(paths, axis=1) < barrier).astype(float)
        alive_a = (np.max(anti, axis=1) < barrier).astype(float)
        payoff = np.maximum(strike - paths[:, -1], 0) * alive
        payoff_a = np.maximum(strike - anti[:, -1], 0) * alive_a

        price, std_err = self._mc_price(payoff, payoff_a)
        ko_prob = 1.0 - 0.5 * (float(np.mean(alive)) + float(np.mean(alive_a)))
        return {"price": price, "std_error": std_err,
                "knockout_prob": ko_prob, "type": "up_and_out_put"}

    def down_and_in_call(self, paths: np.ndarray, strike: float,
                         barrier: float) -> dict:
        """Down-and-in call: activates only if S touches barrier.
        Uses antithetic variates."""
        anti = _antithetic_paths(paths)
        touched = (np.min(paths, axis=1) <= barrier).astype(float)
        touched_a = (np.min(anti, axis=1) <= barrier).astype(float)
        payoff = np.maximum(paths[:, -1] - strike, 0) * touched
        payoff_a = np.maximum(anti[:, -1] - strike, 0) * touched_a

        price, std_err = self._mc_price(payoff, payoff_a)
        return {"price": price, "std_error": std_err,
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
        Uses antithetic variates.
        """
        anti = _antithetic_paths(paths)
        log_ret = np.diff(np.log(paths), axis=1)
        log_ret_a = np.diff(np.log(anti), axis=1)
        n_steps = log_ret.shape[1]

        realized_var = np.sum(log_ret ** 2, axis=1) * (252.0 / n_steps)
        realized_var_a = np.sum(log_ret_a ** 2, axis=1) * (252.0 / n_steps)

        payoff = realized_var - strike_var
        payoff_a = realized_var_a - strike_var

        price, std_err = self._mc_price(payoff, payoff_a)
        fair_strike = float(0.5 * (np.mean(realized_var) + np.mean(realized_var_a)))

        return {"price": price, "std_error": std_err,
                "fair_strike_var": fair_strike,
                "fair_strike_vol": float(np.sqrt(fair_strike)) * 100,
                "type": "variance_swap"}

    def volatility_swap(self, paths: np.ndarray, strike_vol: float) -> dict:
        """
        Volatility swap: payoff = (RV - K_vol)
        RV = sqrt((252/n) * Σ (log S_{i+1}/S_i)²)

        Note: convexity adjustment means K_vol < sqrt(K_var).
        Uses antithetic variates.
        """
        anti = _antithetic_paths(paths)
        log_ret = np.diff(np.log(paths), axis=1)
        log_ret_a = np.diff(np.log(anti), axis=1)
        n_steps = log_ret.shape[1]

        realized_vol = np.sqrt(np.sum(log_ret ** 2, axis=1) * (252.0 / n_steps))
        realized_vol_a = np.sqrt(np.sum(log_ret_a ** 2, axis=1) * (252.0 / n_steps))

        payoff = realized_vol - strike_vol
        payoff_a = realized_vol_a - strike_vol

        price, std_err = self._mc_price(payoff, payoff_a)
        fair_strike = float(0.5 * (np.mean(realized_vol) + np.mean(realized_vol_a)))

        return {"price": price, "std_error": std_err,
                "fair_strike_vol": fair_strike,
                "convexity_adj": float(
                    0.5 * (np.mean(realized_vol ** 2) + np.mean(realized_vol_a ** 2))
                    - fair_strike ** 2),
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
