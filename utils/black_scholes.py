"""
Black-Scholes Utilities
=======================
Single source of truth for BS pricing, implied vol, and Greeks.
Includes both Newton-Raphson and rational approximation IV solvers.

The rational approximation (inspired by Jaeckel 2017) provides a robust
initial guess that handles deep OTM / short maturity cases where
naive Newton-Raphson diverges.
"""

import numpy as np
from scipy.stats import norm


class BlackScholes:
    """Black-Scholes pricing, implied volatility, and Greeks."""

    # ------------------------------------------------------------------
    #  Pricing
    # ------------------------------------------------------------------
    @staticmethod
    def price(S, K, T, r, sigma, opt_type='call'):
        """Black-Scholes option price."""
        if T <= 0:
            return max(S - K, 0) if opt_type == 'call' else max(K - S, 0)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if opt_type == 'call':
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    # ------------------------------------------------------------------
    #  Implied Volatility — hybrid solver
    # ------------------------------------------------------------------
    @staticmethod
    def implied_vol(price, S, K, T, r, opt_type='call', max_iter=100,
                    tol=1e-10) -> float:
        """
        Extract implied volatility via hybrid rational-guess + Newton-Raphson.

        1) Rational initial guess handles extreme moneyness
        2) Newton-Raphson refines to machine precision
        3) Fallback to bisection if Newton diverges
        """
        if price <= 0 or T <= 0 or S <= 0 or K <= 0:
            return np.nan

        F = S * np.exp(r * T)
        x = np.log(F / K)
        p = price * np.exp(r * T)  # undiscounted price

        intrinsic = max(F - K, 0) if opt_type == 'call' else max(K - F, 0)
        if p <= intrinsic + 1e-12:
            return np.nan

        time_value_upper = min(F, K)
        if p >= time_value_upper - 1e-12:
            return np.nan

        # Step 1: rational initial guess
        sigma = BlackScholes._rational_iv_guess(p, F, K, T, opt_type)
        sigma = np.clip(sigma, 0.005, 5.0)

        # Step 2: Newton-Raphson refinement
        for _ in range(max_iter):
            bs_price = BlackScholes.price(S, K, T, r, sigma, opt_type)
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            vega = S * np.sqrt(T) * norm.pdf(d1)

            if vega < 1e-14:
                break
            diff = bs_price - price
            if abs(diff) < tol:
                return sigma
            sigma -= diff / vega
            sigma = np.clip(sigma, 0.005, 5.0)

        # Step 3: bisection fallback
        return BlackScholes._bisection_iv(price, S, K, T, r, opt_type)

    @staticmethod
    def _rational_iv_guess(undiscounted_price, F, K, T, opt_type) -> float:
        """
        Rational approximation for initial IV guess.
        Based on normalized price beta = p / F (call) or p / K (put)
        and Brenner-Subrahmanyam ATM approximation: σ ≈ p·sqrt(2π) / (F·sqrt(T)).
        """
        if opt_type == 'call':
            beta = undiscounted_price / F
        else:
            beta = undiscounted_price / K

        beta = np.clip(beta, 1e-10, 0.99)

        sigma_bs = np.sqrt(2.0 * np.pi / T) * beta

        x = np.log(F / K)
        if abs(x) > 0.01:
            atm_adj = abs(x) / np.sqrt(T)
            sigma_bs = max(sigma_bs, atm_adj * 0.5)

        return np.clip(sigma_bs, 0.01, 3.0)

    @staticmethod
    def _bisection_iv(price, S, K, T, r, opt_type, n_iter=80) -> float:
        """Robust bisection fallback for IV extraction."""
        lo, hi = 0.005, 5.0
        for _ in range(n_iter):
            mid = (lo + hi) / 2.0
            p_mid = BlackScholes.price(S, K, T, r, mid, opt_type)
            if p_mid > price:
                hi = mid
            else:
                lo = mid
            if hi - lo < 1e-10:
                break
        return (lo + hi) / 2.0

    # ------------------------------------------------------------------
    #  d1 / d2
    # ------------------------------------------------------------------
    @staticmethod
    def d1(S, K, T, r, sigma):
        return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    @staticmethod
    def d2(S, K, T, r, sigma):
        return BlackScholes.d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

    # ------------------------------------------------------------------
    #  Greeks
    # ------------------------------------------------------------------
    @staticmethod
    def delta(S, K, T, r, sigma, opt_type='call'):
        if T <= 0:
            if opt_type == 'call':
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        if opt_type == 'call':
            return float(norm.cdf(d1))
        else:
            return float(norm.cdf(d1) - 1)

    @staticmethod
    def gamma(S, K, T, r, sigma):
        if T <= 0:
            return 0.0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        return float(norm.pdf(d1) / (S * sigma * np.sqrt(T)))

    @staticmethod
    def vega(S, K, T, r, sigma):
        """Vega per 1% vol move."""
        if T <= 0:
            return 0.0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        return float(S * np.sqrt(T) * norm.pdf(d1) / 100)

    @staticmethod
    def theta(S, K, T, r, sigma, opt_type='call'):
        """Theta per calendar day."""
        if T <= 0:
            return 0.0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        common = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
        if opt_type == 'call':
            return float((common - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365)
        else:
            return float((common + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365)

    @staticmethod
    def vanna(S, K, T, r, sigma):
        """Vanna: ∂²C/∂S∂σ = ∂Δ/∂σ."""
        if T <= 0:
            return 0.0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        return float(-norm.pdf(d1) * d2 / sigma)

    @staticmethod
    def volga(S, K, T, r, sigma):
        """Volga/Vomma: ∂²C/∂σ²."""
        if T <= 0:
            return 0.0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        vega_val = S * np.sqrt(T) * norm.pdf(d1)
        return float(vega_val * d1 * d2 / sigma)

    @staticmethod
    def rho(S, K, T, r, sigma, opt_type='call'):
        """Rho: ∂C/∂r, per 1% rate move."""
        if T <= 0:
            return 0.0
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        if opt_type == 'call':
            return float(K * T * np.exp(-r * T) * norm.cdf(d2) / 100)
        else:
            return float(-K * T * np.exp(-r * T) * norm.cdf(-d2) / 100)
