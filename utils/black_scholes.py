"""
Shared Black-Scholes Utilities
==============================
Single source of truth for BS pricing, implied vol, and Greeks.
Used across all calibration, backtesting, and pricing modules.
"""

import numpy as np
from scipy.stats import norm


class BlackScholes:
    """Black-Scholes pricing, implied volatility, and Greeks."""

    @staticmethod
    def price(S, K, T, r, sigma, opt_type='call'):
        """
        Black-Scholes option price.
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate
            sigma: Volatility
            opt_type: 'call' or 'put'
        """
        if T <= 0:
            return max(S - K, 0) if opt_type == 'call' else max(K - S, 0)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if opt_type == 'call':
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    @staticmethod
    def implied_vol(price, S, K, T, r, opt_type='call', max_iter=50):
        """
        Extract implied volatility from option price via Newton-Raphson.
        
        Args:
            price: Market option price
            S: Spot price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate
            opt_type: 'call' or 'put'
            max_iter: Maximum Newton iterations
            
        Returns:
            Implied volatility (annualized)
        """
        if price <= 0 or T <= 0:
            return np.nan
        sigma = 0.3
        for _ in range(max_iter):
            bs_price = BlackScholes.price(S, K, T, r, sigma, opt_type)
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            vega = S * np.sqrt(T) * norm.pdf(d1)
            if vega < 1e-10:
                break
            diff = bs_price - price
            if abs(diff) < 1e-8:
                break
            sigma -= diff / vega
            sigma = np.clip(sigma, 0.01, 3.0)
        return sigma

    @staticmethod
    def d1(S, K, T, r, sigma):
        """Compute d1."""
        return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    @staticmethod
    def d2(S, K, T, r, sigma):
        """Compute d2."""
        return BlackScholes.d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

    # --- Greeks ---

    @staticmethod
    def delta(S, K, T, r, sigma, opt_type='call'):
        """Option delta (∂C/∂S)."""
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
        """Option gamma (∂²C/∂S²). Same for call and put."""
        if T <= 0:
            return 0.0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        return float(norm.pdf(d1) / (S * sigma * np.sqrt(T)))

    @staticmethod
    def vega(S, K, T, r, sigma):
        """Option vega (∂C/∂σ). Same for call and put. Returns per 1% move."""
        if T <= 0:
            return 0.0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        return float(S * np.sqrt(T) * norm.pdf(d1) / 100)

    @staticmethod
    def theta(S, K, T, r, sigma, opt_type='call'):
        """Option theta (∂C/∂t). Returns per calendar day."""
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
        """Option vanna (∂²C/∂S∂σ = ∂Δ/∂σ)."""
        if T <= 0:
            return 0.0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        return float(-norm.pdf(d1) * d2 / sigma)

    @staticmethod
    def volga(S, K, T, r, sigma):
        """Option volga/vomma (∂²C/∂σ² = ∂vega/∂σ)."""
        if T <= 0:
            return 0.0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        vega_val = S * np.sqrt(T) * norm.pdf(d1)
        return float(vega_val * d1 * d2 / sigma)
