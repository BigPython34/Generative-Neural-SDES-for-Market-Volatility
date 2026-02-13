"""
Greeks via JAX Automatic Differentiation
=========================================
Compute option Greeks using JAX's autodiff capabilities.
Unlike finite-difference Greeks, these are exact (up to float precision).

Usage:
    from utils.greeks_ad import jax_greeks
    greeks = jax_greeks(S=100, K=100, T=0.25, r=0.05, sigma=0.2, opt_type='call')
"""

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm as jax_norm


def _bs_price_jax(S: float, K: float, T: float, r: float, sigma: float, 
                  is_call: bool = True) -> jnp.ndarray:
    """
    JAX-differentiable Black-Scholes price.
    All inputs are scalars. Returns scalar price.
    """
    d1 = (jnp.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * jnp.sqrt(T))
    d2 = d1 - sigma * jnp.sqrt(T)
    
    call_price = S * jax_norm.cdf(d1) - K * jnp.exp(-r * T) * jax_norm.cdf(d2)
    put_price = K * jnp.exp(-r * T) * jax_norm.cdf(-d2) - S * jax_norm.cdf(-d1)
    
    return jnp.where(is_call, call_price, put_price)


def _price_wrt_spot(S, K, T, r, sigma, is_call):
    """Price as function of spot only (for delta/gamma)."""
    return _bs_price_jax(S, K, T, r, sigma, is_call)


def _price_wrt_vol(S, K, T, r, sigma, is_call):
    """Price as function of vol only (for vega/volga)."""
    return _bs_price_jax(S, K, T, r, sigma, is_call)


def _price_wrt_time(S, K, T, r, sigma, is_call):
    """Price as function of T only (for theta)."""
    return _bs_price_jax(S, K, T, r, sigma, is_call)


def _price_wrt_spot_vol(S, K, T, r, sigma, is_call):
    """Price as function of (S, sigma) for vanna."""
    return _bs_price_jax(S, K, T, r, sigma, is_call)


def jax_greeks(S: float, K: float, T: float, r: float, sigma: float, 
               opt_type: str = 'call') -> dict:
    """
    Compute all Greeks via JAX automatic differentiation.
    
    Args:
        S: Spot price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate
        sigma: Volatility
        opt_type: 'call' or 'put'
        
    Returns:
        Dictionary with: price, delta, gamma, vega, theta, vanna, volga
    """
    is_call = (opt_type == 'call')
    S, K, T, r, sigma = map(float, [S, K, T, r, sigma])
    
    # Price
    price = float(_bs_price_jax(S, K, T, r, sigma, is_call))
    
    # Delta = ∂C/∂S
    delta_fn = jax.grad(_price_wrt_spot, argnums=0)
    delta = float(delta_fn(S, K, T, r, sigma, is_call))
    
    # Gamma = ∂²C/∂S²
    gamma_fn = jax.grad(jax.grad(_price_wrt_spot, argnums=0), argnums=0)
    gamma = float(gamma_fn(S, K, T, r, sigma, is_call))
    
    # Vega = ∂C/∂σ (per 1% move → divide by 100)
    vega_fn = jax.grad(_price_wrt_vol, argnums=4)
    vega = float(vega_fn(S, K, T, r, sigma, is_call)) / 100.0
    
    # Theta = -∂C/∂T (per calendar day → divide by 365)
    theta_fn = jax.grad(_price_wrt_time, argnums=2)
    theta = -float(theta_fn(S, K, T, r, sigma, is_call)) / 365.0
    
    # Vanna = ∂²C/∂S∂σ
    vanna_fn = jax.grad(jax.grad(_price_wrt_spot_vol, argnums=0), argnums=4)
    vanna = float(vanna_fn(S, K, T, r, sigma, is_call))
    
    # Volga = ∂²C/∂σ²
    volga_fn = jax.grad(jax.grad(_price_wrt_vol, argnums=4), argnums=4)
    volga = float(volga_fn(S, K, T, r, sigma, is_call))
    
    return {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta,
        'vanna': vanna,
        'volga': volga,
    }


def jax_greeks_vectorized(S: jnp.ndarray, K: jnp.ndarray, T: float, r: float, 
                           sigma: jnp.ndarray, opt_type: str = 'call') -> dict:
    """
    Vectorized Greeks computation for multiple strikes/vols.
    
    Args:
        S: Spot price (scalar)
        K: Array of strikes
        T: Time to maturity
        r: Risk-free rate
        sigma: Array of implied vols (same shape as K)
        opt_type: 'call' or 'put'
    """
    def single_greeks(k, sig):
        return jax_greeks(float(S), float(k), T, r, float(sig), opt_type)
    
    results = [single_greeks(k, s) for k, s in zip(K, sigma)]
    
    return {
        key: jnp.array([r[key] for r in results])
        for key in results[0].keys()
    }


if __name__ == "__main__":
    # Quick test
    print("JAX Autodiff Greeks Test")
    print("=" * 50)
    
    greeks = jax_greeks(S=100, K=100, T=0.25, r=0.05, sigma=0.20, opt_type='call')
    
    print(f"ATM Call (S=100, K=100, T=3m, σ=20%):")
    for name, value in greeks.items():
        print(f"  {name:>8s}: {value:+.6f}")
    
    print()
    greeks_put = jax_greeks(S=100, K=100, T=0.25, r=0.05, sigma=0.20, opt_type='put')
    
    print(f"ATM Put:")
    for name, value in greeks_put.items():
        print(f"  {name:>8s}: {value:+.6f}")
    
    # Verify put-call parity
    import numpy as np
    pcp = greeks['price'] - greeks_put['price'] - (100 - 100 * np.exp(-0.05 * 0.25))
    print(f"\nPut-Call Parity check: {pcp:.8f} (should be ~0)")
