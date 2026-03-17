"""Shared Volterra kernel utilities for rough-volatility simulators."""

from __future__ import annotations

import jax.numpy as jnp


def build_volterra_kernel_jax(n_steps: int, dt: float, H: jnp.ndarray):
    """
    Build RL-fBM Volterra kernel (differentiable in ``H``).

    A[j,k] = sqrt(2H) * dt^H / (H + 0.5) * ((j-k+1)^(H+0.5) - (j-k)^(H+0.5))
    for k <= j.
    """
    C = jnp.sqrt(2 * H) * dt ** H / (H + 0.5)
    alpha = H + 0.5

    j_idx = jnp.arange(n_steps)[:, None]
    k_idx = jnp.arange(n_steps)[None, :]
    lag = j_idx - k_idx

    mask = (lag >= 0).astype(jnp.float32)
    safe_lag = jnp.maximum(lag, 0)
    A = C * ((safe_lag + 1) ** alpha - safe_lag ** alpha) * mask
    var_wh = jnp.sum(A ** 2, axis=1)
    return A, var_wh


_VOL_KERNEL_CACHE: dict[tuple[int, float, float], tuple[jnp.ndarray, jnp.ndarray]] = {}


def get_cached_volterra_kernel(n_steps: int, H: float, dt: float):
    """Cache wrapper for scalar-parameter calibration loops."""
    key = (int(n_steps), round(float(H), 5), round(float(dt), 10))
    if key not in _VOL_KERNEL_CACHE:
        _VOL_KERNEL_CACHE[key] = build_volterra_kernel_jax(
            int(n_steps), float(dt), jnp.array(float(H), dtype=jnp.float32)
        )
    return _VOL_KERNEL_CACHE[key]


def clear_volterra_kernel_cache():
    """Clear cached Volterra kernels."""
    _VOL_KERNEL_CACHE.clear()
