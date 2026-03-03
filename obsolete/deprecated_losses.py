"""
Deprecated loss functions moved from engine/losses.py.

These are kept for reference only — they are NOT used in the current
training pipeline and should not be imported.

History:
  - feller_condition_loss: Only relevant for CIR/Heston variance-space models.
    Current backbone operates in log-variance, where Feller is automatically satisfied.
  - path_regularity_loss: Penalizes rough increments — directly contradicts the
    rough volatility hypothesis (H ≈ 0.1).
  - mean_penalty_loss_logv: Correct idea (Jensen-safe matching in log-V space)
    but the logic was manually inlined in the trainer, so this function is never called.
"""

import jax.numpy as jnp
from jax import jit


@jit
def feller_condition_loss(kappa: float, theta: float,
                          vol_of_vol: float) -> jnp.ndarray:
    """Penalizes Feller condition violation: 2κθ > σ²."""
    return jnp.maximum(0.0, vol_of_vol ** 2 - 2 * kappa * theta) ** 2


@jit
def path_regularity_loss(paths: jnp.ndarray) -> jnp.ndarray:
    """Penalizes path increments (HARMFUL for rough vol models)."""
    increments = jnp.diff(paths, axis=1)
    return jnp.mean(jnp.square(increments))


@jit
def mean_penalty_loss_logv(fake_paths: jnp.ndarray,
                           real_log_mean: float) -> jnp.ndarray:
    """Mean penalty in log-variance space (avoids Jensen bias).
    NOTE: This logic is inlined in the trainer — this function is never called."""
    fake_log = jnp.log(jnp.maximum(fake_paths, 1e-10))
    return jnp.square(jnp.mean(fake_log) - real_log_mean)
