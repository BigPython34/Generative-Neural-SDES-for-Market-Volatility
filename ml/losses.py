import jax.numpy as jnp
from jax import jit

@jit
def martingale_violation_loss(paths: jnp.ndarray, dt: float, r: float = 0.0) -> jnp.ndarray:
    """
    Computes the squared deviation from the Martingale property.
    Under Risk-Neutral measure, E[S_t] should be S_0 (1.0).
    """
    n_steps = paths.shape[1]
    time_grid = jnp.linspace(0, (n_steps - 1) * dt, n_steps)
    discount_factors = jnp.exp(-r * time_grid)
    
    # S_tilde = exp(-rt) * S_t
    discounted_s = paths * discount_factors
    
    # Calculate empirical mean across all paths at each time step
    mean_path = jnp.mean(discounted_s, axis=0)
    
    # Target is S_0 = 1.0. We penalize the L2 distance.
    return jnp.mean(jnp.square(mean_path - 1.0))
@jit
def feller_condition_loss(kappa: float, theta: float, vol_of_vol: float) -> jnp.ndarray:
    """
    Optional: Penalizes if the Feller condition (2*kappa*theta > vol_of_vol^2) 
    is violated (useful for Heston-like variance processes).
    """
    return jnp.maximum(0.0, vol_of_vol**2 - 2 * kappa * theta)**2

@jit
def signature_mmd_loss(fake_signatures: jnp.ndarray, real_signatures: jnp.ndarray, 
                        sig_std: jnp.ndarray = None) -> jnp.ndarray:
    """
    Computes the MMD distance in the NORMALIZED Signature Space.
    
    Each component is divided by its empirical std from the real data,
    so that all dimensions contribute equally to the loss regardless
    of their physical scale.
    
    Args:
        fake_signatures: Shape (batch_size, sig_dim)
        real_signatures: Shape (batch_size, sig_dim)
        sig_std: Component-wise std of real signatures for normalization.
                 If None, falls back to unnormalized L2 (backward compatible).
    """
    # 1. Compute the "Expected Signature"
    mu_fake = jnp.mean(fake_signatures, axis=0)
    mu_real = jnp.mean(real_signatures, axis=0)
    
    # 2. Normalize by component-wise std (makes all dimensions comparable)
    if sig_std is not None:
        # Pure-time components (s1_t, s2_tt, s3_ttt) have near-zero variance
        # because time is deterministic. Ignore them in the loss by setting
        # weight to 0 for components with std below threshold.
        threshold = 1e-8
        weights = jnp.where(sig_std > threshold, 1.0 / sig_std, 0.0)
        diff = (mu_fake - mu_real) * weights
    else:
        diff = mu_fake - mu_real
    
    # 3. Minimize the normalized Euclidean distance
    loss = jnp.sum(jnp.square(diff))
    
    return loss

@jit
def mean_penalty_loss(fake_paths: jnp.ndarray, real_mean: float) -> jnp.ndarray:
    """
    Penalizes deviation of the generated mean variance from the real mean.
    
    This prevents the Jensen inequality bias: when the model generates
    log-variance, exp(log_v) systematically overshoots if the variance
    of log_v is too large. This penalty corrects that drift.
    
    Args:
        fake_paths: Generated variance paths, shape (batch, n_steps)
        real_mean: Mean of real variance data (scalar)
    """
    fake_mean = jnp.mean(fake_paths)
    return jnp.square(fake_mean - real_mean)