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
def signature_mmd_loss(fake_signatures: jnp.ndarray, real_signatures: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the MMD distance in the Signature Space (The "Signature Kernel").
    
    Theory: If E[Sig(Fake)] == E[Sig(Real)] for all orders, 
    then Law(Fake) == Law(Real).
    
    Args:
        fake_signatures: Shape (batch_size, sig_dim)
        real_signatures: Shape (batch_size, sig_dim)
    """
    # 1. Compute the "Expected Signature" (The geometric mean of the market)
    mu_fake = jnp.mean(fake_signatures, axis=0)
    mu_real = jnp.mean(real_signatures, axis=0)
    
    # 2. Minimize the Euclidean distance between these expected signatures
    # This forces the Neural SDE to match all statistical moments (vol, skew, kurtosis...)
    loss = jnp.sum(jnp.square(mu_fake - mu_real))
    
    return loss