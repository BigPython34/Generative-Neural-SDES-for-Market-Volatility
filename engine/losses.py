import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

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


def _rbf_kernel(x: jnp.ndarray, y: jnp.ndarray, bandwidth: float) -> jnp.ndarray:
    """
    RBF (Gaussian) kernel: k(x, y) = exp(-||x-y||² / (2 * bandwidth²))

    Args:
        x: (n, d) matrix
        y: (m, d) matrix
        bandwidth: Kernel bandwidth (sigma)
    Returns:
        (n, m) kernel matrix
    """
    # ||x_i - y_j||² = ||x_i||² + ||y_j||² - 2 * x_i·y_j
    x_sq = jnp.sum(x ** 2, axis=1, keepdims=True)  # (n, 1)
    y_sq = jnp.sum(y ** 2, axis=1, keepdims=True)  # (m, 1)
    cross = x @ y.T                                  # (n, m)
    dist_sq = x_sq + y_sq.T - 2 * cross              # (n, m)
    return jnp.exp(-dist_sq / (2 * bandwidth ** 2))


@partial(jit, static_argnames=['n_bandwidths'])
def kernel_mmd_loss(fake_signatures: jnp.ndarray, real_signatures: jnp.ndarray,
                    sig_std: jnp.ndarray = None,
                    n_bandwidths: int = 5) -> jnp.ndarray:
    """
    True kernel MMD² (Maximum Mean Discrepancy) with multi-scale RBF kernel.

    MMD²(P, Q) = E[k(x,x')] - 2*E[k(x,y)] + E[k(y,y')]

    where x ~ P (real), y ~ Q (generated), k is a sum of RBF kernels
    at multiple bandwidths (median heuristic * {0.25, 0.5, 1, 2, 4}).

    This compares the FULL DISTRIBUTION of signatures, not just their means.
    It captures higher-order moments and cross-correlations.

    Args:
        fake_signatures: (batch_fake, sig_dim) generated path signatures
        real_signatures: (batch_real, sig_dim) real market path signatures
        sig_std: Component-wise std for normalization (from real data)
        n_bandwidths: Number of bandwidth scales (default 5)
    """
    # 1. Normalize signatures
    if sig_std is not None:
        threshold = 1e-8
        weights = jnp.where(sig_std > threshold, 1.0 / sig_std, 0.0)
        x = real_signatures * weights   # (n_real, d)
        y = fake_signatures * weights   # (n_fake, d)
    else:
        x = real_signatures
        y = fake_signatures

    # 2. Median heuristic: use OFF-DIAGONAL distances only (diagonal zeros bias median down)
    all_pts = jnp.concatenate([x, y], axis=0)
    n_total = all_pts.shape[0]
    dists_sq = (jnp.sum(all_pts ** 2, axis=1, keepdims=True)
                + jnp.sum(all_pts ** 2, axis=1, keepdims=True).T
                - 2 * all_pts @ all_pts.T)
    dists_sq_no_diag = jnp.where(
        jnp.eye(n_total, dtype=bool), jnp.nan, dists_sq
    )
    median_dist = jnp.sqrt(jnp.maximum(jnp.nanmedian(dists_sq_no_diag), 1e-8))

    # 3. Log-spaced bandwidth scales from n_bandwidths
    scales = jnp.logspace(
        jnp.log10(0.25), jnp.log10(4.0), num=n_bandwidths, base=10.0
    )
    bandwidths = median_dist * scales

    mmd_sq = jnp.float32(0.0)

    def add_bandwidth_mmd(carry, bw):
        # Kernel matrices
        K_xx = _rbf_kernel(x, x, bw)  # (n_real, n_real)
        K_yy = _rbf_kernel(y, y, bw)  # (n_fake, n_fake)
        K_xy = _rbf_kernel(x, y, bw)  # (n_real, n_fake)

        # Biased MMD² estimate (always >= 0, better for optimization)
        term_xx = jnp.mean(K_xx)
        term_yy = jnp.mean(K_yy)
        term_xy = jnp.mean(K_xy)

        mmd_bw = term_xx + term_yy - 2 * term_xy
        return carry + mmd_bw, None

    mmd_sq, _ = jax.lax.scan(add_bandwidth_mmd, mmd_sq, bandwidths)

    return mmd_sq


@jit
def signature_mmd_loss(fake_signatures: jnp.ndarray, real_signatures: jnp.ndarray,
                        sig_std: jnp.ndarray = None) -> jnp.ndarray:
    """
    LEGACY: Mean-signature L2 distance (kept for backward compatibility).
    For new training, use kernel_mmd_loss instead.

    Computes the L2 distance between mean signatures in normalized space.

    Args:
        fake_signatures: Shape (batch_size, sig_dim)
        real_signatures: Shape (batch_size, sig_dim)
        sig_std: Component-wise std of real signatures for normalization.
    """
    mu_fake = jnp.mean(fake_signatures, axis=0)
    mu_real = jnp.mean(real_signatures, axis=0)

    if sig_std is not None:
        threshold = 1e-8
        weights = jnp.where(sig_std > threshold, 1.0 / sig_std, 0.0)
        diff = (mu_fake - mu_real) * weights
    else:
        diff = mu_fake - mu_real

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


@jit
def marginal_mean_penalty_loss(fake_paths: jnp.ndarray,
                               real_paths: jnp.ndarray) -> jnp.ndarray:
    """
    Per-step marginal mean penalty (Bayer & Stemper, 2018 style).

    Matches E[V_t] at each time step t independently, providing tighter
    control on the marginal distribution than a single global mean.
    Corrects time-dependent Jensen bias: E[exp(log V_t)] may drift
    differently at each t depending on the accumulated diffusion.

    Args:
        fake_paths: Generated variance paths, shape (batch, n_steps)
        real_paths: Real variance paths, shape (batch_real, n_steps)
    """
    fake_means = jnp.mean(fake_paths, axis=0)   # (n_steps,)
    real_means = jnp.mean(real_paths, axis=0)    # (n_steps,)
    return jnp.mean(jnp.square(fake_means - real_means))
