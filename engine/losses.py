"""
Loss Functions for Neural SDE Training
=======================================
Multi-measure loss library:
  - P-measure (physical): MMD + mean penalty                    → VaR, stress testing
  - Q-measure (risk-neutral): MMD + mean penalty + martingale   → pricing, hedging

Components:
  - kernel_mmd_loss:             Distribution matching (MMD² with multi-scale RBF)
  - signature_mmd_loss:          Legacy mean-signature L2 (backward compat)
  - mean_penalty_loss:           Global E[V] matching
  - marginal_mean_penalty_loss:  Per-step E[V_t] matching (Bayer & Stemper 2018)
  - martingale_violation_loss:   E[e^{-rT} S_T] = S_0 constraint (Q-measure)
  - smile_fit_loss:              IV smile matching on target strikes
  - term_structure_loss:         Vol term structure matching
  - jump_regularization_loss:    Penalize excessive jump frequency
  - feller_condition_loss:       2κθ > σ² stability condition
"""

import jax
import jax.numpy as jnp
from jax import jit
from functools import partial


# =====================================================================
#  Distribution matching
# =====================================================================

def _rbf_kernel(x: jnp.ndarray, y: jnp.ndarray, bandwidth: float) -> jnp.ndarray:
    """RBF (Gaussian) kernel: k(x, y) = exp(-||x-y||² / (2σ²))."""
    x_sq = jnp.sum(x ** 2, axis=1, keepdims=True)
    y_sq = jnp.sum(y ** 2, axis=1, keepdims=True)
    cross = x @ y.T
    dist_sq = x_sq + y_sq.T - 2 * cross
    return jnp.exp(-dist_sq / (2 * bandwidth ** 2))


@partial(jit, static_argnames=['n_bandwidths'])
def kernel_mmd_loss(fake_signatures: jnp.ndarray, real_signatures: jnp.ndarray,
                    sig_std: jnp.ndarray = None,
                    n_bandwidths: int = 5) -> jnp.ndarray:
    """
    True kernel MMD² with multi-scale RBF kernel and median heuristic.
    Compares the FULL DISTRIBUTION of path signatures (not just means).
    """
    if sig_std is not None:
        threshold = 1e-8
        weights = jnp.where(sig_std > threshold, 1.0 / sig_std, 0.0)
        x = real_signatures * weights
        y = fake_signatures * weights
    else:
        x = real_signatures
        y = fake_signatures

    all_pts = jnp.concatenate([x, y], axis=0)
    n_total = all_pts.shape[0]
    dists_sq = (jnp.sum(all_pts ** 2, axis=1, keepdims=True)
                + jnp.sum(all_pts ** 2, axis=1, keepdims=True).T
                - 2 * all_pts @ all_pts.T)
    dists_sq_no_diag = jnp.where(jnp.eye(n_total, dtype=bool), jnp.nan, dists_sq)
    median_dist = jnp.sqrt(jnp.maximum(jnp.nanmedian(dists_sq_no_diag), 1e-8))

    scales = jnp.logspace(jnp.log10(0.25), jnp.log10(4.0), num=n_bandwidths, base=10.0)
    bandwidths = median_dist * scales

    mmd_sq = jnp.float32(0.0)

    def add_bandwidth_mmd(carry, bw):
        K_xx = _rbf_kernel(x, x, bw)
        K_yy = _rbf_kernel(y, y, bw)
        K_xy = _rbf_kernel(x, y, bw)
        mmd_bw = jnp.mean(K_xx) + jnp.mean(K_yy) - 2 * jnp.mean(K_xy)
        return carry + mmd_bw, None

    mmd_sq, _ = jax.lax.scan(add_bandwidth_mmd, mmd_sq, bandwidths)
    return mmd_sq


@jit
def signature_mmd_loss(fake_signatures: jnp.ndarray, real_signatures: jnp.ndarray,
                       sig_std: jnp.ndarray = None) -> jnp.ndarray:
    """LEGACY: Mean-signature L2 distance. Use kernel_mmd_loss for new training."""
    mu_fake = jnp.mean(fake_signatures, axis=0)
    mu_real = jnp.mean(real_signatures, axis=0)
    if sig_std is not None:
        threshold = 1e-8
        weights = jnp.where(sig_std > threshold, 1.0 / sig_std, 0.0)
        diff = (mu_fake - mu_real) * weights
    else:
        diff = mu_fake - mu_real
    return jnp.sum(jnp.square(diff))


# =====================================================================
#  Mean matching
# =====================================================================

@jit
def mean_penalty_loss(fake_paths: jnp.ndarray, real_mean: float) -> jnp.ndarray:
    """
    Penalizes E[V_gen] ≠ E[V_real].
    Corrects Jensen bias from exp(log_v) overshoot.
    """
    return jnp.square(jnp.mean(fake_paths) - real_mean)


@jit
def marginal_mean_penalty_loss(fake_paths: jnp.ndarray,
                               real_paths: jnp.ndarray) -> jnp.ndarray:
    """Per-step marginal mean penalty: match E[V_t] at each t (Bayer & Stemper 2018)."""
    fake_means = jnp.mean(fake_paths, axis=0)
    real_means = jnp.mean(real_paths, axis=0)
    return jnp.mean(jnp.square(fake_means - real_means))


# =====================================================================
#  Q-measure constraints (for pricing / hedging models)
# =====================================================================

@jit
def martingale_violation_loss(spot_paths: jnp.ndarray, dt: float,
                              r: float = 0.0) -> jnp.ndarray:
    """
    Squared deviation from the Q-measure martingale property.
    Under Q: E[e^{-rT} S_T] = S_0, equivalently E[S̃_t] = S̃_0 ∀t.

    Args:
        spot_paths: (n_paths, n_steps) normalized spot (S_t / S_0)
        dt: time step in years
        r: risk-free rate
    """
    n_steps = spot_paths.shape[1]
    time_grid = jnp.linspace(0, (n_steps - 1) * dt, n_steps)
    discount_factors = jnp.exp(-r * time_grid)
    discounted_s = spot_paths * discount_factors
    mean_path = jnp.mean(discounted_s, axis=0)
    return jnp.mean(jnp.square(mean_path - 1.0))


@jit
def smile_fit_loss(model_ivs: jnp.ndarray, market_ivs: jnp.ndarray,
                   vega_weights: jnp.ndarray = None) -> jnp.ndarray:
    """
    Vega-weighted IV smile fitting loss.

    L = Σ_i  w_i · (σ_model(K_i) - σ_market(K_i))²

    Vega weighting focuses calibration on liquid ATM region
    where model accuracy matters most for hedging.
    """
    diff_sq = jnp.square(model_ivs - market_ivs)
    if vega_weights is not None:
        w = vega_weights / jnp.sum(vega_weights)
        return jnp.sum(w * diff_sq)
    return jnp.mean(diff_sq)


@jit
def term_structure_loss(model_atm_vols: jnp.ndarray,
                        market_atm_vols: jnp.ndarray) -> jnp.ndarray:
    """
    ATM vol term structure matching.
    model/market_atm_vols: (n_maturities,) array of ATM implied vols.
    """
    return jnp.mean(jnp.square(model_atm_vols - market_atm_vols))


# =====================================================================
#  Regularization
# =====================================================================

@jit
def feller_condition_loss(kappa: float, theta: float,
                          vol_of_vol: float) -> jnp.ndarray:
    """Penalizes Feller condition violation: 2κθ > σ²."""
    return jnp.maximum(0.0, vol_of_vol ** 2 - 2 * kappa * theta) ** 2


@jit
def jump_regularization_loss(log_lambda: jnp.ndarray,
                             target_annual_rate: float = 3.0) -> jnp.ndarray:
    """
    Soft constraint on jump intensity.
    Prevents the model from using too many or too few jumps.
    target_annual_rate ≈ 3 jumps/year is empirically reasonable for equity vol.
    """
    lam = jnp.exp(log_lambda)
    return jnp.square(lam - target_annual_rate)


@jit
def path_regularity_loss(paths: jnp.ndarray) -> jnp.ndarray:
    """
    Penalizes non-physical spikes in generated variance paths.
    Encourages smooth, continuous paths (appropriate for diffusion models).
    """
    increments = jnp.diff(paths, axis=1)
    return jnp.mean(jnp.square(increments))


# =====================================================================
#  Composite losses (convenience wrappers)
# =====================================================================

def p_measure_loss(fake_sigs, real_sigs, sig_std, fake_paths, real_mean,
                   real_paths=None, lambda_mean=10.0,
                   mean_mode="global") -> jnp.ndarray:
    """
    P-measure composite loss for physical dynamics modeling.
    Use for: VaR, stress testing, realized vol forecasting.
    """
    mmd = kernel_mmd_loss(fake_sigs, real_sigs, sig_std)
    if mean_mode == "marginal" and real_paths is not None:
        m_pen = marginal_mean_penalty_loss(fake_paths, real_paths)
    else:
        m_pen = mean_penalty_loss(fake_paths, real_mean)
    return mmd + lambda_mean * m_pen


def q_measure_loss(fake_sigs, real_sigs, sig_std, fake_paths, real_mean,
                   spot_paths, dt, r=0.0,
                   real_paths=None, lambda_mean=10.0, lambda_martingale=5.0,
                   mean_mode="global") -> jnp.ndarray:
    """
    Q-measure composite loss for risk-neutral pricing.
    Use for: option pricing, hedging, calibration.

    Adds martingale constraint on top of P-measure loss:
      L = L_MMD + λ_mean · L_mean + λ_mart · L_martingale
    """
    base = p_measure_loss(fake_sigs, real_sigs, sig_std, fake_paths,
                          real_mean, real_paths, lambda_mean, mean_mode)
    mart = martingale_violation_loss(spot_paths, dt, r)
    return base + lambda_martingale * mart
