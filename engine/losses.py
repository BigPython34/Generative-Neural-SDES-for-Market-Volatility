"""
Loss Functions for Neural SDE Training
=======================================
Multi-measure loss library:
  - P-measure (physical): MMD + mean penalty                    → VaR, stress testing
  - Q-measure (risk-neutral): IV surface + martingale + MMD reg → pricing, hedging

Architecture (v3.0):
  Under P: The primary objective is distribution matching via kernel MMD on
           path signatures. This is correct — we want generated vol paths
           to have the same statistical properties as historical data.

  Under Q: The primary objective is IV SURFACE MATCHING against market
           option prices (Bayer & Stemper 2018, Gierjatowicz et al. 2022).
           MMD on P-data becomes a REGULARIZER (small weight) to keep
           paths realistic. The martingale constraint enforces no-arbitrage.

Components:
  - kernel_mmd_loss:             Distribution matching (MMD² with multi-scale RBF)
  - signature_mmd_loss:          Mean-signature L2 (used by verify_roughness, compare_frequencies)
  - mean_penalty_loss:           Global E[V] matching
  - marginal_mean_penalty_loss:  Per-step E[V_t] matching (Bayer & Stemper 2018)
  - martingale_violation_loss:   E[e^{-rT} S_T] = S_0 constraint (Q-measure)
  - smile_fit_loss:              IV/price smile matching on target strikes
  - term_structure_loss:         Vol term structure matching
  - jump_regularization_loss:    Penalize excessive jump frequency
  - p_measure_loss:              Composite P-loss (MMD + mean penalty)
  - q_measure_loss:              Composite Q-loss (smile + martingale + MMD reg)

Removed components:
  - feller_condition_loss:       Only for CIR/Heston variance-space (we use log-V)
  - path_regularity_loss:        Contradicts rough volatility (penalizes roughness)
  - mean_penalty_loss_logv:      Correct idea but inlined in trainer, never called

References:
  - Kidger et al. (2021). Neural SDEs as Infinite-Dimensional GANs. ICML.
  - Bayer & Stemper (2018). Deep calibration of rough stochastic vol models.
  - Buehler et al. (2021). Deep Hedging: Learning Risk-Neutral IV Dynamics.
  - Gierjatowicz et al. (2022). Robust pricing and hedging via neural SDEs.
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
    """Mean-signature L2 distance. Prefer kernel_mmd_loss for training."""
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
    Penalizes E[V_gen] ≠ E[V_real] in variance space.

    Note: When paths are in variance (exp of log-V), this loss
    suffers from Jensen bias: E[exp(X)] ≥ exp(E[X]).
    The trainer handles this by matching in log-V space inline.
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
def terminal_martingale_violation_loss(
    spot_terminal_normalized: jnp.ndarray,
    T: float,
    r: float = 0.0,
) -> jnp.ndarray:
    """Terminal-only martingale check: $(E[e^{-rT}S_T/S_0]-1)^2$."""
    disc_mean = jnp.mean(spot_terminal_normalized) * jnp.exp(-r * T)
    return jnp.square(disc_mean - 1.0)


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
def jump_regularization_loss(log_lambda: jnp.ndarray,
                             target_annual_rate: float = 3.0) -> jnp.ndarray:
    """
    Soft constraint on jump intensity.
    Prevents the model from using too many or too few jumps.
    target_annual_rate ≈ 3 jumps/year is empirically reasonable for equity vol.
    """
    lam = jnp.exp(log_lambda)
    return jnp.square(lam - target_annual_rate)


# =====================================================================
#  Composite losses (convenience wrappers)
# =====================================================================

def p_measure_loss(fake_sigs, real_sigs, sig_std, fake_paths, real_mean,
                   real_paths=None, lambda_mean=10.0,
                   mean_mode="global") -> jnp.ndarray:
    """
    P-measure composite loss for physical dynamics modeling.
    Use for: VaR, stress testing, realized vol forecasting.

    L = MMD²(sigs_gen, sigs_real) + λ_mean · L_mean

    This is the correct loss for learning P-measure dynamics:
    the MMD matches the *distribution of path signatures*
    to historical data (VIX or realized vol).
    """
    mmd = kernel_mmd_loss(fake_sigs, real_sigs, sig_std)
    if mean_mode == "marginal" and real_paths is not None:
        m_pen = marginal_mean_penalty_loss(fake_paths, real_paths)
    else:
        m_pen = mean_penalty_loss(fake_paths, real_mean)
    return mmd + lambda_mean * m_pen


def q_measure_loss(spot_paths, dt, r,
                   model_prices=None, market_prices=None, vega_weights=None,
                   fake_sigs=None, real_sigs=None, sig_std=None,
                   lambda_martingale=5.0, lambda_smile=1.0,
                   lambda_mmd_reg=0.1) -> jnp.ndarray:
    """
    Q-measure composite loss for risk-neutral pricing & hedging.

    ARCHITECTURE (per Buehler et al. 2021, Bayer & Stemper 2018):
    ──────────────────────────────────────────────────────────────
    Under Q, option prices are expectations: C(K,T) = E^Q[e^{-rT}(S_T-K)^+].
    The volatility dynamics under Q ≠ under P, so matching historical
    path distributions (MMD on P-measure data) is NOT a valid Q-loss.

    The correct Q-measure loss is driven by **market option prices**:

    L_Q = λ_smile · L_smile(model_prices, market_prices)    [PRIMARY]
        + λ_mart  · L_martingale(spot_paths)                  [CONSTRAINT]
        + λ_mmd   · L_MMD(sigs_gen, sigs_real)                [REGULARIZER]

    Where:
      - L_smile: Vega-weighted IV/price error vs market options  (PRIMARY)
      - L_martingale: E[e^{-rT}S_T] = S_0                      (NO-ARBITRAGE)
      - L_MMD: signature distribution matching                   (REGULARIZER,
              keeps paths realistic, prevents degenerate solutions)

    The MMD serves as a regularizer (small weight) to prevent the model
    from finding degenerate Q-dynamics that fit option prices but produce
    unrealistic paths. This is analogous to the KL-divergence regularization
    in Gierjatowicz et al. (2022).

    Args:
        spot_paths: (n_paths, n_steps) normalized spot (S_t/S_0)
        dt: time step in years
        r: risk-free rate (SOFR)
        model_prices: (n_strikes,) MC option prices from generated paths
        market_prices: (n_strikes,) observed market option prices
        vega_weights: (n_strikes,) optional vega weighting
        fake_sigs: generated path signatures (for MMD regularizer)
        real_sigs: real data signatures (for MMD regularizer)
        sig_std: signature normalization weights
        lambda_martingale: weight for martingale constraint
        lambda_smile: weight for IV/price surface matching
        lambda_mmd_reg: weight for MMD regularizer (small!)
    """
    total = jnp.float32(0.0)

    # PRIMARY: Option price / IV surface matching
    if model_prices is not None and market_prices is not None:
        smile = smile_fit_loss(model_prices, market_prices, vega_weights)
        total = total + lambda_smile * smile

    # CONSTRAINT: Martingale (no-arbitrage under Q)
    mart = martingale_violation_loss(spot_paths, dt, r)
    total = total + lambda_martingale * mart

    # REGULARIZER: MMD on signatures (keeps paths realistic)
    if fake_sigs is not None and real_sigs is not None:
        mmd = kernel_mmd_loss(fake_sigs, real_sigs, sig_std)
        total = total + lambda_mmd_reg * mmd

    return total


