"""
Neural SDE Q-Measure Calibration via Girsanov Drift Correction
==============================================================
Learns a risk-neutral drift correction λ_φ(v,t) on top of a frozen
P-measure Neural SDE, so that the resulting Q-dynamics reprice
SPX options and VIX term structure simultaneously.

Mathematical Framework
======================

Under P (physical):
    dlog V_t = μ^P(V,t) dt + σ(V,t) dW^P_t

Girsanov theorem gives the Q-dynamics:
    dlog V_t = [μ^P(V,t) - λ_φ(V,t)·σ(V,t)] dt + σ(V,t) dW^Q_t

Key insight: σ(V,t) is FROZEN from the P-model (Girsanov preserves diffusion).
Only the drift correction λ_φ is learned — a small MLP (~163 parameters).

The Novikov condition ∫₀ᵀ |λ_φ|² dt < ∞ ensures the measure change is valid.
We enforce it via a soft regularizer w_novikov · E[Σ λ²·dt].

Loss components:
    L = w_smile · L_SPX(model IV, market IV)
      + w_vix   · L_VIX(model VIX, market VIX)
      + w_mart  · L_martingale(E[e^{-rT}S_T] = S_0)
      + w_cal   · L_calendar(total var monotone)
      + w_nov   · E[Σ λ² dt]

References:
    - Gierjatowicz et al. (2022). Robust pricing and hedging via neural SDEs.
    - Buehler et al. (2021). Deep Hedging.
    - Bayer & Stemper (2018). Deep calibration of rough stochastic vol models.
"""

from __future__ import annotations

import time
import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import jax
import jax.numpy as jnp
import jax.nn as jnn
import equinox as eqx
import optax



# ═══════════════════════════════════════════════════════════════════
# 1. GIRSANOV DRIFT CORRECTION
# ═══════════════════════════════════════════════════════════════════

class GirsanovDrift(eqx.Module):
    """
    Learnable market-price-of-risk function λ_φ(log_v, t).

    Architecture: MLP  (2 → 32 → 32 → 1)  with tanh output.
    The tanh bounding ensures Novikov condition: |λ| ≤ λ_max.

    Parameters: 2*32 + 32 + 32*32 + 32 + 32*1 + 1 = 163 params.
    """
    net: eqx.nn.MLP
    lambda_max: float

    def __init__(self, key: jax.random.PRNGKey, lambda_max: float = 3.0):
        self.net = eqx.nn.MLP(
            in_size=2,         # (log_v, t)
            out_size=1,
            width_size=32,
            depth=2,           # 2 hidden layers
            activation=jnn.gelu,
            key=key,
        )
        self.lambda_max = lambda_max

    def __call__(self, log_v: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Return λ_φ(log_v, t), bounded by tanh."""
        inp = jnp.array([log_v, t])
        raw = self.net(inp)
        return self.lambda_max * jnp.tanh(jnp.squeeze(raw))


# ═══════════════════════════════════════════════════════════════════
# 2. NEURAL SDE Q-MODEL (wraps frozen P-model)
# ═══════════════════════════════════════════════════════════════════

class NeuralSDEQModel(eqx.Module):
    """
    Q-measure Neural SDE = frozen P-model + Girsanov drift correction.

    Under Q:
        dlog V_t = [μ^P(V,t) - λ_φ(V,t)·σ^P(V,t)] dt + σ^P(V,t) dW^Q_t
        dS_t/S_t = r dt + √exp(log V_t) · [ρ dW^Q_t + √(1-ρ²) dW⊥_t]

    The P-model drift and diffusion are evaluated via stop_gradient
    to freeze them during Q-training. Only λ_φ has trainable params.
    """
    girsanov: GirsanovDrift

    # P-model parameters (frozen — from existing trained model or OU prior)
    p_kappa: jnp.ndarray
    p_theta: jnp.ndarray
    p_drift_scale: float
    p_diff_min: float
    p_diff_max: float

    # Market parameters
    rho: float
    spot: float
    r: float

    log_v_min: float
    log_v_max: float

    def __init__(
        self,
        girsanov: GirsanovDrift,
        kappa: float = 2.72,
        theta: float = -3.5,
        drift_scale: float = 0.5,
        diff_min: float = 0.1,
        diff_max: float = 1.6,
        rho: float = -0.85,
        spot: float = 5500.0,
        r: float = 0.0373,
        log_v_min: float = -7.0,
        log_v_max: float = 2.0,
    ):
        self.girsanov = girsanov
        self.p_kappa = jnp.array(float(kappa))
        self.p_theta = jnp.array(float(theta))
        self.p_drift_scale = drift_scale
        self.p_diff_min = diff_min
        self.p_diff_max = diff_max
        self.rho = rho
        self.spot = spot
        self.r = r
        self.log_v_min = log_v_min
        self.log_v_max = log_v_max

    def simulate(
        self,
        init_log_v: float,
        n_steps: int,
        dt: float,
        key: jax.random.PRNGKey,
        n_paths: int = 4096,
    ):
        """
        Simulate (log_V, S/S_0) under Q with Girsanov drift correction.

        Returns:
            log_v_paths:  (n_paths, n_steps)
            spot_paths:   (n_paths, n_steps)  — normalized by S_0
            lambda_sq:    (n_paths,) — mean λ² per path (for Novikov penalty)
        """
        k1, k2 = jax.random.split(key)
        # Correlated Brownians: dW_v, dW_s = ρ dW_v + √(1-ρ²) dW⊥
        z1 = jax.random.normal(k1, (n_paths, n_steps))  # vol driver
        z2 = jax.random.normal(k2, (n_paths, n_steps))  # independent
        dW_v = jnp.sqrt(dt) * z1
        dW_s = self.rho * dW_v + jnp.sqrt(1.0 - self.rho**2) * jnp.sqrt(dt) * z2

        def simulate_one_path(dw_v_path, dw_s_path):
            return self._simulate_single(init_log_v, dw_v_path, dw_s_path, dt, n_steps)

        log_v_paths, spot_paths, lambda_sq = jax.vmap(simulate_one_path)(dW_v, dW_s)
        return log_v_paths, spot_paths, lambda_sq

    def _simulate_single(self, init_log_v, dw_v, dw_s, dt, n_steps):
        """Euler-Maruyama for a single path under Q."""
        lo, hi = self.log_v_min, self.log_v_max
        # Freeze P-model parameters
        kappa = jax.lax.stop_gradient(self.p_kappa)
        theta = jax.lax.stop_gradient(self.p_theta)
        total_T = n_steps * dt  # for time normalization

        # Time indices for each step (normalized to [0, 1])
        t_indices = jnp.arange(n_steps, dtype=jnp.float32) * dt / total_T

        def scan_fn(carry, inputs):
            log_v, log_s, lam_sq_accum = carry
            dw_v_t, dw_s_t, t_normed = inputs

            # P-model drift: OU mean-reversion
            mu_p = kappa * (theta - log_v) * dt

            # P-model diffusion (simplified: constant based on P-model bounds)
            sigma_p = jax.lax.stop_gradient(
                (self.p_diff_min + self.p_diff_max) / 2.0
            )

            # Girsanov drift correction (TRAINABLE) — now time-aware
            lam = self.girsanov(log_v, t_normed)

            # Q-drift = P-drift - λ · σ · dt
            drift_q = mu_p - lam * sigma_p * dt

            # Step log-variance
            log_v_next = log_v + drift_q + sigma_p * dw_v_t
            log_v_next = jnp.clip(log_v_next, lo, hi)

            # Step spot (under Q: dS/S = r dt + √V dW_s)
            vol = jnp.sqrt(jnp.maximum(jnp.exp(log_v), 1e-10))
            log_s_next = log_s + (self.r - 0.5 * jnp.exp(log_v)) * dt + vol * dw_s_t

            # Accumulate λ² for Novikov regularizer
            lam_sq_accum = lam_sq_accum + lam ** 2 * dt

            return (log_v_next, log_s_next, lam_sq_accum), (log_v_next, log_s_next)

        init_carry = (jnp.float32(init_log_v), jnp.float32(0.0), jnp.float32(0.0))
        final_carry, (log_v_path, log_s_path) = jax.lax.scan(
            scan_fn, init_carry, (dw_v, dw_s, t_indices)
        )
        _, _, lambda_sq_total = final_carry

        spot_path = jnp.exp(log_s_path)  # S_t / S_0
        return log_v_path, spot_path, lambda_sq_total


# ═══════════════════════════════════════════════════════════════════
# 3. Q-MEASURE LOSS COMPONENTS
# ═══════════════════════════════════════════════════════════════════
def compute_spx_smile_loss(
    spot_paths: jnp.ndarray,
    spx_slices: list[dict],
    S0: float,
    r: float,
) -> jnp.ndarray:
    """
    L_SPX = (1/N_mat) Σ_T (1/N_K) Σ_K (1/vega²) · (C_model - C_market)²

    Price-space loss with vega normalization (≈ IV-space loss to first order).
    This is fully JAX-traceable, unlike IV extraction via bisection.

    Reference: Gierjatowicz et al. (2022), §3.2 — vega-weighted price loss.

    spx_slices: list of dicts with keys:
        T, strikes, market_ivs, vega_weights, option_types
    """
    if not spx_slices:
        return jnp.float32(0.0)

    total_err = jnp.float32(0.0)
    n_mats = 0

    for sl in spx_slices:
        T = sl['T']
        strikes = jnp.array(sl['strikes'], dtype=jnp.float32)
        market_ivs = jnp.array(sl['market_ivs'], dtype=jnp.float32)
        vega_w = jnp.array(sl.get('vega_weights', np.ones(len(strikes))),
                           dtype=jnp.float32)
        option_types = sl.get('option_types', ['call'] * len(strikes))

        n_steps = spot_paths.shape[1]
        dt_est = 1.0 / 252.0
        step_T = min(int(T / dt_est), n_steps - 1)
        if step_T < 1:
            continue

        # S_T / S_0 from MC paths
        S_T = spot_paths[:, step_T]  # (n_paths,)
        disc = jnp.exp(-r * T)

        # Pre-compute market BS prices from market IVs (for loss comparison)
        # and MC model prices — both in price space
        sqrt_T = jnp.sqrt(jnp.maximum(T, 1e-8))
        slice_loss = jnp.float32(0.0)
        n_valid = 0

        for i in range(len(strikes)):
            K_abs = strikes[i]
            K_rel = K_abs / S0
            sigma = market_ivs[i]

            # Market BS price (analytical)
            d1 = (jnp.log(S0 / K_abs) + (r + 0.5 * sigma ** 2) * T) / (
                sigma * sqrt_T + 1e-10)
            d2 = d1 - sigma * sqrt_T

            if option_types[i] == 'call':
                # MC model price
                payoff = jnp.maximum(S_T - K_rel, 0.0)
                # BS market price (normalized by S0)
                bs_price = (jax.scipy.stats.norm.cdf(d1)
                            - K_rel * jnp.exp(-r * T)
                            * jax.scipy.stats.norm.cdf(d2))
            else:
                payoff = jnp.maximum(K_rel - S_T, 0.0)
                bs_price = (K_rel * jnp.exp(-r * T)
                            * jax.scipy.stats.norm.cdf(-d2)
                            - jax.scipy.stats.norm.cdf(-d1))

            mc_price = disc * jnp.mean(payoff)

            # Vega normalization: (C_model - C_market)² / vega²
            # This ≈ (σ_model - σ_market)² to first order
            v = jnp.maximum(vega_w[i] / S0, 1e-8)  # normalize
            err = ((mc_price - bs_price) / v) ** 2
            slice_loss = slice_loss + err
            n_valid += 1

        if n_valid > 0:
            total_err = total_err + slice_loss / n_valid
            n_mats += 1

    if n_mats == 0:
        return jnp.float32(0.0)
    return total_err / n_mats


def compute_vix_loss(
    log_v_paths: jnp.ndarray,
    market_vix_ts: dict,
    dt: float,
) -> jnp.ndarray:
    """
    L_VIX = Σ_τ [(VIX_model(τ) - VIX_market(τ))² / VIX_market(τ)²]

    VIX(τ) = √(E[1/Δ ∫_τ^{τ+Δ} V_t dt])  with Δ = 30/365 (30 days).
    """
    n_steps = log_v_paths.shape[1]
    total_T = n_steps * dt
    delta_vix = 30.0 / 365.0  # VIX averages over 30 days

    loss = jnp.float32(0.0)
    count = 0

    for tau_days, mkt_vix in market_vix_ts.items():
        tau = tau_days / 365.0
        if tau + delta_vix > total_T:
            continue

        # Find step indices for [τ, τ+Δ]
        start_step = int(tau / dt)
        end_step = min(int((tau + delta_vix) / dt), n_steps)
        if end_step <= start_step:
            continue

        # E[1/Δ ∫_τ^{τ+Δ} V_t dt] ≈ mean over paths of mean V in window
        V_window = jnp.exp(log_v_paths[:, start_step:end_step])
        mean_V = jnp.mean(V_window)
        model_vix = jnp.sqrt(jnp.maximum(mean_V, 1e-10)) * 100.0

        # Relative squared error
        rel_err = (model_vix - mkt_vix) / jnp.maximum(mkt_vix, 1.0)
        loss = loss + rel_err ** 2
        count += 1

    if count > 0:
        loss = loss / count
    return loss


def compute_martingale_loss(
    spot_paths: jnp.ndarray,
    r: float,
    T: float,
) -> jnp.ndarray:
    """L_mart = (E[e^{-rT} S_T / S_0] - 1)²."""
    S_T = spot_paths[:, -1]
    disc_mean = jnp.mean(S_T) * jnp.exp(-r * T)
    return (disc_mean - 1.0) ** 2


def compute_calendar_loss(
    log_v_paths: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """
    Calendar spread constraint: total implied variance must be non-decreasing.
    L_cal = Σ_t max(0, cumvar_t - cumvar_{t+1})²
    """
    V_paths = jnp.exp(log_v_paths)
    mean_V = jnp.mean(V_paths, axis=0)  # (n_steps,)
    cum_var = jnp.cumsum(mean_V * dt)
    violations = jnp.maximum(cum_var[:-1] - cum_var[1:], 0.0)
    return jnp.mean(violations ** 2)


# ═══════════════════════════════════════════════════════════════════
# 4. COMPOSITE Q-LOSS
# ═══════════════════════════════════════════════════════════════════

@dataclass
class QLossWeights:
    """Weights for each loss component."""
    w_smile: float = 1.0
    w_vix: float = 2.0
    w_mart: float = 5.0
    w_cal: float = 0.5
    w_novikov: float = 0.1


def compute_q_loss(
    model: NeuralSDEQModel,
    init_log_v: float,
    n_steps: int,
    dt: float,
    key: jax.random.PRNGKey,
    n_paths: int,
    # Market data
    spx_slices: Optional[list[dict]] = None,
    market_vix_ts: Optional[dict] = None,
    # Weights
    weights: Optional[QLossWeights] = None,
) -> tuple[jnp.ndarray, dict]:
    """
    Full Q-measure loss with 5 components.

    Args:
        spx_slices: list of per-maturity dicts from _prepare_spx_targets,
                    each with T, strikes, market_ivs, vega_weights, option_types.
        market_vix_ts: {tenor_days: vix_level} dict.

    Returns (total_loss, loss_dict) where loss_dict has component-wise values.
    """
    if weights is None:
        weights = QLossWeights()

    # Simulate under Q
    log_v_paths, spot_paths, lambda_sq = model.simulate(
        init_log_v, n_steps, dt, key, n_paths
    )

    total = jnp.float32(0.0)
    loss_dict = {}

    # 1. SPX multi-maturity smile loss
    if spx_slices:
        l_smile = compute_spx_smile_loss(
            spot_paths, spx_slices, model.spot, model.r,
        )
        total = total + weights.w_smile * l_smile
        loss_dict['smile'] = l_smile
    else:
        loss_dict['smile'] = jnp.float32(0.0)

    # 2. VIX term structure loss
    if market_vix_ts is not None and len(market_vix_ts) > 0:
        l_vix = compute_vix_loss(log_v_paths, market_vix_ts, dt)
        total = total + weights.w_vix * l_vix
        loss_dict['vix'] = l_vix
    else:
        loss_dict['vix'] = jnp.float32(0.0)

    # 3. Martingale constraint
    T_total = n_steps * dt
    l_mart = compute_martingale_loss(spot_paths, model.r, T_total)
    total = total + weights.w_mart * l_mart
    loss_dict['martingale'] = l_mart

    # 4. Calendar spread constraint
    l_cal = compute_calendar_loss(log_v_paths, dt)
    total = total + weights.w_cal * l_cal
    loss_dict['calendar'] = l_cal

    # 5. Novikov regularizer: E[∫₀ᵀ λ² dt]
    l_novikov = jnp.mean(lambda_sq)
    total = total + weights.w_novikov * l_novikov
    loss_dict['novikov'] = l_novikov

    loss_dict['total'] = total
    return total, loss_dict


# ═══════════════════════════════════════════════════════════════════
# 5. RESULT DATACLASS
# ═══════════════════════════════════════════════════════════════════

@dataclass
class NeuralSDEQResult:
    """Results from Neural SDE Q-calibration."""
    # Parameters
    rho: float
    H_bergomi: float       # from rBergomi seed
    eta_bergomi: float     # from rBergomi seed

    # Loss history
    final_loss: float
    loss_components: dict
    loss_history: list

    # Model info
    n_girsanov_params: int
    n_epochs: int
    n_paths: int
    elapsed_seconds: float

    # Diagnostics
    model_vix_ts: dict = field(default_factory=dict)
    market_vix_ts: dict = field(default_factory=dict)
    martingale_error: float = 0.0

    def summary(self) -> str:
        lines = []
        lines.append("=" * 65)
        lines.append("  NEURAL SDE Q-CALIBRATION RESULTS (Girsanov)")
        lines.append("=" * 65)
        lines.append(f"  Girsanov params:  {self.n_girsanov_params}")
        lines.append(f"  Epochs:           {self.n_epochs}")
        lines.append(f"  MC paths/epoch:   {self.n_paths:,}")
        lines.append(f"  Time:             {self.elapsed_seconds:.1f}s")
        lines.append("")
        lines.append("  Seed Parameters (from rBergomi):")
        lines.append(f"    H_Q   = {self.H_bergomi:.4f}")
        lines.append(f"    η     = {self.eta_bergomi:.3f}")
        lines.append(f"    ρ     = {self.rho:.3f}")
        lines.append("")
        lines.append("  Loss Components:")
        for k, v in self.loss_components.items():
            lines.append(f"    {k:>15} = {v:.6f}")
        lines.append("")

        if self.model_vix_ts and self.market_vix_ts:
            lines.append("  VIX Term Structure (Q-model):")
            lines.append(f"    {'Tenor':>8} {'Market':>8} {'Model':>8} {'Diff':>8}")
            lines.append("    " + "─" * 36)
            for tau in sorted(self.market_vix_ts.keys()):
                mkt = self.market_vix_ts.get(tau, None)
                mdl = self.model_vix_ts.get(tau, None)
                if mkt is not None and mdl is not None:
                    lines.append(f"    {tau:>6}d {mkt:>8.2f} {mdl:>8.2f} {mdl - mkt:>+8.2f}")

        lines.append(f"\n  Martingale error: {self.martingale_error:.6f}")
        lines.append("=" * 65)
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# 6. TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════

def train_neural_sde_q(
    # Market data
    market_vix_ts: dict,
    spot: float = 5500.0,
    r: float = 0.0373,
    spx_slices: Optional[list[dict]] = None,
    vix_futures: Optional[dict] = None,
    vvix: Optional[float] = None,
    # rBergomi seed
    H_bergomi: float = 0.03,
    eta_bergomi: float = 1.0,
    rho: float = -0.90,
    # Training config
    n_epochs: int = 200,
    n_paths: int = 4096,
    n_steps: int = 252,
    lr: float = 3e-3,
    seed: int = 42,
    verbose: bool = True,
    # Loss weights
    weights: Optional[QLossWeights] = None,
) -> tuple[NeuralSDEQModel, NeuralSDEQResult]:
    """
    Train the Girsanov drift correction to produce Q-consistent dynamics.

    Args:
        market_vix_ts: {tenor_days: vix_level} — VIX index term structure.
        spx_slices: list of per-maturity dicts (from _prepare_spx_targets).
        vix_futures: {days_to_exp: price} — VIX futures (merged into VIX TS).
        vvix: VVIX level (used for η prior / diagnostics).

    Returns (trained_model, result).
    """
    if weights is None:
        weights = QLossWeights()

    # Merge VIX futures into VIX targets (longer tenors)
    combined_vix_ts = dict(market_vix_ts)
    if vix_futures:
        for days, price in vix_futures.items():
            if days not in combined_vix_ts:
                combined_vix_ts[days] = price
            # If both exist, prefer VIX index (more direct measure)

    if verbose:
        print("\n" + "=" * 65)
        print("  NEURAL SDE Q-CALIBRATION (Girsanov Drift Correction)")
        print("=" * 65)
        print(f"  Seed params: H={H_bergomi:.4f}, η={eta_bergomi:.3f}, ρ={rho:.3f}")
        print(f"  Market: spot={spot:.0f}, r={r:.4f}")
        if vvix is not None:
            print(f"  VVIX: {vvix:.1f}")
        print(f"  Training: {n_epochs} epochs, {n_paths:,} paths, lr={lr:.0e}")
        print(f"  VIX index tenors:   {sorted(market_vix_ts.keys())}")
        if vix_futures:
            print(f"  VIX futures tenors: {sorted(vix_futures.keys())}")
        print(f"  Combined VIX targets: {len(combined_vix_ts)} tenors")
        if spx_slices:
            n_opts = sum(len(s['strikes']) for s in spx_slices)
            mats = [s['dte'] for s in spx_slices]
            print(f"  SPX surface: {len(spx_slices)} maturities, {n_opts} options")
            print(f"    DTEs: {mats}")
        else:
            print(f"  SPX surface: none (VIX-only calibration)")
        print()

    t0 = time.time()

    # Initialize model
    key = jax.random.PRNGKey(seed)
    k_init, k_train = jax.random.split(key)

    girsanov = GirsanovDrift(k_init, lambda_max=3.0)
    n_params = sum(x.size for x in jax.tree.leaves(eqx.filter(girsanov, eqx.is_array)))
    if verbose:
        print(f"  Girsanov MLP: {n_params} trainable parameters")

    # Map rBergomi H, η → OU-equivalent kappa, theta
    # For rough models (H ≪ 0.5), the fractional kernel decays as a power law,
    # not exponentially. The OU κ approximation breaks down for small H.
    # We use a moderate κ that preserves VIX TS dynamics:
    #   κ ∈ [1, 8] — enough mean-reversion for stationarity, not enough to
    #   kill the term structure. The Girsanov MLP will learn the rest.
    kappa_ou = float(np.clip(eta_bergomi * 2.0, 1.0, 8.0))

    # Diffusion ≈ η (vol-of-vol from rBergomi)
    diff_scale = float(np.clip(eta_bergomi, 0.3, 3.0))

    # Initial log-variance from VIX level
    vix_30d = market_vix_ts.get(30, market_vix_ts.get(9, 20.0))
    init_log_v = float(np.log(max((vix_30d / 100.0) ** 2, 1e-10)))

    model = NeuralSDEQModel(
        girsanov=girsanov,
        kappa=kappa_ou,
        theta=init_log_v,
        drift_scale=0.5,
        diff_min=diff_scale * 0.5,
        diff_max=diff_scale * 1.5,
        rho=rho,
        spot=spot,
        r=r,
    )

    if verbose:
        print(f"  OU prior: κ={kappa_ou:.2f}, θ={init_log_v:.3f} "
              f"(init V = {np.exp(init_log_v):.4f}, σ_range=[{diff_scale*0.5:.2f}, {diff_scale*1.5:.2f}])")
        print()

    # Optimizer: Adam + cosine decay
    schedule = optax.cosine_decay_schedule(lr, n_epochs, alpha=0.01)
    optimizer = optax.adam(schedule)

    # Only optimize Girsanov params (freeze everything else)
    # Use eqx.partition with a filter that selects only array leaves
    # in the girsanov sub-module
    def _is_girsanov_param(path, leaf):
        """Return True only for array leaves inside .girsanov."""
        path_str = jax.tree_util.keystr(path)
        return '.girsanov.' in path_str and eqx.is_array(leaf)

    filter_spec = jax.tree_util.tree_map_with_path(
        lambda path, leaf: _is_girsanov_param(path, leaf),
        model,
    )

    trainable, frozen = eqx.partition(model, filter_spec)
    opt_state = optimizer.init(trainable)

    dt = 1.0 / 252.0  # daily steps

    # Loss function — takes trainable part, combines with frozen
    def loss_fn(trainable_part, frozen_part, key_epoch):
        full_model = eqx.combine(trainable_part, frozen_part)
        loss, _ = compute_q_loss(
            full_model, init_log_v, n_steps, dt, key_epoch, n_paths,
            spx_slices=spx_slices,
            market_vix_ts=combined_vix_ts,
            weights=weights,
        )
        return loss

    # JIT-compiled value_and_grad (only w.r.t. first arg = trainable)
    # Use eqx.filter_jit so non-array leaves (activation functions etc.)
    # are treated as static and don't get traced by JAX.
    @eqx.filter_jit
    def train_step(trainable_part, frozen_part, opt_st, key_epoch):
        loss_val, grads = jax.value_and_grad(loss_fn)(trainable_part, frozen_part, key_epoch)
        updates, new_opt_st = optimizer.update(grads, opt_st, trainable_part)
        new_trainable = eqx.apply_updates(trainable_part, updates)
        return new_trainable, new_opt_st, loss_val

    # Training loop
    loss_history = []
    best_loss = float('inf')
    best_trainable = trainable
    patience = 60
    patience_counter = 0

    for epoch in range(n_epochs):
        k_train, k_epoch = jax.random.split(k_train)

        trainable, opt_state, loss_val = train_step(
            trainable, frozen, opt_state, k_epoch
        )
        loss_val = float(loss_val)

        loss_history.append(loss_val)

        # Early stopping
        if loss_val < best_loss - 1e-6:
            best_loss = loss_val
            best_trainable = trainable
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch % 20 == 0 or epoch == n_epochs - 1):
            print(f"  Epoch {epoch:>4d}/{n_epochs}  loss = {loss_val:.6f}  "
                  f"best = {best_loss:.6f}  patience = {patience_counter}/{patience}")

        if patience_counter >= patience:
            if verbose:
                print(f"\n  Early stopping at epoch {epoch} (patience={patience})")
            break

    elapsed = time.time() - t0
    model = eqx.combine(best_trainable, frozen)

    # Final evaluation with more paths
    if verbose:
        print(f"\n  Final evaluation with {n_paths * 2:,} paths...")

    k_final = jax.random.PRNGKey(seed + 999)
    _, final_loss_dict = compute_q_loss(
        model, init_log_v, n_steps, dt, k_final, n_paths * 2,
        spx_slices=spx_slices,
        market_vix_ts=combined_vix_ts,
        weights=weights,
    )
    # Convert JAX arrays to Python floats for serialization
    final_loss_dict = {k: float(v) for k, v in final_loss_dict.items()}

    # Extract model VIX term structure
    log_v_paths, spot_paths, _ = model.simulate(
        init_log_v, n_steps, dt, k_final, n_paths * 2
    )
    model_vix_ts = {}
    delta_vix = 30.0 / 365.0
    total_T = n_steps * dt
    for tau_days in sorted(combined_vix_ts.keys()):
        tau = tau_days / 365.0
        if tau + delta_vix > total_T:
            continue
        start_step = int(tau / dt)
        end_step = min(int((tau + delta_vix) / dt), n_steps)
        if end_step <= start_step:
            continue
        V_window = jnp.exp(log_v_paths[:, start_step:end_step])
        mean_V = float(jnp.mean(V_window))
        model_vix_ts[tau_days] = float(np.sqrt(max(mean_V, 1e-10)) * 100.0)

    # Martingale error
    S_T = spot_paths[:, -1]
    mart_err = float(abs(jnp.mean(S_T) * np.exp(-r * total_T) - 1.0))

    result = NeuralSDEQResult(
        rho=rho,
        H_bergomi=H_bergomi,
        eta_bergomi=eta_bergomi,
        final_loss=best_loss,
        loss_components=final_loss_dict,
        loss_history=loss_history,
        n_girsanov_params=n_params,
        n_epochs=len(loss_history),
        n_paths=n_paths,
        elapsed_seconds=elapsed,
        model_vix_ts=model_vix_ts,
        market_vix_ts=dict(combined_vix_ts),
        martingale_error=mart_err,
    )

    if verbose:
        print(result.summary())

    return model, result


# ═══════════════════════════════════════════════════════════════════
# 7. MODEL PERSISTENCE
# ═══════════════════════════════════════════════════════════════════

def save_q_model(
    model: NeuralSDEQModel,
    path: str = "models/neural_sde_q_model.eqx",
    config_path: str = "models/neural_sde_q_config.json",
):
    """
    Save trained Q-model (Girsanov drift correction) to disk.

    Saves:
      - EQX model weights  → path
      - Reconstruction config (architecture + scalar params) → config_path
    """
    from pathlib import Path as _P
    _P(path).parent.mkdir(exist_ok=True)
    eqx.tree_serialise_leaves(path, model)

    config = {
        'lambda_max': float(model.girsanov.lambda_max),
        'kappa': float(model.p_kappa),
        'theta': float(model.p_theta),
        'drift_scale': float(model.p_drift_scale),
        'diff_min': float(model.p_diff_min),
        'diff_max': float(model.p_diff_max),
        'rho': float(model.rho),
        'spot': float(model.spot),
        'r': float(model.r),
        'log_v_min': float(model.log_v_min),
        'log_v_max': float(model.log_v_max),
    }

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  Q-model saved: {path}")
    print(f"  Q-config saved: {config_path}")


def load_q_model(
    path: str = "models/neural_sde_q_model.eqx",
    config_path: str = "models/neural_sde_q_config.json",
) -> Optional[NeuralSDEQModel]:
    """
    Load trained Q-model from disk.

    Returns None if model file doesn't exist.
    """
    from pathlib import Path as _P
    model_path = _P(path)
    cfg_path = _P(config_path)

    if not model_path.exists():
        return None

    # Load reconstruction config
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
    else:
        cfg = {
            'lambda_max': 3.0, 'kappa': 2.72, 'theta': -3.5,
            'drift_scale': 0.5, 'diff_min': 0.5, 'diff_max': 2.0,
            'rho': -0.9, 'spot': 5500.0, 'r': 0.0373,
            'log_v_min': -7.0, 'log_v_max': 2.0,
        }

    # Create skeleton with matching architecture
    key = jax.random.PRNGKey(0)
    girsanov = GirsanovDrift(key, lambda_max=cfg.get('lambda_max', 3.0))
    skeleton = NeuralSDEQModel(
        girsanov=girsanov,
        kappa=cfg.get('kappa', 2.72),
        theta=cfg.get('theta', -3.5),
        drift_scale=cfg.get('drift_scale', 0.5),
        diff_min=cfg.get('diff_min', 0.5),
        diff_max=cfg.get('diff_max', 2.0),
        rho=cfg.get('rho', -0.9),
        spot=cfg.get('spot', 5500.0),
        r=cfg.get('r', 0.0373),
        log_v_min=cfg.get('log_v_min', -7.0),
        log_v_max=cfg.get('log_v_max', 2.0),
    )

    try:
        return eqx.tree_deserialise_leaves(model_path, skeleton)
    except Exception as e:
        print(f"  [WARN] Could not load Q-model: {e}")
        return None
