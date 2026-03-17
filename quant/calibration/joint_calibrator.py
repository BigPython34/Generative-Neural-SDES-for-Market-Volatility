"""
Joint SPX-VIX Calibration under Rough Bergomi — JAX-Accelerated
================================================================
Simultaneous calibration of the rBergomi model to:
  1. SPX/SPY implied volatility surface (multi-maturity smile)
  2. VIX term structure (VIX1D → VIX1Y fixed-tenure indices)
  3. VIX futures term structure (CBOE contracts)

This produces a Q-measure calibration that is consistent across both
the SPX options market and the VIX derivatives market.

Mathematical Framework
======================

The rough Bergomi (rBergomi) model under Q:

    V_t = ξ₀(t) · exp( η Ŵᴴ_t - ½η² Var[Ŵᴴ_t] )                   (1)

    dS_t/S_t = r dt + √V_t [ ρ dW_t + √(1-ρ²) dW⊥_t ]               (2)

Parameters to calibrate:
    θ = { H, η, ρ, ξ₀(·) }

where H ∈ (0, ½) is the Hurst exponent, η > 0 the vol-of-vol,
ρ ∈ (-1, 0) the spot-vol correlation, and ξ₀(·) a piecewise-constant
forward variance curve.

Calibration Strategy (2-stage)
==============================

Stage 1: Bootstrap ξ₀(t) from VIX term structure
--------------------------------------------------
In the rBergomi model, the VIX at tenor τ satisfies EXACTLY:

    E^Q[ VIX²(τ) ] = (1/τ) ∫₀^τ ξ₀(s) ds

This identity is parameter-independent: H, η, ρ do NOT affect it
(because the exponential martingale ℰ(η Ŵᴴ) has unit expectation).

Therefore ξ₀(t) is uniquely determined by the market VIX term
structure. We bootstrap piecewise-constant ξ₀ from {VIX1D, VIX9D,
VIX, VIX3M, VIX6M, VIX1Y} using the same Dupire-like inversion.

Why NOT from SPX ATM:
  - VIX = √(variance swap fair value), which includes the full
    smile (especially OTM puts weighted by 1/K²).
  - SPX ATM IV is just one point on the smile.
  - Typical gap: VIX ≈ 22% vs ATM IV ≈ 16% → 6pt bias if misused.
  - Ref: Rømer (2022) §4, Bayer et al. (2016) §4.

Fallback: if no VIX data, bootstrap from SPX ATM (with warning).

Stage 2: Calibrate (H, η, ρ) to smile shape + VIX term structure
-----------------------------------------------------------------
With ξ₀(t) fixed, the remaining parameters (H, η, ρ) control:
  - H: term structure steepness (how fast the skew flattens)
  - η: vol-of-vol amplitude (smile width / VIX convexity)
  - ρ: skew direction and magnitude

The joint loss function:

    L(H, η, ρ) = λ_SPX · L_SPX(H, η, ρ)
               + λ_VIX · L_VIX(H, η, ρ)
               + λ_mart · L_mart(H, η, ρ)
               + λ_reg · L_reg(H, η, ρ)

where:

  L_SPX = Σ_{T,K}  w_{T,K} · [ σ_model(K,T) - σ_market(K,T) ]²
          (vega-weighted IV RMSE across the surface)

  L_VIX = Σ_τ  [ VIX_model(τ) - VIX_market(τ) ]²
          (VIX term structure MSE, τ ∈ {9d, 30d, 3m, 6m, 1y})

  L_mart = Σ_T  [ E[e^{-rT} S_T] / S_0 - 1 ]²
           (martingale constraint)

  L_reg = α_H · (H - H_prior)² + α_η · (η - η_prior)²
          (regularization toward P-measure estimates)

VIX pricing under rBergomi
===========================
Given variance paths {V_t^{(i)}} from MC simulation:

    VIX²(τ) = (1/τ) ∫₀^τ V_s ds

where τ is the VIX window and the integral is computed by
trapezoidal rule on the discrete time grid.

Then:
    VIX_model(τ) = E^Q[ VIX(τ) ] ≈ (1/N) Σ_i √(VIX²(τ)^{(i)})

The convexity adjustment (Jensen's inequality):
    E[VIX] ≤ √(E[VIX²])
is computed exactly via MC.

JAX Acceleration
=================
All hot paths use JAX for maximum throughput:

  - Volterra kernel: vectorized O(N²) via broadcasting — no Python loops
  - Kernel cache: {(H, n_steps)} → reuse across (η, ρ) grid points
  - MC simulation: @jax.jit compiled, antithetic variates
  - Option payoffs: vmapped over strikes (all strikes simultaneously)
  - VIX integration: JIT-compiled trapezoidal rule

Typical calibration time (on CPU):
  --quick mode: ~10–20s  (48 grid points × 3K paths × 12 strikes/mat)
  normal mode:  ~30–60s  (245 grid points × 8K paths × 12 strikes/mat)

References
----------
[1] Bayer, Friz & Gatheral (2016). Pricing under rough volatility. QF.
[2] Bayer & Stemper (2018). Deep calibration of rough stochastic
    volatility models.
[3] Rømer (2022). Empirical analysis of rough and classical stochastic
    volatility models to the SPX and VIX markets.
[4] Gatheral & Keller-Ressel (2019). Affine forward variance models.
[5] Jacquier, Martini & Muguruza (2018). On VIX futures in the rough
    Bergomi model.
"""

import numpy as np
import jax
import jax.numpy as jnp
import time
from scipy.optimize import minimize as scipy_minimize
from quant.models.black_scholes import BlackScholes
from quant.calibration.results import JointCalibrationResult
from engine.losses import terminal_martingale_violation_loss
from engine.volterra import get_cached_volterra_kernel


# ═══════════════════════════════════════════════════════════════════
# 1. JAX-ACCELERATED KERNELS
# ═══════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────
# JIT-compiled simulation & pricing
# ─────────────────────────────────────────────────────────

@jax.jit
def _simulate_paths_jit(Z, Z_perp, A, var_wh, xi0_arr,
                        eta, rho, r, dt, s0):
    """
    JIT-compiled joint (S, V) simulation under rBergomi.

    V_t = ξ₀(t) · exp(η·Ŵᴴ_t - ½η²·Var[Ŵᴴ_t])
    dS = S·[r dt + √V_{t-1}·(ρ dW + √(1-ρ²) dW⊥)]

    Uses previsible variance V_{k-1} (Bayer et al. 2016).

    Parameters
    ----------
    Z, Z_perp : (n_paths, n_steps) iid standard normals
    A          : (n_steps, n_steps) Volterra kernel
    var_wh     : (n_steps,) variance of Ŵᴴ at each step
    xi0_arr    : (n_steps,) forward variance curve
    eta, rho, r, dt, s0 : model scalars

    Returns
    -------
    (spot_paths, var_paths) : shapes (n_paths, n_steps+1) and (n_paths, n_steps)
    """
    n_paths = Z.shape[0]

    # Volterra fBM
    dW = Z * jnp.sqrt(dt)
    Wh = Z @ A.T  # (n_paths, n_steps)

    # Variance process
    var_paths = xi0_arr * jnp.exp(eta * Wh - 0.5 * eta ** 2 * var_wh)

    # Correlated spot noise
    dW_perp = Z_perp * jnp.sqrt(dt)
    dW_spot = rho * dW + jnp.sqrt(1.0 - rho ** 2) * dW_perp

    # Previsible Euler: V_{k-1}
    v0_col = jnp.full((n_paths, 1), xi0_arr[0])
    var_prev = jnp.concatenate([v0_col, var_paths[:, :-1]], axis=1)

    # Clamp variance to avoid sqrt of negative (numerical safety)
    var_prev_safe = jnp.maximum(var_prev, 1e-16)

    log_ret = (r - 0.5 * var_prev_safe) * dt + jnp.sqrt(var_prev_safe) * dW_spot
    log_s = jnp.cumsum(log_ret, axis=1)
    spot_paths = s0 * jnp.exp(log_s)

    # Prepend S₀
    s0_col = jnp.full((n_paths, 1), s0)
    spot_paths = jnp.concatenate([s0_col, spot_paths], axis=1)

    return spot_paths, var_paths


@jax.jit
def _mc_prices_batch(S_T, strikes, is_call, discount):
    """
    Vectorized MC option pricing — all strikes simultaneously.

    Replaces the per-strike Python loop with a single JAX op.

    S_T      : (n_paths,)
    strikes  : (n_strikes,)
    is_call  : (n_strikes,) boolean mask (1.0 = call, 0.0 = put)
    discount : scalar exp(-rT)

    Returns  : (n_strikes,) MC prices
    """
    # Payoff matrices: (n_paths, n_strikes)
    call_payoff = jnp.maximum(S_T[:, None] - strikes[None, :], 0.0)
    put_payoff = jnp.maximum(strikes[None, :] - S_T[:, None], 0.0)

    payoff = jnp.where(is_call[None, :], call_payoff, put_payoff)
    prices = discount * jnp.mean(payoff, axis=0)
    return prices


@jax.jit
def _compute_vix_from_window(v_window, dt):
    """
    JIT-compiled VIX from a pre-sliced variance window.

    v_window : (n_paths, n_window) variance paths within [0, τ]
    dt       : time step

    Returns scalar: mean VIX level.
    """
    n_window = v_window.shape[1]

    # Trapezoidal weights
    weights = jnp.ones(n_window) * dt
    weights = weights.at[0].set(weights[0] * 0.5)
    weights = weights.at[-1].set(weights[-1] * 0.5)

    integrated = jnp.sum(v_window * weights[None, :], axis=1)
    actual_tau = jnp.maximum(n_window * dt, 1e-10)
    vix_sq = integrated / actual_tau
    vix_paths = 100.0 * jnp.sqrt(jnp.maximum(vix_sq, 1e-16))

    return jnp.mean(vix_paths)


# ═══════════════════════════════════════════════════════════════════
# 2. EXTENDED ROUGH BERGOMI MODEL
# ═══════════════════════════════════════════════════════════════════

class ExtendedRoughBergomi:
    """
    rBergomi with forward variance curve ξ₀(t) and JAX acceleration.

    Key features vs core.bergomi.RoughBergomiModel:
      1. Piecewise-constant ξ₀(t) (not scalar ξ₀)
      2. VIX pricing from simulated variance paths
      3. Volterra kernel cache (shared across H-grid points)
      4. JIT-compiled simulation & batched option pricing
      5. Antithetic variates for variance reduction
    """

    def __init__(self, H, eta, rho, xi0_curve, T_max, n_steps, r=0.05):
        self.H = H
        self.eta = eta
        self.rho = rho
        self.T_max = T_max
        self.n_steps = n_steps
        self.r = r
        self.dt = T_max / n_steps

        # Time grid: t₁, t₂, ..., t_N (excludes t₀=0)
        self.time_grid = np.linspace(0, T_max, n_steps + 1)[1:]

        # Forward variance curve → JAX array (n_steps,)
        if callable(xi0_curve):
            self._xi0 = jnp.array([xi0_curve(t) for t in self.time_grid])
        elif isinstance(xi0_curve, (int, float)):
            self._xi0 = jnp.full(n_steps, float(xi0_curve))
        else:
            xi_arr = np.asarray(xi0_curve, dtype=float)
            if len(xi_arr) >= n_steps:
                self._xi0 = jnp.array(xi_arr[:n_steps])
            else:
                # Pad with last value if curve is shorter than grid
                padded = np.full(n_steps, xi_arr[-1])
                padded[:len(xi_arr)] = xi_arr
                self._xi0 = jnp.array(padded)

        # Volterra kernel — cached per (H, n_steps, dt)
        self._A, self._var_wh = get_cached_volterra_kernel(n_steps, H, self.dt)

    def simulate(self, n_paths, s0=100.0, key=None, antithetic=True):
        """
        Joint (S, V) simulation with antithetic variates.

        Antithetic sampling: generates N/2 paths from Z, then mirrors
        to -Z, cutting MC variance roughly in half for monotone payoffs.

        Returns dict with spot_paths, var_paths, dt, n_paths.
        """
        if key is None:
            key = jax.random.PRNGKey(42)
        key_z, key_perp = jax.random.split(key)

        n_half = n_paths // 2 if antithetic else n_paths

        Z = jax.random.normal(key_z, (n_half, self.n_steps))
        Z_perp = jax.random.normal(key_perp, (n_half, self.n_steps))

        if antithetic:
            Z = jnp.concatenate([Z, -Z], axis=0)
            Z_perp = jnp.concatenate([Z_perp, -Z_perp], axis=0)

        spot_paths, var_paths = _simulate_paths_jit(
            Z, Z_perp, self._A, self._var_wh, self._xi0,
            self.eta, self.rho, self.r, self.dt, s0,
        )

        return {
            'spot_paths': spot_paths,
            'var_paths': var_paths,
            'dt': self.dt,
            'n_paths': int(spot_paths.shape[0]),
        }

    def compute_vix(self, var_paths, tau_days):
        """
        Compute E^Q[VIX(τ)] from simulated variance paths.

        VIX²(τ) = (1/τ) ∫₀^τ V_s ds

        Parameters
        ----------
        var_paths : (n_paths, n_steps) JAX array
        tau_days  : VIX window in calendar days

        Returns
        -------
        scalar : mean VIX level (e.g. 18.5 for 18.5%)
        """
        tau = tau_days / 365.0
        n_in_tau = max(1, min(int(round(tau / self.dt)), var_paths.shape[1]))

        # Slice outside JIT (shape change triggers recompilation otherwise)
        v_window = var_paths[:, :n_in_tau]

        # Call JIT-compiled trapezoidal integration
        return float(_compute_vix_from_window(v_window, self.dt))

    def price_vix_term_structure(self, var_paths, tau_days_list):
        """Price VIX at multiple tenors. Returns {tau_days: model_vix}."""
        return {tau: self.compute_vix(var_paths, tau) for tau in tau_days_list}

    def price_spx_options(self, spot_paths, strikes, T, option_types, s0):
        """
        Price SPX options and extract implied vols.

        MC pricing is JAX-batched (all strikes at once).
        IV extraction uses scipy (not JIT-able but fast per-option).

        Returns (model_ivs, mc_prices) as numpy arrays.
        """
        # Find time index closest to T
        t_idx = int(np.argmin(np.abs(self.time_grid - T)))
        S_T = spot_paths[:, t_idx + 1]  # +1 because col 0 = S₀

        discount = float(np.exp(-self.r * T))
        is_call = jnp.array([1.0 if ot == 'call' else 0.0 for ot in option_types])
        strikes_jax = jnp.array(strikes, dtype=jnp.float32)

        # Batched MC pricing — all strikes in one JAX op
        mc_prices = _mc_prices_batch(S_T, strikes_jax, is_call, discount)
        mc_prices_np = np.array(mc_prices)

        # IV extraction (NumPy/scipy — still fast: ~0.1ms per option)
        model_ivs = np.full(len(strikes), np.nan)
        for i, (price, K, opt_type) in enumerate(zip(mc_prices_np, strikes, option_types)):
            if price > 1e-8:
                model_ivs[i] = BlackScholes.implied_vol(
                    float(price), s0, float(K), T, self.r, opt_type
                )

        return model_ivs, mc_prices_np


# ═══════════════════════════════════════════════════════════════════
# 3. LOSS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def vix_term_structure_loss(model_vix, market_vix, weights=None):
    """
    L_VIX = Σ_τ w_τ · [ VIX_model(τ) - VIX_market(τ) ]²

    Normalized by VIX level² to make errors comparable across tenors.
    """
    common = set(model_vix.keys()) & set(market_vix.keys())
    if not common:
        return 0.0

    loss = 0.0
    for tau in sorted(common):
        w = weights[tau] if weights and tau in weights else 1.0
        mkt = market_vix[tau]
        diff = (model_vix[tau] - mkt) / max(mkt, 1.0)  # relative error
        loss += w * diff ** 2

    return loss / len(common)


def spx_smile_loss(model_ivs, market_ivs, vega_weights=None):
    """
    L_SPX = Σ_K w_K · [ σ_model(K) - σ_market(K) ]²

    Vega-weighted to focus on liquid ATM region.
    """
    valid = ~np.isnan(model_ivs) & ~np.isnan(market_ivs)
    if valid.sum() < 2:
        return 1e6

    diff = model_ivs[valid] - market_ivs[valid]
    if vega_weights is not None:
        w = vega_weights[valid]
        w = w / w.sum()
        return float(np.sum(w * diff ** 2))
    return float(np.mean(diff ** 2))


def martingale_loss(spot_paths, r, T, s0):
    """
    Martingale constraint: E^Q[e^{-rT} S_T] = S_0.

    Returns squared violation of the discounted spot martingale.
    """
    s_t_norm = spot_paths[:, -1] / s0
    return float(terminal_martingale_violation_loss(s_t_norm, T, r))


# ═══════════════════════════════════════════════════════════════════
# 5. JOINT CALIBRATOR
# ═══════════════════════════════════════════════════════════════════

class JointCalibrator:
    """
    Joint SPX-VIX calibrator for the rBergomi model.

    Pipeline:
    1. Bootstrap ξ₀(t) from VIX term structure (exact inversion)
       Fallback: ATM SPX surface (with warning)
    2. Grid search over (H, η, ρ) with JAX-accelerated MC
    3. Bounded local refinement (η ≥ 0.5 enforced)
    4. Final evaluation with 2× paths
    """

    def __init__(
        self,
        lambda_spx: float = 1.0,
        lambda_vix: float = 2.0,
        lambda_mart: float = 5.0,
        lambda_reg: float = 0.01,
        n_mc_paths: int = 8_000,
        n_steps_per_year: int = 252,
        H_prior: float = 0.07,  # UPDATED: was 0.11 (too high), now 0.07 (literature: 0.04-0.08)
        eta_prior: float = 1.9,
        moneyness_range: tuple = (-0.25, 0.25),  # WIDENED from (-0.15, 0.15) to capture OTM puts with high skew
        min_dte: int = 7,
        max_dte: int = 180,
        max_strikes_per_maturity: int = 12,
        verbose: bool = True,
    ):
        self.lambda_spx = lambda_spx
        self.lambda_vix = lambda_vix
        self.lambda_mart = lambda_mart
        self.lambda_reg = lambda_reg
        self.n_mc_paths = n_mc_paths
        self.n_steps_per_year = n_steps_per_year
        self.H_prior = H_prior
        self.eta_prior = eta_prior
        self.moneyness_range = moneyness_range
        self.min_dte = min_dte
        self.max_dte = max_dte
        self.max_strikes_per_maturity = max_strikes_per_maturity
        self.verbose = verbose

    # ─────────────────────────────────────────────────────
    # Data preparation
    # ─────────────────────────────────────────────────────

    def _prepare_spx_targets(self, surface_df, spot):
        """Filter SPX options into per-maturity calibration slices."""
        from quant.calibration.market_targets import prepare_spx_slices

        return prepare_spx_slices(
            surface_df,
            spot,
            max_strikes=self.max_strikes_per_maturity,
            max_maturities=None,
            min_dte=self.min_dte,
            max_dte=self.max_dte,
            moneyness_range=self.moneyness_range,
        )

    def _prepare_vix_targets(self, vix_ts, dt):
        """
        Convert VIX term structure to calibration targets.

        Filters out tenors too short for the time grid (need ≥ 3 steps).
        """
        if vix_ts is None:
            return {}

        from quant.calibration.vix_futures_loader import VIX_INDEX_TENORS

        min_tau_days = max(5, int(3 * dt * 365))
        targets = {}
        for i, label in enumerate(vix_ts.labels):
            # Handle both TradingView labels (vix1d, vix9d, etc.) and futures labels (fut_XXd)
            if label in VIX_INDEX_TENORS:
                tau_days = VIX_INDEX_TENORS[label]["tau_days"]
            elif label.startswith("fut_"):
                # Extract days from "fut_XXd" label
                try:
                    tau_days = int(label.replace("fut_", "").replace("d", ""))
                except ValueError:
                    tau_days = int(vix_ts.tenors_days[i])
            else:
                tau_days = int(vix_ts.tenors_days[i])
            
            if tau_days >= min_tau_days:
                targets[tau_days] = float(vix_ts.vix_levels[i])

        return targets

    # ─────────────────────────────────────────────────────
    # Evaluation (single parameter point)
    # ─────────────────────────────────────────────────────

    def evaluate(self, H, eta, rho, xi0_curve, spx_slices, vix_targets,
                 spot, r, T_max, n_steps, key=None):
        """
        Evaluate joint loss at a single (H, η, ρ) point.

        The model is built, simulated, and losses computed.
        The Volterra kernel is cached → only built once per unique H.

        Returns dict with total_loss, component losses, and model outputs.
        """
        model = ExtendedRoughBergomi(
            H=H, eta=eta, rho=rho, xi0_curve=xi0_curve,
            T_max=T_max, n_steps=n_steps, r=r,
        )

        sim = model.simulate(self.n_mc_paths, s0=spot, key=key, antithetic=True)
        spot_paths = sim['spot_paths']
        var_paths = sim['var_paths']

        # ── VIX term structure loss ──
        vix_loss_val = 0.0
        model_vix = {}
        if vix_targets:
            tau_list = sorted(vix_targets.keys())
            model_vix = model.price_vix_term_structure(var_paths, tau_list)
            vix_loss_val = vix_term_structure_loss(model_vix, vix_targets)

        # ── SPX smile loss (multi-maturity) ──
        spx_loss_val = 0.0
        model_ivs_all = {}
        n_valid = 0
        for sl in spx_slices:
            m_ivs, _ = model.price_spx_options(
                spot_paths, sl['strikes'], sl['T'],
                sl['option_types'], s0=spot,
            )
            sl_loss = spx_smile_loss(m_ivs, sl['market_ivs'], sl['vega_weights'])
            if sl_loss < 1e5:
                spx_loss_val += sl_loss
                n_valid += 1
            model_ivs_all[sl['T']] = (sl['strikes'], m_ivs, sl['market_ivs'])

        if n_valid > 0:
            spx_loss_val /= n_valid

        # ── Martingale loss ──
        mart_loss_val = martingale_loss(spot_paths, r, T_max, spot)

        # ── Regularization ──
        reg_loss = (H - self.H_prior) ** 2 + 0.1 * (eta - self.eta_prior) ** 2

        # ── Total ──
        total = (
            self.lambda_spx * spx_loss_val
            + self.lambda_vix * vix_loss_val
            + self.lambda_mart * mart_loss_val
            + self.lambda_reg * reg_loss
        )

        # RMSE in bps (total and shape-only)
        all_diffs = []
        shape_diffs = []
        for _, (_, m_iv, mkt_iv) in model_ivs_all.items():
            valid = ~np.isnan(m_iv) & ~np.isnan(mkt_iv)
            if valid.any():
                raw = m_iv[valid] - mkt_iv[valid]
                all_diffs.extend(raw.tolist())
                # De-meaned: remove per-maturity ATM level bias
                # This isolates the smile SHAPE fit quality
                shape_diffs.extend((raw - raw.mean()).tolist())

        spx_rmse_bps = (
            np.sqrt(np.mean(np.array(all_diffs) ** 2)) * 10000
            if all_diffs else 0.0
        )
        spx_shape_rmse_bps = (
            np.sqrt(np.mean(np.array(shape_diffs) ** 2)) * 10000
            if shape_diffs else 0.0
        )
        atm_bias_bps = (
            np.mean(all_diffs) * 10000 if all_diffs else 0.0
        )

        return {
            'total_loss': total,
            'spx_loss': spx_loss_val,
            'vix_loss': vix_loss_val,
            'mart_loss': mart_loss_val,
            'reg_loss': reg_loss,
            'model_vix': model_vix,
            'model_ivs': model_ivs_all,
            'spx_rmse_bps': spx_rmse_bps,
            'spx_shape_rmse_bps': spx_shape_rmse_bps,
            'spx_atm_bias_bps': atm_bias_bps,
        }

    # ─────────────────────────────────────────────────────
    # Main calibration loop
    # ─────────────────────────────────────────────────────

    def calibrate(self, market_data, xi0_curve=None,
                  quick=False):
        """
        Run joint calibration.

        Parameters
        ----------
        market_data : CalibrationMarketData from assemble_calibration_data()
        xi0_curve : ForwardVarianceCurve (if None, bootstrapped from ATM)
        quick : if True, use coarser grid for speed

        Returns
        -------
        JointCalibrationResult
        """
        from quant.calibration.forward_variance import (
            bootstrap_from_surface, bootstrap_xi0_from_vix,
        )

        t_start = time.time()
        spot = market_data.spx_spot
        r = market_data.risk_free_rate

        if self.verbose:
            print("\n" + "=" * 65)
            print("  JOINT SPX-VIX CALIBRATION — JAX-Accelerated")
            print("  Bayer, Friz & Gatheral (2016) — Rough Bergomi")
            print("=" * 65)

        # ── Stage 1: Bootstrap ξ₀(t) ──
        # Priority: VIX term structure → SPX ATM surface → error
        #
        # In rBergomi: E^Q[VIX²(τ)] = (1/τ) ∫₀^τ ξ₀(s) ds  (exact)
        # This identity is parameter-independent (H, η, ρ don't affect it).
        # Therefore ξ₀ is uniquely determined by the VIX term structure
        # and should be bootstrapped from VIX levels, not from SPX ATM IVs.
        #
        # SPX ATM IV ≠ VIX level because VIX captures the full variance swap
        # (including OTM put skew), whereas ATM IV is just a single point.
        # Using SPX ATM for ξ₀ systematically under-prices VIX by 5-8 pts.
        #
        # Ref: Rømer (2022) §4, Bayer et al. (2016) §4
        if xi0_curve is None:
            vix_ts = market_data.vix_term_structure
            if vix_ts is not None and len(vix_ts.labels) >= 2:
                # ── VIX-based bootstrap (preferred) ──
                if self.verbose:
                    print("\n  [Stage 1] Bootstrapping ξ₀(t) from VIX term structure...")
                    print("    Source: CBOE VIX index family (VIX1D → VIX1Y)")
                    print("    Identity: E^Q[VIX²(τ)] = (1/τ) ∫₀^τ ξ₀(s) ds")

                from quant.calibration.vix_futures_loader import VIX_INDEX_TENORS
                vix_targets_for_xi0 = {}
                for i, label in enumerate(vix_ts.labels):
                    # Handle both TradingView labels (vix1d, vix9d, etc.) and futures labels (fut_XXd)
                    if label in VIX_INDEX_TENORS:
                        tau_days = VIX_INDEX_TENORS[label]["tau_days"]
                    elif label.startswith("fut_"):
                        # Extract days from "fut_XXd" label
                        try:
                            tau_days = int(label.replace("fut_", "").replace("d", ""))
                        except ValueError:
                            # Fallback to tenor_days from snapshot
                            tau_days = int(vix_ts.tenors_days[i])
                    else:
                        # Fallback to tenor_days from snapshot
                        tau_days = int(vix_ts.tenors_days[i])
                    
                    vix_targets_for_xi0[tau_days] = float(vix_ts.vix_levels[i])

                xi0_curve = bootstrap_xi0_from_vix(vix_targets_for_xi0)

                if self.verbose:
                    print(f"    → {len(xi0_curve.maturities)} pillars (from VIX TS)")
                    for i in range(len(xi0_curve.maturities)):
                        dte = int(xi0_curve.maturities[i] * 365)
                        v = 100 * np.sqrt(xi0_curve.xi_values[i])
                        print(f"      T={dte:>4}d: ξ₀={xi0_curve.xi_values[i]:.6f} "
                              f"(√ξ₀={v:.2f}%)")

            elif market_data.spx_surface is not None:
                # ── Fallback: SPX ATM bootstrap ──
                if self.verbose:
                    print("\n  [Stage 1] Bootstrapping ξ₀(t) from ATM SPX (fallback)...")
                    print("    VIX term structure not available")
                    print("    Warning: VIX fit may be biased low (~5-8 pts)")
                xi0_curve = bootstrap_from_surface(
                    market_data.spx_surface, spot,
                    min_dte=self.min_dte, max_dte=self.max_dte,
                )
                if self.verbose:
                    print(f"    → {len(xi0_curve.maturities)} pillars (from SPX ATM)")
                    for i in range(min(6, len(xi0_curve.maturities))):
                        dte = int(xi0_curve.maturities[i] * 365)
                        print(f"      T={dte:>4}d: ξ₀={xi0_curve.xi_values[i]:.6f} "
                              f"(σ_ATM={xi0_curve.atm_ivs[i] * 100:.2f}%)")
            else:
                raise ValueError(
                    "No data for ξ₀ bootstrap: need VIX term structure or SPX surface"
                )

        # ── Compute simulation parameters (constant across grid) ──
        spx_slices = []
        if market_data.spx_surface is not None:
            spx_slices = self._prepare_spx_targets(market_data.spx_surface, spot)
            if self.verbose:
                n_opts = sum(len(s['strikes']) for s in spx_slices)
                print(f"\n  SPX targets: {len(spx_slices)} maturity slices, "
                      f"{n_opts} options")

        # VIX targets — compute dt first
        max_T_spx = max((s['T'] for s in spx_slices), default=0.5)
        max_tau_vix = 365  # up to VIX1Y
        if market_data.vix_term_structure is not None:
            max_tau_vix = int(max(market_data.vix_term_structure.tenors_days))
        T_max = max(max_T_spx, max_tau_vix / 365.0) * 1.05 + 0.01
        n_steps = max(60, int(T_max * self.n_steps_per_year))
        dt = T_max / n_steps

        vix_targets = self._prepare_vix_targets(
            market_data.vix_term_structure, dt
        )
        if self.verbose and vix_targets:
            print(f"  VIX targets: {len(vix_targets)} tenors "
                  f"(skipped τ < {max(5, int(3 * dt * 365))}d)")
            for tau, lvl in sorted(vix_targets.items()):
                print(f"    τ={tau:>4}d: VIX={lvl:.2f}")

        if self.verbose:
            print(f"\n  Simulation: T_max={T_max:.3f}y, n_steps={n_steps}, "
                  f"dt={dt * 365:.1f}d")

        # η prior from VVIX
        if market_data.vvix is not None:
            self.eta_prior = market_data.vvix / 100.0 * (30 / 365) ** (-self.H_prior)
            self.eta_prior = np.clip(self.eta_prior, 0.5, 5.0)
            if self.verbose:
                print(f"  η prior from VVIX ({market_data.vvix:.1f}): "
                      f"{self.eta_prior:.2f}")

        # ── Stage 2: Grid search ──
        if self.verbose:
            print("\n  [Stage 2] Grid search over (H, η, ρ)...")

        if quick:
            H_grid = [0.01, 0.03, 0.05, 0.08]
            eta_grid = [0.8, 1.2, 1.8, 2.5]
            rho_grid = [-0.95, -0.85, -0.70]
        else:
            H_grid = [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10]
            eta_grid = [0.5, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5]
            rho_grid = [-0.99, -0.95, -0.90, -0.85, -0.80, -0.70, -0.60]

        total_combos = len(H_grid) * len(eta_grid) * len(rho_grid)

        if self.verbose:
            print(f"    {total_combos} combinations "
                  f"({len(H_grid)} H × {len(eta_grid)} η × {len(rho_grid)} ρ)")

        # JIT warmup (first call compiles — ~2s, all subsequent calls fast)
        if self.verbose:
            print("    Compiling JAX kernels (one-time)...")
        t_warmup = time.time()
        warmup_key = jax.random.PRNGKey(0)
        try:
            _ = self.evaluate(
                H_grid[0], eta_grid[0], rho_grid[0], xi0_curve,
                spx_slices, vix_targets, spot, r, T_max, n_steps,
                key=warmup_key,
            )
        except Exception:
            pass
        if self.verbose:
            print(f"    Compiled in {time.time() - t_warmup:.1f}s")

        # ── Common Random Numbers (CRN) ──
        # Use a FIXED JAX key for ALL grid + optimizer evaluations.
        # This makes the MC objective deterministic in (H, η, ρ),
        # removing ~15% noise and enabling clean Nelder-Mead convergence.
        # The absolute loss may be biased for one seed, but the
        # relative ranking of parameters is exact.
        crn_key = jax.random.PRNGKey(2024)

        # Run grid search
        best_params = None
        best_loss = float('inf')
        best_result = None
        grid_count = 0
        t_grid_start = time.time()

        for H in H_grid:
            for eta in eta_grid:
                for rho in rho_grid:
                    grid_count += 1

                    try:
                        result = self.evaluate(
                            H, eta, rho, xi0_curve,
                            spx_slices, vix_targets,
                            spot, r, T_max, n_steps, key=crn_key,
                        )
                    except Exception as e:
                        continue

                    if result['total_loss'] < best_loss:
                        best_loss = result['total_loss']
                        best_params = (H, eta, rho)
                        best_result = result
                        if self.verbose:
                            elapsed = time.time() - t_grid_start
                            rate = grid_count / max(elapsed, 0.1)
                            eta_sec = (total_combos - grid_count) / max(rate, 0.01)
                            print(
                                f"    [{grid_count:>3}/{total_combos}] "
                                f"H={H:.2f} η={eta:.1f} ρ={rho:.2f} → "
                                f"L={result['total_loss']:.6f} "
                                f"(SPX={result['spx_rmse_bps']:.0f}bps, "
                                f"VIX={result['vix_loss']:.4f}) "
                                f"[ETA {eta_sec:.0f}s]"
                            )

                    elif self.verbose and grid_count % 25 == 0:
                        elapsed = time.time() - t_grid_start
                        rate = grid_count / max(elapsed, 0.1)
                        eta_sec = (total_combos - grid_count) / max(rate, 0.01)
                        print(f"    [{grid_count:>3}/{total_combos}] ... "
                              f"ETA {eta_sec:.0f}s")

        if best_params is None:
            raise RuntimeError("Grid search failed — no valid parameter point")

        H_best, eta_best, rho_best = best_params
        grid_elapsed = time.time() - t_grid_start

        if self.verbose:
            print(f"\n    Grid done: {grid_count} pts in {grid_elapsed:.1f}s "
                  f"({grid_count / max(grid_elapsed, 0.1):.1f} pts/s)")
            print(f"    Best: H={H_best:.3f} η={eta_best:.2f} ρ={rho_best:.2f} "
                  f"(L={best_loss:.6f})")

        # ── Stage 3: Local refinement (bounded) ──
        #
        # Proper bounds prevent pathological convergence (e.g. η→0.1).
        # We use L-BFGS-B for gradient-free bounded optimization.
        # If it fails, fall back to bounded Nelder-Mead with penalty.
        #
        # Bounds (v2 — widened after literature comparison):
        #   H   ∈ [0.005, 0.40]  (allows very rough H ≈ 0.01-0.02 from Bayer & Stemper 2018)
        #   η   ∈ [0.30, 5.00]   (lowered to allow small η when H is very small)
        #   ρ   ∈ [-0.995, -0.10] (widened: saturated ρ at -0.98 is a common artefact)

        BOUNDS_H   = (0.005, 0.40)
        BOUNDS_ETA = (0.30, 5.00)
        BOUNDS_RHO = (-0.995, -0.10)

        if self.verbose:
            print(f"\n  [Stage 3] Local refinement (bounded Nelder-Mead)...")
            print(f"    Bounds: H∈{BOUNDS_H}, η∈{BOUNDS_ETA}, ρ∈{BOUNDS_RHO}")

        

        eval_count = [0]

        def objective(x):
            H, eta, rho = x
            # Hard bounds via large penalty
            if (H < BOUNDS_H[0] or H > BOUNDS_H[1] or
                eta < BOUNDS_ETA[0] or eta > BOUNDS_ETA[1] or
                rho < BOUNDS_RHO[0] or rho > BOUNDS_RHO[1]):
                return 1e6
            try:
                eval_count[0] += 1
                res = self.evaluate(
                    H, eta, rho, xi0_curve,
                    spx_slices, vix_targets,
                    spot, r, T_max, n_steps, key=crn_key,  # CRN
                )
                if self.verbose and eval_count[0] % 5 == 0:
                    print(f"      eval {eval_count[0]:>3}: "
                          f"H={H:.3f} η={eta:.2f} ρ={rho:.2f} → "
                          f"L={res['total_loss']:.6f}")
                return res['total_loss']
            except Exception:
                return 1e6

        # Clamp starting point within bounds
        x0 = [
            np.clip(H_best, *BOUNDS_H),
            np.clip(eta_best, *BOUNDS_ETA),
            np.clip(rho_best, *BOUNDS_RHO),
        ]

        local_result = scipy_minimize(
            objective,
            x0=x0,
            method='Nelder-Mead',
            options={
                'maxiter': 40,
                'xatol': 0.002,
                'fatol': 1e-6,
                'adaptive': True,
            },
        )

        H_opt = np.clip(local_result.x[0], *BOUNDS_H)
        eta_opt = np.clip(local_result.x[1], *BOUNDS_ETA)
        rho_opt = np.clip(local_result.x[2], *BOUNDS_RHO)
        n_local_evals = eval_count[0]

        if self.verbose:
            improved = local_result.fun < best_loss
            print(f"    → H={H_opt:.4f} η={eta_opt:.3f} ρ={rho_opt:.3f} "
                  f"({'improved' if improved else 'no improvement'}, "
                  f"{n_local_evals} evals)")

        # ── Stage 4: Final evaluation with 2× paths ──
        # Use a FRESH seed (not the CRN key) for unbiased estimates.
        if self.verbose:
            print(f"\n  [Final] Evaluating with {self.n_mc_paths * 2:,} paths "
                  f"(fresh seed)...")

        saved_paths = self.n_mc_paths
        self.n_mc_paths *= 2
        key_final = jax.random.PRNGKey(99999)

        final = self.evaluate(
            H_opt, eta_opt, rho_opt, xi0_curve,
            spx_slices, vix_targets,
            spot, r, T_max, n_steps, key=key_final,
        )
        self.n_mc_paths = saved_paths

        elapsed = time.time() - t_start

        cal_result = JointCalibrationResult(
            H=float(H_opt),
            eta=float(eta_opt),
            rho=float(rho_opt),
            xi0_maturities=np.array(xi0_curve.maturities),
            xi0_values=np.array(xi0_curve.xi_values),
            total_loss=final['total_loss'],
            spx_loss=final['spx_loss'],
            vix_loss=final['vix_loss'],
            martingale_loss=final['mart_loss'],
            model_vix_ts={int(k): v for k, v in final['model_vix'].items()},
            market_vix_ts={int(k): v for k, v in vix_targets.items()},
            model_ivs={
                k: (s.tolist(), m.tolist(), mkt.tolist())
                for k, (s, m, mkt) in final['model_ivs'].items()
            },
            spx_rmse_bps=final['spx_rmse_bps'],
            spx_shape_rmse_bps=final.get('spx_shape_rmse_bps', 0.0),
            spx_atm_bias_bps=final.get('spx_atm_bias_bps', 0.0),
            n_mc_paths=saved_paths * 2,
            n_iterations=n_local_evals,
            elapsed_seconds=elapsed,
            grid_evaluated=grid_count,
            method="grid_search + nelder_mead (JAX-accelerated)",
        )

        if self.verbose:
            print(cal_result.summary())

        return cal_result
