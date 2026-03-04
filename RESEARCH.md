# DeepRoughVol — Research Journey

> *This document traces the intellectual path behind [DeepRoughVol](README.md): the false starts, the critical insights, and the design decisions that shaped the project. It is written as a narrative for anyone who wants to understand not just what the code does, but why it is built this way.*

---

## Table of Contents

1. [Why This Project Exists](#why-this-project-exists)
2. [What DeepRoughVol Does That Black-Scholes Cannot](#what-deeproughvol-does-that-black-scholes-cannot)
3. [The Research Phases](#the-research-phases)
4. [Key Code Insights](#key-code-insights)
5. [Lessons Learned](#lessons-learned)

---

## Why This Project Exists

Most quantitative finance projects fall into one of three buckets: implement Black-Scholes (trivial, unrealistic flat vol), use Heston/SABR (better, but smooth vol — misses the fractal structure of real markets), or train a neural network on prices (no financial theory, no interpretability).

DeepRoughVol started from a different premise: **volatility is rough**. Gatheral, Jaisson & Rosenbaum proved it empirically in 2018 — log-realized-volatility of equity indices behaves like fractional Brownian motion with Hurst exponent $H \approx 0.1$, far below the $H = 0.5$ of standard Brownian motion. This is not a minor correction. It means volatility paths are vastly more irregular than anything Heston or SABR can produce, and that increments are negatively correlated (anti-persistent) — a positive vol shock is more likely followed by a negative one, creating the jagged, erratic behavior seen in real markets.

Every design choice in this project follows from that fact:

| Feature | Standard Quant Projects | DeepRoughVol |
|---|---|---|
| Vol dynamics | Smooth (Heston, SABR) | **Rough** ($H \approx 0.1$, verified on real SPX data) |
| Memory | Markovian (no path memory) | **Path-dependent** (signature conditioning via Chen's identity) |
| Architecture | Parametric model OR black-box NN | **Neural SDE = OU/Volterra prior + learned corrections** |
| Measures | Single measure | **P-measure (risk) and Q-measure (pricing) with separate losses** |
| Calibration | Manual / grid search | **Auto-calibrated** from VVIX, VIX futures, SOFR, SPX returns |
| Verification | "Trust the model" | **Quantitative proofs**: variogram, structure function, signature correlation, MMD |

---

## What DeepRoughVol Does That Black-Scholes Cannot

### Learning Volatility Dynamics from Data

Black-Scholes assumes constant vol. Heston assumes a fixed parametric form (mean-reverting CIR). The Neural SDE learns dynamics directly from market data. The signature conditioning gives it memory of the entire path history, not just the current state. You prove roughness on your data (`bin/verify_roughness.py`), train the model — it learns the rough dynamics automatically (`bin/train_multi.py --measure P`), and the generated paths exhibit realistic long memory, fat tails, and leverage effect.

### Separate P-Measure and Q-Measure Models

In most projects, a single model is trained on historical data and used for pricing. This is mathematically wrong: the probability measure under which volatility moves in the real world ($\mathbb{P}$) is not the measure under which derivatives are priced ($\mathbb{Q}$). DeepRoughVol trains separate models with different loss functions:

- **P-measure**: trained on realized vol (SPX 5-min returns) with MMD loss → for VaR, CVaR, stress testing
- **Q-measure**: trained on VIX (implied vol) with IV smile fit + martingale constraint $E^Q[e^{-rT}S_T] = S_0$ → for option pricing, hedging

Mixing measures gives biased results in both directions.

### Exotic Pricing Where Rough Vol Matters Most

Exotic payoffs depend on the entire path, not just terminal values. Cliquets depend on forward skew. Autocallables depend on vol term structure. Variance swaps depend on realized vol dynamics. BS gets all of these wrong because it assumes flat vol. The impact is massive:

| Product | Why Rough Vol Matters | Typical BS Error |
|---|---|---|
| **Cliquet** | Forward skew completely wrong under flat vol | 50–200% |
| **Variance swap** | RV dynamics ≠ constant σ² | 30–100% |
| **Autocallable** | Barrier crossing probability is path-dependent | 20–50% |
| **Lookback** | Max/min depend on path roughness | 15–30% |

### Risk Management with Realistic Tail Behavior

Parametric VaR assumes normal returns. Historical VaR uses past data directly. Neither captures the non-linear, path-dependent tail risk of a derivatives portfolio. Monte Carlo VaR using the P-measure Neural SDE generates paths with realistic fat tails, leverage effect ($\rho \approx -0.7$), and vol clustering — all learned from data.

### Neural SDE Greeks via JAX Autodiff

BS delta assumes flat vol. In reality, vol moves with spot ($\rho \approx -0.7$) — BS delta systematically underhedges in crashes and overhedges in rallies. DeepRoughVol computes $\partial C / \partial S$ via JAX autodiff through the full MC pipeline, capturing the stochastic vol structure exactly.

### Regime-Adaptive Parameters

Vol dynamics change across market regimes. A model calibrated in calm markets (VIX ≈ 12) performs poorly in crisis (VIX ≈ 45). The 5-signal regime detector (VIX level, VVIX, term structure, VRP, VIX percentile) provides per-regime parameter overrides:

| Regime | VIX Range | $H$ | $\eta$ | $\rho$ |
|---|:---:|:---:|:---:|:---:|
| Calm | < 13 | 0.10 | 1.5 | −0.65 |
| Normal | 13–20 | 0.07 | 1.9 | −0.70 |
| Stressed | 20–30 | 0.06 | 2.5 | −0.80 |
| Crisis | > 30 | 0.03 | 3.5 | −0.90 |

In crisis: roughness increases ($H$ drops), vol-of-vol explodes ($\eta$ triples), correlation deepens ($\rho \to -0.9$).

### Summary

| Capability | Black-Scholes | Heston/SABR | **DeepRoughVol** |
|---|:---:|:---:|:---:|
| Realistic vol dynamics | No (flat) | Partial (smooth) | **Yes** (rough, learned) |
| Path memory | No (Markov) | No (Markov) | **Yes** (signatures) |
| Forward skew (cliquets) | No | Partial | **Yes** |
| Separate P/Q measures | No | No | **Yes** |
| Auto-calibration | No | Manual | **Yes** (VVIX, SOFR, VIX futures) |
| Regime adaptation | No | No | **Yes** (5-signal detector) |
| Neural SDE Greeks | No | FD only | **Yes** (exact JAX AD) |
| Walk-forward validation | n/a | No | **Yes** (per-fold recalibration) |

---

## The Research Phases

### Phase 1–2: Initial Implementation

The project began with a straightforward Neural SDE implementation conditioned on path signatures (Kidger et al. 2021). Early on, two basic training bugs were discovered: weight persistence across training epochs and a dimension mismatch in the signature input. Fixing these was the first taste of a recurring lesson — numerical code that runs without errors can still be deeply wrong.

### Phase 3: The P ≠ Q Problem

This was the first critical insight of the project. VIX is not "volatility" — it is the risk-neutral expectation of future realized variance (a $\mathbb{Q}$-measure object). Training on VIX therefore means training in the risk-neutral measure directly, which is correct for pricing but wrong for risk management. This led to the fundamental architectural decision: separate P-measure and Q-measure models with different loss functions.

### Phase 4–8: Robustness Audit

A systematic audit revealed multiple issues:
- **Leverage effect**: The correlation $\rho(\text{SPX returns}, \Delta\text{VIX}) = -0.86$ must be preserved in the model. This rules out architectures that generate spot and vol independently.
- **VIX smoothing trap**: VIX is a 30-day integral of variance. Integration smooths roughness — VIX shows $H \approx 0.5$ even though the underlying variance process has $H \approx 0.1$. Roughness must be estimated from realized vol computed from intraday SPX returns, not from VIX.
- **Bergomi implementation bugs**: The Davies-Harte variance was 500× too low, the skew was reversed due to fGn/BM confusion, and the Euler scheme used $V_k$ instead of $V_{k-1}$ (adaptedness violation).

### Phase 9–13: The Critical Fixes

This was the humbling phase — discovering that multiple components had subtle but material numerical bugs:
- A 328× temporal mismatch: the time step was $dt = 1/N$ instead of $T/N$ (in annual units), meaning the model was simulating at the wrong time scale.
- Jensen bias: the mean penalty was computed in variance space ($E[V]$), but the model works in log-variance space. Since $E[e^X] > e^{E[X]}$, this created a systematic upward bias. The fix: penalize in log-variance space.
- Signature normalization: the time component dominated the variance component, starving the neural network of useful information.
- Non-deterministic validation: different random seeds between training epochs made early stopping noisy.
- Signature state reset: resetting the signature between data blocks destroyed the path memory that makes the model non-Markovian.

### Phase 14 (v2.0): Multi-Measure Architecture

With the foundations fixed, the full quant toolkit was built:
- Separate P and Q models with appropriate losses (MMD for P, smile fit + martingale for Q)
- Real SOFR rates replacing hardcoded $r = 5\%$
- VVIX-calibrated vol-of-vol ($\eta = 1.33$ from VVIX vs $1.9$ hardcoded — a 30% difference)
- Merton jump-diffusion for crisis modeling
- Full exotic pricing suite (Asian, lookback, barrier, autocallable, cliquet, variance/vol swaps)
- Risk engine (VaR/CVaR/stressed VaR), hedging simulator (BS/Bartlett), P&L attribution
- Regime detection with adaptive parameters
- REST API (FastAPI) with interactive dashboard

### Phase 15 (v3.0): Mathematical Audit

A comprehensive mathematical review produced 13 improvements. The most impactful:
- **Fractional backbone**: A Volterra kernel with learnable $(H, \eta)$ that nests rBergomi as a special case. When the neural corrections are zero, the model exactly recovers Bayer, Friz & Gatheral (2016). When they are nonzero, it goes beyond rBergomi.
- **Exact Chen's identity**: The signature computation was rewritten from scratch for orders 2, 3, and 4, with mathematically exact tensor updates using `jax.lax.scan`.
- **Jensen bias fix**: Mean penalty moved to log-variance space.
- **Walk-forward recalibration**: Per-fold $(\xi_0, H, \eta)$ recalibration using only past data.
- **Neural SDE Greeks**: Model-implied $\Delta$, $\Gamma$, Vega via JAX autodiff through the full MC pipeline.

### Phase 16 (v3.3): Multi-Scale Hurst Estimation

The centerpiece empirical result. A rigorous 1050-line estimation library (`quant/hurst_estimation.py`) implementing 4 independent estimators (variogram, structure function, ratio, DMA) across 5 time scales (5-min to daily), with 500-iteration block bootstrap confidence intervals and inverse-variance weighted consensus. The result:

$$\boxed{H_{\text{consensus}} = 0.110 \pm 0.003}$$

Key findings:
- Roughness is universal across assets: SPX ($H = 0.110$), SPY ($H = 0.111$), CAC 40 ($H = 0.090$)
- The process is monofractal ($\zeta(q) = qH$ with $R^2 > 0.998$) — a single $H$ suffices
- VIX shows $H \approx 0.47$ as expected (integration smoothing, not a contradiction)
- The DMA estimator gives $H \approx 0.4$ (known upward bias for rough processes) — naturally downweighted by inverse-variance scheme

### Phase 17 (v3.4): Formal Mathematical Proofs

Seven formal theorems and propositions were written to ground every estimator in the Hurst pipeline. All turn out to be standard results in the literature:
- Kolmogorov–Čentsov regularity → Hölder exponent of fBM equals $H$ a.s.
- Variogram consistency (Gatheral et al. 2018 §3)
- Structure function & monofractality (Frisch 1995)
- Ratio estimator (Istas & Lang 1997)
- TSRV bias correction (Zhang, Mykland & Aït-Sahalia 2005)
- Integration smoothing → VIX $H \approx 0.5$ (Bennedsen et al. 2016)
- Inverse-variance BLUE (Gauss-Markov theorem)
- Block bootstrap validity (Politis & Romano 1994)

Full proofs are available in the git history; the README now cites the original papers.

### Phase 18 (v3.5): Joint SPX-VIX Calibration

The most technically challenging phase. Simultaneous calibration of rBergomi to SPX options smile + VIX term structure under $\mathbb{Q}$:

1. **ξ₀ bootstrap from VIX**: The forward variance curve is uniquely determined by the VIX term structure via $E^Q[\text{VIX}^2(\tau)] = (1/\tau)\int_0^\tau \xi_0(s)\,ds$ — this identity is parameter-independent.
2. **JAX-accelerated grid search**: Vectorized Volterra kernel, batched option pricing, Common Random Numbers for deterministic optimization.
3. **The SPX-VIX puzzle**: The calibrated model fits VIX to < 0.5 pts at 30d+ tenors, but SPX ATM IV shows a +478 bps bias — the well-documented joint calibration puzzle (Guyon 2019, Rømer 2022). The rBergomi model cannot simultaneously match VIX levels and SPX ATM IV because the gap reflects the variance swap convexity premium from OTM put skew. This is a model limitation, not a bug.
4. **Performance**: 10× speedup (242s → 27s) via strike subsampling, kernel cache, and CRN.

Calibrated Q-measure parameters: $H_Q = 0.005$, $\eta = 1.341$, $\rho = -0.959$ (with refined grid, bounds widened to $H \in [0.005, 0.40]$, $\eta \in [0.30, 5.00]$, $\rho \in [-0.995, -0.10]$).

### Phase 19 (v3.6): Neural SDE Q-Calibration via Girsanov

The rBergomi calibration produces a fixed parametric model. This phase extends it by learning a **risk-neutral drift correction** $\lambda_\phi(v,t)$ via Girsanov's theorem — a neural network P→Q bridge.

**Core idea**: The P-measure Neural SDE has drift $\mu^P$ and diffusion $\sigma$. Girsanov preserves the diffusion (this is a theorem, not an approximation) — only the drift changes:

$$d\log V_t = [\mu^P - \lambda_\phi(V,t)\cdot\sigma]\,dt + \sigma\,dW^Q_t$$

The **market price of variance risk** $\lambda_\phi$ is parameterized as a small MLP (2→32→32→1, GELU, tanh × $\lambda_{\max}$) — only **1,185 trainable parameters**.

**Key technical challenges solved**:
1. **equinox freezing**: P-model parameters frozen via `eqx.partition` + `jax.tree_util.tree_map_with_path` — only Girsanov weights trained
2. **JIT compatibility**: Replaced `@jax.jit` with `@eqx.filter_jit` so activation functions (non-array leaves) pass through JAX compilation
3. **Price-space loss**: Replaced IV-extraction (scipy bisection, not differentiable) with vega-normalized price loss (fully JAX-traceable, equivalent to first order)
4. **OU parameter mapping**: Naive $\kappa = \eta^2/(2H)$ diverges for $H \ll 0.5$ — clipped to moderate range and let the Girsanov MLP compensate
5. **Time-aware MLP**: Added normalized time $t/T$ as input to the Girsanov MLP (not constant 0.5)

**Data consumption**: 12 VIX tenors (6 indices + 6 futures), 12 SPX maturities × 8 strikes = 96 options, VVIX, SOFR.

**Results**: Total Q-loss = 0.097, martingale error < $10^{-6}$, VIX 30d fit ±0.24 pts, training time 94s on CPU.

---

## Key Code Insights

The following snippets illustrate the core technical ideas. They are extracted from the actual codebase.

### 1. Exact Chen's Identity for Path Signatures

Path signatures make the Neural SDE non-Markovian: the drift and diffusion networks receive the full history of the path, not just the current state. The implementation uses `jax.lax.scan` for a differentiable, GPU-compatible computation of Chen's identity up to order 4.

```python
# From engine/signature_engine.py — Chen's identity, order 3 (exact)

@staticmethod
def _make_scan_step_order3():
    """
    For a path X_t with increment dx = X_{t+1} - X_t:
      S^1 += dx
      S^2 += S^1 ⊗ dx + ½ dx ⊗ dx
      S^3 += S^2 ⊗ dx + S^1 ⊗ (½ dx⊗dx) + (1/6) dx⊗dx⊗dx

    This is mathematically exact (Chen 1957).
    """
    def scan_step(carry, dx):
        s0, s1, s2, s3, s4 = carry
        dx_outer = jnp.outer(dx, dx).flatten()

        new_s1 = s1 + dx
        new_s2 = s2 + jnp.outer(s1, dx).flatten() + 0.5 * dx_outer
        new_s3 = (s3
                  + jnp.kron(s2, dx)
                  + jnp.kron(s1, 0.5 * dx_outer)
                  + (1.0 / 6.0) * jnp.kron(dx, jnp.kron(dx, dx)))
        return (s0, new_s1, new_s2, new_s3, s4), None
    return scan_step
```

At order 3 with 2D input (time + log-variance), this produces a 14-dimensional signature feature vector — a compact, mathematically principled summary of the entire path history.

### 2. JAX-Vectorized Volterra Kernel

The rBergomi variance process depends on the Volterra (Riemann-Liouville) representation of fractional Brownian motion. The naive implementation uses Python loops — $O(N^2)$ in Python. The JAX version vectorizes the entire kernel construction into a single broadcasting operation, compiling to XLA for ~1000× speedup.

```python
# From quant/calibration/joint_calibrator.py — Volterra kernel

def build_volterra_kernel(n_steps: int, H: float, dt: float):
    """
    A[j,k] = √(2H) · dt^H / (H+½) · [(j−k+1)^{H+½} − (j−k)^{H+½}]
    for k ≤ j (lower triangular).

    O(N²) memory, O(1) Python ops (all vectorized in XLA).
    For N=252: ~1ms (vs ~1s with loops).
    """
    alpha = H + 0.5
    C = jnp.sqrt(2.0 * H) * dt ** H / alpha

    j_idx = jnp.arange(n_steps, dtype=jnp.float32)[:, None]  # (n, 1)
    k_idx = jnp.arange(n_steps, dtype=jnp.float32)[None, :]  # (1, n)
    lag = jnp.maximum(j_idx - k_idx, 0.0)                     # (n, n)

    A_raw = C * ((lag + 1.0) ** alpha - lag ** alpha)
    mask = k_idx <= j_idx
    A = jnp.where(mask, A_raw, 0.0)

    var_wh = jnp.sum(A ** 2, axis=1)  # Var[Ŵᴴ(tⱼ)]
    return A, var_wh
```

The kernel is cached per $(H, N, \Delta t)$ — during grid search over $H$, each kernel is built once and reused for all $(\eta, \rho)$ combinations.

### 3. Multi-Scale RBF MMD Loss

The training objective for the P-measure model is the Maximum Mean Discrepancy (MMD) between the distribution of generated and real path signatures. Multi-scale RBF kernels with median heuristic bandwidth selection make the test sensitive across all signature scales simultaneously.

```python
# From engine/losses.py — kernel MMD² (simplified)

@partial(jit, static_argnames=['n_bandwidths'])
def kernel_mmd_loss(fake_signatures, real_signatures, sig_std=None, n_bandwidths=5):
    """
    True kernel MMD² with multi-scale RBF kernel and median heuristic.
    Compares the FULL DISTRIBUTION of path signatures (not just means).
    """
    if sig_std is not None:
        weights = jnp.where(sig_std > 1e-8, 1.0 / sig_std, 0.0)
        x, y = real_signatures * weights, fake_signatures * weights
    else:
        x, y = real_signatures, fake_signatures

    # Median heuristic: bandwidth = median of pairwise distances
    all_pts = jnp.concatenate([x, y], axis=0)
    # ... compute pairwise distances, set bandwidth grid ...
    # MMD² = E[k(x,x')] + E[k(y,y')] - 2·E[k(x,y)]
    for sigma in bandwidth_grid:
        K_xx = _rbf_kernel(x, x, sigma)
        K_yy = _rbf_kernel(y, y, sigma)
        K_xy = _rbf_kernel(x, y, sigma)
        mmd += K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
    return mmd / n_bandwidths
```

### 4. Fractional Backbone — Nesting rBergomi in a Neural SDE

The fractional backbone uses learnable parameters $(H, \eta)$ stored in logit/log space for unconstrained optimization. When the neural drift and diffusion corrections are zero, the model exactly recovers the rBergomi variance process $V_t = \xi_0 \exp(\eta \hat{W}^H_t - \frac{1}{2}\eta^2 \text{Var}[\hat{W}^H_t])$.

```python
# From engine/neural_sde.py — Learnable fractional parameters

class FractionalParams(eqx.Module):
    log_H: jnp.ndarray    # logit(H/0.5) → H ∈ (0, 0.5) via sigmoid
    log_eta: jnp.ndarray  # log(η) → η > 0

    def __init__(self, key, H_init=0.10, eta_init=1.9):
        self.log_H = jnp.array(jnp.log(H_init / (0.5 - H_init)))
        self.log_eta = jnp.array(jnp.log(eta_init))

    @property
    def H(self):
        """Hurst exponent, constrained to (0, 0.5)."""
        return 0.5 * jax.nn.sigmoid(self.log_H)

    @property
    def eta(self):
        """Vol-of-vol, constrained > 0."""
        return jnp.exp(self.log_eta)
```

This is the key architectural idea: a **physics-informed prior** (rBergomi) augmented with **learned corrections** (neural networks). The model can always fall back to the known-good parametric form, but it can also capture dynamics that rBergomi misses.

---

## Lessons Learned

These are the hard-won insights from 18 phases of development. Each cost real debugging time.

1. **P ≠ Q**: Training on realized vol and testing on implied vol is fundamentally wrong. Use VIX for Q-measure, SPX RV for P-measure. Mixing measures gives biased results in both directions. The Girsanov theorem provides the correct P→Q bridge: it preserves the diffusion and only changes the drift by $-\lambda_\phi \cdot \sigma$.

2. **Temporal scale must be physical**: `dt = T/n_steps` in annual units, not `1/n_steps`. Getting this wrong introduces a 328× error that is invisible in the code but catastrophic in the outputs.

3. **VIX ≠ Volatility for roughness**: VIX is a 30-day integral of variance. Integration smooths roughness ($H_{\text{VIX}} \approx 0.5$). True roughness ($H \approx 0.1$) can only be measured from realized vol computed from intraday SPX returns.

4. **Jensen's inequality bites silently**: $E[e^X] > e^{E[X]}$. If you penalize $E[V]$ but the model works in log-variance, the mean penalty is systematically biased upward. The fix: penalize in log-variance space directly.

5. **fBM ≠ fGn for correlation**: The Volterra kernel representation of fBM preserves the spot-vol correlation structure. The Davies-Harte (spectral) method for fGn does not — it can reverse the skew sign.

6. **Previsible variance in Euler schemes**: The variance at step $k$ should use $V_{k-1}$ (known at time $t_{k-1}$), not $V_k$ (which depends on the current Brownian increment). This adaptedness requirement is easy to violate and hard to diagnose.

7. **Martingale matters for Q-measure**: Without the explicit constraint $E^Q[e^{-rT}S_T] = S_0$, risk-neutral pricing is systematically biased. The error grows with maturity.

8. **Real rates matter**: SOFR (3.73%) vs hardcoded 5% makes a material difference for maturities beyond 3 months.

9. **Vol-of-vol from VVIX beats heuristics**: Direct calibration ($\eta = 1.33$ from VVIX) is 30% different from the initial manual estimate ($\eta = 1.9$).

10. **Signatures propagate across blocks**: Resetting the signature state between data blocks destroys the path memory that makes the model non-Markovian. The signature must carry forward via Chen's identity.

---

*← Back to [README.md](README.md)*
