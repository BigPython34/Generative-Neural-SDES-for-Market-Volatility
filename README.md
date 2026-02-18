# DeepRoughVol: Neural Stochastic Volatility with Market Calibration

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![JAX](https://img.shields.io/badge/JAX-0.4-orange)
![Equinox](https://img.shields.io/badge/Equinox-Neural%20SDE-green)
![License](https://img.shields.io/badge/License-MIT-grey)

> **A research project exploring Neural SDEs for volatility modeling, from theoretical foundations to market-calibrated option pricing.**

---

## Table of Contents

1. [Abstract](#abstract)
2. [Key Results](#key-results)
3. [Methodology](#methodology)
4. [Data Sources](#data-sources)
5. [Research Journey](#research-journey)
6. [Market Insights](#market-insights)
7. [Architecture](#architecture)
8. [Usage](#usage)
9. [File Structure](#file-structure)
10. [Lessons Learned](#lessons-learned)
11. [Future Work](#future-work)
12. [References](#references)

---

## Abstract

**DeepRoughVol** is a non-parametric generative framework that learns volatility dynamics directly from market data using **Neural Stochastic Differential Equations (Neural SDEs)** conditioned on **Path Signatures**.

Unlike traditional stochastic volatility models (Heston, SABR, Bergomi) which impose rigid parametric forms, this approach:
- Learns drift and diffusion functions from data via signature-conditioned MLPs
- Captures rough volatility behavior naturally ($H < 0.5$) through path-dependent dynamics
- Calibrates to real SPY option prices
- Incorporates market observables (VVIX, VIX Futures term structure)

The model is benchmarked against a corrected **Rough Bergomi** (rBergomi) implementation using the Volterra kernel for exact spot-vol correlation, and evaluated on real SPY options surfaces from Yahoo Finance.

---

## Key Results

### Statistical Fit (Generated vs Real VIX Variance Paths)

| Metric | Market | Neural SDE | Bergomi | Status |
|--------|--------|------------|---------|--------|
| **Mean Variance** | 0.0372 | **0.0373** | 0.019 | < 0.3% error |
| **Median Variance** | 0.0294 | **0.0293** | 0.018 | Exact match |
| **Kurtosis (marginal)** | 38.5 | **38.9** | 0.3 | Fat tails captured |
| **Hurst H** (on training data) | 0.475 | **0.476** | 0.256 | Gap = 0.001 |
| **ACF(1)** | 0.69 | **0.67** | 0.24 | Memory structure preserved |
| **Signature Correlation** | — | **0.986** | — | Distribution match |

*Source: `outputs/roughness_verification.json`*

### Risk-Neutral IV Calibration

| Model | IV RMSE | |
|-------|---------|---|
| **Neural SDE** | **2.95%** | Winner |
| Black-Scholes | 3.94% | Baseline |

*Source: `outputs/risk_neutral_calibration.json`, calibrated on 31-DTE SPY smile.*

### Options Surface Calibration (31 DTE, 138 options)

| Model | IV RMSE | |
|-------|---------|---|
| **Neural SDE** | **5.29%** | Winner |
| Black-Scholes | 6.41% | Baseline |
| Rough Bergomi | 7.92% | H=0.2, eta=1.0, rho=-0.8 |

*Source: `outputs/calibration_report.json`*

### Walk-Forward Backtest (Feb 2026, 24 scenarios)

| Model | Mean IV RMSE | |
|-------|-------------|---|
| **Rough Bergomi** | **3.46%** | Winner |
| Neural SDE | 5.85% | — |
| Black-Scholes | 6.13% | Baseline |

*Source: `outputs/walk_forward_results.json`, out-of-sample test on latest SPY options.*

### Calibrated Parameters (from Market Data)

| Parameter | Symbol | Value | Source |
|-----------|--------|-------|--------|
| Vol-of-Vol (implied) | $\eta$ | 0.96 | VVIX/100 mean |
| Vol-of-Vol (realized) | $\eta$ | 1.14 | Realized vol of VIX |
| Mean Reversion | $\kappa$ | 2.72 | VIX Futures term structure |
| Long-term VIX | $\theta$ | 18.6% | Historical VIX mean (2024-2026) |
| Half-life | $t_{1/2}$ | 93 days | $\ln(2)/\kappa$ |
| **Hurst (True RV, SPX 5-min)** | $H$ | **0.201** | Variogram on daily RV |
| **Hurst (True RV, SPX 30-min)** | $H$ | **0.101** | Variogram on daily RV |
| Hurst (VIX paths) | $H$ | ~0.47 | Expected: VIX is a 30-day integral |
| Forward Variance | $\xi_0$ | 0.029 | VIX Futures 30d settlement |
| Contango frequency | — | 77% | VIX F2 > F1 |
| VVIX-VIX correlation | — | 0.83 | Panic amplification |

*Sources: `outputs/advanced_calibration.json`, `outputs/enhanced_calibration.json`*

---

## Methodology

### 1. Signature-Conditional Neural SDE

The variance process $V_t$ is modeled in log-space for guaranteed positivity:

$$dX_t = \underbrace{\kappa(\theta - X_t)}_{\text{OU Prior}} dt + \underbrace{\mathcal{N}_\mu(\mathbb{S}_{0,t}, X_t)}_{\text{Neural Drift}} dt + \underbrace{\mathcal{N}_\sigma(\mathbb{S}_{0,t}, X_t)}_{\text{Neural Diffusion}} dW_t$$

where $X_t = \log(V_t)$ and $\mathbb{S}_{0,t}$ is the **running path signature** of the generated path up to time $t$:

$$\mathbb{S}_{0,t} = \text{Sig}\left((s, X_s)_{s \in [0,t]}\right) \in T^{(3)}(\mathbb{R}^2) \cong \mathbb{R}^{14}$$

The running signature is updated at each step via **Chen's identity** (Lyons, 1998):

$$\mathbb{S}_{0,t+dt} = \mathbb{S}_{0,t} \otimes \mathbb{S}_{t,t+dt}$$

This makes the SDE genuinely **path-dependent** (non-Markovian), which is essential for capturing rough volatility dynamics where the future depends not just on the current state but on the entire path history.

**Why running signatures?**
- **Path dependence**: Conditions on full path history, essential for rough vol ($H < 0.5$)
- **Universal approximation**: Signatures form a basis for continuous functionals on paths (Lyons, 1998)
- **Gradient stability**: No vanishing/exploding gradients (unlike RNNs)
- **Incremental updates**: Chen's identity gives $O(d^3)$ update cost per step

**Log-variance normalization**: The input to the neural networks is standardized as $(X_t - c) / s$ where $c$ and $s$ are derived from the clipping bounds (configurable in `params.yaml`). This maps typical log-variance values to $\approx [-1, 1]$, ensuring good utilization of the `tanh` activations (LeCun et al., 1998).

### 2. Unsupervised Training via Normalized MMD

No labeled data needed. We minimize a **Maximum Mean Discrepancy** (Gretton et al., 2012) on path signatures with multi-scale RBF kernels:

$$\mathcal{L} = \text{MMD}^2_k\left(\mathbb{S}(\text{fake}), \mathbb{S}(\text{real})\right) + \lambda \cdot \text{MeanPenalty}$$

where the **mean penalty** has two modes (configurable via `training.mean_penalty_mode`):

- **Global** (default): $\lambda \cdot (\bar{V}^{\text{fake}} - \bar{V}^{\text{real}})^2$ — matches the overall mean variance
- **Marginal**: $\lambda \cdot \frac{1}{T}\sum_t (E[V_t^{\text{fake}}] - E[V_t^{\text{real}}])^2$ — matches $E[V_t]$ at each time step independently, providing tighter control on the marginal distribution and correcting time-dependent Jensen bias (Bayer & Stemper, 2018)

The signature MMD is **component-wise normalized** by the empirical standard deviation $\sigma_k^{\text{real}}$ of each component. Pure-time components (near-zero variance across paths) receive zero weight to prevent numerical instability.

### 3. Temporal Consistency

All dynamics use the **physical temporal scale**:

$$dt = \frac{T}{n_{\text{steps}}} \approx 1.53 \times 10^{-4} \text{ years} = 15 \text{ min}$$

Brownian increments: $dW_t = \sqrt{dt} \cdot Z$, $Z \sim \mathcal{N}(0,1)$.

The signature engine uses real time increments (not `linspace(0,1)`) to ensure consistency between the time axis in the signature and the SDE dynamics.

### 4. Rough Bergomi (Benchmark)

Implementation follows Bayer, Friz & Gatheral (2016). Two fBM simulation methods are available (configurable via `bergomi.fBm_method`):

- **Volterra** (default): Discretized Riemann-Liouville kernel $\hat{W}^H(t_j) = \sum_k A_{jk} Z_k$ where $Z_k$ are the **same** i.i.d. normals driving the spot BM. Guarantees exact spot-vol correlation by construction. $O(N^2)$ memory.
- **Davies-Harte**: Standard fBM via circulant embedding / FFT. $O(N \log N)$ but produces standard Mandelbrot-Van Ness fBM (different covariance for $H \neq 0.5$) and cannot share the driving BM with the spot.

The spot Euler scheme uses **previsible** (lagged) variance $V_{k-1}$ to avoid adaptedness bias:

$$\log S_k = \log S_{k-1} + (r - \tfrac{1}{2}V_{k-1})\Delta t + \sqrt{V_{k-1}}\,\Delta W_k^{\text{spot}}$$

### 5. Option Pricing via Monte Carlo

1. Generate $N$ variance paths: $V^{(i)}_t$ for $i = 1, \ldots, N$
2. Simulate correlated spot: $dS = rS\,dt + \sqrt{V}S\,(\rho\,dW^V + \sqrt{1-\rho^2}\,dW^\perp)$
3. Price option: $C = e^{-rT} \frac{1}{N}\sum_i \max(S^{(i)}_T - K, 0)$
4. Invert to IV via Black-Scholes formula (Newton-Raphson)

---

## Data Sources

### Primary Data

| Dataset | Source | Frequency | Period | Points |
|---------|--------|-----------|--------|--------|
| **VIX Spot** | TradingView | 5/10/15/30-min | 2023–2026 | ~20,000 each |
| **SPX** | TradingView | 5-min | 2025–2026 | 20,112 |
| **SPX** | TradingView | 30-min | 2020–2026 | 21,402 |
| **VVIX** | CBOE/TradingView | 5/15-min | 2023–2026 | ~20,000 |
| **VIX Futures** | CBOE | Daily | 2013–2026 | 29,477 |
| **S&P 500** | Yahoo Finance | Daily | 2010–2023 | ~3,200 |
| **SPY Options** | Yahoo Finance | Snapshots | Feb 2026 | ~2,400/snapshot |

### Derived Metrics

- **Realized Volatility**: $\text{RV}_t = \sum_i r_{i,\text{intraday}}^2$ per trading day from SPX intraday returns. Annualization: $\times 252 \cdot \text{bars/day} / \text{window}$ (Andersen & Bollerslev, 1998; Barndorff-Nielsen & Shephard, 2002)
- **Hurst Exponent**: Variogram method on $\log(\text{RV})$ — gold standard per Gatheral et al. (2018)
- **Forward Variance**: $\xi_0$ from VIX Futures settlement prices
- **Vol-of-Vol**: From VVIX (implied) and realized vol of VIX (realized)

---

## Research Journey

This section documents the iterative research process — including mistakes, discoveries, and pivots.

### Phase 1–2: Initial Implementation & First Option Pricing Attempt

**Goal**: Build a Neural SDE generating realistic volatility paths, then price options.

**Result**: Model generated plausible variance paths but option pricing was inconsistent due to training bugs (model weights not persisted, incorrect signature dimension 15 vs 14, prior parameters not aligned).

### Phase 3: The P vs Q Problem

The critical conceptual insight:

| Measure | Name | What it represents | Training data |
|---------|------|-------------------|---------------|
| **P** | Physical | Historical dynamics | Realized Vol (~19%) |
| **Q** | Risk-Neutral | No-arbitrage pricing | VIX (~19%) |

The **Variance Risk Premium** $VRP = IV^2 - \mathbb{E}^P[RV^2] > 0$ is typically 2–4 pp. In our dataset (2024–2026), the VRP was unusually small: RV $\approx$ VIX $\approx$ 19%. The real bugs were implementation issues, not the P/Q gap.

### Phase 4: Robustness Audit

**Key discoveries:**
- **Leverage effect confirmed**: $\rho(\text{Return}_{\text{SPX}}, \Delta\text{VIX}) = -0.86$, consistent with literature (Bekaert, Bouchaud)
- **VIX $\neq$ Realized Vol for roughness**: VIX is a 30-day integrated implied variance. Integration **smooths** the process, destroying roughness. Measuring $H$ on VIX gives $H \approx 0.5$ (not rough), while the correct measurement on daily RV from SPX returns gives $H \approx 0.1$ (rough). This is fully consistent with Gatheral et al. (2018).

### Phase 5: The VIX Solution

The VIX is computed from SPX option prices:

$$VIX^2 = \frac{2}{T}\sum_i \frac{\Delta K_i}{K_i^2} e^{rT} Q(K_i)$$

It's already a risk-neutral expectation. Training on VIX means training in Q-measure directly — no Girsanov change of measure needed.

### Phase 6: Market-Calibrated Parameters

Using VVIX and VIX Futures term structure, we extracted model parameters from market observables rather than assuming them:

- **Vol-of-vol**: VVIX mean = 96% (implied), realized vol of VIX = 114% (realized). The market **underprices** vol-of-vol by ~16%.
- **Mean reversion**: $\kappa = 2.72$ from futures term structure (contango 77% of days). Half-life = 93 days.
- **Forward variance**: $\xi_0 = 0.029$ from VIX Futures 30d settlement.

### Phase 9: Critical Training Fixes

Four fundamental issues in the training pipeline:

1. **328× temporal mismatch**: `dt = 1/n_steps` instead of physical `dt = T/n_steps`. The OU drift was absurdly strong.
2. **Wrong kurtosis metric**: `kurtosis(data.flatten())` mixed inter-path and intra-path variance. Correct: per-step marginal kurtosis.
3. **Signatures dominated by time**: `linspace(0,1)` time augmentation made time increments 50× larger than variance increments. Fix: component-wise normalization, zero-weight on pure-time dims.
4. **No mean penalty (Jensen bias)**: $E[e^X] > e^{E[X]}$ caused 2× upward bias in mean variance. Fix: explicit mean penalty with $\lambda = 10$.

### Phase 11: Correct Hurst Methodology

True roughness ($H \approx 0.1$) is only visible on **realized volatility** from high-frequency returns:

$$\text{RV}_{\text{day}} = \sum_{i=1}^{n} r_{i,\text{intraday}}^2$$

One RV point per day, then Hurst on $\log(\text{RV})$ via variogram.

| Source | $H$ (variogram) | $R^2$ |
|--------|:---:|:---:|
| SPX 5-min → daily RV | **0.201** | 0.91 |
| SPX 30-min → daily RV | **0.101** | 0.99 |
| VIX 15-min (old method) | 0.47 | — |

Additional fixes: variogram using all lags (not "middle portion" subset), DMA $-0.5$ correction for cumulative sum integration.

### Phase 12: Rough Bergomi Simulation Fix (Three Critical Bugs)

1. **Davies-Harte fBM variance 500× too low**: Missing `sqrt(2N)` factor and complex Gaussian (Dieker 2004).
2. **Spot-vol correlation wrong**: fGn increments $\neq$ the driving BM. For $H = 0.07$, fGn is anti-persistent — Cholesky mixing produces **reverse skew**. Fix: joint $(dW, \hat{W}^H)$ via Volterra kernel.
3. **Adaptedness bias**: $V_k$ depends on $Z_k$, so it's not $\mathcal{F}_{k-1}$-measurable. Using $V_k$ in the spot Euler step biases $E[S_T]$ by ~6%. Fix: use previsible $V_{k-1}$.

### Phase 13: Mathematical Audit & Corrections

A systematic audit identified and fixed:

- **Fixed PRNGKey(0)** in training: same v0 samples every epoch → overfitting. Now uses a fresh key per epoch.
- **Deterministic validation**: Pre-generated fixed noise for validation loss, eliminating noise in early stopping.
- **RV annualization factor**: Was $252 \times \text{bars/day}$ (inflating RV by ~78×), corrected to $252 \times \text{bars/day} / \text{window}$.
- **`learn_ou_params` flag**: Was a no-op (both branches stored `jnp.array`). Now uses `eqx.partition` to freeze $\kappa, \theta$ when disabled.
- **Log-variance normalization**: Changed from simple shift (+4.0) to proper standardization $(x - c)/s$ with configurable clipping bounds.
- **Signature propagation across blocks**: Multi-maturity backtesting was resetting the signature to zero between blocks, destroying path memory. Now propagates $(s_1, s_2, s_3)$ via `generate_variance_path_with_state`.
- **Bergomi fBM consistency**: Added `bergomi.fBm_method` config parameter to choose between Volterra (correct for rBergomi) and Davies-Harte (faster, different process).
- **Risk-neutral calibration**: Added proper risk-free rate drift and discounting (was implicitly $r = 0$).
- **Marginal mean penalty**: Added per-step $E[V_t]$ matching option to correct time-dependent Jensen bias.
- **OU ablation baseline**: Was CIR-like on $V$; fixed to OU on $\log V$ with same $(κ, θ)$ as the Neural SDE prior.
- **LR schedule**: `decay_steps` was 2000 for 500 epochs (cosine barely decayed). Set to match `n_epochs`.
- **Hardcoded values**: Replaced all hardcoded `r=0.05`, `rho=-0.7` with config reads across `options_calibration.py`, `risk_neutral_calibration.py`, `advanced_calibration.py`.

---

## Market Insights

### 1. Vol-of-Vol Gap

The market consistently underprices volatility-of-volatility:

| Metric | Value |
|--------|-------|
| VVIX (implied η, 30-day) | 0.96 |
| Realized Vol of VIX (η) | 1.14 |
| Gap | ~16% underpriced |
| VVIX-VIX correlation | 0.83 |

### 2. VIX Futures Term Structure as Regime Indicator

| Regime | Frequency | Avg VIX | Interpretation |
|--------|-----------|---------|----------------|
| Contango (F2 > F1) | 77% | 17.0 | Normal, calm |
| Backwardation (F1 > F2) | 23% | 23.6 | Stress, crisis |

### 3. Roughness Confirmed

True roughness ($H \approx 0.1$) is only visible on realized volatility from intraday returns, never on VIX:

| Source | $H$ | Method |
|--------|-----|--------|
| SPX 30-min → daily RV | **0.101** | Variogram ($R^2 = 0.99$) |
| SPX 5-min → daily RV | **0.201** | Variogram ($R^2 = 0.91$) |
| VIX 15-min | 0.47 | Variogram (expected: integration adds ~0.5) |

---

## Architecture

### Neural Network

```
NeuralRoughSimulator
├── Input: running signature Sig(X_{0:t}) (dim=14) + normalized log-variance (dim=1)
├── drift_net: MLP [15 → 64 → 64 → 64 → 1], tanh activation
├── diff_net:  MLP [15 → 64 → 64 → 64 → 1], sigmoid output
├── Output scaling: drift × drift_scale (tanh), diffusion ∈ [min, max] (sigmoid)
├── OU prior: κ, θ from config (optionally learnable via eqx.partition)
├── Log-variance: proper (x-c)/s standardization, configurable clipping bounds
├── Signature propagation: generate_variance_path_with_state for multi-block chaining
└── Integration: Euler-Maruyama with running signature via Chen's identity
```

### Signature Update (Chen's Identity)

For a 2D path (time, log-variance) with truncation order 3:
- Order 1: $S^1 \in \mathbb{R}^2$ → $S^1 \leftarrow S^1 + dx$
- Order 2: $S^2 \in \mathbb{R}^4$ → $S^2 \leftarrow S^2 + S^1 \otimes dx + \frac{1}{2} dx \otimes dx$
- Order 3: $S^3 \in \mathbb{R}^8$ → $S^3 \leftarrow S^3 + S^2 \otimes dx + S^1 \otimes \frac{dx^{\otimes 2}}{2} + \frac{dx^{\otimes 3}}{6}$

Total: 14 features per step, updated in $O(d^3)$ via `jax.lax.scan`.

### Training Loop

```
GenerativeTrainer
├── Data: VIX variance paths × 120 steps (configurable train/val split)
├── Target: Signatures of real paths (component-wise normalized)
├── Loss: MMD (multi-scale RBF) + mean penalty (global or marginal)
├── Noise: dW = √dt · Z, fresh PRNGKey per epoch
├── Validation: deterministic (pre-generated fixed noise + indices)
├── LR: Linear warmup (50 steps) → cosine decay (decay_steps = n_epochs)
├── Optimizer: Adam + gradient clipping
├── Early stopping: patience on deterministic validation loss
├── Parameter freezing: eqx.partition for optional κ, θ freezing
└── Convergence: ~200 epochs, best loss ≈ 0.017
```

---

## Usage

### Installation

```bash
git clone https://github.com/BigPython34/Generative-Neural-SDES-for-Market-Volatility
cd DeepRoughVol

python -m venv .venv
.venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### Entry Points

All executable scripts live in `bin/` (plus `main.py` at root):

```bash
python main.py                          # Full pipeline: train + benchmark + roughness check
python bin/train.py                     # Train Neural SDE
python bin/calibrate.py                 # Calibrate Bergomi params from market data
python bin/backtest.py                  # Historical options backtest
python bin/walk_forward.py              # Walk-forward backtest
python bin/fetch_options.py             # Fetch and cache SPY options data
python bin/dashboard.py                 # Generate HTML dashboard
python bin/options_calibration.py       # Full options surface calibration
python bin/risk_neutral_calibration.py  # IV surface calibration
```

### Diagnostics

```bash
python bin/verify_roughness.py          # Roughness + signatures + ablation
python bin/coherence_check.py           # Full coherence audit
python bin/compare_frequencies.py       # VIX frequency comparison
python bin/compare_vix_vs_rv.py         # VIX vs Realized Vol roughness
python bin/robustness_check.py          # Full robustness diagnostics
python bin/hurst_diagnostic.py          # VIX vs RV Hurst diagnostic
```

### Configuration

All parameters are centralized in `config/params.yaml`:

```yaml
data:
  data_type: "vix"                    # "vix" (Q-measure) or "realized_vol" (P-measure)
  source: "data/TVC_VIX, 15.csv"
  segment_length: 120
  stride_ratio: 0.5

simulation:
  n_steps: 120
  T: 0.01832                         # 120 × 15min in years

training:
  n_epochs: 500
  batch_size: 256
  decay_steps: 500                    # = n_epochs for full cosine utilization
  lambda_mean: 10.0
  mean_penalty_mode: "global"         # "global" or "marginal"
  early_stopping_patience: 50

neural_sde:
  learn_ou_params: true
  kappa: 2.72                         # From VIX futures
  theta: -3.5                         # log(0.03) ≈ 17% vol
  log_v_clip_min: -7.0                # vol ≈ 3%
  log_v_clip_max: 2.0                 # vol ≈ 272%

bergomi:
  hurst: 0.07
  eta: 1.9
  rho: -0.7
  xi0: 0.0225
  fBm_method: "volterra"              # "volterra" or "davies_harte"

pricing:
  risk_free_rate: 0.05
```

---

## File Structure

```
DeepRoughVol/
│
├── main.py                              # Full demo: train + benchmark + roughness
│
├── bin/                                 # ALL executable entry points
│   ├── train.py                         # Train the Neural SDE
│   ├── calibrate.py                     # Run advanced calibration
│   ├── backtest.py                      # Run historical backtest
│   ├── walk_forward.py                  # Walk-forward backtest
│   ├── fetch_options.py                 # Download & cache SPY options
│   ├── dashboard.py                     # Generate interactive dashboard
│   ├── verify_roughness.py              # Roughness verification + ablation
│   ├── options_calibration.py           # Full options surface calibration
│   ├── risk_neutral_calibration.py      # IV surface calibration
│   ├── robustness_check.py              # Diagnostic audit
│   ├── hurst_diagnostic.py              # VIX vs RV Hurst diagnostic
│   ├── coherence_check.py               # Full coherence audit
│   ├── compare_frequencies.py           # VIX frequency comparison
│   └── compare_vix_vs_rv.py             # VIX vs Realized Vol comparison
│
├── config/
│   └── params.yaml                      # Central config
│
├── core/                                # Stochastic models (library only)
│   ├── bergomi.py                       # rBergomi (Volterra + Davies-Harte, configurable)
│   └── stochastic_process.py            # Fractional Brownian Motion (Davies-Harte)
│
├── engine/                              # ML engine (library only)
│   ├── __init__.py
│   ├── neural_sde.py                    # NeuralRoughSimulator (with signature propagation)
│   ├── signature_engine.py              # Path signature computation (esig + JAX)
│   ├── generative_trainer.py            # Training loop (deterministic val, param freezing)
│   └── losses.py                        # MMD, mean penalty, marginal mean penalty
│
├── quant/                               # Quant library (no standalone scripts)
│   ├── calibration/
│   │   ├── hurst.py                     # Hurst estimation (variogram, DMA)
│   │   └── bergomi_optimizer.py         # Bergomi parameter grid search
│   ├── pricing.py                       # MC pricing engine (per-path v0, previsible var)
│   ├── mc_pricer.py                     # MonteCarloOptionPricer (European options)
│   ├── advanced_calibration.py          # Market parameter extraction (H, η, κ, ξ₀)
│   ├── backtesting.py                   # Historical backtest (signature-aware blocks)
│   ├── walk_forward_backtest.py         # Walk-forward backtest
│   ├── dashboard_v2.py                  # Interactive HTML dashboard (Plotly)
│   └── options_cache.py                 # SPY options cache (Yahoo Finance)
│
├── utils/                               # Utilities (library only)
│   ├── config.py                        # load_config (cached)
│   ├── black_scholes.py                 # BS pricing, IV, Greeks
│   ├── greeks_ad.py                     # JAX autodiff Greeks
│   ├── data_loader.py                   # VIX + RV loading (corrected annualization)
│   └── diagnostics.py                   # Statistics, Hurst, daily RV
│
├── data/
│   ├── TVC_VIX, *.csv                   # VIX at 5/10/15/30-min
│   ├── SP_SPX, *.csv                    # SPX intraday (5/30-min)
│   ├── CBOE_DLY_VVIX, *.csv            # VVIX at 5/15-min
│   ├── cboe_vix_futures_full/           # VIX futures (1M/2M/3M front, all contracts)
│   ├── options_cache/                   # Cached SPY option surfaces (10 snapshots)
│   └── data_^GSPC_*.csv                # S&P 500 daily
│
├── outputs/                             # Generated results (JSON + HTML)
├── models/                              # Trained models (.eqx)
├── tests/                               # Structural tests
└── obsolete/                            # Superseded scripts & outputs
```

---

## Lessons Learned

### 1. P ≠ Q (Measure Matters)
Training on realized vol and testing on implied vol is fundamentally wrong. The VIX is already Q-measure — training on it avoids the need for a Girsanov change of measure.

### 2. Temporal Scale Must Be Physical
Using `dt = 1/n_steps` instead of real physical time ($15 \text{min} \approx 1.53 \times 10^{-4} \text{yr}$) introduced a **328× scaling error**. All SDE parameters ($\kappa, \sigma$, drift) are in annual units — the $dt$ must match.

### 3. VIX ≠ Volatility (for Roughness)
The VIX is a 30-day integral of implied variance. Integration smooths the process: $H_\text{VIX} \approx 0.5$ regardless of $H_\text{RV} \approx 0.1$. Roughness must be measured on **realized vol from intraday returns**.

### 4. Normalize Your Loss Components
Signature components span 8 orders of magnitude. Without normalization, the loss is blind to stochastic components. Pure-time components must be excluded (zero weight).

### 5. Jensen's Inequality Bites
When the model generates log-variance and you exponentiate: $E[e^X] > e^{E[X]}$. An explicit mean penalty is necessary. The marginal (per-step) variant provides tighter control.

### 6. fBM ≠ fGn for Correlation
Fractional Gaussian noise increments ($\Delta W^H$) for $H < 0.5$ are anti-persistent. Treating them as the "driving BM" inverts the leverage effect. The Volterra kernel is the correct approach: correlate the spot with the underlying $W$ (not $W^H$).

### 7. Previsible Variance in Euler Schemes
When $V_k$ depends on the same noise increment $Z_k$, using $V_k$ in $\sqrt{V_k}\,dW_k$ creates a Jensen bias: $E[S_T]$ can be 6% below the forward. Use $V_{k-1}$ instead.

### 8. Deterministic Validation for Early Stopping
Resampling noise at each validation step makes the validation loss non-deterministic, causing early stopping to respond to noise rather than overfitting. Pre-generate fixed validation noise once.

### 9. Seed Diversity in Training
A fixed `PRNGKey(0)` for selecting initial conditions means the model sees the same $v_0$ samples every epoch, limiting diversity and encouraging overfitting.

### 10. Signature State Must Propagate
When chaining Neural SDE blocks for long maturities, reinitializing the signature to zero between blocks destroys the path memory that signatures are designed to capture (Chevyrev & Kormilitzin, 2016).

---

## Future Work

- **Maturity-dependent η**: Constant $\eta$ produces excess skew at ~17d; a term-structure $\eta(T)$ would improve intermediate maturities
- **Multi-frequency training**: Train on 5/10/15/30-min simultaneously to capture multi-scale roughness
- **Joint Loss**: Train on VIX paths + option prices simultaneously for end-to-end calibration
- **Regime-aware training**: Use VVIX/VIX ratio or contango/backwardation as conditioning variable
- **Neural SDE retraining**: Current model uses segment_length=120 but needs retraining with the corrected pipeline

---

## References

### Academic Papers

1. Gatheral, J., Jaisson, T. & Rosenbaum, M. (2018). *Volatility is Rough*. Quantitative Finance, 18(6), 933–949.
2. Bayer, C., Friz, P. & Gatheral, J. (2016). *Pricing under rough volatility*. Quantitative Finance, 16(6), 887–904.
3. Kidger, P., Foster, J., Li, X. & Lyons, T. (2021). *Neural SDEs as Infinite-Dimensional GANs*. ICML.
4. Lyons, T. (1998). *Differential equations driven by rough paths*. Rev. Mat. Iberoamericana, 14(2), 215–310.
5. Chevyrev, I. & Kormilitzin, A. (2016). *A primer on the signature method in machine learning*. arXiv:1603.03788.
6. Gretton, A., Borgwardt, K., Rasch, M., Schölkopf, B., & Smola, A. (2012). *A Kernel Two-Sample Test*. JMLR, 13, 723–773.
7. Cuchiero, C., Khosrawi, W. & Teichmann, J. (2023). *A generative adversarial network approach to calibration of local stochastic volatility models*. Risks, 8(4), 101.
8. Bennedsen, M., Lunde, A. & Pakkanen, M.S. (2017). *Hybrid scheme for Brownian semistationary processes*. Finance and Stochastics, 21(4), 931–965.
9. Andersen, T.G. & Bollerslev, T. (1998). *Answering the Skeptics: Yes, Standard Volatility Models Do Provide Accurate Forecasts*. International Economic Review, 39(4), 885–905.
10. Barndorff-Nielsen, O.E. & Shephard, N. (2002). *Econometric analysis of realized volatility*. Journal of the Royal Statistical Society B, 64(2), 253–280.
11. Dieker, T. (2004). *Simulation of fractional Brownian motion*. MSc thesis, University of Twente.
12. LeCun, Y., Bottou, L., Orr, G. & Müller, K.R. (1998). *Efficient BackProp*. Neural Networks: Tricks of the Trade, Springer.
13. Bayer, C. & Stemper, B. (2018). *Deep calibration of rough stochastic volatility models*. arXiv:1810.03399.

### Technical Documentation

- [JAX Documentation](https://jax.readthedocs.io/)
- [Equinox](https://docs.kidger.site/equinox/)
- [esig Library](https://esig.readthedocs.io/)
- [CBOE VIX White Paper](https://www.cboe.com/tradable_products/vix/)

---

## License

MIT License — see [LICENSE](LICENSE).

---

*Last updated: February 2026*
