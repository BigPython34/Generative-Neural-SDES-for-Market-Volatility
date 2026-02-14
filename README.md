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
3. [Project Evolution & Thought Process](#project-evolution--thought-process)
4. [Methodology](#methodology)
5. [Data Sources](#data-sources)
6. [Market Insights Discovered](#market-insights-discovered)
7. [Architecture](#architecture)
8. [Usage](#usage)
9. [File Structure](#file-structure)
10. [Lessons Learned](#lessons-learned)
11. [Future Work](#future-work)
12. [References](#references)

---

## Abstract

**DeepRoughVol** is a non-parametric generative framework that learns volatility dynamics directly from market data using **Neural Stochastic Differential Equations (Neural SDEs)** conditioned on **Path Signatures**.

Unlike traditional stochastic volatility models (Heston, Bergomi) which impose rigid parametric forms, this approach:
- Learns drift and diffusion functions from data
- Captures rough volatility behavior naturally (H < 0.5)
- Calibrates to real option prices
- Incorporates market observables (VVIX, VIX Futures)

**Latest Achievement**: Fixed three critical bugs in the rBergomi simulation (Davies-Harte fBM variance, spot–vol correlation via Volterra kernel, adaptedness bias). Bergomi now produces correct negative equity skew and **wins 70% of scenarios** against real SPY options (mean RMSE 4.58% vs BS 6.01%).

---

## Key Results

### Statistical Fit (Generated vs Real VIX Variance Paths)

After the Phase 9 corrections (temporal scaling, normalized MMD, mean penalty):

| Metric | Market | Neural SDE | Bergomi | Status |
|--------|--------|------------|---------|--------|
| **Mean Variance** | 0.0372 | **0.0373** | 0.019 | ✅ < 0.3% error |
| **Median Variance** | 0.0294 | **0.0293** | 0.018 | ✅ Perfect |
| **Kurtosis (marginal)** | 38.5 | **38.9** | 0.3 | ✅ Fat tails captured |
| **Hurst H** | 0.475 | **0.476** | 0.256 | ✅ Gap = 0.001 |
| **ACF(1)** | 0.69 | **0.67** | 0.24 | ✅ Memory structure |
| **Sig Correlation** | — | **0.985** | — | ✅ Distribution match |

### Option Pricing Performance

| Model | IV RMSE | Improvement |
|-------|---------|-------------|
| **Neural SDE (VIX-trained)** | **2.95%** | ✅ Winner |
| Black-Scholes | 3.94% | Baseline |
| Neural SDE (RV-trained) | 5.85% | ❌ Wrong measure |

### Real Options Backtest (SPY, 5 snapshots × 4 maturities)

After fixing the rBergomi simulation (Phase 12: Volterra kernel + previsible variance):

| Model | Mean RMSE | Win Rate |
|-------|-----------|----------|
| **Rough Bergomi** | **4.58%** | **70%** (14/20) |
| Neural SDE | 5.35% | 25% (5/20) |
| Black-Scholes | 6.01% | 5% (1/20) |

Bergomi dominates at 7d, 31d, 45d maturities. Neural SDE fills the gap at 17d where constant-η Bergomi overestimates skew.

### Calibrated Parameters (from Market Data)

| Parameter | Symbol | Value | Source |
|-----------|--------|-------|--------|
| Vol-of-Vol | η | 0.96 (implied) / 1.14 (realized) | VVIX/100 vs realized vol(VIX) |
| Mean Reversion | κ | 2.72 | VIX Futures term structure |
| Long-term VIX | θ | 18.6% | Historical VIX mean (2024-2026) |
| Half-life | t½ | 93 days | ln(2)/κ |
| **Hurst (True RV)** | H | **0.10 – 0.13** | Variogram on daily RV from SPX 5/30-min |
| Hurst (VIX paths) | H | ~0.47 | VIX is smoothed (30-day integral) → H ≈ 0.5 expected |
| Optimal Bergomi H | H | **0.151** | Median of SPX variogram estimates |

### Ablation: Running Signatures Matter

| Model | Norm. MMD | Hurst Gap | Mean Error |
|-------|-----------|-----------|------------|
| **Neural SDE** | 0.027 | **0.038** ✅ | 0.27% |
| OU baseline | 0.025 | 0.059 | 0.10% |

Neural SDE wins on **roughness** (the core objective), while OU has a marginal edge on simple marginal statistics. The OU process cannot capture path-dependent dynamics.

---

## Project Evolution & Thought Process

This section documents the research journey, including mistakes, discoveries, and pivots. It reflects the iterative nature of quantitative research.

### Phase 1: Initial Implementation

**Goal**: Build a Neural SDE that generates realistic volatility paths.

**Approach**:
- Train on VIX intraday data (30-min)
- Use Path Signatures to condition the SDE (avoiding RNN gradient issues)
- MMD loss to match distribution of generated vs real paths

**Result**: ✅ Model generates plausible variance paths with correct statistical properties (fat tails, memory, realistic levels).

---

### Phase 2: Option Pricing Attempt

**Goal**: Use the trained model to price SPX options via Monte Carlo.

**Approach**:
- Generate 50,000 variance paths with Neural SDE
- Simulate correlated spot paths
- Price options and invert to implied volatility
- Compare to market IV

**Result**: ❌ **Initial failure** — Model produced inconsistent volatility levels due to training bugs.

---

### Phase 3: Diagnosis — The P vs Q Problem

This was the critical insight of the entire project.

**Key Realization**: We needed to understand the difference between **Realized Volatility** (P-measure) and **Implied Volatility** (Q-measure).

| Measure | Name | What it represents | Our dataset |
|---------|------|-------------------|-------------|
| **P** | Physical / Real-World | What actually happens historically | ~19% (RV) |
| **Q** | Risk-Neutral | What's priced into options (no-arbitrage) | ~19% (VIX) |

**The Variance Risk Premium (VRP)**

$$VRP = IV^2 - \mathbb{E}^P[RV^2] > 0$$

Investors pay a premium for volatility protection, so IV > E[RV] **ex-ante** on average. The VRP is typically 2-4 percentage points (VIX ~19%, long-term RV ~17%). This is one of the most robust findings in empirical finance.

**In our dataset** (2024-2026): RV ≈ 19% and VIX ≈ 19% — very close! This period had elevated realized vol matching implied expectations.

**Our actual mistake**: Not the P/Q gap (which was small), but **implementation bugs**:
1. Model weights not persisted between training and pricing
2. Incorrect signature dimension (15 vs 14)
3. Prior parameters not aligned with market calibration

---

### Phase 4: Robustness Audit

Before fixing the P/Q issue, we audited the entire pipeline for other problems.

**Script**: `quant/robustness_check.py`

**Discoveries**:

#### 4.1 Correlation ρ Confirms Literature

Classical stochastic volatility models use ρ ≈ -0.7 for the "leverage effect" (stocks down → vol up).

**Our finding from data** (SPX return vs VIX change, 2024-2026):
```
ρ(Return_SPX, ΔVIX) = -0.86   ← Consistent with literature!
```

*Note: An earlier version incorrectly reported ρ = -0.07. This was due to computing correlation between returns and realized volatility derived from the same returns (which is ~0 by construction). The correct measure uses VIX (implied vol) vs SPX returns.*

Volatility clustering is also confirmed:
```
ρ(|Return|, Vol) = +0.50  ← Large moves predict high vol
```

**Interpretation**: The leverage effect is robust at ρ ≈ -0.7 to -0.9 for SPX-VIX, consistent with Bekaert, Bouchaud, and others.

#### 4.2 Hurst Exponent: VIX ≠ Realized Volatility

A critical mistake was measuring Hurst on VIX and calling it "roughness". The VIX is a 30-day integrated implied variance — integration **smooths** the process, destroying roughness.

| Data Source | Method | H estimate | Interpretation |
|-------------|--------|-----------|----------------|
| SPX 5-min → **daily RV** | Variogram | **0.130** (R²=0.91) | ✅ Rough (Gatheral sense) |
| SPX 30-min → **daily RV** | Variogram | **0.116** (R²=0.99) | ✅ Rough (Gatheral sense) |
| VIX 15-min (log) | Variogram | 0.47 | ≈ 0.5 → **NOT rough** (expected) |
| SPX rolling RV | Variogram | 0.10 – 0.20 | ✅ Rough (depends on window) |

**Key insight (Gatheral et al. 2018)**: True roughness ($H \approx 0.05-0.14$) is only visible on **realized volatility computed from high-frequency returns**, not on VIX or smoothed vol proxies. VIX $H \approx 0.5$ is mathematically expected and says nothing about roughness.

**Methodology**: Daily RV = $\sum_i r_i^2$ over all intraday returns within each trading day. Hurst is measured on the $\log(\text{RV})$ time series via variogram: $\mathbb{E}[|\Delta \log \text{RV}|^2] \sim \tau^{2H}$.

#### 4.3 Model Not Persisted

Each calibration script reinitialized the model with random weights. The trained model from `main.py` was never saved or reused.

---

### Phase 5: The VIX Solution

**Insight**: The VIX **is already in Q-measure**!

The VIX is computed from SPX option prices via:

$$VIX^2 = \frac{2}{T}\sum_i \frac{\Delta K_i}{K_i^2} e^{rT} Q(K_i)$$

It's a risk-neutral expectation of variance. Training on VIX means training in Q-measure directly.

**Action**: Switch training data from Realized Volatility to VIX.

```yaml
# config/params.yaml - BEFORE
data:
  data_type: "realized_vol"
  source: "data/SP_SPX, 30.csv"

# config/params.yaml - AFTER  
data:
  data_type: "vix"
  source: "data/TVC_VIX, 15.csv"
```

**Result**: Both RV and VIX give ~19% volatility in our dataset, but VIX is the **correct** training target for option pricing because:
1. VIX is directly in Q-measure (no measure change needed)
2. VIX has cleaner dynamics (no microstructure noise)
3. VIX better captures forward-looking expectations

---

### Phase 6: Enhanced Calibration with Market Data

With new data (VVIX, VIX Futures), we could extract model parameters from the market itself rather than assuming them.

**Script**: `quant/enhanced_calibration.py`

#### 6.1 Vol-of-Vol from VVIX

The VVIX measures expected volatility of the VIX (30-day horizon).

```
VVIX mean (implied):     96%   → η_implied = 0.96
Realized Vol(VIX):       114%  → η_realized = 1.14
Ratio (implied/realized): 0.84x
```

**Discovery**: The market **underprices** vol-of-vol by ~16%. Realized VIX volatility consistently exceeds what VVIX implies.

*Note: An earlier version reported 150% realized and 40% gap. The correct annualized realized vol of VIX over 2024-2026 is ~114%, computed as std(daily log-returns of VIX) × √252.*

#### 6.2 Mean Reversion from Futures

VIX Futures term structure encodes mean-reversion expectations:

```
Contango (F2 > F1):      77% of days → κ ≈ 2.7
Backwardation (F1 > F2): 23% of days → Stress signal
```

When in backwardation, VIX averages 23.6 (crisis). When in contango, VIX averages 17.0 (normal).

**Calibrated κ**: 2.72 → Half-life of 93 days

---

### Phase 7: Final Victory

With all pieces in place:
1. ✓ Training on VIX (Q-measure)
2. ✓ Correct signature dimension (14, not 15)
3. ✓ Using trained model (not random weights)

**Result**:

```
┌─────────────────────────────────────────┐
│     RISK-NEUTRAL CALIBRATION RESULTS    │
├─────────────────────────────────────────┤
│  Neural SDE IV RMSE:    2.95%           │
│  Black-Scholes RMSE:    3.94%           │
│  Improvement:           +25%            │
└─────────────────────────────────────────┘
```

---

### Phase 8: Closing the Loop — Roughness Verification

**Goal**: Verify that the full pipeline is coherent:
- Does the model **generate** rough paths (H < 0.5)?
- Do the **signatures** of generated paths match real data?

**Script**: `quant/verify_roughness.py`

**Results**:

| Check | Real Data | Neural SDE | Status |
|-------|-----------|------------|--------|
| Hurst H (VIX paths) | 0.475 | 0.476 ± 0.10 | ✅ Matches training data |
| **True H (daily RV from SPX)** | — | — | **0.130** ✅ Rough |
| Signature Correlation | — | 0.986 | ✅ Match |
| Ablation: Hurst gap | — | 0.038 (Neural) vs 0.059 (OU) | ✅ Neural closer |

**Important correction**: The H ≈ 0.475 above is measured on VIX (training data) — it measures how well the model reproduces VIX dynamics, **not** roughness. True roughness ($H \approx 0.13$) was later confirmed on SPX daily RV (Phase 11).

---

### Phase 9: Temporal & Loss Corrections (Critical Fix)

**Discovery**: A systematic audit revealed **4 fundamental issues** in the training pipeline.

#### 9.1 The 328× Temporal Mismatch

The model was treating each 15-minute bar as 18.2 days:

```
Real dt per step:  15 min = 1/6552 years ≈ 0.000153 years
Model dt per step: 1/20   = 0.05         ≈ 18.2 days     ← 328× too large!
```

The OU drift `κ(θ − x)·dt` was absurdly strong, and Brownian increments `dW = √dt · Z` were 18× too large.

**Fix**: Compute `dt = T / n_steps` where `T = 0.00305` years (the real horizon of 20 bars × 15 min).

#### 9.2 Wrong Kurtosis Metric

The diagnostic computed `kurtosis(data.flatten())` on all paths flattened — mixing inter-path variance with intra-path dynamics:
- Flattened kurtosis: **84.6** (meaningless — inflated by mixing VIX levels across time)
- True marginal kurtosis: **38.5** (excess kurtosis at each time step, averaged)

**Fix**: Compute per-time-step kurtosis and average: $\bar{\kappa} = \frac{1}{T}\sum_t \kappa_4(V_t)$

#### 9.3 Signatures Dominated by Time Components

With `linspace(0, 1)` time augmentation:
- Time increments ≈ 0.053 vs variance increments ≈ 0.001
- Time components dominated the L2 loss by **50×**
- Pure-time components (s¹ₜ, s²ₜₜ, s³ₜₜₜ) have near-zero variance across paths → dividing by their std explodes

**Fix**: Component-wise normalization of the MMD loss, with zero-weight on pure-time components (std < 10⁻⁸).

#### 9.4 No Mean Penalty (Jensen Bias)

The model generates log-variance then exponentiates: $V = e^X$. Without explicit mean control, Jensen's inequality causes systematic upward bias:

$$\mathbb{E}[e^X] > e^{\mathbb{E}[X]}$$

Before fix: mean variance = 0.074 (2× too high). After fix: mean variance = 0.037 (< 0.3% error).

**Fix**: Add penalty $\lambda \cdot (\bar{V}_{fake} - \bar{V}_{real})^2$ with $\lambda = 10$.

#### 9.5 Impact Summary

| Metric | Before (Phase 8) | After (Phase 9) | Improvement |
|--------|----------|---------|-------------|
| Mean Variance | 0.074 ❌ | **0.037** ✅ | 2× bias → 0.3% error |
| Kurtosis | 84.6 (wrong metric) | **38.9** ✅ | Correct metric, matches real |
| Hurst Gap | 0.137 | **0.001** ✅ | 137× closer |
| Sig Correlation | 1.000 | **0.985** | Still excellent |

---

### Phase 10: Architecture Improvements

Sync all hardcoded params with config, add validation/early stopping, learnable OU parameters, JAX autodiff Greeks, shared Black-Scholes, and real Monte Carlo backtesting. See the [Completed checklist](#-completed-phase-10) for details.

---

### Phase 11: Hurst Methodology Correction (Critical Fix)

**Discovery**: All previous Hurst measurements were **fundamentally wrong** — they measured H on VIX paths instead of realized volatility.

#### 11.1 The VIX Smoothing Problem

The VIX is defined as:

$$\text{VIX}^2 = \frac{2}{T}\int_0^T \xi_0(t)\,dt$$

This 30-day integration **smooths** the underlying variance process. If $\xi_0(t)$ has Hurst exponent $H_{\text{true}} \approx 0.1$, the integrated VIX has $H_{\text{VIX}} \approx 0.5$ (integration adds ~0.5 to the scaling exponent). Measuring H on VIX and calling it "roughness" is like measuring the roughness of a moving average and concluding the signal is smooth.

#### 11.2 Correct Methodology: Daily Realized Variance

True roughness must be measured on **realized volatility** computed directly from intraday returns:

$$\text{RV}_{\text{day}} = \sum_{i=1}^{n} r_{i,\text{intraday}}^2$$

One RV point per trading day, then Hurst on $\log(\text{RV})$ via variogram.

**Results**:

| Source | H (variogram) | R² | Points |
|--------|:---:|:---:|:---:|
| SPX 5-min → daily RV | **0.130** | 0.911 | 256 days |
| SPX 30-min → daily RV | **0.116** | 0.987 | 1,522 days |
| VIX 15-min (old method) | 0.47 | — | — |

This is fully consistent with Gatheral et al. (2018): $H \approx 0.05-0.14$.

#### 11.3 Variogram Fix

The variogram method was using a "middle portion" subset (`lags[n//5:4*n//5]`) that excluded the short lags where roughness is most visible. This caused SPX 5-min to give $H = -0.074$ (nonsensical). After using all lags: $H = 0.201$ (5-min rolling RV) / $H = 0.101$ (30-min rolling RV).

#### 11.4 DMA Correction

The DMA (Detrended Moving Average) method operates on `cumsum(returns)`, which adds +0.5 to the scaling exponent. The correction $H = \alpha - 0.5$ was missing, causing DMA to report $H \approx 0.85-1.19$ instead of the correct $H \approx 0.35-0.46$.

#### 11.5 H Selection Logic

`get_optimal_bergomi_params()` now prefers variogram estimates (gold standard per Gatheral 2018) over DMA. The optimal Bergomi $H$ is **0.151**, median of SPX variogram estimates.

#### 11.6 Config Centralization

All hardcoded paths (`"data/SP_SPX, 30.csv"`, `"data/TVC_VIX, 15.csv"`, etc.) and parameters (`r=0.05`, `s0=100`, `rho=-0.7`, `n_mc=3000`) were moved to `config/params.yaml`. A new `utils/config.py` module provides `load_config()` with caching.

New config sections: `data.vix_files`, `data.spx_files`, `backtesting.*`, `pricing.n_mc_paths`, `outputs.*`.

#### 11.7 RV Loader Adaptive Window

The `RealizedVolatilityLoader` now auto-detects bar frequency from timestamps and scales `rv_window` proportionally. Config's `rv_window=78` (for 5-min) is automatically adjusted to 13 for 30-min data.

---

### Phase 12: Rough Bergomi Simulation Fix (Three Critical Bugs)

**Problem**: Backtest results were catastrophic — Black-Scholes (flat ATM vol) beat both Bergomi (16.0% RMSE) and Neural SDE (14.5% RMSE) on every single scenario (20/20 wins). Stochastic volatility models were *worse* than constant vol.

#### 12.1 Bug #1: Davies-Harte fBM Variance — 500× Too Low

The `_single_path()` implementation used real-valued Gaussian noise and a missing `sqrt(2N)` factor:

```python
# BEFORE (wrong):
f = jnp.fft.ifft(jnp.sqrt(w) * z).real[:n]

# AFTER (Dieker 2004 algorithm):
v = jnp.sqrt(lam / M) * (n1 + 1j * n2)  # complex Gaussian
z = (jnp.fft.ifft(v) * M).real[:n]       # proper IFFT scaling
```

**Impact**: $\text{Var}(W^H_T) = 0.0013$ instead of $T^{2H} = 0.705$ — a **500× underestimation**. The previous `eta = 1.9` was calibrated to this broken fBM, masking the error.

#### 12.2 Bug #2: Spot–Variance Correlation (Incorrect Cholesky Mixing)

The old code extracted fGn increments from Davies-Harte fBM, normalized them, and mixed via Cholesky with independent spot noise:

```python
# BEFORE (wrong): dwh_unit ≠ the BM that drives W^H
dwh = jnp.diff(wh, axis=1)
dwh_unit = dwh / jnp.sqrt(dt ** (2*H))
dw_spot = rho * dwh_unit + sqrt(1-rho²) * z_indep
```

**Problem**: The fGn increments $\Delta W^H_k$ are **not** the Brownian motion $W$ that drives $W^H$ via the Volterra kernel. For $H = 0.07$, fGn increments are extremely anti-persistent: $\text{std}\!\left[\sum_k \Delta W^H_k / \sigma_k\right] = 1.36$ vs $\sqrt{n} = 9.49$ for independent BM. This Cholesky mixing creates wrong correlation sign at the path level, producing **reverse skew** (OTM calls more expensive than puts).

**Fix**: Joint simulation of $(dW, W^H)$ via the Volterra kernel matrix:

$$W^H(t_j) = \sqrt{2H} \sum_{k=0}^{j} \frac{(j-k+1)^{H+\frac{1}{2}} - (j-k)^{H+\frac{1}{2}}}{H + \frac{1}{2}} \cdot \Delta t^H \cdot Z_k$$

where $Z_k$ are **the same** iid $N(0,1)$ that generate the spot BM: $dW_k = \sqrt{\Delta t} \cdot Z_k$. The covariance matrix $\Sigma = M M^\top$ is PSD by construction — no eigenvalue surgery.

#### 12.3 Bug #3: Adaptedness Bias (Non-Previsible Variance)

In the Volterra discretization, $V_k$ depends on $Z_k$ (the current driving noise), so $V_k$ is **not** $\mathcal{F}_{k-1}$-measurable. Using $V_k$ in the spot Euler step:

$$\log S_k = \log S_{k-1} + (r - \tfrac{1}{2}V_k)\Delta t + \sqrt{V_k}\,\Delta W_k^{\text{spot}}$$

creates a systematic downward bias because $V_k$ and $\Delta W_k$ are correlated (Jensen's inequality). With $\rho = -0.7$: $\mathbb{E}[S_T] = 655$ instead of $697$.

**Fix**: Use **previsible** (lagged) variance $V_{k-1}$:

$$\log S_k = \log S_{k-1} + (r - \tfrac{1}{2}V_{k-1})\Delta t + \sqrt{V_{k-1}}\,\Delta W_k^{\text{spot}}$$

Result: $\mathbb{E}[S_T] = 696.9$ (exact).

#### 12.4 Backtest Results After Fix

| Model | Mean RMSE | Win Rate | Improvement |
|-------|-----------|----------|-------------|
| **Rough Bergomi** | **4.58%** | **14/20 (70%)** | ↓ from 16.0% |
| Neural SDE | 5.35% | 5/20 (25%) | ↓ from 14.5% |
| Black-Scholes | 6.01% | 1/20 (5%) | unchanged |

**RMSE by Maturity**:

| DTE | BS | Bergomi | Neural SDE |
|-----|-----|---------|------------|
| 7d | 5.90% | **2.87%** | 5.81% |
| 17d | **7.76%** | 10.33% | 5.20% |
| 31d | 5.34% | **2.74%** | 6.25% |
| 45d | 5.02% | **2.37%** | 4.13% |

Bergomi dominates at 7d, 31d, and 45d. The 17d weakness is due to the constant-$\eta$ model producing excessive skew at intermediate maturities (a known limitation of flat-$\eta$ rBergomi). Neural SDE fills this gap.

#### 12.5 Smile Validation

With $H = 0.07$, $\eta = 1.9$, $\rho = -0.7$, $\xi_0 = 0.030$ (30-day maturity):

| Moneyness | Model IV | Market IV |
|-----------|----------|-----------|
| −10% | 24.2% | 23.3% |
| −5% | 20.2% | 18.5% |
| ATM | 16.2% | 16.2% |
| +5% | 13.0% | 11.1% |
| +10% | 12.6% | 13.3% |

---

## Methodology

### 1. Signature-Conditional Neural SDE

The variance process $V_t$ is modeled in log-space for guaranteed positivity:

$$X_t = \log(V_t)$$

$$dX_t = \underbrace{\kappa(\theta - X_t)}_{\text{OU Prior}} dt + \underbrace{\mathcal{N}_\mu(\mathbb{S}_{0,t}, X_t)}_{\text{Neural Drift}} dt + \underbrace{\mathcal{N}_\sigma(\mathbb{S}_{0,t}, X_t)}_{\text{Neural Diffusion}} dW_t$$

Where $\mathbb{S}_{0,t}$ is the **running path signature** of the generated path up to time $t$:

$$\mathbb{S}_{0,t} = \text{Sig}\left((s, X_s)_{s \in [0,t]}\right) \in T^{(3)}(\mathbb{R}^2) \cong \mathbb{R}^{14}$$

The running signature is updated at each step via **Chen's identity**, making the SDE genuinely **path-dependent** (non-Markovian).

**Why running signatures?**
- **Path dependence**: The model conditions on full path history, essential for rough volatility
- **Universal approximation**: Signatures form a basis for continuous functions on paths
- **Gradient stability**: No vanishing/exploding gradients (unlike RNNs)
- **Incremental updates**: Chen's identity gives O(d³) update cost per step

### 2. Unsupervised Training via Normalized MMD

No labeled data needed. We minimize a **normalized Maximum Mean Discrepancy**:

$$\mathcal{L} = \sum_k \left(\frac{\mu_k^{\text{fake}} - \mu_k^{\text{real}}}{\sigma_k^{\text{real}}}\right)^2 + \lambda \cdot \left(\bar{V}^{\text{fake}} - \bar{V}^{\text{real}}\right)^2$$

Where:
- $\mu_k$ are the expected signature components (matching moments up to order 3)
- $\sigma_k^{\text{real}}$ normalizes each component by its empirical standard deviation
- Pure-time components (near-zero variance) are excluded (weight = 0)
- The mean penalty with $\lambda = 10$ prevents Jensen bias from the exponential map

### 3. Temporal Consistency

All dynamics use the **real temporal scale**:

$$dt = \frac{T}{n_{\text{steps}}} = \frac{0.00305}{20} \approx 1.53 \times 10^{-4} \text{ years} = 15 \text{ min}$$

Brownian increments are properly scaled: $dW_t = \sqrt{dt} \cdot Z$, $Z \sim \mathcal{N}(0,1)$.

The signature engine uses the real time increments (not `linspace(0,1)`) to ensure consistency between the time axis in the signature and the SDE dynamics.

### 4. Option Pricing via Monte Carlo

Given trained model:

1. Generate $N$ variance paths: $V^{(i)}_t$ for $i = 1, \ldots, N$
2. Simulate correlated spot: $dS = rS\,dt + \sqrt{V}S\,(\rho\,dW^V + \sqrt{1-\rho^2}\,dW^\perp)$
3. Price option: $C = e^{-rT} \frac{1}{N}\sum_i \max(S^{(i)}_T - K, 0)$
4. Invert to IV via Black-Scholes formula

### 5. Calibration

Optimize initial variance $v_0$ to minimize IV RMSE:

$$v_0^* = \arg\min_{v_0} \sqrt{\frac{1}{N}\sum_i \left(\sigma^{model}_i - \sigma^{market}_i\right)^2}$$

---

## Data Sources

### Primary Data

| Dataset | Source | Frequency | Period | Points |
|---------|--------|-----------|--------|--------|
| **VIX Spot** | TradingView | 5/10/15/30-min | 2023-01 → 2026-02 | ~20,000 each |
| **SPX** | TradingView | 5-min | 2025-02 → 2026-02 | 20,112 |
| **SPX** | TradingView | 30-min | 2020-01 → 2026-02 | 21,402 |
| **VVIX** | CBOE/TradingView | 15-min | 2023-02 → 2026-02 | 20,386 |
| **VIX Futures** | CBOE | Daily | 2013-01 → 2026-02 | 29,477 |
| **S&P 500** | Yahoo Finance | Daily | 2010-01 → 2023-01 | ~3,200 |

### Derived Metrics

- **Realized Volatility (daily)**: $\text{RV}_t = \sum_i r_{i,\text{intraday}}^2$ per trading day from SPX 5-min returns
- **Hurst Exponent**: Variogram method on $\log(\text{RV})$ → $H \approx 0.13$ (5-min) / $H \approx 0.12$ (30-min)
- **Term Structure Slope**: F2 - F1 from VIX futures
- **Vol-of-Vol**: From VVIX (implied) and realized vol of VIX (realized)

---

## Market Insights Discovered

### 1. The Vol-of-Vol Gap

```
VVIX (implied, 30-day):  ~96%   → η_implied  = 0.96
Realized Vol of VIX:     ~114%  → η_realized = 1.14
Gap:                     ~16% underpriced
```

**Implication**: The market consistently underprices volatility-of-volatility. Realized VIX swings exceed what VVIX implies.

### 2. Contango as Regime Indicator

| Regime | Frequency | VIX Level | Interpretation |
|--------|-----------|-----------|----------------|
| Contango | 77% | 17.0 | Normal, calm |
| Backwardation | 23% | 23.6 | Stress, crisis |

### 3. Roughness Confirmed (on Realized Vol, not VIX)

Hurst measured on **daily realized variance** from SPX intraday returns:
```
H (SPX 5-min → daily RV)   = 0.130  (R² = 0.91)   ✅ Rough
H (SPX 30-min → daily RV)  = 0.116  (R² = 0.99)   ✅ Rough
H (VIX 15-min log)          ≈ 0.47                  Expected (smoothed)
```

**Important**: VIX H ≈ 0.5 does **not** mean volatility is smooth. The VIX is a 30-day integral of implied variance — integration kills roughness. True roughness (H ≈ 0.1) is only visible on realized vol from intraday returns. This is consistent with Gatheral et al. (2018).

### 4. VIX-VVIX Correlation

```
ρ(VIX, VVIX) = 0.83
```

Panic amplifies: when VIX spikes, VVIX spikes even more.

---

## Architecture

### Neural Network

```
NeuralRoughSimulator (with running signatures, learnable OU params)
├── Input: running signature Sig(X_{0:t}) (dim=14) + log-variance (dim=1)
├── drift_net: MLP [15 → 64 → 64 → 64 → 1], tanh activation
├── diff_net:  MLP [15 → 64 → 64 → 64 → 1], sigmoid output
├── Output scaling: drift × drift_scale (tanh), diffusion × (max-min)(sigmoid) + min
├── OU prior: κ, θ read from config/params.yaml (learnable during training)
├── All hyperparams (width, depth, scales) loaded from YAML — no hardcoded values
└── Integration: Euler-Maruyama with running signature via Chen's identity
```

### Running Signature (Chen's Identity)

At each step $t$, the path signature $\mathbb{S}_{0,t}$ is updated incrementally:

$$\mathbb{S}_{0,t+dt} = \mathbb{S}_{0,t} \otimes \mathbb{S}_{t,t+dt}$$

For a 2D path (time, log-variance) with truncation order 3:
- Order 1: $S^1 \in \mathbb{R}^2$ → $S^1_{new} = S^1 + dx$
- Order 2: $S^2 \in \mathbb{R}^4$ → $S^2_{new} = S^2 + S^1 \otimes dx + \frac{1}{2} dx \otimes dx$
- Order 3: $S^3 \in \mathbb{R}^8$ → $S^3_{new} = S^3 + S^2 \otimes dx + S^1 \otimes \frac{dx^{\otimes 2}}{2} + \frac{dx^{\otimes 3}}{6}$

This gives the model genuine path-dependent (non-Markovian) dynamics, essential for rough volatility.

### Signature Computation (for loss)

```
SignatureFeatureExtractor (order=3, dt=real_dt)
├── Time augmentation: path → (time, value) with REAL dt increments
├── NumPy input → esig C++ library (15 features, with constant)
├── JAX input → custom JAX implementation (14 features, no constant)
├── Normalization: component-wise by std, zero-weight on pure-time dims
└── Output: 2 + 4 + 8 = 14 features (JAX, used in training)
```

### Training Loop

```
GenerativeTrainer (with validation, early stopping, LR warmup)
├── Data: 1,304 train + 230 validation VIX variance paths × 20 steps (15%)
├── Target: Signatures of real paths
├── Generated: Neural SDE with running signatures (computed internally)
├── Loss: Normalized MMD + mean penalty (λ=10)
│   ├── MMD: component-wise normalized by real sig std
│   ├── Pure-time components: zero-weight (std < 1e-8)
│   └── Mean penalty: λ·(mean_fake - mean_real)²
├── Brownian: dW = √dt · Z with dt = T/n_steps = 0.000153 years
├── LR: Linear warmup (50 steps) → cosine decay
├── Optimizer: Adam + grad clipping (1.0)
├── Early stopping: patience=50 on validation loss
├── Learnable OU: κ, θ optimized during training (regularized by prior)
└── Convergence: ~200 epochs, best loss ≈ 0.017
```

---

## Usage

### Installation

```bash
git clone https://github.com/your-repo/DeepRoughVol.git
cd DeepRoughVol

python -m venv .venv
.venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### Training

```bash
# Train Neural SDE on VIX (saves best model to models/)
python main.py
```

### Verification & Diagnostics

```bash
# Roughness + signatures + ablation study
python quant/verify_roughness.py

# Full coherence audit (dimensions, parameters, dt)
python quant/coherence_check.py
```

### Calibration

```bash
# Market parameter extraction (VVIX, Futures)
python quant/enhanced_calibration.py

# Option pricing vs Black-Scholes
python quant/risk_neutral_calibration.py

# Full robustness diagnostics
python quant/robustness_check.py

# Historical backtesting
python quant/backtesting.py
```

### Configuration

`config/params.yaml`:
```yaml
data:
  data_type: "vix"                    # "vix" or "realized_vol"
  source: "data/TVC_VIX, 15.csv"     # VIX for Q-measure training
  rv_source: "data/SP_SPX, 5.csv"    # SPX for P-measure roughness
  segment_length: 20
  vix_files:                          # All VIX frequencies
    5:  "data/TVC_VIX, 5.csv"
    15: "data/TVC_VIX, 15.csv"
    30: "data/TVC_VIX, 30.csv"
  spx_files:                          # SPX for Hurst estimation
    5:  "data/SP_SPX, 5.csv"
    30: "data/SP_SPX, 30.csv"

simulation:
  n_steps: 20
  T: 0.00305                         # Real horizon: 20 × 15min in years
  bars_per_day: 26
  bar_interval_min: 15

training:
  n_epochs: 300
  batch_size: 256
  learning_rate_init: 0.001
  gradient_clip: 1.0
  lambda_mean: 10.0                   # Mean penalty weight
  warmup_steps: 50                    # Linear LR warmup
  validation_split: 0.15             # 15% held out
  early_stopping_patience: 50        # Stop if val loss stalls

neural_sde:
  sig_truncation_order: 3
  mlp_width: 64
  mlp_depth: 3
  drift_scale: 0.5
  diffusion_min: 0.1
  diffusion_max: 1.6
  learn_ou_params: true              # Allow κ, θ to be learned
  kappa: 2.72                        # Initial mean-reversion (from VIX futures)
  theta: -3.5                        # Initial log-variance target
```

---

## File Structure

```
DeepRoughVol/
│
├── main.py                           # Entry point: train Neural SDE
├── compare_frequencies.py            # VIX training quality + SPX true roughness
├── compare_vix_vs_rv.py              # VIX vs Realized Vol statistical comparison
├── config/
│   └── params.yaml                   # Central config (all paths, params, hyperparams)
│
├── core/
│   ├── bergomi.py                    # Rough Bergomi baseline (reads mu from params)
│   └── stochastic_process.py         # Fractional Brownian Motion generation
│
├── ml/
│   ├── neural_sde.py                 # NeuralRoughSimulator (learnable κ,θ from YAML)
│   ├── signature_engine.py           # Path signatures (esig + JAX, real dt augmentation)
│   ├── losses.py                     # Normalized MMD + mean penalty loss
│   ├── generative_trainer.py         # Training (validation split, early stopping, LR warmup)
│   └── model.py                      # Legacy
│
├── models/                           # Trained models
│   └── neural_sde_best.eqx           # Best trained Neural SDE
│
├── quant/
│   ├── pricing.py                    # Monte Carlo option pricing (reads config)
│   ├── risk_neutral_calibration.py   # IV surface calibration vs BS
│   ├── enhanced_calibration.py       # VVIX + Futures market parameter extraction
│   ├── advanced_calibration.py       # Multi-method Hurst & forward variance (reads config)
│   ├── multi_maturity_calibration.py # Multi-maturity IV surface
│   ├── backtesting.py                # Historical backtesting (reads config)
│   ├── dashboard_v2.py               # Interactive dashboard
│   ├── robustness_check.py           # Diagnostic audit
│   ├── verify_roughness.py           # Roughness (VIX + true RV) + ablation verification
│   └── coherence_check.py            # Full coherence audit
│
├── scripts/
│   └── hurst_diagnostic.py           # Comprehensive VIX vs RV Hurst diagnostic
│
├── utils/
│   ├── config.py                     # Centralized config loader (load_config, cached)
│   ├── black_scholes.py              # Shared BS: pricing, IV, Greeks (single source)
│   ├── greeks_ad.py                  # Greeks via JAX autodiff (Δ, Γ, vega, θ, vanna, volga)
│   ├── data_loader.py                # VIX, RV, SPX loading (auto-detect frequency)
│   └── diagnostics.py                # Stats, Hurst (variogram/DMA), daily RV approach
│
├── data/                             # Raw market data only
│   ├── TVC_VIX, 5/10/15/30.csv       # VIX spot at multiple frequencies
│   ├── SP_SPX, 5.csv                 # SPX 5-min intraday (roughness)
│   ├── SP_SPX, 30.csv                # SPX 30-min intraday (roughness)
│   ├── data_^GSPC_*.csv              # SPX daily prices
│   └── options_cache/                # Cached option chains
│
├── outputs/                          # Generated results (JSON + HTML)
│   ├── advanced_calibration.json     # H, η, ρ, ξ₀ calibrated from market
│   ├── roughness_verification.json   # Roughness + ablation results
│   ├── backtest_results.json         # 30-day backtesting results
│   ├── risk_neutral_calibration.json # IV RMSE comparison
│   └── enhanced_calibration.json     # VVIX + futures calibration
│
└── research/
    ├── theory_draft.tex              # Math derivations
    ├── maths_proofs.tex              # Signature theory
    └── proof.tex                     # Additional proofs
```

---

## Lessons Learned

### 1. P ≠ Q (Measure Matters!)
Training on realized vol and testing on implied vol is fundamentally wrong. Always know which probability measure you're in.

### 2. Temporal Scale Must Be Physical
Using `dt = 1/n_steps` instead of the real physical time step (15 min ≈ 0.000153 years) introduced a **328× scaling error**. All SDE parameters (κ, σ, drift) are calibrated in annual units — the dt must match.

### 3. VIX ≠ Volatility (for Roughness)
The VIX is a 30-day integrated implied variance. Integration smooths the process, destroying roughness: $H_\text{VIX} \approx 0.5$ regardless of the true $H_\text{RV} \approx 0.1$. Roughness must be measured on **realized vol from intraday returns**, never on VIX. This error persisted for several phases.

### 4. Normalize Your Loss Components
Signature components span 8 orders of magnitude (from 10⁻¹⁴ for s³ₜₜₜ to 10⁻² for s¹ᵥ). Without normalization, the loss is blind to the stochastic components. Pure-time components must be excluded entirely.

### 5. Jensen's Inequality Bites
When the model generates log-variance and you exponentiate, $\mathbb{E}[e^X] > e^{\mathbb{E}[X]}$. An explicit mean penalty is necessary to control the first moment.

### 6. Kurtosis Requires Care
`kurtosis(data.flatten())` on multi-path data conflates the marginal distribution with the cross-sectional distribution. The correct metric is the average marginal kurtosis per time step.

### 7. Variogram Method Sensitivity
The variogram log-log fit is sensitive to which lag range is used. Excluding short lags (where roughness lives) can give H < 0 or H > 0.5. Similarly, DMA on cumulative sums requires a $-0.5$ correction that is easy to forget.

### 8. Market Data > Assumptions
Extract parameters (η, κ, θ) from observables (VVIX, futures) rather than assuming standard values.

### 9. Centralize Configuration
Hardcoded paths and parameters scattered across 15+ files is a maintenance nightmare and a source of inconsistencies. A single YAML config with a cached loader prevents drift.

### 10. Previsible Variance in Euler Schemes
In the Volterra discretization, $V_k$ depends on the current noise increment $Z_k$. Using $V_k$ in $\sqrt{V_k}\,dW_k$ creates a Jensen's-inequality bias: $\mathbb{E}[S_T]$ can be 6% below the forward. The fix is to use $V_{k-1}$ (the lagged, $\mathcal{F}_{k-1}$-measurable variance). This is standard in Euler-Maruyama but easy to forget when generating variance and spot from shared noise.

### 11. fBM ≠ fGn for Correlation
Fractional Gaussian noise increments ($\Delta W^H$) for $H < 0.5$ are **anti-persistent**: $\text{Corr}(\Delta W^H_k, \Delta W^H_{k+1}) < 0$. Treating them as the "driving BM" and applying Cholesky mixing inverts the leverage effect. The correct approach is to generate the **underlying standard BM** $W$ and build $W^H$ via the Volterra kernel, then correlate the spot with $W$ (not $W^H$).

---

## Future Work

### ✅ Completed (Phase 10)
- [x] **Sync hardcoded params with config**: κ, θ, drift_scale, diffusion range all read from `params.yaml`
- [x] **Fix dt/noise in calibration scripts**: `risk_neutral_calibration.py`, `multi_maturity_calibration.py`, `compare_frequencies.py` now use `dW = √dt · Z`
- [x] **Real backtesting**: `backtesting.py` loads trained model and runs actual Monte Carlo pricing
- [x] **Validation split + early stopping**: 15% held-out set, patience=50 epochs
- [x] **LR warmup**: Linear warmup (50 steps) before cosine decay
- [x] **Learnable OU parameters**: κ and θ stored as JAX arrays, differentiated through during training
- [x] **Greeks via AD**: `utils/greeks_ad.py` — Δ, Γ, vega, θ, vanna, volga via `jax.grad`
- [x] **Shared BS utilities**: `utils/black_scholes.py` — single source, used by all calibration scripts

### ✅ Completed (Phase 11)
- [x] **Correct Hurst methodology**: Daily RV from SPX returns instead of VIX → H = 0.13 (not 0.48)
- [x] **Fix variogram**: Remove middle-portion subsetting that excluded short lags
- [x] **Fix DMA**: Apply $-0.5$ correction for cumulative sum integration
- [x] **H selection logic**: Prefer variogram over DMA in `get_optimal_bergomi_params()`
- [x] **Config centralization**: `utils/config.py` + enriched `params.yaml` (vix_files, spx_files, backtesting, outputs)
- [x] **Adaptive RV window**: Auto-detect bar frequency, scale rv_window proportionally
- [x] **UTF-8 terminal fix**: All scripts handle Windows CP1252 encoding

### ✅ Completed (Phase 12)
- [x] **Fix Davies-Harte fBM**: Correct Dieker (2004) algorithm — complex Gaussian, proper IFFT scaling. Var ratio 1.000.
- [x] **Fix spot–vol correlation**: Joint (dW, W^H) via Volterra kernel matrix $M$ (PSD by construction, no eigenvalue surgery)
- [x] **Fix adaptedness bias**: Previsible variance $V_{k-1}$ in spot Euler scheme. $\mathbb{E}[S_T]$ bias: 6% → 0%.
- [x] **Real SPY options backtest**: 5 snapshots × 4 maturities vs real Yahoo Finance option chains
- [x] **VIX futures xi0 calibration**: Forward variance from CBOE term structure (30d/60d/90d)
- [x] **Bergomi wins 70%**: Mean RMSE 4.58% (was 16.0%), beating BS (6.01%) and Neural SDE (5.35%)

### Remaining
- [ ] **Maturity-dependent eta**: Constant η produces excess skew at ~17d; a term-structure η(T) would improve intermediate maturities
- [ ] **Neural SDE retraining**: Current model uses segment_length=120 but hasn't been retrained since the fBM fix
- [ ] **Multi-frequency training**: Train on 5/10/15/30-min simultaneously to capture multi-scale roughness
- [ ] **Joint Loss**: Train on VIX paths + option prices simultaneously for end-to-end calibration
- [ ] **Regime-aware training**: Use VVIX/VIX ratio or contango/backwardation as conditioning variable
- [ ] **Live Dashboard**: Real-time calibration monitoring with Streamlit

---

## References

### Academic Papers

1. Kidger, P. et al. (2021). *Neural SDEs as Infinite-Dimensional GANs*. ICML.
2. Gatheral, J. et al. (2018). *Volatility is Rough*. Quantitative Finance.
3. Bayer, C. et al. (2016). *Pricing under rough volatility*. Quantitative Finance.
4. Lyons, T. (1998). *Differential equations driven by rough paths*. Rev. Mat. Iberoamericana.
5. Chevyrev, I. & Kormilitzin, A. (2016). *A primer on the signature method*.

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

*"The market can stay irrational longer than you can stay solvent, but a well-calibrated model helps."*
