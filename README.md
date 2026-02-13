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

**Latest Achievement**: After fixing temporal scaling (328× dt mismatch), adding signature normalization and mean penalty, the Neural SDE now matches real VIX variance distribution with **< 0.3% mean error** and **near-identical kurtosis**.

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

### Calibrated Parameters (from Market Data)

| Parameter | Symbol | Value | Source |
|-----------|--------|-------|--------|
| Vol-of-Vol | η | 0.96 (implied) / 1.14 (realized) | VVIX/100 vs realized vol(VIX) |
| Mean Reversion | κ | 2.72 | VIX Futures term structure |
| Long-term VIX | θ | 18.6% | Historical VIX mean (2024-2026) |
| Half-life | t½ | 93 days | ln(2)/κ |
| Hurst Exponent | H | 0.05 (SPX 30min) / 0.48 (VIX 15min) | Variogram method |

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

#### 4.2 Hurst Exponent Estimation is Fragile

We tested multiple methods and data sources:

| Data Source | Method | H estimate | Interpretation |
|-------------|--------|-----------|----------------|
| SPX 30-min returns | Variogram | 0.05 | Very rough ✓ |
| VIX 15-min (log) | Variogram | 0.48 | Marginally rough ✓ |
| SPX 30-min returns | R/S | 0.51 | Not rough ✗ |

**Conclusion**: H depends heavily on data source and method. SPX high-frequency returns show H ≈ 0.05 (very rough), while VIX intraday gives H ≈ 0.48 (marginally rough but still < 0.5). The robust finding is H < 0.5 across all sources.

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
| Hurst Exponent H | 0.475 | 0.438 ± 0.09 | ✅ Both rough |
| Signature Norm | 1.131 | 1.131 | ✅ Match |
| Signature Correlation | — | 1.000 | ✅ Perfect |

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
| **VIX Spot** | TradingView | 15-min | 2024-08 → 2026-02 | 20,225 |
| **SPX** | TradingView | 5/30-min | 2019 → 2026 | 21,402 |
| **VVIX** | CBOE/TradingView | 15-min | 2023-02 → 2026-02 | 20,386 |
| **VIX Futures** | CBOE | Daily | 2013-01 → 2026-02 | 29,477 |

### Derived Metrics

- **Realized Volatility**: From SPX 5-min returns, exponential weighting
- **Hurst Exponent**: Variogram method on SPX 30-min
- **Term Structure Slope**: F2 - F1 from VIX futures

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

### 3. Roughness Confirmed

High-frequency data shows:
```
H (SPX 30-min returns)  ≈ 0.05 << 0.5   (very rough)
H (VIX 15-min log)      ≈ 0.48 < 0.5    (marginally rough)
```

Volatility is **rough** (H < 0.5) — far from the smooth diffusions (H = 0.5) in Heston/Black-Scholes.
The exact H value depends on frequency and data source, but the qualitative finding is robust.

### 4. VIX-VVIX Correlation

```
ρ(VIX, VVIX) = 0.83
```

Panic amplifies: when VIX spikes, VVIX spikes even more.

---

## Architecture

### Neural Network

```
NeuralRoughSimulator (with running signatures)
├── Input: running signature Sig(X_{0:t}) (dim=14) + log-variance (dim=1)
├── drift_net: MLP [15 → 64 → 64 → 64 → 1], tanh activation
├── diff_net:  MLP [15 → 64 → 64 → 64 → 1], sigmoid output
├── Output scaling: drift × 0.5 (tanh), diffusion × 1.5 (sigmoid) + 0.1
├── OU prior: κ=2.72, θ=-3.5 (log-variance ~ 17% vol)
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
GenerativeTrainer
├── Data: 1,534 VIX variance paths × 20 steps (15-min bars)
├── Target: Signatures of real paths
├── Generated: Neural SDE with running signatures (computed internally)
├── Loss: Normalized MMD + mean penalty (λ=10)
│   ├── MMD: component-wise normalized by real sig std
│   ├── Pure-time components: zero-weight (std < 1e-8)
│   └── Mean penalty: λ·(mean_fake - mean_real)²
├── Brownian: dW = √dt · Z with dt = T/n_steps = 0.000153 years
├── Optimizer: Adam + cosine LR decay + grad clipping
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
  source: "data/TVC_VIX, 15.csv"
  segment_length: 20

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

neural_sde:
  sig_truncation_order: 3
  mlp_width: 64
  mlp_depth: 3
```

---

## File Structure

```
DeepRoughVol/
│
├── main.py                           # Entry point: train Neural SDE
├── config/
│   └── params.yaml                   # Central configuration (temporal, training, model)
│
├── core/
│   ├── bergomi.py                    # Rough Bergomi baseline
│   └── stochastic_process.py         # SDE utilities
│
├── ml/
│   ├── neural_sde.py                 # NeuralRoughSimulator (Equinox, Chen's identity)
│   ├── signature_engine.py           # Path signatures (esig + JAX, real dt augmentation)
│   ├── losses.py                     # Normalized MMD + mean penalty loss
│   ├── generative_trainer.py         # Training loop (real dt, proper dW scaling)
│   └── model.py                      # Legacy
│
├── models/                           # Trained models
│   └── neural_sde_best.eqx           # Best trained Neural SDE
│
├── quant/
│   ├── pricing.py                    # Monte Carlo option pricing (barrier, vanilla)
│   ├── risk_neutral_calibration.py   # IV surface calibration vs BS
│   ├── enhanced_calibration.py       # VVIX + Futures market parameter extraction
│   ├── advanced_calibration.py       # Multi-method calibration
│   ├── multi_maturity_calibration.py # Multi-maturity IV surface
│   ├── backtesting.py                # Historical backtesting
│   ├── dashboard_v2.py               # Interactive dashboard
│   ├── robustness_check.py           # Diagnostic audit
│   ├── verify_roughness.py           # Roughness + ablation verification
│   └── coherence_check.py            # Full coherence audit
│
├── utils/
│   ├── data_loader.py                # VIX, RV, SPX loading (temporal coherence)
│   └── diagnostics.py                # Stats (marginal kurtosis, Hurst, ACF)
│
├── data/                             # Raw market data only
│   ├── TVC_VIX, 15.csv               # VIX spot 15-min (training)
│   ├── TVC_VIX, 5/10/30.csv          # VIX at other frequencies
│   ├── data_^GSPC_*.csv              # SPX prices
│   └── options_cache/                # Cached option chains
│
├── outputs/                          # Generated results (JSON + HTML)
│   ├── roughness_verification.json
│   ├── coherence_check.json
│   ├── risk_neutral_calibration.json
│   ├── enhanced_calibration.json
│   └── backtest_results.json
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

### 3. Normalize Your Loss Components
Signature components span 8 orders of magnitude (from 10⁻¹⁴ for s³ₜₜₜ to 10⁻² for s¹ᵥ). Without normalization, the loss is blind to the stochastic components. Pure-time components must be excluded entirely.

### 4. Jensen's Inequality Bites
When the model generates log-variance and you exponentiate, $\mathbb{E}[e^X] > e^{\mathbb{E}[X]}$. An explicit mean penalty is necessary to control the first moment.

### 5. Kurtosis Requires Care
`kurtosis(data.flatten())` on multi-path data conflates the marginal distribution with the cross-sectional distribution. The correct metric is the average marginal kurtosis per time step.

### 6. High-Frequency Data Reveals Structure
Daily data masks roughness. Intraday data (≤30min) shows H < 0.5, with exact values depending on data source (SPX returns: H≈0.05, VIX levels: H≈0.48).

### 7. Market Data > Assumptions
Extract parameters (η, κ, θ) from observables (VVIX, futures) rather than assuming standard values.

### 8. Persist Your Models
A trained model is worthless if you reinitialize random weights in downstream scripts.

---

## Future Work

### High Priority
- [ ] **Sync hardcoded params with config**: Model hardcodes κ=2.72, θ=-3.5, drift_scale=0.5 — these should be read from `params.yaml`
- [ ] **Fix dt/noise in calibration scripts**: `risk_neutral_calibration.py`, `advanced_calibration.py`, `multi_maturity_calibration.py` still use old unscaled noise
- [ ] **Real backtesting**: `backtesting.py` currently uses synthetic Neural SDE smiles — should load the actual trained model and run MC pricing
- [ ] **Validation split + early stopping**: Training uses all data with no held-out set for overfitting detection

### Architecture
- [ ] **Longer paths**: Increase from 20 to 50-100 steps for richer signature information
- [ ] **LR warmup**: Add linear warmup (50 steps) before cosine decay — stabilizes early training when the signature feedback loop is fragile
- [ ] **Learnable OU parameters**: Allow κ and θ to be learned during training (with regularization toward market-calibrated values)
- [ ] **Multi-frequency training**: Train on 5/10/15/30-min simultaneously to capture multi-scale roughness

### Pricing & Calibration
- [ ] **Multi-Maturity Calibration**: Fit full IV surface across strikes and expiries simultaneously
- [ ] **Greeks via AD**: Automatic differentiation for Δ, Γ, Vega, Vanna using JAX's `grad`
- [ ] **Joint Loss**: Train on VIX paths + option prices simultaneously for end-to-end calibration

### Infrastructure
- [ ] **Shared BS utilities**: Extract `BlackScholes` class (duplicated in 4+ files) into `utils/black_scholes.py`
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
