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
10. [References](#references)

---

## Abstract

**DeepRoughVol** is a non-parametric generative framework that learns volatility dynamics directly from market data using **Neural Stochastic Differential Equations (Neural SDEs)** conditioned on **Path Signatures**.

Unlike traditional stochastic volatility models (Heston, Bergomi) which impose rigid parametric forms, this approach:
- Learns drift and diffusion functions from data
- Captures rough volatility behavior naturally
- Calibrates to real option prices
- Incorporates market observables (VVIX, VIX Futures)

**Final Achievement**: The Neural SDE **beats Black-Scholes** on option pricing with a **2.95% IV RMSE** vs **3.94%** for BS (25% error reduction).

---

## Key Results

### Option Pricing Performance

| Model | IV RMSE | Improvement |
|-------|---------|-------------|
| **Neural SDE (VIX-trained)** | **2.95%** | ✅ Winner |
| Black-Scholes | 3.94% | Baseline |
| Neural SDE (RV-trained) | 5.85% | ❌ Wrong measure |

### Calibrated Parameters (from Market Data)

| Parameter | Symbol | Value | Source |
|-----------|--------|-------|--------|
| Vol-of-Vol | η | 0.44 (implied) / 1.5 (realized) | VVIX |
| Mean Reversion | κ | 2.72 | VIX Futures |
| Long-term VIX | θ | 18.5% | Historical |
| Half-life | t½ | 93 days | Term Structure |
| Hurst Exponent | H | 0.05 | High-freq SPX |

### Statistical Fit (Generated vs Real VIX Paths)

| Metric | Market | Neural SDE | Analysis |
|--------|--------|------------|----------|
| Median Variance | 0.0151 | 0.0119 | ✓ Captures base regime |
| Kurtosis | 15.93 | 13.93 | ✓ Fat tails reproduced |
| ACF(1) | 0.97 | 0.85 | ✓ Long memory |
| MMD Loss | — | 0.0003 | ✓ Converged |

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

**Result**: ❌ **Catastrophic failure** — Model produced ~50% volatility while market IV was ~16%.

```
Model output:  σ ≈ 50%
Market IV:     σ ≈ 16%
Error:         3x overestimate!
```

---

### Phase 3: Diagnosis — The P vs Q Problem

This was the critical insight of the entire project.

**Key Realization**: We had trained on **Realized Volatility** (P-measure) but were testing on **Implied Volatility** (Q-measure).

| Measure | Name | What it represents | Typical σ |
|---------|------|-------------------|-----------|
| **P** | Physical / Real-World | What actually happens historically | ~50% (RV) |
| **Q** | Risk-Neutral | What's priced into options (no-arbitrage) | ~16% (IV) |

**Why they differ**: The **Variance Risk Premium (VRP)**

$$VRP = \mathbb{E}^P[RV] - IV \approx 50\% - 16\% = 34\%$$

Investors pay a premium for volatility protection, so IV < RV on average. This is one of the most robust findings in empirical finance.

**Our mistake**: Training on realized volatility (high) and expecting to match implied volatility (low).

---

### Phase 4: Robustness Audit

Before fixing the P/Q issue, we audited the entire pipeline for other problems.

**Script**: `quant/robustness_check.py`

**Discoveries**:

#### 4.1 Correlation ρ is NOT -0.7

Classical stochastic volatility models use ρ ≈ -0.7 for the "leverage effect" (stocks down → vol up).

**Our finding from data**:
```
ρ(Return, ΔVol) = -0.07   ← Much weaker than assumed!
```

However, volatility clustering is confirmed:
```
ρ(|Return|, Vol) = +0.50  ← Large moves predict high vol
```

**Interpretation**: The leverage effect may be weaker in recent markets, or requires different measurement. But the clustering effect is robust.

#### 4.2 Hurst Exponent Estimation is Fragile

We tested multiple methods for estimating H:

| Method | H estimate | Interpretation |
|--------|-----------|----------------|
| Variogram | 0.046 | Very rough ✓ |
| R/S | 0.51 | Not rough ✗ |
| Periodogram | 0.99 | Noise |

**Conclusion**: High-frequency data (30-min) confirms H ≈ 0.05, but estimation requires care. Daily data gives H ≈ 0.1-0.2.

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

**Result**: 
```
BEFORE (RV training): Mean variance = 0.25  (σ = 50%)
AFTER (VIX training): Mean variance = 0.037 (σ = 19%)
```

Now matches market IV scale! ✓

---

### Phase 6: Enhanced Calibration with Market Data

With new data (VVIX, VIX Futures), we could extract model parameters from the market itself rather than assuming them.

**Script**: `quant/enhanced_calibration.py`

#### 6.1 Vol-of-Vol from VVIX

The VVIX measures expected volatility of the VIX (30-day horizon).

```
VVIX mean:           96%
Realized Vol(VIX):   150%
Ratio:               0.64x
```

**Discovery**: The market systematically **underprices** vol-of-vol by ~40%!

This is a negative risk premium on volatility-of-volatility. Sellers of VIX options receive less than fair compensation.

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
│                                         │
│  ✅ NEURAL SDE BEATS BLACK-SCHOLES      │
└─────────────────────────────────────────┘
```

---

## Methodology

### 1. Signature-Conditional Neural SDE

The variance process $V_t$ is modeled in log-space for guaranteed positivity:

$$X_t = \log(V_t)$$

$$dX_t = \underbrace{\kappa(\theta - X_t)}_{\text{OU Prior}} dt + \underbrace{\mathcal{N}_\mu(\mathbb{S}_t)}_{\text{Neural Drift}} dt + \underbrace{\mathcal{N}_\sigma(\mathbb{S}_t)}_{\text{Neural Diffusion}} dW_t$$

Where $\mathbb{S}_t$ is the truncated **path signature** of order 3:

$$\mathbb{S}_t = \left(1, \int_0^t dW, \int_0^t W dW, \int \int dW dW, \ldots \right)$$

**Why signatures?**
- **Universal approximation**: Signatures form a basis for continuous functions on paths
- **Gradient stability**: No vanishing/exploding gradients (unlike RNNs)
- **Geometric encoding**: Naturally captures roughness and path irregularity

### 2. Unsupervised Training via MMD

No labeled data needed. We minimize the **Maximum Mean Discrepancy**:

$$\mathcal{L}_{MMD} = \left\| \mathbb{E}[\phi(\text{Generated})] - \mathbb{E}[\phi(\text{Market})] \right\|^2_{\mathcal{H}}$$

Where $\phi$ maps paths to their signatures. This matches distributions without requiring path-by-path correspondence.

### 3. Option Pricing via Monte Carlo

Given trained model:

1. Generate $N$ variance paths: $V^{(i)}_t$ for $i = 1, \ldots, N$
2. Simulate correlated spot: $dS = rS\,dt + \sqrt{V}S\,(\rho\,dW^V + \sqrt{1-\rho^2}\,dW^\perp)$
3. Price option: $C = e^{-rT} \frac{1}{N}\sum_i \max(S^{(i)}_T - K, 0)$
4. Invert to IV via Black-Scholes formula

### 4. Calibration

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
VVIX (implied, 30-day):  ~96%
Realized Vol of VIX:     ~150%
Gap:                     ~40% underpriced
```

**Implication**: Systematically selling VIX options may appear profitable but doesn't compensate for tail risk.

### 2. Contango as Regime Indicator

| Regime | Frequency | VIX Level | Interpretation |
|--------|-----------|-----------|----------------|
| Contango | 77% | 17.0 | Normal, calm |
| Backwardation | 23% | 23.6 | Stress, crisis |

### 3. Roughness Confirmed

High-frequency data shows:
```
H (Hurst) ≈ 0.05 << 0.5
```

Volatility is **rough** — far from the smooth diffusions in Heston/Black-Scholes.

### 4. VIX-VVIX Correlation

```
ρ(VIX, VVIX) = 0.83
```

Panic amplifies: when VIX spikes, VVIX spikes even more.

---

## Architecture

### Neural Network

```
NeuralRoughSimulator
├── Input: signature (dim=14) + log-variance (dim=1)
├── drift_net: MLP [15 → 64 → 64 → 1], tanh activation
├── diff_net:  MLP [15 → 64 → 64 → 1], softplus output
├── Output scaling: drift × 0.2, diffusion × 0.3
└── Integration: Euler-Maruyama with dt = T/n_steps
```

### Signature Computation

```
SignatureFeatureExtractor (order=3)
├── Time augmentation: path → (time, value)
├── NumPy input → esig C++ library (fast)
├── JAX input → custom JAX implementation (differentiable)
└── Output: 2 + 4 + 8 = 14 features
```

### Training Loop

```
GenerativeTrainer
├── Data: 1,534 VIX variance paths × 20 steps
├── Target: Signatures of real paths
├── Generated: Neural SDE with random noise
├── Loss: MMD between signature distributions
├── Optimizer: Adam + cosine LR decay + grad clipping
└── Convergence: ~100 epochs, MMD < 0.001
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
# Train Neural SDE on VIX
python main.py
```

### Calibration

```bash
# Market parameter extraction (VVIX, Futures)
python quant/enhanced_calibration.py

# Option pricing vs Black-Scholes
python quant/risk_neutral_calibration.py

# Full robustness diagnostics
python quant/robustness_check.py
```

### Configuration

`config/params.yaml`:
```yaml
data:
  data_type: "vix"           # "vix" or "realized_vol"
  source: "data/TVC_VIX, 15.csv"

neural_sde:
  sig_truncation_order: 3
  hidden_dim: 64

training:
  n_epochs: 200
  batch_size: 128
  learning_rate_init: 0.001
  gradient_clip: 1.0
```

---

## File Structure

```
DeepRoughVol/
│
├── main.py                           # Entry point: train Neural SDE
├── config/
│   └── params.yaml                   # Central configuration
│
├── core/
│   ├── bergomi.py                    # Rough Bergomi baseline
│   └── stochastic_process.py         # SDE utilities
│
├── ml/
│   ├── neural_sde.py                 # NeuralRoughSimulator (Equinox)
│   ├── signature_engine.py           # Path signatures (esig + JAX)
│   ├── losses.py                     # MMD loss
│   ├── generative_trainer.py         # Main training loop
│   └── model.py                      # Legacy
│
├── quant/
│   ├── pricing.py                    # Monte Carlo option pricing
│   ├── risk_neutral_calibration.py   # IV surface calibration
│   ├── enhanced_calibration.py       # VVIX + Futures analysis
│   ├── market_constrained_trainer.py # eta-constrained training
│   ├── robustness_check.py           # Diagnostic audit
│   ├── advanced_calibration.py       # Hurst estimation
│   └── empirical_vs_grid_calibration.py
│
├── utils/
│   ├── data_loader.py                # VIX, RV, SPX loading
│   └── diagnostics.py                # Statistics, plots
│
├── data/                             # Raw market data only
│   ├── TVC_VIX, 15.csv               # VIX spot
│   ├── SP_SPX, 30.csv                # SPX prices
│   ├── CBOE_DLY_VVIX, 15.csv         # VVIX
│   ├── cboe_vix_futures_full/        # VIX futures
│   └── options_cache/                # Cached option chains
│
├── outputs/                          # Generated results
│   ├── enhanced_calibration.json     # Calibrated parameters
│   ├── risk_neutral_calibration.json # Pricing results
│   ├── robustness_check.json         # Diagnostic results
│   └── *.html                        # Interactive plots
│
└── research/
    ├── theory_draft.tex              # Math derivations
    └── maths_proofs.tex              # Signature theory
```

---

## Lessons Learned

### 1. P ≠ Q (Measure Matters!)
Training on realized vol and testing on implied vol is fundamentally wrong. Always know which probability measure you're in.

### 2. High-Frequency Data Reveals Structure
Daily data masks roughness. You need ≤30min frequency to see H < 0.1.

### 3. Market Data > Assumptions
Extract parameters (η, κ, θ) from observables (VVIX, futures) rather than assuming standard values.

### 4. Tensor Shape Consistency
`esig` and JAX signature engines have different output dimensions. Always verify shapes at interfaces.

### 5. Persist Your Models
A trained model is worthless if you reinitialize random weights in downstream scripts.

---

## Future Work

- [ ] **Multi-Maturity Calibration**: Fit full IV surface across strikes and expiries
- [ ] **Greeks via AD**: Automatic differentiation for Δ, Γ, Vega, Vanna
- [ ] **Regime Switching**: Use VVIX/VIX ratio as regime indicator
- [ ] **Joint Loss**: Train on VIX paths + option prices simultaneously
- [ ] **Live Dashboard**: Streamlit app for real-time calibration

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
