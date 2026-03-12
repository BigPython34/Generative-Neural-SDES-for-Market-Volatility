# DeepRoughVol: Neural Stochastic Volatility Engine

![Python](https://img.shields.io/badge/Python-3.12-blue)
![JAX](https://img.shields.io/badge/JAX-0.8-orange)
![Equinox](https://img.shields.io/badge/Equinox-Neural%20SDE-green)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-teal)
![License](https://img.shields.io/badge/License-MIT-grey)

> **A production-grade Neural SDE framework for rough volatility modeling, multi-measure option pricing, portfolio risk management, and exotic derivatives valuation.**

---

## Table of Contents

1. [Abstract](#abstract)
2. [Key Results](#key-results)
3. [Joint SPX-VIX Calibration](#joint-spx-vix-calibration-q-measure)
4. [Neural SDE Q-Calibration](#neural-sde-q-calibration-girsanov)
5. [Multi-Scale Hurst Estimation](#multi-scale-hurst-estimation)
6. [Architecture Overview](#architecture-overview)
7. [Multi-Measure Framework](#multi-measure-framework)
8. [Modules](#modules)
9. [Methodology](#methodology)
10. [Data Sources](#data-sources)
11. [Usage](#usage)
12. [API Reference](#api-reference)
13. [File Structure](#file-structure)
14. [References](#references)

> **Research narrative**: For the full story — design decisions, false starts, code insights, and lessons learned — see [RESEARCH.md](RESEARCH.md).

---

## Abstract

**DeepRoughVol** is a non-parametric generative framework that learns volatility dynamics directly from market data using **Neural Stochastic Differential Equations (Neural SDEs)** conditioned on **Path Signatures**.

- **Volatility modeling**: Signature-conditioned Neural SDE with **dual backbone** (OU or Volterra/fractional — nests rBergomi exactly) and optional Merton jump-diffusion
- **Roughness proof**: Empirical verification of $H \approx 0.1$ from SPX realized vol, with variogram, structure function, signature correlation (0.9996), and MMD ablation
- **Multi-measure training**: Separate P-measure (physical) and Q-measure (risk-neutral) models with appropriate loss functions
- **Exact path signatures**: Chen's identity at orders 2, 3, 4 — making the SDE genuinely non-Markovian
- **Option pricing**: European vanillas, barriers, Asian, lookback, autocallable, cliquet, variance/volatility swaps
- **Neural SDE Greeks**: Model-implied Δ, Γ, Vega via JAX autodiff through the full MC pipeline
- **Risk management**: Monte Carlo VaR/CVaR, parametric VaR, stressed VaR, component VaR, tail risk (Hill estimator)
- **Hedging**: Delta hedging simulator comparing BS, Bartlett (vanna/volga-corrected), and sticky-strike strategies
- **P&L attribution**: Second-order Taylor decomposition into Greeks contributions
- **Regime detection**: Multi-signal classifier (VIX, VVIX, term structure, VRP) with adaptive model parameters
- **Market data integration**: Real SOFR rates, VVIX-calibrated vol-of-vol, VIX futures term structure, cached SPY options
- **REST API**: FastAPI server with Swagger docs + interactive dashboard

The model is benchmarked against **Rough Bergomi** (rBergomi) and **Black-Scholes** on real SPY options surfaces.

---

## Key Results

### Model Comparison — IV Smile RMSE (SPY Options)

| Model | Mean IV RMSE | Win Rate | Best At |
|-------|:---:|:---:|---|
| **Rough Bergomi** | **5.63%** | 75% (9/12) | Medium maturities (14–41 DTE) |
| Neural SDE | 7.27% | 25% (3/12) | Short maturities (≤14 DTE) |
| Black-Scholes | 7.73% | 0% | Baseline |

### Calibrated Parameters

| Parameter | Symbol | Value | Source |
|-----------|:---:|:---:|---|
| **Hurst (P-measure consensus)** | $H_P$ | **0.110 ± 0.003** | Multi-scale variogram + structure function + ratio (5m → daily, 500 bootstrap) |
| **Hurst (Q-measure)** | $H_Q$ | **0.005** | Joint SPX-VIX rBergomi calibration (refined grid) |
| **Vol-of-Vol (Q-cal)** | $\eta_Q$ | **1.341** | Joint SPX-VIX calibration |
| **Correlation (Q-cal)** | $\rho_Q$ | **-0.959** | Joint SPX-VIX calibration |
| **Girsanov Q-loss** | $L_Q$ | **0.097** | Neural SDE Q-calibration (1,185 params, Girsanov MLP) |
| Vol-of-Vol (VVIX) | $\eta$ | 1.33 | VVIX with H-correction |
| Mean Reversion | $\kappa$ | 2.72 | VIX Futures term structure |
| SOFR Rate | $r$ | 3.73% | NY Fed SOFR daily |
| Market Regime | — | elevated | Multi-signal consensus |

### Risk Metrics (Neural SDE, P-measure)

| Metric | Value |
|--------|:---:|
| VaR 95% (~5 days) | 4.05% |
| VaR 99% | 5.80% |
| CVaR 95% (Expected Shortfall) | 5.13% |
| CVaR 99% | 6.53% |

---

## Joint SPX-VIX Calibration (Q-Measure)

> **P2 of the research roadmap**: simultaneous calibration of the rBergomi model to SPX options smile + VIX term structure under the risk-neutral measure.

### Method

The forward variance curve $\xi_0(t)$ is bootstrapped from the **VIX term structure** (VIX1D → VIX1Y) using the exact model identity:

$$E^Q\big[\text{VIX}^2(\tau)\big] = \frac{1}{\tau} \int_0^\tau \xi_0(s)\,ds$$

This identity is **parameter-independent** — $H$, $\eta$, $\rho$ do not affect it. Therefore $\xi_0(t)$ is uniquely determined by the observed VIX levels, giving a model-free anchor.

With $\xi_0$ fixed, the remaining parameters $(H, \eta, \rho)$ are calibrated by minimizing the joint loss:

$$L(H,\eta,\rho) = \lambda_{SPX} \cdot L_{SPX} + \lambda_{VIX} \cdot L_{VIX} + \lambda_{mart} \cdot L_{mart} + \lambda_{reg} \cdot L_{reg}$$

via JAX-accelerated Monte Carlo: vectorized Volterra kernel, batched option pricing, Common Random Numbers (CRN) for noise-free optimization.

### VIX Term Structure Fit

| Tenor | Market | Model | Δ |
|:---:|:---:|:---:|:---:|
| 9d | 23.77 | 22.13 | -1.64 |
| 30d | 22.66 | 22.12 | -0.54 |
| 90d | 23.24 | 22.93 | -0.31 |
| 180d | 24.70 | 24.43 | -0.27 |
| 365d | 24.86 | 24.61 | -0.25 |

**VIX fit**: < 0.5 pts at 30d+ (excellent). The 9d gap (1.6 pts) comes from the coarse time grid.

### SPX-VIX Joint Calibration Puzzle

The SPX smile fit shows **Total RMSE** = 618 bps, with an **ATM level bias** of +478 bps and a **Shape RMSE** of 372 bps (de-meaned — actual smile shape quality).

The +478 bps ATM bias is the well-documented **SPX-VIX joint calibration puzzle** (Guyon 2019, Rømer 2022): the rBergomi model cannot simultaneously match VIX levels and SPX ATM IV because $\text{VIX} \approx 22\%$ while ATM IV $\approx 16\%$. The gap reflects the variance swap convexity premium from OTM put skew. This is a **model limitation**, not a calibration bug. Solutions include jumps (Bates), multi-factor extensions (Guyon 2019), or LSV hybrids.

### Performance

| Mode | Grid | Paths | Strikes/mat | Time | Speed |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Quick | 48 pts | 3,000 | 12 | **27s** | 4.0 pts/s |
| Normal | 245 pts | 8,000 | 12 | ~90s | ~2.7 pts/s |

Key optimizations: JAX-vectorized Volterra kernel, kernel cache, strike subsampling (12/maturity), CRN, bounded Nelder-Mead ($\eta \geq 0.5$).

---

## Neural SDE Q-Calibration (Girsanov)

> **P2b of the research roadmap**: learn a risk-neutral drift correction $\lambda_\phi(v,t)$ via Girsanov's theorem, on top of the frozen P-measure Neural SDE. This produces Q-dynamics that reprice the full SPX options surface + VIX term structure + VIX futures simultaneously.

### Method

Under the physical measure P, the Neural SDE has drift $\mu^P(V,t)$ and diffusion $\sigma(V,t)$. Girsanov's theorem gives the Q-dynamics:

$$d\log V_t = \big[\mu^P(V,t) - \lambda_\phi(V,t)\cdot\sigma(V,t)\big]dt + \sigma(V,t)\,dW^Q_t$$

**Key insight**: $\sigma(V,t)$ is **frozen** from the P-model — Girsanov preserves the diffusion coefficient. Only the market price of risk $\lambda_\phi$ is learned via a small MLP (2→32→32→1, GELU activation, tanh output × $\lambda_{\max}$). The Novikov condition $E\big[\exp\big(\frac{1}{2}\int_0^T |\lambda_\phi|^2\,dt\big)\big] < \infty$ ensures the measure change is valid and is enforced via soft penalty.

### Loss Function

The composite Q-loss trains on **5 simultaneous objectives**:

$$L = w_{\text{smile}}\cdot L_{SPX} + w_{\text{vix}}\cdot L_{VIX} + w_{\text{mart}}\cdot L_{\text{mart}} + w_{\text{cal}}\cdot L_{\text{cal}} + w_{\text{nov}}\cdot E\Big[\sum \lambda^2\,dt\Big]$$

where $L_{SPX}$ is the multi-maturity **vega-normalized price loss** (≈ IV loss to first order, fully JAX-differentiable), and $L_{VIX}$ uses relative squared error across 12 combined tenors.

### Data Used

| Source | # Points | Tenors/Maturities |
|--------|:---:|---|
| **VIX Index TS** | 6 | 1d, 9d, 30d, 90d, 180d, 365d |
| **VIX Futures (CBOE)** | 6 | 11d, 42d, 103d, 164d, 195d, 225d |
| **SPX/SPY Options Surface** | 96 | 12 maturities × 8 strikes (5d → 179d DTE) |
| **VVIX** | 1 | Vol-of-vol prior for η |
| **SOFR** | 1 | Risk-free rate = 3.73% |

### Results

| Component | Loss | Assessment |
|-----------|:---:|---|
| **Smile (SPX)** | 0.062 | Vega-normalized, 12 maturities |
| **VIX TS** | 0.014 | 12 combined tenors |
| **Martingale** | < 1e-6 | $E^Q[e^{-rT}S_T] = S_0$ |
| **Calendar** | < 1e-6 | Total var monotone |
| **Novikov** | 0.060 | MLP active, bounded |
| **Total** | **0.097** | — |

### VIX Term Structure Fit (Q-model)

| Tenor | Market | Model | Δ |
|:---:|:---:|:---:|:---:|
| 1d | 17.10 | 22.76 | +5.66 |
| 9d | 23.77 | 22.81 | -0.96 |
| 30d | 22.66 | 22.90 | **+0.24** |
| 90d | 23.24 | 23.04 | **-0.20** |
| 180d | 24.70 | 23.20 | -1.50 |
| 225d (futures) | 22.60 | 23.41 | +0.81 |

30d–90d fit within ±0.3 pts. The 1d gap reflects the VIX1D intraday regime (very short-dated). The Girsanov MLP has only **1,185 trainable parameters** and trains in **94 seconds** on CPU.

### Diagnostic Plots

| Plot | File |
|---|---|
| Training loss curve | `outputs/plots/neural_q_loss_curve.png` |
| VIX TS fit | `outputs/plots/neural_q_vix_fit.png` |
| Loss decomposition | `outputs/plots/neural_q_loss_components.png` |

```bash
python bin/calibration/neural_q_calibration.py --quick                # Quick: 50 epochs, 2048 paths
python bin/calibration/neural_q_calibration.py --epochs 300 --paths 4096  # Full calibration
python bin/calibration/neural_q_calibration.py --H 0.03 --rho -0.95  # Custom seed params
```

---

## Multi-Scale Hurst Estimation

> **Key result**: $H_{\text{consensus}} = 0.110 \pm 0.003$ — firmly in the rough volatility regime. Universal across assets (SPX, SPY, CAC40) and monofractal.

### Mathematical Framework

#### The Rough Volatility Hypothesis

Gatheral, Jaisson & Rosenbaum (2018) discovered empirically that log-realized-volatility behaves like fractional Brownian motion with $H \approx 0.1$:

$$X_t = \log \sigma_t \approx X_0 + \eta\, B^H_t$$

A fractional Brownian motion $B^H$ with Hurst parameter $H \in (0,1)$ has covariance $\text{Cov}(B^H_t, B^H_s) = \tfrac{1}{2}(|t|^{2H} + |s|^{2H} - |t-s|^{2H})$ and satisfies the self-similarity property $(B^H_{ct}) \overset{d}{=} c^H (B^H_t)$. For $H < 1/2$, increments are **negatively correlated** — creating the jagged, anti-persistent paths observed in real volatility. By the Kolmogorov–Čentsov theorem, sample paths have Hölder regularity exactly $H$ almost surely (Arcones 1995).

For $H = 0.11$, paths are vastly rougher than Brownian motion ($H = 0.5$): they have infinite $p$-variation for $p < 1/H \approx 9$.

#### Variogram Estimator

The primary estimator exploits fBM scaling $E[|X_{t+\tau} - X_t|^2] = C\,\tau^{2H}$. The empirical variogram:

$$m(2, \tau) = \frac{1}{N-\tau}\sum_{t=1}^{N-\tau}(X_{t+\tau} - X_t)^2 \;\xrightarrow{\;p\;}\; C\tau^{2H}$$

by the ergodic theorem, so $\log m(2,\tau) \to 2H\log\tau + \log C$. The OLS slope divided by 2 gives a consistent estimator $\hat{H}_{\text{var}}$ (Gatheral et al. 2018 §3). We observe $R^2 > 0.93$ at all frequencies.

#### Structure Function and Monofractality

The generalized structure function $m(q,\tau) = \frac{1}{N-\tau}\sum_t |X_{t+\tau} - X_t|^q$ converges to $c_q \cdot \tau^{\zeta(q)}$. For a **monofractal** process (single $H$), $\zeta(q) = qH$ is linear in $q$. If $\zeta(q)$ is concave, multiple exponents coexist (multifractality). We test by fitting $\zeta(q) = a_1 q + a_2 q^2$ for $q \in \{0.5, 1, 1.5, 2, 3, 4\}$:

Our results: $R^2_{\text{linear}} > 0.998$ and curvature $|a_2| < 0.006$ at all frequencies → **single $H$ confirmed** → validates using one Hurst parameter in the rBergomi backbone. (Ref: Frisch 1995, Gatheral et al. 2018 §4.)

#### Ratio Estimator

A regression-free alternative (Istas & Lang 1997): $\hat{H}(\tau) = \frac{1}{2}\log_2\big(m(2,2\tau)/m(2,\tau)\big)$. If $m(2,\tau) = C\tau^{2H}$, then $m(2,2\tau)/m(2,\tau) = 2^{2H}$ and $\hat{H}(\tau) = H$ exactly. Each lag gives a local estimate, then averaged. Higher variance but no regression assumptions.

#### Microstructure Correction (TSRV)

At ultra-high frequency, observed prices $Y_{t_i} = X_{t_i} + \varepsilon_{t_i}$ are contaminated by microstructure noise. Naive RV diverges: $E[RV^{(n)}] = \int_0^T \sigma^2_t\,dt + 2n\sigma^2_\varepsilon$. The Two-Scale RV (Zhang, Mykland & Aït-Sahalia 2005) eliminates this bias:

$$\hat{\text{TSRV}} = RV^{(K)} - \tfrac{K}{n}\,RV^{(n)}$$

with $K \asymp n^{2/3}$ (optimal subsampling rate). Applied for $\Delta \leq 1$min; for $\Delta \geq 5$min noise is negligible.

#### Consensus Aggregation

Given $K$ estimators $\hat{H}_k$ with bootstrap variances $\sigma^2_k$, the inverse-variance weighted average $\hat{H}_w = \sum_k \hat{H}_k/\sigma^2_k \big/ \sum_k 1/\sigma^2_k$ is BLUE (minimum variance among linear unbiased combinations, by the Gauss-Markov theorem). Block bootstrap with $\ell = \lceil n^{1/3} \rceil$ preserves temporal dependence (Politis & Romano 1994).

#### Why VIX Shows $H \approx 0.5$

VIX measures expected integrated variance over 30 days. For a moving average $\bar{X}_t^{(\Delta)} = \frac{1}{\Delta}\int_t^{t+\Delta} X_s\,ds$ of an fBM($H$) process:
- Lags $\tau \gg \Delta$: roughness preserved, variogram $\sim \tau^{2H}$
- Lags $\tau \ll \Delta$: appears Lipschitz, variogram $\sim \tau^2$ ($H_{\text{eff}} \to 1$)

Our measurements confirm: VIX 15-min → $H \approx 0.47$, VIX 30-min → $H \approx 0.45$. Roughness must be estimated from **realized vol** (SPX intraday returns), not VIX. (Ref: Bennedsen et al. 2016.)

### Results — SPX Multi-Scale

| Frequency | Days | $H_{\text{var}}$ | $H_{\text{struct}}$ | $H_{\text{ratio}}$ | $R^2$ |
|:---------:|:----:|:---:|:---:|:---:|:---:|
| **5m** | 510 | 0.119 | 0.114 | 0.076 | 0.933 |
| **15m** | 1,509 | 0.101 | 0.099 | 0.092 | 0.969 |
| **30m** | 2,974 | 0.117 | 0.114 | 0.098 | 0.992 |
| **1h** | 2,974 | 0.094 | 0.091 | 0.082 | 0.991 |
| **daily** | 1,821 | 0.087 | 0.087 | 0.091 | 0.978 |

$$\boxed{H_{\text{consensus}} = 0.110 \pm 0.003 \quad \text{(inverse-variance weighted, 95\% CI)}}$$

### Cross-Asset Universality

| Asset | Consensus $H$ | 95% CI |
|:-----:|:---:|:---:|
| **S&P 500** | 0.110 ± 0.003 | [0.105, 0.115] |
| **SPY (ETF)** | 0.111 ± 0.009 | [0.093, 0.129] |
| **CAC 40** | 0.090 ± 0.011 | [0.068, 0.112] |

### Diagnostic Plots

| Plot | File |
|---|---|
| Variogram log-log | `outputs/plots/hurst_variogram_loglog.png` |
| Hurst across scales | `outputs/plots/hurst_across_scales.png` |
| Cross-asset comparison | `outputs/plots/hurst_cross_asset.png` |
| Multifractal spectrum | `outputs/plots/hurst_multifractal_spectrum.png` |
| Bootstrap distributions | `outputs/plots/hurst_bootstrap_distributions.png` |
| Log(RV) time series | `outputs/plots/hurst_logrv_timeseries.png` |

```bash
python bin/analysis/hurst_multiscale.py                    # Full analysis (~3 min)
python bin/analysis/hurst_multiscale.py --cross-asset      # + SPY & CAC40
python bin/analysis/hurst_multiscale.py --update-config    # Write consensus H to params.yaml
```

### Proving Roughness: ACF / Variogram Evidence

| Metric | Real Data | Neural SDE | Pure OU |
|---|:---:|:---:|:---:|
| Path signature correlation | 1.000 | **0.9996** | — |
| Hurst (VIX paths) | 0.475 | 0.475 | 0.465 |
| MMD (distribution distance) | — | **0.0280** | 0.0296 |

The Neural SDE beats the OU baseline on MMD and perfectly matches the signature distribution — generated paths are statistically indistinguishable from real VIX variance paths.

---

## Experimental Results

All results are reproducible via `python bin/experiments/paper_results.py` which generates 13 publication-quality figures and 4 JSON data files in `outputs/paper_results/`.

### VRP Backtest — Regime-Adaptive Volatility Trading

| Strategy | Sharpe | Return (bps) | MaxDD (bps) | Win Rate |
|:---:|:---:|:---:|:---:|:---:|
| always_sell | 0.77 | 4,025 | 2,989 | — |
| bergomi | 0.76 | 3,983 | 2,989 | — |
| neural_sde | 0.71 | 3,714 | 2,989 | — |
| **regime_bergomi** | **0.88** | **4,622** | **1,527** | — |
| regime_neural | 0.80 | 4,198 | 1,650 | — |

The regime-adaptive rBergomi strategy improves Sharpe by **+14%** and cuts maximum drawdown by **-49%** vs the static baseline, confirming that regime conditioning adds genuine economic value.

### Exotic Pricing — Neural SDE Q-Model vs Benchmarks

| Product | Black-Scholes | rBergomi | Neural SDE Q | Spread (NSDE-Berg) |
|:---:|:---:|:---:|:---:|:---:|
| Asian (arithmetic) | $2.81 | $2.22 | $2.41 | +8.6% |
| Cliquet | $5.05 | $3.88 | $4.24 | +9.3% |
| Lookback (fixed) | $8.83 | $6.72 | $7.26 | +8.0% |
| Variance swap | $1.83 | $1.46 | $1.58 | +8.2% |
| Barrier DOC | $1.61 | $1.32 | $1.41 | +6.8% |
| Autocallable | $7.40 | $7.17 | $7.25 | +1.1% |

Neural SDE prices sit between BS and rBergomi, with a consistent +6–9% premium over Bergomi reflecting learned higher-order dynamics (skewness, kurtosis).

### Stress Test — Tail Risk Comparison

| Scenario | Metric | Neural SDE | rBergomi | Ratio |
|:---:|:---:|:---:|:---:|:---:|
| Panic (σ×3) | Kurtosis | **6.17** | 1.13 | 5.5× |
| Panic (σ×3) | Skew | **-2.00** | -0.75 | 2.7× |
| Crash (μ-5σ) | P(loss > 3σ) | **4.2%** | 0.3% | 14× |
| Vol explosion | CVaR 99% | **-28.1%** | -12.4% | 2.3× |

The Neural SDE produces **5× fatter tails** than rBergomi under extreme stress, capturing non-Gaussian dynamics that Gaussian-kernel rough models miss. This matters for tail-risk hedging and regulatory capital.

### Regime Timeline (2007–2026)

Full 18-year regime classification from 7 TradingView signals:

| Regime | Days | Share | Example Periods |
|:---:|:---:|:---:|---|
| Normal | 2,369 | 53% | 2013–2015, 2017, 2019 |
| Calm | 983 | 22% | 2005–2006, late 2017 |
| Stressed | 894 | 20% | 2008, 2011, 2015Q3, 2022 |
| Crisis | 224 | 5% | Sep 2008, Mar 2020, Aug 2024 |

### Literature Comparison

| Parameter | This Work | Literature | Reference |
|:---:|:---:|:---:|---|
| $H_P$ (P-measure) | $0.110 \pm 0.003$ | $0.10 \pm 0.01$ | Gatheral, Jaisson & Rosenbaum (2018) |
| $\eta$ (vol-of-vol) | $1.341$ | $1.9$ | Bayer, Friz & Gatheral (2016) |
| $\rho$ (spot-vol corr) | $-0.959$ | $-0.7$ to $-0.9$ | Bayer, Friz & Gatheral (2016) |
| $H_Q$ (Q-measure) | $0.005$ ⚠ | $0.03$–$0.08$ | Rømer (2022) |

> **Note on $H_Q$**: The joint calibration grid has lower bound $H = 0.005$. The optimizer hitting this boundary suggests the Q-measure Hurst exponent may be even rougher than the grid allows, or that $\eta$ and $\rho$ absorb some of the roughness signal. Widening the grid or using a continuous optimizer is left for future work.

---

## Architecture Overview

```
                    ┌──────────────────────────────────────────┐
                    │           DeepRoughVol Engine             │
                    └──────────────────────────────────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
     ┌────────▼────────┐    ┌─────────▼─────────┐   ┌─────────▼─────────┐
     │  P-Measure Model │    │  Q-Measure Model  │   │  Q + Jump Model   │
     │  (Physical)      │    │  (Risk-Neutral)   │   │  (Crisis)         │
     └────────┬────────┘    └─────────┬─────────┘   └─────────┬─────────┘
              │                        │                        │
     ┌────────▼────────────────────────▼────────────────────────▼─────────┐
     │                        Shared Infrastructure                       │
     ├────────────────────────────────────────────────────────────────────┤
     │  SOFR Rates  │  VVIX Calibration  │  Regime Detection  │  API    │
     └────────────────────────────────────────────────────────────────────┘
              │                        │                        │
     ┌────────▼────────┐    ┌─────────▼─────────┐   ┌─────────▼─────────┐
     │  Risk Engine     │    │  Exotic Pricer    │   │  Hedging Sim      │
     │  VaR/CVaR/Stress │    │  Asian/Auto/Var   │   │  BS/Bartlett      │
     └─────────────────┘    └───────────────────┘   └───────────────────┘
```

---

## Multi-Measure Framework

| Measure | Training Data | Loss Components | Model File | Use Cases |
|:---:|---|---|---|---|
| **P** (physical) | Realized vol (SPX 5-min) | MMD + mean penalty | `neural_sde_best_p.eqx` | VaR, CVaR, stress testing |
| **Q** (risk-neutral) | VIX (implied vol) | MMD + mean + **martingale** | `neural_sde_best_q.eqx` | Option pricing, hedging |
| **Q + jumps** | VIX + Merton jumps | MMD + mean + martingale + **jump reg** | `neural_sde_best_q_jump.eqx` | Crisis pricing, tail risk |

---

## Modules

### Pricing — `quant/exotic_pricer.py`

| Product | Key Sensitivity |
|---|---|
| **Asian** (arithmetic/geometric) | Vol term structure, autocorrelation |
| **Lookback** (fixed/floating) | Vol level, path continuity |
| **Barrier** (DOC, DIC, UOP) | Skew, tail risk |
| **Autocallable** | Forward vol, correlation |
| **Cliquet** | Forward skew, vol-of-vol |
| **Variance / Volatility swap** | Realized vol dynamics |

### Risk — `quant/risk_engine.py`

VaR (95%, 99%), CVaR / ES, Parametric VaR, Stressed VaR, Component VaR, Tail Index (Hill), Stress Testing (Black Monday, Lehman, COVID, Flash Crash, Volmageddon, rate shock).

### Hedging — `quant/hedging_simulator.py`

| Strategy | Delta Formula |
|---|---|
| **Black-Scholes** | $\Delta^{BS}(\sigma_{ATM})$ |
| **Bartlett** | $\Delta^{BS} + \text{Vanna} \cdot d\sigma/dS$ |
| **Sticky-Strike** | $\Delta^{BS}(\sigma_t^{local})$ |

### P&L Attribution — `quant/pnl_attribution.py`

Second-order Taylor decomposition: $\Delta C \approx \Delta \cdot \delta S + \tfrac{1}{2}\Gamma(\delta S)^2 + \nu\,\delta\sigma + \Theta\,\delta t + \text{cross terms}$

Also: `NeuralSDEGreeks` — model-implied Δ, Γ, Vega via **JAX autodiff through the full MC pipeline**.

### Regime Detection — `quant/regimes/regime_detector.py`

7 weighted signals from TradingView market data:

| Signal | Weight | Calm | Normal | Stressed | Crisis |
|---|:---:|:---:|:---:|:---:|:---:|
| VIX level | 0.25 | <13 | <20 | <30 | ≥30 |
| VVIX | 0.15 | <80 | <100 | <130 | ≥130 |
| VIX term slope | 0.15 | >0.05 | >0 | >-0.05 | ≤-0.05 |
| VRP (IV-RV) | 0.10 | <2 | <5 | <10 | ≥10 |
| VIX 1Y percentile | 0.10 | <25 | <50 | <75 | ≥75 |
| VIX1D/VIX ratio | 0.15 | <0.9 | <1.1 | <1.3 | ≥1.3 |
| SKEW index | 0.10 | >135 | >125 | >115 | ≤115 |

Regime → adaptive rBergomi $(H, \eta, \rho)$ mapping via `config/params.yaml`.

**Historical regime distribution** (2007–2026, 4,470 trading days):
Normal 53% · Calm 22% · Stressed 20% · Crisis 5%

---

## Methodology

### Neural SDE with Path Signatures

Two backbone architectures (`neural_sde.backbone` in config):

**OU backbone** (default — fast):
$$dX_t = \kappa(\theta - X_t)\,dt + f_\theta(\mathbb{S}_{0,t}, X_t)\,dt + g_\theta(\mathbb{S}_{0,t}, X_t)\,dW_t + J\,dN_t$$

**Fractional backbone** (nests rBergomi):
$$X_t = \eta\hat{W}^H_t - \tfrac{1}{2}\eta^2\text{Var}[\hat{W}^H_t] + \int_0^t f_\theta\,ds + \int_0^t g_\theta\,dW_s$$

where $\hat{W}^H_t = \sqrt{2H}\int_0^t (t-s)^{H-1/2}dW_s$ is RL-fBM with **learnable** $(H, \eta)$, and $\mathbb{S}_{0,t}$ is the running path signature of $(t, \log V)$ up to order $M \in \{2,3,4\}$ (exact Chen's identity).

### Training

- **Distribution matching**: Kernel MMD² with multi-scale RBF (median heuristic)
- **Mean correction**: In **log-variance space** ($E[\log V]$ matching, avoiding Jensen bias)
- **Martingale** (Q only): $E^Q[e^{-rT}S_T] = S_0$ with real SOFR rate
- **Smile fit** (Q pricing): Vega-weighted IV smile loss from cached SPY options
- **Optimization**: Adam + gradient clipping, linear warmup + cosine decay, deterministic early stopping

### Rough Bergomi Benchmark

Volterra kernel implementation (Bayer, Friz & Gatheral 2016) with exact spot-vol correlation, previsible variance Euler scheme, and per-fold walk-forward recalibration.

---

## Data Sources

| Dataset | Source | Format | Volume | Used By |
|---------|--------|--------|--------|---------|
| **VIX Term Structure** | CBOE / TradingView | VIX1D, VIX9D, VIX, VIX3M, VIX6M, VIX1Y | 6 tenors, 5m–daily | rBergomi + Neural Q |
| **VIX Futures** | CBOE | 6 contracts (front month → 3M) | 25,618 rows (2013–2026) | Neural Q |
| **SPX / SPY** | TradingView | 5s, 5m, 15m, 30m, 1h, daily | ~40k rows per freq | Hurst + calibration |
| **SPY Options Surface** | Yahoo Finance | Cached snapshots (5 dates) | 2,345 options × 20 mats | rBergomi + Neural Q |
| **VVIX** | CBOE / TradingView | 5m–daily | ~40k rows | η prior |
| **SOFR Rate** | NY Fed | Daily | 1,968 rows (2018–2026) | Risk-free rate |
| **VIX Futures (TradingView)** | TradingView | VX1!, VX2! (5m–daily) | Continuous contracts | Regime detection |
| **SKEW Index** | TradingView | Daily | 9,044 rows | Future: smile prior |
| **Put/Call Ratio** | TradingView | Daily | 4,858 rows | Future: sentiment |
| **Implied Correlation** | TradingView | COR1M (5m–daily) | ~40k rows | Future: dispersion |
| **US Rates** | TradingView | 2Y, 5Y, 10Y, 30Y | Multiple freq | Future: macro regime |
| **CAC 40** | TradingView | 5m, 15m, 30m | ~40k rows | Cross-asset Hurst |

---

## Usage

### Prerequisites

- **Python 3.12+** · Windows/Linux/macOS · ~4 GB disk

### Step 0 — Install

```bash
git clone <repo-url>
cd "Projet IA & quant"
pip install -r requirements.txt
```

### Step 1 — Download Market Data

```bash
python bin/data/regenerate_data.py --mode yahoo      # Yahoo Finance + CBOE
# Or: --mode tradingview if you have TradingView CSV exports in data/
```

### Step 2 — Fetch SPY Options Surface

```bash
python bin/data/fetch_options.py                # Cached in data/options_cache/
```

### Step 3 — Calibrate Parameters

```bash
python bin/calibration/calibrate.py             # -> outputs/advanced_calibration.json
python bin/analysis/hurst_multiscale.py --update-config  # Multi-scale H estimation -> params.yaml
```

### Step 4 — Train Neural SDE

```bash
python bin/training/train_multi.py              # P + Q models (~15 min total)
python bin/training/train_multi.py --measure Q --jumps   # Optional: crisis model
```

| Flag | Model File | Use For |
|---|---|---|
| (default) | `neural_sde_best_p.eqx` + `_q.eqx` | Full coverage |
| `--measure P` | `neural_sde_best_p.eqx` | VaR, stress testing |
| `--measure Q` | `neural_sde_best_q.eqx` | Pricing, hedging |
| `--measure Q --jumps` | `neural_sde_best_q_jump.eqx` | Crisis pricing |

### Step 5 — Backtest & Validate

```bash
python bin/backtesting/backtest.py              # IV RMSE: Neural SDE vs Bergomi vs BS
python bin/analysis/verify_roughness.py         # Roughness proof + signature + ablation
python bin/backtesting/walk_forward.py          # Out-of-sample temporal backtest
```

### Step 6 — Joint SPX-VIX Calibration

```bash
python bin/calibration/joint_calibration.py --quick         # ~27s, Q-measure rBergomi calibration
python bin/calibration/joint_calibration.py                 # Full grid (~90s)
python bin/calibration/neural_q_calibration.py --quick      # Neural SDE Girsanov Q-calibration
```

### Step 7 — Reports & Dashboard

```bash
python bin/backtesting/model_suite.py --run-usecases   # VaR, stress, exotic pricing, regime
python bin/apps/dashboard.py                    # -> outputs/dashboard.html
python bin/apps/api_server.py                   # -> http://localhost:8000 (UI + API)
```

### Step 8 — Publication Results

```bash
python bin/experiments/paper_results.py         # -> outputs/paper_results/ (13 figures + 4 JSONs)
python bin/experiments/exotic_comparison.py      # Standalone exotic pricing comparison
python bin/experiments/stress_test_comparison.py # Standalone stress test comparison
python bin/backtesting/pnl_backtest.py          # VRP backtest with regime strategies
```

### Step 9 — Diagnostics (optional)

```bash
python bin/analysis/hurst_diagnostic.py         # VIX vs RV Hurst comparison
python bin/analysis/compare_vix_vs_rv.py        # Side-by-side roughness analysis
python bin/analysis/robustness_check.py         # Full robustness: MMD, Hurst, smiles
python main.py                                  # Legacy all-in-one demo
```

### Output Files

| Output | Script | Contents |
|---|---|---|
| `outputs/backtest_results.json` | `backtest.py` | RMSE per model/maturity |
| `outputs/model_usecases_report.json` | `model_suite.py` | VaR, CVaR, stress, regime |
| `outputs/advanced_calibration.json` | `calibrate.py` | H, η, ξ₀, ρ estimates |
| `outputs/roughness_verification.json` | `verify_roughness.py` | Roughness proof |
| `outputs/hurst_multiscale_report.json` | `hurst_multiscale.py` | Multi-scale Hurst |
| `outputs/joint_calibration.json` | `joint_calibration.py` | Joint SPX-VIX Q-calibration |
| `outputs/pnl_backtest.json` | `pnl_backtest.py` | VRP strategy Sharpe/DD |
| `outputs/dashboard.html` | `dashboard.py` | Interactive dashboard |
| `models/neural_sde_best_*.eqx` | `train_multi.py` | Trained models |
| **`outputs/paper_results/`** | **`paper_results.py`** | **Publication data & figures** |
| `  ├── exotic_comparison.json` | | BS vs Berg vs NSDE prices |
| `  ├── stress_test_comparison.json` | | Tail metrics per scenario |
| `  ├── regime_timeline.json` | | 18-year daily regime labels |
| `  ├── calibration_summary.json` | | P & Q parameters + literature |
| `  └── *.png (×13)` | | Publication-ready figures |

The JSON files in `outputs/paper_results/` are structured data exports designed for inclusion in a future research paper or report. Each contains full numerical results with metadata (dates, parameters, model versions) for reproducibility.

---

## API Reference

Launch: `python bin/apps/api_server.py` → Swagger at `http://localhost:8000/docs`

| Endpoint | Method | Description |
|---|:---:|---|
| `/health` | GET | Health check |
| `/regime` | GET | Market regime + signals |
| `/rates/sofr` | GET | Current SOFR rate |
| `/price/vanilla` | POST | BS price + Greeks |
| `/price/exotic` | POST | MC exotic pricing |
| `/risk/var` | POST | Portfolio VaR/CVaR |
| `/risk/stress` | POST | Stress test scenarios |
| `/pnl/attribute` | POST | Greeks P&L decomposition |
| `/hedge/simulate` | POST | Delta hedging comparison |
| `/calibrate/eta` | POST | VVIX-based η calibration |

---

## File Structure

Top-level overview (run `tree /F` for full listing):

```
DeepRoughVol/
├── main.py                    # Full demo pipeline
├── config/params.yaml         # Central config (200+ keys)
├── bin/                       # CLI entry points (domain-first)
│   ├── calibration/           # joint_calibration, neural_q, options, RN
│   ├── analysis/              # hurst_multiscale, verify_roughness, diagnostics
│   ├── training/              # train.py, train_multi.py (P/Q)
│   ├── backtesting/           # backtest, walk_forward, pnl_backtest, model_suite
│   ├── experiments/           # paper_results, exotic_comparison, stress_test
│   ├── apps/                  # api_server, dashboard
│   ├── data/                  # regenerate_data, fetch_options, refresh_all_data
│   └── ops/                   # compare_frequencies
├── engine/                    # ML engine (Neural SDE, signatures, losses, trainer)
├── quant/                     # Quant library
│   ├── models/                # rBergomi, fBM (canonical stochastic models)
│   ├── calibration/           # Calibration engines and data targets
│   ├── workflows/             # Backtesting and walk-forward workflows
│   ├── pricers/               # Vanilla and exotic pricing engines
│   ├── hedging/               # Hedging simulator and P&L attribution
│   ├── risk/                  # Risk engine (VaR, CVaR, stress testing)
│   ├── regimes/               # 7-signal regime detector (TradingView)
│   ├── analysis/              # Hurst and diagnostics analysis
│   └── data/                  # Quant-side data helpers
├── utils/                     # SOFR, VVIX calibration, BS, Greeks AD, data pipeline
├── data/                      # Market data (~115 CSVs, options cache, SOFR)
│   ├── trading_view/          # 33+ files: volatility/, equity_indices/, rates/, etc.
│   ├── market/                # Yahoo-sourced VIX, SPX, VVIX
│   ├── cboe_vix_futures_full/ # 25k+ rows of VIX futures
│   └── options_cache/         # 5 SPY surface snapshots (2,345 options)
├── models/                    # Trained models (.eqx) + Q-config
├── outputs/                   # Results, reports, plots, dashboard
│   ├── paper_results/         # 13 figures + 4 JSONs for publication
│   ├── plots/                 # Hurst, calibration, backtest plots
│   └── model_suite/           # Per-use-case detailed reports
└── research/                  # LaTeX proofs (maths_proofs.tex, proof.tex)
```

---

## References

### Academic

1. Gatheral, Jaisson & Rosenbaum (2018). *Volatility is Rough*. Quantitative Finance.
2. Bayer, Friz & Gatheral (2016). *Pricing under rough volatility*. Quantitative Finance.
3. Kidger, Foster, Li & Lyons (2021). *Neural SDEs as Infinite-Dimensional GANs*. ICML.
4. Lyons (1998). *Differential equations driven by rough paths*. Rev. Mat. Iberoamericana.
5. Merton (1976). *Option pricing when underlying stock returns are discontinuous*. JFE.
6. Gretton et al. (2012). *A Kernel Two-Sample Test*. JMLR.
7. Chevyrev & Kormilitzin (2016). *A primer on the signature method in ML*.
8. Bayer & Stemper (2018). *Deep calibration of rough stochastic volatility models*.
9. Jaeckel (2017). *Let's be rational*. Wilmott.
10. Bennedsen, Lunde & Pakkanen (2017). *Hybrid scheme for BSS processes*. Finance and Stochastics.
11. Zhang, Mykland & Aït-Sahalia (2005). *A Tale of Two Time Scales*. JASA.
12. Bennedsen, Lunde & Pakkanen (2016). *Decoupling the short- and long-term behavior of stochastic volatility*.
13. Politis & Romano (1994). *The Stationary Bootstrap*. JASA.
14. Fukasawa (2021). *Volatility has to be rough*. Quantitative Finance.
15. Guyon (2019). *The Joint S&P 500/VIX Smile Calibration Puzzle Solved*. Risk.
16. Rømer (2022). *Empirical analysis of rough and classical stochastic volatility models*. Quantitative Finance.
17. Jacquier, Martini & Muguruza (2018). *On VIX Futures in the Rough Bergomi Model*. SIAM J. Financial Mathematics.
18. Istas & Lang (1997). *Quadratic variations and estimation of the local Hölder index of a Gaussian process*. Annales de l'I.H.P.
19. Gierjatowicz, Sabate-Vidales, Šiška, Szpruch & Žurič (2022). *Robust pricing and hedging via neural SDEs*. JCAM.
20. Buehler, Gonon, Teichmann, Wood & Mohan (2021). *Deep Hedging: Hedging Derivatives Under Generic Market Frictions Using Reinforcement Learning*. Swiss Finance Institute.
21. Girsanov (1960). *On transforming a certain class of stochastic processes by absolutely continuous substitution of measures*. Theory of Probability.

### Technical

- [JAX](https://jax.readthedocs.io/) · [Equinox](https://docs.kidger.site/equinox/) · [esig](https://esig.readthedocs.io/) · [FastAPI](https://fastapi.tiangolo.com/)

---

## License

MIT License — see [LICENSE](LICENSE).

---

*Last updated: June 2025 — v4.0 (Regime-adaptive VRP backtest, exotic Neural SDE Q-pricing, stress test comparison, paper_results pipeline, 7-signal regime detector with TradingView data)*
