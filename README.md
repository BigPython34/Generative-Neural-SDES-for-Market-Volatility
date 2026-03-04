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
4. [Multi-Scale Hurst Estimation](#multi-scale-hurst-estimation)
5. [Architecture Overview](#architecture-overview)
6. [Multi-Measure Framework](#multi-measure-framework)
7. [Modules](#modules)
8. [Methodology](#methodology)
9. [Data Sources](#data-sources)
10. [Usage](#usage)
11. [API Reference](#api-reference)
12. [File Structure](#file-structure)
13. [References](#references)

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
| **Hurst (Q-measure)** | $H_Q$ | **0.020** | Joint SPX-VIX calibration |
| **Vol-of-Vol (Q-cal)** | $\eta_Q$ | **0.959** | Joint SPX-VIX calibration |
| **Correlation (Q-cal)** | $\rho_Q$ | **−0.955** | Joint SPX-VIX calibration |
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
| 9d | 23.77 | 22.13 | −1.64 |
| 30d | 22.66 | 22.12 | −0.54 |
| 90d | 23.24 | 22.93 | −0.31 |
| 180d | 24.70 | 24.43 | −0.27 |
| 365d | 24.86 | 24.61 | −0.25 |

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

## Multi-Scale Hurst Estimation

> **Key result**: $H_{\text{consensus}} = 0.110 \pm 0.003$ — firmly in the rough volatility regime. Universal across assets (SPX, SPY, CAC40) and monofractal.

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

### Estimation Methods

Four independent estimators, each grounded in standard results:

| Estimator | Reference | Method |
|---|---|---|
| **Variogram** | Gatheral et al. 2018 §3 | OLS slope of $\log m(2,\tau)$ vs $\log\tau$, divided by 2 |
| **Structure function** | Frisch 1995, Gatheral et al. 2018 §4 | Generalized $m(q,\tau) \propto \tau^{qH}$; monofractality test via $\zeta(q) = qH$ linearity |
| **Ratio** | Istas & Lang 1997 | $\hat{H}(\tau) = \frac{1}{2}\log_2\big(m(2,2\tau)/m(2,\tau)\big)$ — regression-free, local |
| **TSRV correction** | Zhang, Mykland & Aït-Sahalia 2005 | Two-Scale RV for microstructure debiasing at $\Delta \leq 1$min |

Consensus via **inverse-variance weighting** (BLUE — Gauss-Markov theorem). Block bootstrap CI with $\ell = \lceil n^{1/3} \rceil$ (Politis & Romano 1994).

### Why VIX Shows $H \approx 0.5$

VIX is a 30-day moving average of variance. Integration smooths roughness: at lags $\tau \ll 30$d, the smoothed process appears Lipschitz ($H_{\text{eff}} \to 1$). Our measurements confirm: VIX 15-min → $H \approx 0.47$, VIX 30-min → $H \approx 0.45$. Roughness must be estimated from **realized vol** (SPX intraday returns), not VIX. (Ref: Bennedsen et al. 2016.)

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
python bin/hurst_multiscale.py                    # Full analysis (~3 min)
python bin/hurst_multiscale.py --cross-asset      # + SPY & CAC40
python bin/hurst_multiscale.py --update-config    # Write consensus H to params.yaml
```

### Proving Roughness: ACF / Variogram Evidence

| Metric | Real Data | Neural SDE | Pure OU |
|---|:---:|:---:|:---:|
| Path signature correlation | 1.000 | **0.9996** | — |
| Hurst (VIX paths) | 0.475 | 0.475 | 0.465 |
| MMD (distribution distance) | — | **0.0280** | 0.0296 |

The Neural SDE beats the OU baseline on MMD and perfectly matches the signature distribution — generated paths are statistically indistinguishable from real VIX variance paths.

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

### Regime Detection — `quant/regime_detector.py`

5 weighted signals (VIX 30%, VVIX 20%, term structure 20%, VRP 15%, VIX percentile 15%) → consensus regime → adaptive $(H, \eta, \rho)$.

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

| Dataset | Source | Frequency | Period |
|---------|--------|-----------|--------|
| VIX Spot | TradingView / Yahoo | 5/10/15/30-min | 2023–2026 |
| SPX | TradingView / Yahoo | 5/30-min | 2023–2026 |
| SPX Daily | Yahoo Finance | Daily | 2010–2026 |
| VVIX | Yahoo Finance | Daily | 2013–2026 |
| VIX Futures | CBOE | Daily | 2013–2026 |
| SOFR Rate | NY Fed | Daily | 2018–2026 |
| SPY Options | Yahoo Finance | Snapshots | Latest cache |

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
python bin/regenerate_data.py --mode yahoo      # Yahoo Finance + CBOE
# Or: --mode tradingview if you have TradingView CSV exports in data/
```

### Step 2 — Fetch SPY Options Surface

```bash
python bin/fetch_options.py                     # Cached in data/options_cache/
```

### Step 3 — Calibrate Parameters

```bash
python bin/calibrate.py                         # → outputs/advanced_calibration.json
python bin/hurst_multiscale.py --update-config  # Multi-scale H estimation → params.yaml
```

### Step 4 — Train Neural SDE

```bash
python bin/train_multi.py                       # P + Q models (~15 min total)
python bin/train_multi.py --measure Q --jumps   # Optional: crisis model
```

| Flag | Model File | Use For |
|---|---|---|
| (default) | `neural_sde_best_p.eqx` + `_q.eqx` | Full coverage |
| `--measure P` | `neural_sde_best_p.eqx` | VaR, stress testing |
| `--measure Q` | `neural_sde_best_q.eqx` | Pricing, hedging |
| `--measure Q --jumps` | `neural_sde_best_q_jump.eqx` | Crisis pricing |

### Step 5 — Backtest & Validate

```bash
python bin/backtest.py                          # IV RMSE: Neural SDE vs Bergomi vs BS
python bin/verify_roughness.py                  # Roughness proof + signature + ablation
python bin/walk_forward.py                      # Out-of-sample temporal backtest
```

### Step 6 — Joint SPX-VIX Calibration

```bash
python bin/joint_calibration.py --quick         # ~27s, Q-measure rBergomi calibration
python bin/joint_calibration.py                 # Full grid (~90s)
```

### Step 7 — Reports & Dashboard

```bash
python bin/model_suite.py --run-usecases        # VaR, stress, exotic pricing, regime
python bin/dashboard.py                         # → outputs/dashboard.html
python bin/api_server.py                        # → http://localhost:8000 (UI + API)
```

### Step 8 — Diagnostics (optional)

```bash
python bin/hurst_diagnostic.py                  # VIX vs RV Hurst comparison
python bin/compare_vix_vs_rv.py                 # Side-by-side roughness analysis
python bin/robustness_check.py                  # Full robustness: MMD, Hurst, smiles
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
| `outputs/dashboard.html` | `dashboard.py` | Interactive dashboard |
| `models/neural_sde_best_*.eqx` | `train_multi.py` | Trained models |

---

## API Reference

Launch: `python bin/api_server.py` → Swagger at `http://localhost:8000/docs`

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
├── bin/                       # CLI entry points (train, backtest, calibrate, API, ...)
├── core/                      # Stochastic models (rBergomi, fBM)
├── engine/                    # ML engine (Neural SDE, signatures, losses, trainer)
├── quant/                     # Quant library (pricing, risk, hedging, P&L, regime)
│   └── calibration/           # Joint SPX-VIX calibrator, forward variance, VIX loader
├── utils/                     # SOFR, VVIX calibration, BS, Greeks AD, data pipeline
├── data/                      # Market data (VIX, SPX, VVIX, SOFR, options, futures)
├── models/                    # Trained models (.eqx)
├── outputs/                   # Results, reports, plots, dashboard
├── research/                  # LaTeX proofs (maths_proofs.tex, proof.tex)
└── obsolete/                  # Superseded code
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

### Technical

- [JAX](https://jax.readthedocs.io/) · [Equinox](https://docs.kidger.site/equinox/) · [esig](https://esig.readthedocs.io/) · [FastAPI](https://fastapi.tiangolo.com/)

---

## License

MIT License — see [LICENSE](LICENSE).

---

*Last updated: March 2026 — v3.5 (Joint SPX-VIX calibration, multi-scale Hurst, README restructured → see [RESEARCH.md](RESEARCH.md) for narrative)*
