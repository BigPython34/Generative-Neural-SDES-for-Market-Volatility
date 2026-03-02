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
3. [Why This Project Is Different](#why-this-project-is-different)
4. [Proving Roughness: The ACF / Variogram Evidence](#proving-roughness-the-acf--variogram-evidence)
5. [Architecture Overview](#architecture-overview)
6. [Multi-Measure Framework](#multi-measure-framework)
7. [Modules](#modules)
8. [Methodology](#methodology)
9. [Data Sources](#data-sources)
10. [Usage](#usage)
11. [API Reference](#api-reference)
12. [File Structure](#file-structure)
13. [Research Journey](#research-journey)
14. [Lessons Learned](#lessons-learned)
15. [References](#references)

---

## Abstract

**DeepRoughVol** is a non-parametric generative framework that learns volatility dynamics directly from market data using **Neural Stochastic Differential Equations (Neural SDEs)** conditioned on **Path Signatures**.

The project provides a complete quantitative finance toolkit:

- **Volatility modeling**: Signature-conditioned Neural SDE with **dual backbone** (OU or Volterra/fractional — nests rBergomi exactly) and optional Merton jump-diffusion
- **Roughness proof**: Empirical verification of $H \approx 0.1$ from SPX realized vol, with ACF, variogram, signature correlation (0.9996), and MMD ablation
- **Multi-measure training**: Separate P-measure (physical) and Q-measure (risk-neutral) models with appropriate loss functions
- **Exact path signatures**: Chen's identity implemented at orders 2, 3, 4 — making the SDE genuinely non-Markovian
- **Option pricing**: European vanillas, barriers, Asian, lookback, autocallable, cliquet, variance/volatility swaps
- **Neural SDE Greeks**: Model-implied Δ, Γ, Vega via JAX autodiff through the full MC pipeline
- **Risk management**: Monte Carlo VaR/CVaR, parametric VaR, stressed VaR, component VaR, tail risk (Hill estimator)
- **Hedging**: Delta hedging simulator comparing BS, Bartlett (vanna/volga-corrected), and sticky-strike strategies
- **P&L attribution**: Second-order Taylor decomposition into Greeks contributions
- **Regime detection**: Multi-signal classifier (VIX, VVIX, term structure, VRP) with adaptive model parameters
- **Market data integration**: Real SOFR rates, VVIX-calibrated vol-of-vol (η auto-calibration), VIX futures term structure, cached SPY options
- **REST API**: FastAPI server with Swagger docs for all pricing/risk/regime endpoints
- **Interactive dashboard**: HTML dashboard with regime, risk metrics, calibration, and IV surface plots

The model is benchmarked against **Rough Bergomi** (rBergomi) with Volterra kernel and **Black-Scholes**, evaluated on real SPY options surfaces.

---

## Key Results

### Model Comparison — IV Smile RMSE (SPY Options)

| Model | Mean IV RMSE | Win Rate | Best At |
|-------|:---:|:---:|---|
| **Rough Bergomi** | **5.63%** | 75% (9/12) | Medium maturities (14–41 DTE) |
| Neural SDE | 7.27% | 25% (3/12) | Short maturities (≤14 DTE) |
| Black-Scholes | 7.73% | 0% | Baseline |

*Source: `outputs/backtest_results.json` — 12 scenarios across 3 snapshots × 4 maturities. Ran on 2026-03-02.*

### Risk Metrics (Neural SDE, P-measure)

| Metric | Value |
|--------|:---:|
| VaR 95% (~5 days) | 4.05% |
| VaR 99% | 5.80% |
| CVaR 95% (Expected Shortfall) | 5.13% |
| CVaR 99% | 6.53% |
| Terminal Vol (mean) | 17.8% |
| Terminal Vol (P95) | 21.1% |

*Source: `outputs/model_usecases_report.json`*

### Calibrated Parameters (from Market Data)

| Parameter | Symbol | Value | Source |
|-----------|:---:|:---:|---|
| Hurst (SPX 5-min RV) | $H$ | **0.201** | Variogram on daily RV (258 days) |
| Hurst (SPX 30-min RV) | $H$ | **0.101** | Variogram on daily RV (1646 days) |
| Vol-of-Vol (VVIX-calibrated) | $\eta$ | 1.33 | VVIX with H-correction |
| Vol-of-Vol (config) | $\eta$ | 1.9 | Bergomi benchmark |
| Mean Reversion | $\kappa$ | 2.72 | VIX Futures term structure |
| Long-term log-var | $\theta$ | -3.5 | Historical VIX mean |
| Correlation | $\rho$ | -0.7 | SPX-VIX leverage effect |
| VVIX (current) | — | 109 | CBOE VVIX index |
| SOFR Rate | $r$ | 3.73% | NY Fed SOFR daily |
| Market Regime | — | elevated | Multi-signal consensus |

---

## Why This Project Is Different

Most quantitative finance projects either:
- Implement Black-Scholes (trivial, unrealistic flat vol)
- Use Heston/SABR (better, but smooth vol — misses the fractal structure of real markets)
- Train a neural network on prices (no financial theory, no interpretability)

**DeepRoughVol** is fundamentally different because it is built on the empirical discovery that **volatility is rough** (Gatheral, Jaisson & Rosenbaum, 2018) — and every design choice follows from that fact:

| Feature | Standard Quant Projects | DeepRoughVol |
|---|---|---|
| Vol dynamics | Smooth (Heston, SABR) | **Rough** ($H \approx 0.1$, verified on real SPX data) |
| Memory | Markovian (no path memory) | **Path-dependent** (signature conditioning via Chen's identity) |
| Architecture | Parametric model OR black-box NN | **Neural SDE = OU/Volterra prior + learned corrections** |
| Measures | Single measure | **P-measure (risk) and Q-measure (pricing) with separate losses** |
| Calibration | Manual / grid search | **Auto-calibrated** from VVIX, VIX futures, SOFR, SPX returns |
| Backbone | Fixed | **Dual**: OU (fast) or Fractional/Volterra (nests rBergomi exactly) |
| Verification | "Trust the model" | **Quantitative proofs**: ACF, variogram, signature correlation, MMD ablation |

### Proving Roughness: The ACF / Variogram Evidence

The single most important claim is that volatility is **rough** (Hölder exponent $H \approx 0.1 \ll 0.5$). Here is how we prove it on our actual data:

#### 1. Hurst Exponent from SPX Realized Volatility

Using the variogram method on log-realized-vol computed from 5-min SPX returns (Gatheral et al. 2018):

| Source | $H_{\text{variogram}}$ | $R^2$ | Interpretation |
|---|:---:|:---:|---|
| SPX 5-min (258 days) | **0.201** | 0.91 | Sub-diffusive, borderline rough |
| SPX 30-min (1646 days) | **0.101** | — | **Rough volatility confirmed** ($H < 0.2$) |

The longer the history, the more clearly roughness appears — consistent with the literature.

#### 2. VIX $H \approx 0.5$ Is Expected (Not a Bug)

| Source | $H_{\text{variogram}}$ | Why |
|---|:---:|---|
| VIX 15-min | 0.466 | VIX integrates IV over 30 days → **smoothing kills roughness** |
| VIX 30-min | 0.445 | Same effect, slightly less data |

This is the **P ≠ Q trap**: VIX is a Q-measure object (risk-neutral expectation of future RV). Its apparent smoothness ($H \approx 0.5$) does not contradict the roughness of actual volatility.

#### 3. Neural SDE Reproduces the Right Dynamics

| Metric | Real Data | Neural SDE | Pure OU | Winner |
|---|:---:|:---:|:---:|---|
| Path signature correlation | 1.000 | **0.9996** | — | Neural SDE |
| Hurst (VIX paths) | 0.475 | 0.475 | 0.465 | Neural SDE |
| MMD (distribution distance) | — | **0.0280** | 0.0296 | Neural SDE (−5.6%) |
| Mean variance | 0.0341 | 0.0346 | 0.0339 | OU (closer) |

**The Neural SDE beats the OU baseline on MMD** (the training objective) and **perfectly matches the path signature distribution** (correlation = 0.9996). This means the generated paths are statistically indistinguishable from real VIX variance paths in terms of their sequential structure.

#### 4. Auto-Correlation Function (ACF)

The ACF of variance paths shows **long memory** — slow decay characteristic of rough processes:

- Real ACF(lag 1) ≈ 0.99 (strong persistence)
- Neural SDE ACF matches the real ACF curve closely
- This is visible in the `main.py` comparison plot (Row 3)

#### 5. Running This Yourself

```bash
python bin/verify_roughness.py        # Full roughness + signature + ablation report
python bin/hurst_diagnostic.py        # VIX vs RV Hurst comparison with plots
python bin/compare_vix_vs_rv.py       # Side-by-side roughness analysis
python main.py                        # Visual comparison: Real vs Bergomi vs Neural SDE (ACF plot)
```

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
              │ Loss: MMD + Mean       │ Loss: MMD + Mean       │ Loss: MMD + Mean
              │                        │       + Martingale     │       + Martingale
              │                        │                        │       + Jump Reg
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

Different use cases require different probability measures and loss functions:

| Measure | Training Data | Loss Components | Model File | Use Cases |
|:---:|---|---|---|---|
| **P** (physical) | Realized vol (SPX 5-min) | MMD + mean penalty | `neural_sde_best_p.eqx` | VaR, CVaR, stress testing, vol forecasting |
| **Q** (risk-neutral) | VIX (implied vol) | MMD + mean penalty + **martingale** | `neural_sde_best_q.eqx` | Option pricing, delta hedging, calibration |
| **Q + jumps** | VIX + Merton jumps | MMD + mean + martingale + **jump reg** | `neural_sde_best_q_jump.eqx` | Crisis pricing, tail risk, crash scenarios |

### Why separate models?

- **P-measure**: Captures real-world dynamics including the variance risk premium. VaR and stress tests must reflect how volatility *actually* moves, not how derivatives price it.
- **Q-measure**: Enforces the martingale property ($E^Q[e^{-rT}S_T] = S_0$) required for arbitrage-free pricing. Uses real SOFR rates instead of hardcoded $r$.
- **Q + jumps**: Adds Merton (1976) compound Poisson jumps to the log-variance process with learnable intensity $\lambda$, mean jump $\mu_J$, and jump vol $\sigma_J$. Includes a jump compensator to maintain drift correctness.

---

## Modules

### Pricing — `quant/exotic_pricer.py`

Monte Carlo pricing for path-dependent exotics:

| Product | Description | Key Sensitivity |
|---|---|---|
| **Asian** (arithmetic/geometric) | Payoff on average price | Vol term structure, autocorrelation |
| **Lookback** (fixed/floating) | Payoff on max/min price | Vol level, path continuity |
| **Barrier** (DOC, DIC, UOP) | Knock-out/knock-in | Skew, tail risk |
| **Autocallable** | Early redemption + coupon + KI put | Forward vol, correlation |
| **Cliquet** | Locally capped/floored periodic returns | Forward skew, vol-of-vol |
| **Variance swap** | RV² vs strike | Realized vol dynamics |
| **Volatility swap** | RV vs strike (convexity adj.) | Vol-of-vol, roughness |

### Risk — `quant/risk_engine.py`

| Metric | Method |
|---|---|
| **VaR** (95%, 99%) | Monte Carlo on Neural SDE paths |
| **CVaR / ES** | Conditional tail expectation |
| **Parametric VaR** | Delta-normal (quick approximation) |
| **Stressed VaR** | Conditional on spot drop > 5% |
| **Component VaR** | Marginal contribution per position |
| **Tail Index** | Hill estimator on extreme losses |
| **Stress Testing** | Black Monday, Lehman, COVID, Flash Crash, Volmageddon, rate shock |

### Hedging — `quant/hedging_simulator.py`

| Strategy | Delta Formula | Key Advantage |
|---|---|---|
| **Black-Scholes** | $\Delta^{BS}(\sigma_{ATM})$ | Simple, fast |
| **Bartlett** | $\Delta^{BS} + \text{Vanna} \cdot d\sigma/dS$ | Minimum-variance under stoch vol |
| **Sticky-Strike** | $\Delta^{BS}(\sigma_t^{local})$ | Adapts to realized vol |

### P&L Attribution — `quant/pnl_attribution.py`

Second-order Taylor decomposition:

$$\Delta C \approx \underbrace{\Delta \cdot \delta S}_{\text{Delta}} + \underbrace{\tfrac{1}{2}\Gamma \cdot (\delta S)^2}_{\text{Gamma}} + \underbrace{\nu \cdot \delta\sigma}_{\text{Vega}} + \underbrace{\Theta \cdot \delta t}_{\text{Theta}} + \underbrace{\text{Vanna} \cdot \delta S \cdot \delta\sigma}_{\text{Cross}} + \underbrace{\tfrac{1}{2}\text{Volga} \cdot (\delta\sigma)^2}_{\text{Convexity}} + \underbrace{\rho \cdot \delta r}_{\text{Rho}}$$

Also includes `NeuralSDEGreeks`: model-implied $\Delta$, $\Gamma$, Vega via **JAX autodiff through the full MC pricing pipeline** (pathwise differentiation). These capture stochastic vol effects that BS Greeks miss.

### Regime Detection — `quant/regime_detector.py`

5 weighted signals → consensus regime → adaptive parameters:

| Signal | Weight | Source |
|---|:---:|---|
| VIX level | 30% | VIX spot |
| VVIX level | 20% | Vol-of-vol index |
| Term structure | 20% | VIX futures spread |
| VRP | 15% | Implied - Realized |
| VIX percentile | 15% | Historical rank |

Regimes: **calm** → **normal** → **stressed** → **crisis**, each with recommended $(H, \eta, \rho)$.

---

## Methodology

### Neural SDE with Path Signatures

The model supports two backbone architectures, selectable via `neural_sde.backbone` in config:

**OU backbone** (default — fast, good for P-measure / stress testing):

$$dX_t = \underbrace{\kappa(\theta - X_t)}_{\text{OU Prior}} dt + \underbrace{f_\theta(\mathbb{S}_{0,t}, X_t)}_{\text{Neural Drift}} dt + \underbrace{g_\theta(\mathbb{S}_{0,t}, X_t)}_{\text{Neural Diffusion}} dW_t + \underbrace{J \cdot dN_t}_{\text{Jumps (optional)}}$$

**Fractional backbone** (nests rBergomi exactly — for Q-measure / pricing):

$$X_t = \eta \cdot \hat{W}^H_t - \tfrac{1}{2}\eta^2 \text{Var}[\hat{W}^H_t] + \int_0^t f_\theta(\mathbb{S}_{0,s}, X_s) ds + \int_0^t g_\theta(\mathbb{S}_{0,s}, X_s) dW_s$$

where $\hat{W}^H_t = \sqrt{2H} \int_0^t (t-s)^{H-1/2} dW_s$ is the Riemann-Liouville fBM with **learnable** $(H, \eta)$. When $f_\theta = g_\theta = 0$, this exactly recovers rBergomi.

In both cases:
- $X_t = \log(V_t)$ is log-variance
- $\mathbb{S}_{0,t} \in T^{(M)}(\mathbb{R}^2)$ is the running path signature of $(t, X)$ up to order $M \in \{2,3,4\}$ (configurable), computed via **exact Chen's identity**
- At order 3: $\dim(\mathbb{S}) = 14$ features; at order 4: $\dim(\mathbb{S}) = 30$ features
- The signature makes the SDE genuinely **non-Markovian** (path-dependent) — essential for rough volatility

### Training

- **Distribution matching**: Kernel MMD² with multi-scale RBF (median heuristic)
- **Mean correction**: In **log-variance space** to avoid Jensen bias ($E[e^X] > e^{E[X]}$)
- **Marginal mode**: Optional per-step $E[V_t]$ matching (Bayer & Stemper 2018)
- **Martingale** (Q only): $E^Q[e^{-rT}S_T] = S_0$ constraint with real SOFR rate
- **Smile fit** (Q pricing mode): Vega-weighted IV smile loss from cached SPY options
- **Jump regularization**: Soft constraint on jump intensity (~3 jumps/year)
- **η auto-calibration**: When `bergomi.eta_source: vvix`, η is calibrated from VVIX at training time
- **Optimization**: Adam + gradient clipping, linear warmup + cosine decay, deterministic early stopping
- **Multi-scale** (optional): Train on multiple VIX frequencies simultaneously for horizon consistency

### Rough Bergomi Benchmark

Volterra kernel implementation (Bayer, Friz & Gatheral 2016) with exact spot-vol correlation and previsible variance in Euler scheme. Walk-forward backtest recalibrates $(\xi_0, H, \eta)$ per fold to avoid look-ahead bias.

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

- **Python 3.12+** (tested on 3.12)
- **Windows/Linux/macOS** — all scripts are cross-platform
- ~4 GB disk space for data + models

### Step 0 — Installation

```bash
git clone <repo-url>
cd "Projet IA & quant"
pip install -r requirements.txt
```

All dependencies (JAX, Equinox, FastAPI, Plotly, etc.) are pinned in `requirements.txt`.

---

### Step 1 — Download Market Data

Before anything else, populate the `data/` folder with historical VIX, SPX, VVIX, VIX Futures, and SOFR rates.

```bash
python bin/regenerate_data.py --mode yahoo
```

This downloads from Yahoo Finance and CBOE. Files are cached — rerun with `--force` to refresh.

**If you have TradingView CSV exports** in `data/`, use `--mode tradingview` instead (it parses local files).

After completion, verify the data folder:

```
data/
├── market/vix/           # VIX 5-min, 30-min, daily
├── market/spx/           # SPX 5-min, 30-min, daily
├── market/vvix/          # VVIX daily
├── rates/sofr_daily_nyfed.csv   # SOFR daily rates (NY Fed)
└── cboe_vix_futures_full/       # VIX futures term structure
```

### Step 2 — Fetch SPY Options Surface

Downloads and caches the latest SPY options chain (calls + puts, all expirations) from Yahoo Finance:

```bash
python bin/fetch_options.py
```

The data is saved in `data/options_cache/`. This is needed for backtesting and IV calibration.

### Step 3 — Calibrate Market Parameters

Estimate Hurst exponent, vol-of-vol, and other Rough Bergomi parameters from real SPX/VIX data:

```bash
python bin/calibrate.py
```

**Output**: `outputs/advanced_calibration.json` — contains estimated $H$, $\eta$, $\xi_0$, $\rho$ from intraday data.

### Step 4 — Train the Neural SDE Models

This is the core step. Train separate models for different purposes:

```bash
# Train both P-measure AND Q-measure models (recommended for first run)
python bin/train_multi.py

# Or train a specific measure:
python bin/train_multi.py --measure P          # Physical measure (for VaR, risk)
python bin/train_multi.py --measure Q          # Risk-neutral (for pricing)
python bin/train_multi.py --measure Q --jumps  # Risk-neutral with crisis jumps
```

| Flag | Model File Created | Use For |
|---|---|---|
| (default) | `neural_sde_best_p.eqx` + `neural_sde_best_q.eqx` | Full coverage |
| `--measure P` | `neural_sde_best_p.eqx` | VaR, stress testing, vol forecasting |
| `--measure Q` | `neural_sde_best_q.eqx` | Option pricing, hedging, calibration |
| `--measure Q --jumps` | `neural_sde_best_q_jump.eqx` | Crisis pricing, tail risk |

**Training time**: ~5–15 min per model on CPU (depends on `n_epochs` in `config/params.yaml`).

### Step 5 — Run Backtest on Real Options

Compare Neural SDE vs Rough Bergomi vs Black-Scholes on historical SPY option smiles:

```bash
python bin/backtest.py
```

**Output**: `outputs/backtest_results.json` + `outputs/plots/backtest_smiles.html` — model comparison across multiple maturities and snapshots.

The backtest uses **CBOE VIX futures** when available: `data/cboe_vix_futures_full/` (e.g. `vix_futures_front_month.csv`, or `vix_futures_all.csv`) to calibrate the forward variance xi0 from the term structure (30/60/90 DTE). This improves Bergomi and Neural SDE pricing. The API exposes `GET /data/cboe/term-structure` and `GET /data/cboe/futures-history` for the dashboard.

### Step 6 — Options Surface Calibration (optional)

Calibrate the Bergomi model to the real SPY smile, and run Neural SDE IV surface calibration:

```bash
python bin/options_calibration.py             # Bergomi smile calibration
python bin/risk_neutral_calibration.py        # Neural SDE IV surface fit
```

**Output**: `outputs/calibration_report.json`, `outputs/risk_neutral_calibration.json`

### Step 7 — Generate Use-Case Reports (Scenarios, VaR, Pricing, Regime)

Run the full model suite pipeline to get concrete risk and pricing outputs:

```bash
python bin/model_suite.py --run-usecases
```

**Output**: `outputs/model_usecases_report.json` — contains VaR/CVaR, stress tests, vol scenarios, regime classification, exotic pricing, etc.

You can also retrain profiles and run use-cases in one command:

```bash
python bin/model_suite.py --train-suite --run-usecases
```

### Step 8 — Generate the Interactive Dashboard

Aggregates all results into a single HTML dashboard:

```bash
python bin/dashboard.py
```

**Output**: `outputs/dashboard.html` — open in any browser. Shows regime status, risk KPIs, IV surfaces, model comparison charts.

### Step 9 — Launch the Interactive Dashboard & API

Start the FastAPI server with the built-in web UI:

```bash
python bin/api_server.py
```

Open **http://localhost:8000** for the interactive dashboard with:

- **Overview**: Market regime, SOFR rate, VIX history, report summary KPIs
- **Monte Carlo Paths**: Animated path generation with terminal distribution histogram
- **Vol Surface 3D**: Interactive 3D implied volatility surface from cached options
- **Hedging Simulation**: Animated hedge P&L fan chart (mean, confidence bands, sample paths)
- **Pricing**: Vanilla (BS + Greeks) and exotic (MC) option pricing forms
- **Risk**: VaR/CVaR calculator and stress test with bar chart visualization
- **P&L Attribution**: Greeks decomposition with waterfall chart
- **Regime Detection**: Radar chart of market signals
- **Script Runner**: Launch any pipeline script (train, backtest, etc.) from the UI with live console output
- **Reports**: Browse all JSON output reports

Swagger API docs are still available at **http://localhost:8000/docs**.

#### API Examples (Swagger / curl)

**Price a vanilla option:**

```json
POST /price/vanilla
{
  "spot": 684.17,
  "strike": 680,
  "T": 0.08,
  "sigma": 0.18,
  "opt_type": "call"
}
→ {"price": 15.42, "greeks": {"delta": 0.56, "gamma": 0.012, "vega": 0.98, "theta": -0.47}, "r_used": 0.0373}
```

**Price an exotic (Asian call):**

```json
POST /price/exotic
{
  "product": "asian_call",
  "spot": 100,
  "strike": 100,
  "T": 0.5,
  "n_mc_paths": 50000,
  "extra_params": {"sigma": 0.25}
}
→ {"price": 4.82, "std_error": 0.05}
```

Available products: `asian_call`, `asian_put`, `lookback_call`, `lookback_put`, `autocallable`, `cliquet`, `variance_swap`, `volatility_swap`.

**Compute VaR / CVaR:**

```json
POST /risk/var
{
  "spot": 684.17,
  "positions": [{"opt_type": "call", "strike": 680, "T": 0.08, "quantity": 10}],
  "n_mc_paths": 50000,
  "horizon_days": 1
}
→ {"var_95": -12.34, "var_99": -18.56, "cvar_95": -15.78, "cvar_99": -22.10, ...}
```

**Run stress tests:**

```json
POST /risk/stress
{
  "spot": 684.17,
  "positions": [{"opt_type": "call", "strike": 680, "T": 0.08, "quantity": 10}]
}
→ {"black_monday": {"pnl": -890.5}, "lehman": {"pnl": -312.1}, "covid": {"pnl": -567.3}, ...}
```

**P&L attribution (Greeks decomposition):**

```json
POST /pnl/attribute
{
  "spot": 684.17,
  "strike": 680,
  "T": 0.08,
  "r": 0.0373,
  "sigma": 0.18,
  "opt_type": "call",
  "spot_new": 680,
  "sigma_new": 0.20,
  "dt": 0.003968
}
→ {"total_pnl": -1.23, "delta_pnl": -2.35, "gamma_pnl": 0.10, "vega_pnl": 0.98, "theta_pnl": -0.47, ...}
```

**Detect market regime:**

```json
GET /regime
→ {"regime": "normal", "confidence": 0.72, "signals": {"vix_level": 17.1, "vvix": 92, ...},
   "recommended_params": {"H": 0.07, "eta": 1.9, "rho": -0.70}}
```

**Simulate delta hedging (compare strategies):**

```json
POST /hedge/simulate
{
  "spot": 684.17,
  "strike": 680,
  "T": 0.25,
  "sigma": 0.18,
  "opt_type": "call",
  "n_mc_paths": 10000,
  "hedge_freq": "daily"
}
→ {"Black-Scholes": {"mean_pnl": 0.03, "std_pnl": 1.54, "tracking_error": 3.08, ...},
   "Bartlett": {"mean_pnl": 0.01, "std_pnl": 1.12, "tracking_error": 2.24, ...}}
```

**Calibrate eta from VVIX:**

```json
POST /calibrate/eta
{"H": 0.07, "window_days": 252}
→ {"eta_estimate": 1.87, "eta_std": 0.14, "vvix_mean": 93.2, "regime": "normal"}
```

### Step 10 — Diagnostics & Research (optional)

Deep-dive scripts for verifying mathematical properties:

```bash
python bin/verify_roughness.py        # Roughness + signatures + ablation study
python bin/hurst_diagnostic.py        # VIX vs RV Hurst comparison
python bin/compare_frequencies.py     # VIX frequency analysis (5m/15m/30m)
python bin/compare_vix_vs_rv.py       # VIX vs Realized Vol roughness
python bin/robustness_check.py        # Full robustness: MMD, Hurst, smiles, mean
python bin/walk_forward.py            # Walk-forward temporal backtest
```

### Step 11 — Full Demo Pipeline (legacy)

The original all-in-one demo — trains a model, generates paths, prices an exotic, and shows comparison plots:

```bash
python main.py
```

---

### Where to Find Results

| Output file | Generated by | Contents |
|---|---|---|
| `outputs/backtest_results.json` | `bin/backtest.py` | RMSE per model/maturity, win rates |
| `outputs/model_usecases_report.json` | `bin/model_suite.py` | VaR, CVaR, stress tests, regime, vol scenarios |
| `outputs/advanced_calibration.json` | `bin/calibrate.py` | Estimated H, eta, xi0, rho |
| `outputs/roughness_verification.json` | `bin/verify_roughness.py` | **Roughness proof**: H, MMD, sig corr, ablation |
| `outputs/calibration_report.json` | `bin/options_calibration.py` | Bergomi smile calibration |
| `outputs/risk_neutral_calibration.json` | `bin/risk_neutral_calibration.py` | Neural SDE IV surface fit |
| `outputs/dashboard.html` | `bin/dashboard.py` | Interactive dashboard (open in browser) |
| `outputs/plots/backtest_smiles.html` | `bin/backtest.py` | IV smile comparison charts |
| `models/neural_sde_best_p.eqx` | `bin/train_multi.py` | P-measure model (risk) |
| `models/neural_sde_best_q.eqx` | `bin/train_multi.py` | Q-measure model (pricing) |
| `models/neural_sde_best_q_jump.eqx` | `bin/train_multi.py --jumps` | Q + jumps model (crisis) |

### Recommended First-Time Workflow (TL;DR)

**Without TradingView** (anyone can run this):

```bash
# 1. Get data
python bin/regenerate_data.py --mode yahoo    # ~60 days intraday from Yahoo
python bin/fetch_options.py                   # SPY options surface

# 2. Calibrate + Train
python bin/calibrate.py
python bin/train_multi.py

# 3. Backtest + Reports
python bin/backtest.py
python bin/model_suite.py --run-usecases

# 4. Prove it works (roughness + signature quality)
python bin/verify_roughness.py                # → outputs/roughness_verification.json

# 5. Visualize + API
python bin/dashboard.py                       # → open outputs/dashboard.html
python bin/api_server.py                      # → open http://localhost:8000/docs
```

**With TradingView exports** (richer data, years of history):

Place your TradingView CSV exports in `data/trading_view/` with names like `TVC_VIX,5min.csv`, `SP_SPX,30min.csv`, then:

```bash
python bin/regenerate_data.py --mode tradingview   # uses local TV files + downloads daily/SOFR
python bin/fetch_options.py
python bin/calibrate.py
python bin/train_multi.py                          # many more training paths
python bin/backtest.py
python bin/model_suite.py --run-usecases
python bin/verify_roughness.py                     # roughness proof + ablation
python bin/dashboard.py
```

---

### Configuration

All parameters are centralized in `config/params.yaml` (200+ lines). Key sections:

```yaml
# Backbone architecture
neural_sde:
  backbone: "ou"                   # "ou" (fast) or "fractional" (nests rBergomi)
  sig_truncation_order: 3          # Path signature order (2, 3, or 4)
  fractional:                      # Used when backbone=fractional
    hurst_init: 0.10               # Initial H (learnable)
    eta_init: 1.9                  # Initial η (learnable)
    learn_hurst: true
    learn_eta: true

# Training mode determines active losses
training:
  training_mode: "general"         # "pricing" | "stress_test" | "general"
  mean_penalty_space: "log_v"      # "log_v" (avoids Jensen bias) or "variance"
  q_measure:
    lambda_martingale: 5.0
    lambda_smile: 1.0              # Active in pricing mode only

# Auto-calibration
bergomi:
  eta_source: "config"             # "config" (manual) or "vvix" (auto-calibrate from VVIX)

# Multi-scale training
data:
  multi_scale:
    enabled: false
    scales:
      - {freq_min: 15, segment_length: 120, weight: 1.0}
      - {freq_min: 30, segment_length: 240, weight: 0.5}

# Temporal coherence validation
simulation:
  coherence_test:
    enabled: true
    horizons_days: [5, 10, 20, 30]
    tolerance_mean: 0.15
```

---

## API Reference

Launch: `python bin/api_server.py` → Swagger at `http://localhost:8000/docs`

| Endpoint | Method | Description |
|---|:---:|---|
| `/health` | GET | Health check |
| `/regime` | GET | Current market regime + signals |
| `/rates/sofr` | GET | Current SOFR rate |
| `/price/vanilla` | POST | BS price + Greeks |
| `/price/exotic` | POST | MC exotic pricing (asian, lookback, autocall, cliquet, var/vol swap) |
| `/risk/var` | POST | Portfolio VaR/CVaR |
| `/risk/stress` | POST | Stress test scenarios |
| `/pnl/attribute` | POST | Greeks P&L decomposition |
| `/calibrate/eta` | POST | VVIX-based eta calibration |

---

## File Structure

```
DeepRoughVol/
│
├── main.py                              # Full demo pipeline
│
├── bin/                                  # Executable entry points
│   ├── train.py                          # Train single Neural SDE
│   ├── train_multi.py                    # ★ Multi-measure training (P/Q/jumps)
│   ├── calibrate.py                      # Bergomi calibration
│   ├── backtest.py                       # Historical options backtest
│   ├── walk_forward.py                   # Walk-forward backtest
│   ├── fetch_options.py                  # SPY options cache
│   ├── dashboard.py                      # Dashboard generator
│   ├── api_server.py                     # ★ FastAPI REST server
│   ├── model_suite.py                    # Multi-profile training
│   ├── options_calibration.py            # Options surface calibration
│   ├── risk_neutral_calibration.py       # IV surface calibration
│   ├── regenerate_data.py                # Data pipeline
│   ├── verify_roughness.py              
│   ├── hurst_diagnostic.py              
│   ├── compare_frequencies.py           
│   ├── compare_vix_vs_rv.py            
│   └── robustness_check.py             
│
├── config/
│   └── params.yaml                       # ★ Central config (incl. risk, hedging, regime, jumps)
│
├── core/                                 # Stochastic models
│   ├── bergomi.py                        # rBergomi (Volterra + Davies-Harte)
│   └── stochastic_process.py             # Fractional Brownian Motion
│
├── engine/                               # ML engine
│   ├── neural_sde.py                     # ★★ NeuralRoughSimulator + FractionalBackbone + JumpParams
│   ├── signature_engine.py               # ★★ Exact Chen's identity (order 2-4)
│   ├── generative_trainer.py             # ★★ Multi-measure trainer (P/Q, smile fit, log-V mean penalty)
│   └── losses.py                         # ★★ MMD, martingale, smile_fit, jump reg, composites
│
├── quant/                                # Quant library
│   ├── exotic_pricer.py                  # ★ Asian, Lookback, Barrier, Autocall, Cliquet, Var/Vol swap
│   ├── risk_engine.py                    # ★ VaR/CVaR/Stressed VaR/Component VaR/stress test
│   ├── hedging_simulator.py              # ★ BS/Bartlett/Sticky-strike delta hedging
│   ├── pnl_attribution.py               # ★ Greeks P&L decomposition (2nd order Taylor)
│   ├── regime_detector.py                # ★ Multi-signal regime classifier
│   ├── pricing.py                        # MC pricing engine
│   ├── mc_pricer.py                      # European option pricer
│   ├── backtesting.py                    # Historical backtest
│   ├── walk_forward_backtest.py          # Walk-forward backtest
│   ├── advanced_calibration.py           # Market parameter extraction
│   ├── dashboard_v2.py                   # ★ Enhanced dashboard (regime + risk panels)
│   ├── options_cache.py                  # SPY options cache
│   └── calibration/
│       ├── hurst.py                      # Hurst estimation (variogram, DMA)
│       └── bergomi_optimizer.py          # Bergomi grid search
│
├── utils/                                # Utilities
│   ├── sofr_loader.py                    # ★ NY Fed SOFR rates integration
│   ├── vvix_calibrator.py                # ★★ VVIX auto-calibration of η (with H-correction)
│   ├── black_scholes.py                  # ★ BS pricing + hybrid IV solver (rational + Newton + bisection)
│   ├── greeks_ad.py                      # ★★ Neural SDE Greeks (Δ, Γ, Vega via JAX autodiff through MC)
│   ├── options_iv_loader.py              # ★★ Options-based IV extraction (VIX replication + Dupire)
│   ├── config.py                         # Config loader
│   ├── data_loader.py                    # ★★ Multi-scale data loading (multi-frequency VIX)
│   ├── data_pipeline.py                  # Data regeneration
│   ├── data_scrapper.py                  # Market data fetcher
│   ├── coherence_check.py                # Data coherence audit
│   └── diagnostics.py                    # ★★ Statistics, Hurst, ACF + temporal coherence test
│
├── data/
│   ├── market/{vix,spx,vvix}/            # Intraday + daily market data
│   ├── rates/sofr_daily_nyfed.csv        # SOFR rates (2018–2026)
│   ├── cboe_vix_futures_full/            # VIX futures term structure
│   └── options_cache/                    # Cached SPY option surfaces
│
├── models/                               # Trained models (.eqx)
│   ├── neural_sde_best.eqx              # Latest best model (backward compat)
│   ├── neural_sde_best_p.eqx            # P-measure model
│   ├── neural_sde_best_q.eqx            # Q-measure model
│   └── neural_sde_best_q_jump.eqx       # Q-measure with jumps
│
├── outputs/                              # Generated results
├── research/                             # LaTeX proofs
│   └── maths_proofs.tex                  # Signature-kernel convergence bound
│
└── obsolete/                             # Superseded code
```

★ = new or significantly modified in v2.0 — ★★ = new or rewritten in v3.0 (mathematical audit)

---

## Research Journey

*(Condensed — see git history for full details)*

### Phase 1–2: Initial Implementation
Built Neural SDE with path signatures, discovered training bugs (weight persistence, dimension mismatch).

### Phase 3: The P vs Q Problem
Critical insight: VIX is already Q-measure. Training on VIX = training in risk-neutral measure directly.

### Phase 4–8: Robustness Audit
- **Leverage effect**: $\rho(\text{SPX returns}, \Delta\text{VIX}) = -0.86$
- **VIX smoothing**: VIX is a 30-day integral → $H \approx 0.5$ (not rough). True roughness ($H \approx 0.1$) only on realized vol from SPX returns.
- **Bergomi bugs**: Davies-Harte variance 500× too low, reversed skew from fGn/BM confusion, adaptedness bias from non-previsible variance.

### Phase 9–13: Critical Fixes
- 328× temporal mismatch ($dt = 1/N$ vs $T/N$)
- Jensen bias (missing mean penalty)
- Signature normalization (time components dominated)
- Deterministic validation noise for early stopping
- Signature state propagation across blocks

### Phase 14 (v2.0): Multi-Measure Architecture
- Separate P and Q models with appropriate losses
- Martingale constraint for Q-measure
- SOFR integration replacing hardcoded rates
- VVIX-calibrated eta
- Jump-diffusion component for crisis modeling
- Full exotic pricing, risk, hedging, P&L attribution suite
- Regime detection with adaptive parameters
- REST API and enhanced dashboard

### Phase 15 (v3.0): Mathematical Audit & Rigorous Foundations
Comprehensive mathematical review → 13 improvements:

- **Fractional backbone**: Volterra kernel with learnable $(H, \eta)$ — nests rBergomi as special case. Neural corrections on top allow the model to go beyond rBergomi.
- **Exact Chen's identity**: Signature computation rewritten for orders 2, 3, 4 with mathematically exact tensor updates (replacing the approximate order-3 implementation).
- **Jensen bias fix**: Mean penalty moved from variance space to **log-variance space** ($E[\log V]$ matching instead of $E[V]$), eliminating the systematic upward bias from $E[e^X] > e^{E[X]}$.
- **η auto-calibration from VVIX**: When `eta_source: vvix`, η is estimated from market data at training time (1.33 from VVIX vs 1.9 hardcoded — 30% difference).
- **Smile fit loss**: Vega-weighted IV smile matching from cached SPY options (Q-measure pricing mode).
- **Walk-forward recalibration**: Per-fold $(\xi_0, H, \eta)$ recalibration using only past data (no look-ahead).
- **Multi-scale data loading**: Train on multiple VIX frequencies simultaneously.
- **Neural SDE Greeks**: Model-implied Δ, Γ, Vega via JAX autodiff through the MC pricing pipeline.
- **Temporal coherence test**: Validates generated moments at T = 5, 10, 20, 30 days against market.
- **Options-based variance loader**: Extract instantaneous variance under Q from cached SPY options (CBOE VIX methodology + Dupire).
- **Dead loss audit**: `feller_condition_loss` and `path_regularity_loss` marked deprecated (irrelevant for log-V backbone / contradicts roughness).
- **All hardcoded params → `config/params.yaml`**: 15+ new config keys for full reproducibility.

---

## Lessons Learned

1. **P ≠ Q**: Training on realized vol and testing on implied vol is fundamentally wrong. Use VIX for Q-measure, SPX RV for P-measure.
2. **Temporal scale must be physical**: `dt = T/n_steps` in annual units, not `1/n_steps`.
3. **VIX ≠ Volatility for roughness**: VIX is a 30-day integral; roughness requires realized vol from intraday returns.
4. **Jensen's inequality bites**: $E[e^X] > e^{E[X]}$ — explicit mean penalty needed.
5. **fBM ≠ fGn for correlation**: Volterra kernel preserves spot-vol correlation; Davies-Harte does not.
6. **Previsible variance**: Use $V_{k-1}$ in Euler scheme, not $V_k$.
7. **Martingale matters**: Without the martingale constraint, Q-measure pricing is systematically biased.
8. **Real rates matter**: SOFR vs hardcoded 5% makes a material difference for longer maturities.
9. **Vol-of-vol from VVIX**: Direct calibration from market data beats heuristic estimation.
10. **Signatures propagate**: Resetting signatures between blocks destroys path memory.

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

### Technical

- [JAX](https://jax.readthedocs.io/) · [Equinox](https://docs.kidger.site/equinox/) · [esig](https://esig.readthedocs.io/) · [FastAPI](https://fastapi.tiangolo.com/)

---

## License

MIT License — see [LICENSE](LICENSE).

---

*Last updated: March 2026 — v3.0 (fractional backbone, exact signatures, Jensen fix, VVIX auto-calibration, Neural SDE Greeks)*
