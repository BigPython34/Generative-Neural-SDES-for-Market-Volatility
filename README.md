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
3. [Architecture Overview](#architecture-overview)
4. [Multi-Measure Framework](#multi-measure-framework)
5. [Modules](#modules)
6. [Methodology](#methodology)
7. [Data Sources](#data-sources)
8. [Usage](#usage)
9. [API Reference](#api-reference)
10. [File Structure](#file-structure)
11. [Research Journey](#research-journey)
12. [Lessons Learned](#lessons-learned)
13. [References](#references)

---

## Abstract

**DeepRoughVol** is a non-parametric generative framework that learns volatility dynamics directly from market data using **Neural Stochastic Differential Equations (Neural SDEs)** conditioned on **Path Signatures**.

The project provides a complete quantitative finance toolkit:

- **Volatility modeling**: Signature-conditioned Neural SDE with optional Merton jump-diffusion
- **Multi-measure training**: Separate P-measure (physical) and Q-measure (risk-neutral) models with appropriate loss functions
- **Option pricing**: European vanillas, barriers, Asian, lookback, autocallable, cliquet, variance/volatility swaps
- **Risk management**: Monte Carlo VaR/CVaR, parametric VaR, stressed VaR, component VaR, tail risk (Hill estimator)
- **Hedging**: Delta hedging simulator comparing BS, Bartlett (vanna/volga-corrected), and sticky-strike strategies
- **P&L attribution**: Second-order Taylor decomposition into Greeks contributions
- **Regime detection**: Multi-signal classifier (VIX, VVIX, term structure, VRP) with adaptive model parameters
- **Market data integration**: Real SOFR rates, VVIX-calibrated vol-of-vol, VIX futures term structure
- **REST API**: FastAPI server with Swagger docs for all pricing/risk/regime endpoints
- **Interactive dashboard**: HTML dashboard with regime, risk metrics, calibration, and IV surface plots

The model is benchmarked against **Rough Bergomi** (rBergomi) with Volterra kernel and **Black-Scholes**, evaluated on real SPY options surfaces.

---

## Key Results

### Model Comparison — IV Smile RMSE (SPY Options)

| Model | Mean IV RMSE | Win Rate | Best At |
|-------|:---:|:---:|---|
| **Rough Bergomi** | **{{BERGOMI_RMSE}}%** | {{BERGOMI_WINRATE}} | Medium maturities (15–42 DTE) |
| Neural SDE | {{NEURAL_RMSE}}% | {{NEURAL_WINRATE}} | Short maturities (7 DTE), crisis |
| Black-Scholes | {{BS_RMSE}}% | {{BS_WINRATE}} | Baseline |

*Source: `outputs/backtest_results.json` — {{N_SCENARIOS}} scenarios across {{N_SNAPSHOTS}} snapshots × multiple maturities.*

### Risk Metrics (Neural SDE, P-measure)

| Metric | Value |
|--------|:---:|
| VaR 95% ({{HORIZON}} days) | {{VAR95}}% |
| VaR 99% | {{VAR99}}% |
| CVaR 95% (Expected Shortfall) | {{ES95}}% |
| CVaR 99% | {{ES99}}% |
| Terminal Vol (mean) | {{TERM_VOL_MEAN}}% |
| Terminal Vol (P95) | {{TERM_VOL_P95}}% |

*Source: `outputs/model_usecases_report.json`*

### Calibrated Parameters (from Market Data)

| Parameter | Symbol | Value | Source |
|-----------|:---:|:---:|---|
| Hurst (SPX 5-min RV) | $H$ | **{{H_SPX5}}** | Variogram on daily RV |
| Hurst (SPX 30-min RV) | $H$ | **{{H_SPX30}}** | Variogram on daily RV |
| Vol-of-Vol (VVIX-calibrated) | $\eta$ | {{ETA_VVIX}} | VVIX with H-correction |
| Vol-of-Vol (config) | $\eta$ | 1.9 | Bergomi benchmark |
| Mean Reversion | $\kappa$ | 2.72 | VIX Futures term structure |
| Long-term log-var | $\theta$ | -3.5 | Historical VIX mean |
| Correlation | $\rho$ | -0.7 | SPX-VIX leverage effect |
| VVIX (current) | — | {{VVIX_CURRENT}} | CBOE VVIX index |
| SOFR Rate | $r$ | {{SOFR_RATE}}% | NY Fed SOFR daily |
| Market Regime | — | {{REGIME}} | Multi-signal consensus |

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

$$dX_t = \underbrace{\kappa(\theta - X_t)}_{\text{OU Prior}} dt + \underbrace{f_\theta(\mathbb{S}_{0,t}, X_t)}_{\text{Neural Drift}} dt + \underbrace{g_\theta(\mathbb{S}_{0,t}, X_t)}_{\text{Neural Diffusion}} dW_t + \underbrace{J \cdot dN_t}_{\text{Jumps (optional)}}$$

where $X_t = \log(V_t)$, $\mathbb{S}_{0,t} \in T^{(3)}(\mathbb{R}^2) \cong \mathbb{R}^{14}$ is the running path signature, and $J \cdot dN_t$ is a compound Poisson process with compensator.

The running signature captures the full path history through Chen's identity, making the SDE genuinely non-Markovian — essential for rough volatility ($H < 0.5$).

### Training

- **Distribution matching**: Kernel MMD² with multi-scale RBF (median heuristic)
- **Mean correction**: Global or per-step marginal penalty (Jensen bias correction)
- **Martingale** (Q only): $E[e^{-rT}S_T] = S_0$ constraint with SOFR rate
- **Jump regularization**: Soft constraint on jump intensity (~3 jumps/year)
- **Optimization**: Adam + gradient clipping, linear warmup + cosine decay, deterministic early stopping

### Rough Bergomi Benchmark

Volterra kernel implementation (Bayer, Friz & Gatheral 2016) with exact spot-vol correlation and previsible variance in Euler scheme.

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

# 4. Visualize + API
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
python bin/dashboard.py
```

---

### Configuration

All parameters are centralized in `config/params.yaml`:

```yaml
# Measure-specific training
training:
  mean_penalty_mode: "global"      # or "marginal"
  q_measure:
    lambda_martingale: 5.0         # Martingale constraint weight
    lambda_jump_reg: 0.1           # Jump regularization

# Neural SDE with jumps
neural_sde:
  learn_ou_params: true
  kappa: 2.72
  theta: -3.5
  jumps:
    lambda_init: 2.0               # Jump intensity (jumps/year)
    mu_j_init: 0.5                 # Mean jump size
    sigma_j_init: 0.3              # Jump volatility

# Risk engine
risk:
  confidence_levels: [0.95, 0.99]
  n_mc_paths: 50000

# Real rates
pricing:
  use_sofr: true                   # Use NY Fed SOFR instead of fixed r
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
│   ├── neural_sde.py                     # ★ NeuralRoughSimulator + JumpParams
│   ├── signature_engine.py               # Path signatures (esig + JAX)
│   ├── generative_trainer.py             # ★ Multi-measure trainer (P/Q/jumps)
│   └── losses.py                         # ★ MMD, martingale, smile, jump reg, composites
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
│   ├── vvix_calibrator.py                # ★ VVIX-based eta calibration
│   ├── black_scholes.py                  # ★ BS pricing + hybrid IV solver (rational + Newton + bisection)
│   ├── greeks_ad.py                      # JAX autodiff Greeks
│   ├── config.py                         # Config loader
│   ├── data_loader.py                    # VIX + RV loading
│   ├── data_pipeline.py                  # Data regeneration
│   ├── data_scrapper.py                  # Market data fetcher
│   ├── coherence_check.py                # Data coherence audit
│   └── diagnostics.py                    # Statistics, Hurst, ACF
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

★ = new or significantly modified in v2.0

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

*Last updated: February 2026 — v2.0 (multi-measure, risk, exotics, API)*
