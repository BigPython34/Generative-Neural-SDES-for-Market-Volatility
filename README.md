# DeepRoughVol: Generative Neural SDEs for Market Volatility

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![JAX](https://img.shields.io/badge/JAX-0.4-orange)
![License](https://img.shields.io/badge/License-MIT-grey)

## Abstract

**DeepRoughVol** is a non-parametric generative framework designed to simulate S&P 500 Realized Volatility paths. Unlike traditional stochastic volatility models (Heston, Bergomi) which rely on rigid SDE specifications, this project leverages **Neural Stochastic Differential Equations (Neural SDEs)** conditioned by **Path Signatures** to learn market dynamics directly from historical data.

The model addresses the gradient explosion problem common in Neural SDE training by operating in a constrained **Log-Variance space** augmented with a physical mean-reverting prior.

## Methodology

### 1. Signature-Conditional Generation
Instead of using Recurrent Neural Networks (RNNs) which suffer from vanishing gradients over long horizons, the drift and diffusion components of the SDE are conditioned on the **Path Signature** of the driving noise. Signatures provide a graded basis for the space of continuous functions, effectively encoding the geometric roughness and history of the process.

### 2. Unsupervised Learning via MMD
The model is trained in an unsupervised manner by minimizing the **Maximum Mean Discrepancy (MMD)** between the expected signature of the generated paths and the empirical signature of the S&P 500 realized volatility.

### 3. Physical Prior Stabilization
To ensure numerical stability and realistic outputs, the architecture combines a learned component with a physical prior:
*   **Log-Space Modeling:** Ensures strictly positive variance.
*   **Ornstein-Uhlenbeck Prior:** Provides a baseline mean-reversion force, preventing volatility collapse or explosion during the early stages of training.

## Performance Metrics

The model was trained on S&P 500 realized volatility data (2010-2023). It successfully reproduces key stylized facts of financial time series, specifically the non-Gaussian distribution (fat tails).

| Metric | Real Market (Target) | AI Generated (Ours) | Analysis |
| :--- | :--- | :--- | :--- |
| **Median Level** | 0.0151 | **0.0119** | Accurately captures the base market regime. |
| **Kurtosis** | 15.93 | **13.93** | Successfully reproduces extreme tail risks (Fat Tails). |
| **Memory (ACF-1)**| 0.97 | 0.85 | Captures significant long-range dependence. |

## Technical Stack

*   **JAX / Equinox:** High-performance differentiable programming and Neural SDE solver.
*   **Esig:** Computation of Path Signatures (Order 3).
*   **Optax:** Optimization with gradient clipping.
*   **Plotly:** Interactive visualization for statistical validation.

## Mathematical Framework

The model approximates the dynamics of the log-variance process $X_t = \log(V_t)$ as:

$$ dX_t = \kappa(\theta - X_t)dt + \mathcal{N}_{\mu}(\mathbb{S}_t)dt + \mathcal{N}_{\sigma}(\mathbb{S}_t)dW_t $$

Where:
*   $\kappa(\theta - X_t)$ is the physical mean-reverting prior.
*   $\mathcal{N}_{\mu, \sigma}$ are neural networks parameterizing the drift and diffusion corrections.
*   $\mathbb{S}_t$ is the truncated signature of the driving Brownian motion $W_{[0,t]}$.

## Usage

### Installation
```bash
pip install -r requirements.txt
```
### Execution
Run the full pipeline (Data loading -> Training -> Generation -> Diagnostics):
```bash
python main.py
```
References
Kidger, P. et al. (2021). "Neural SDEs as Infinite-Dimensional GANs".
Levin, D. et al. (2013). "Learning from the past, predicting the statistics for the future, learning an evolving system".