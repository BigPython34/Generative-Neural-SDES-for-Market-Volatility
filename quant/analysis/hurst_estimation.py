"""
Multi-Scale Hurst Exponent Estimation for Rough Volatility
===========================================================

Rigorous implementation of Hurst exponent estimation across time scales,
following the methodology of:

  [GJR18] Gatheral, Jaisson & Rosenbaum (2018):
          "Volatility is rough", Quantitative Finance 18(6), 933-949.
  [BLP16] Bennedsen, Lunde & Pakkanen (2016):
          "Decoupling the short- and long-term behavior of stochastic volatility."
  [ZMLS05] Zhang, Mykland & Aït-Sahalia (2005):
          "A tale of two time scales", JASA 100(472), 1394-1411.

Mathematical framework
----------------------
Let σ²(t) be the instantaneous variance process and define the daily
Realized Variance at sampling frequency Δ:

    RV_d^{(Δ)} = Σᵢ (log S_{tᵢ+Δ} - log S_{tᵢ})²

where the sum runs over all intraday return intervals of length Δ on day d.

The roughness of σ(t) is characterised by the Hurst exponent H ∈ (0, ½):

    E[|log σ²(t+τ) - log σ²(t)|^q] ∝ τ^{qH}

for small τ. We estimate H via:

  1. Variogram (q=2):   m(2,τ) = (1/N) Σ (X_{t+τ} - X_t)²  →  log m ∝ 2H log τ
  2. Structure function: m(q,τ) = (1/N) Σ |X_{t+τ} - X_t|^q  →  log m ∝ qH log τ
  3. DMA (Detrended Moving Average): σ²_DMA(n) ∝ n^{2H+1}
  4. Wavelet estimator (optional, for cross-validation)

Microstructure noise correction
-------------------------------
At very high frequencies (Δ ≤ 1 min), market microstructure noise biases
RV upward. We implement:

  TSRV = RV^{(slow)} - (n̄_slow / n̄_fast) · RV^{(fast)}

per Zhang, Mykland & Aït-Sahalia (2005), where the slow scale averages over
K subsamples of size ≈ n^{2/3}.

Confidence intervals
--------------------
We use block-bootstrap (Künsch 1989) to obtain non-parametric confidence
intervals for H, respecting the temporal dependence in log(RV).

Author: DeepRoughVol Project
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from scipy.stats import linregress
from typing import Optional
DEFAULT_CROSS_ASSETS = {
    "SPY_5m": "data/market/equity_etfs/spy_5m.csv",
    "SPY_15m": "data/market/equity_etfs/spy_15m.csv",
    "SPY_daily": "data/market/equity_etfs/spy_daily.csv",
    "CAC40_5m": "data/market/equity_indices/cac40_5m.csv",
    "CAC40_15m": "data/market/equity_indices/cac40_15m.csv",
    "CAC40_30m": "data/market/equity_indices/cac40_30m.csv",
}

DEFAULT_SPX_PATHS = {
    "5s": "data/market/equity_indices/spx_5s.csv",
    "5m": "data/market/equity_indices/spx_5m.csv",
    "15m": "data/market/equity_indices/spx_15m.csv",
    "30m": "data/market/equity_indices/spx_30m.csv",
    "1h": "data/market/equity_indices/spx_1h.csv",
    "daily": "data/market/equity_indices/spx_daily.csv",
}
try:
    from utils.config import load_config  # type: ignore
except Exception:  # pragma: no cover
    load_config = None  # type: ignore

from utils.loader.RealizedVariance import (
    RVSeriesSettings,
    compute_log_rv_series_from_file,
    rv_series_settings_from_config,
)


# ═══════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════

@dataclass
class HurstEstimate:
    """Result of a single Hurst estimation."""
    H: float                       # Point estimate
    r_squared: float               # R² of the log-log regression
    method: str                    # 'variogram', 'structure_q1', 'dma', etc.
    lags_used: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    log_moments: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    slope: float = 0.0             # Raw slope (before dividing by q)
    intercept: float = 0.0
    std_err: float = 0.0           # Standard error of the slope
    n_points: int = 0              # Number of data points in the series


@dataclass
class HurstBootstrapResult:
    """Hurst estimate with bootstrap confidence intervals."""
    n_obs: int                    # Number of observations in the (log RV) series
    H_point: float                 # Point estimate (full sample)
    H_mean: float                  # Bootstrap mean
    H_std: float                   # Bootstrap standard deviation
    ci_lower: float                # Lower CI bound
    ci_upper: float                # Upper CI bound
    alpha: float                   # Significance level (default 0.05)
    method: str
    n_bootstrap: int
    r_squared: float               # R² from the point estimate
    all_H: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))


@dataclass
class MultiScaleResult:
    """Result of multi-scale Hurst analysis across frequencies."""
    asset: str
    estimates: dict[str, list[HurstBootstrapResult]]  # method -> list over scales
    rv_series: dict[str, np.ndarray]                   # freq_label -> daily log(RV)
    tsrv_series: dict[str, np.ndarray]                 # freq_label -> daily log(TSRV)
    dates: dict[str, pd.DatetimeIndex]                 # freq_label -> dates for each RV point
    consensus_H: float                                 # Weighted consensus H
    consensus_std: float                               # Uncertainty on consensus


# ═══════════════════════════════════════════════════════════════════
# 1. REALIZED VARIANCE COMPUTATION
# ═══════════════════════════════════════════════════════════════════
# Centralised in utils/loader/realized_variance.py


# ═══════════════════════════════════════════════════════════════════
# 2. HURST ESTIMATORS
# ═══════════════════════════════════════════════════════════════════

def _validate_series(series: np.ndarray, name: str = "series") -> np.ndarray:
    """Validate and clean input series."""
    series = np.asarray(series, dtype=np.float64)
    finite_mask = np.isfinite(series)
    if not np.all(finite_mask):
        n_bad = np.sum(~finite_mask)
        series = series[finite_mask]
        if len(series) < 20:
            raise ValueError(f"{name}: only {len(series)} finite values remain (removed {n_bad})")
    return series


def hurst_variogram(
    series: np.ndarray,
    min_lag: int = 1,
    max_lag: Optional[int] = None,
    fit_range: Optional[tuple[int, int]] = None,
) -> HurstEstimate:
    """
    Variogram (second-order structure function) Hurst estimator.

    Computes  m(2, τ) = (1/N_τ) Σ_{t} (X_{t+τ} - X_t)²

    Then fits  log m(2, τ) = 2H · log(τ) + c  via OLS.

    Parameters
    ----------
    series : 1-D array (e.g., log RV time series)
    min_lag, max_lag : lag range for regression
    fit_range : if given, (lag_low, lag_high) restricts the fitting interval
                (useful to exclude microstructure-contaminated short lags)

    Returns
    -------
    HurstEstimate with H = slope/2, R², etc.
    """
    series = _validate_series(series, "variogram")
    n = len(series)
    if max_lag is None:
        max_lag = min(100, n // 4)
    max_lag = min(max_lag, n // 2)

    lags = np.arange(min_lag, max_lag + 1)
    variogram = np.zeros(len(lags))

    for i, tau in enumerate(lags):
        diffs = series[tau:] - series[:-tau]
        variogram[i] = np.mean(diffs ** 2)

    # Select fitting range
    if fit_range is not None:
        mask = (lags >= fit_range[0]) & (lags <= fit_range[1])
    else:
        mask = np.ones(len(lags), dtype=bool)

    if mask.sum() < 3:
        mask = np.ones(len(lags), dtype=bool)

    log_lags = np.log(lags[mask].astype(float))
    log_var = np.log(np.maximum(variogram[mask], 1e-30))

    slope, intercept, r_value, _, std_err = linregress(log_lags, log_var)
    H = slope / 2.0

    return HurstEstimate(
        H=H, r_squared=r_value ** 2, method="variogram",
        lags_used=lags, log_moments=np.log(np.maximum(variogram, 1e-30)),
        slope=slope, intercept=intercept, std_err=std_err,
        n_points=n,
    )


def hurst_structure_function(
    series: np.ndarray,
    q: float = 1.0,
    min_lag: int = 1,
    max_lag: Optional[int] = None,
    fit_range: Optional[tuple[int, int]] = None,
) -> HurstEstimate:
    """
    q-th order structure function estimator.

    Computes  m(q, τ) = (1/N_τ) Σ_{t} |X_{t+τ} - X_t|^q

    Then fits  log m(q, τ) = qH · log(τ) + c  via OLS  ⟹  H = slope / q.

    The q=1 estimator is more robust to heavy tails than the variogram (q=2).

    Parameters
    ----------
    series : 1-D array
    q : moment order (default 1)
    min_lag, max_lag : lag range
    fit_range : optional restricted fitting interval

    Returns
    -------
    HurstEstimate
    """
    series = _validate_series(series, f"structure_q{q}")
    n = len(series)
    if max_lag is None:
        max_lag = min(100, n // 4)
    max_lag = min(max_lag, n // 2)

    lags = np.arange(min_lag, max_lag + 1)
    moments = np.zeros(len(lags))

    for i, tau in enumerate(lags):
        diffs = np.abs(series[tau:] - series[:-tau])
        moments[i] = np.mean(diffs ** q)

    if fit_range is not None:
        mask = (lags >= fit_range[0]) & (lags <= fit_range[1])
    else:
        mask = np.ones(len(lags), dtype=bool)

    if mask.sum() < 3:
        mask = np.ones(len(lags), dtype=bool)

    log_lags = np.log(lags[mask].astype(float))
    log_mom = np.log(np.maximum(moments[mask], 1e-30))

    slope, intercept, r_value, _, std_err = linregress(log_lags, log_mom)
    H = slope / q

    return HurstEstimate(
        H=H, r_squared=r_value ** 2, method=f"structure_q{q}",
        lags_used=lags, log_moments=np.log(np.maximum(moments, 1e-30)),
        slope=slope, intercept=intercept, std_err=std_err,
        n_points=n,
    )


def hurst_dma(
    series: np.ndarray,
    min_lag: int = 2,
    max_lag: Optional[int] = None,
    theta: float = 0.0,
) -> HurstEstimate:
    """
    Detrended Moving Average (DMA) estimator.

    Computes σ²_DMA(n) for windows of size n, then fits:
        log σ_DMA(n) = (H + 0.5) · log(n) + c
    so  H = slope - 0.5.

    Parameters
    ----------
    series : 1-D array
    min_lag, max_lag : window sizes
    theta : position of the average (0 = backward, 0.5 = centered)

    Returns
    -------
    HurstEstimate
    """
    series = _validate_series(series, "DMA")
    n = len(series)
    if max_lag is None:
        max_lag = min(100, n // 4)
    max_lag = min(max_lag, n // 2)

    # Cumulative sum
    Y = np.cumsum(series - np.mean(series))

    window_sizes = np.arange(min_lag, max_lag + 1)
    sigma_dma = np.zeros(len(window_sizes))

    for i, w in enumerate(window_sizes):
        if theta == 0.0:
            # Backward DMA: compare Y(t) with its backward moving average
            ma = np.convolve(Y, np.ones(w) / w, mode='valid')
            # Align: ma[j] = mean of Y[j], Y[j+1], ..., Y[j+w-1]
            # We compare Y[j + w - 1] with ma[j]
            residual = Y[w - 1: w - 1 + len(ma)] - ma
        else:
            # Centered DMA
            half = int(w * theta)
            ma = np.convolve(Y, np.ones(w) / w, mode='valid')
            offset = w - 1 - half
            end_idx = offset + len(ma)
            if end_idx > len(Y):
                end_idx = len(Y)
                ma = ma[:end_idx - offset]
            residual = Y[offset:end_idx] - ma

        sigma_dma[i] = np.sqrt(np.mean(residual ** 2)) if len(residual) > 0 else 0

    valid = sigma_dma > 0
    if valid.sum() < 3:
        return HurstEstimate(H=np.nan, r_squared=0, method="DMA", n_points=n)

    log_w = np.log(window_sizes[valid].astype(float))
    log_s = np.log(sigma_dma[valid])

    slope, intercept, r_value, _, std_err = linregress(log_w, log_s)
    H = slope - 0.5

    return HurstEstimate(
        H=H, r_squared=r_value ** 2, method="DMA",
        lags_used=window_sizes, log_moments=np.log(np.maximum(sigma_dma, 1e-30)),
        slope=slope, intercept=intercept, std_err=std_err,
        n_points=n,
    )


def hurst_ratio_estimator(
    series: np.ndarray,
    lag_base: int = 1,
    max_ratio_lag: Optional[int] = None,
    max_lag: Optional[int] = None
) -> HurstEstimate:
    """
    Ratio-based (non-regression) Hurst estimator.

    For each lag pair (τ, 2τ), compute:
        Ĥ(τ) = (1/2) · log₂( m(2, 2τ) / m(2, τ) )

    This is the 'local' variogram slope and avoids long-range regression bias.
    Reference: Bennedsen, Lunde & Pakkanen (2016).

    Returns the median H across all valid lag pairs.
    """
    series = _validate_series(series, "ratio")
    n = len(series)
    # Accept max_lag as alias for max_ratio_lag (for uniform API)
    if max_ratio_lag is None:
        max_ratio_lag = max_lag if max_lag is not None else min(50, n // 8)

    h_local = []
    lags_used = []

    for tau in range(lag_base, max_ratio_lag + 1):
        if 2 * tau >= n:
            break
        diffs_tau = series[tau:] - series[:-tau]
        diffs_2tau = series[2 * tau:] - series[:-2 * tau]

        m_tau = np.mean(diffs_tau ** 2)
        m_2tau = np.mean(diffs_2tau ** 2)

        if m_tau > 1e-30 and m_2tau > 1e-30:
            H_local = 0.5 * np.log2(m_2tau / m_tau)
            h_local.append(H_local)
            lags_used.append(tau)

    if len(h_local) < 2:
        return HurstEstimate(H=np.nan, r_squared=0, method="ratio", n_points=n)

    h_local = np.array(h_local)
    H_median = np.median(h_local)

    return HurstEstimate(
        H=H_median, r_squared=0.0, method="ratio",
        lags_used=np.array(lags_used), log_moments=h_local,
        slope=0, intercept=0, std_err=np.std(h_local) / np.sqrt(len(h_local)),
        n_points=n,
    )


def hurst_rescaled_range(
    series: np.ndarray,
    min_window: int = 10,
    max_window: Optional[int] = None,
) -> HurstEstimate:
    """
    Rescaled Range (R/S) Analysis — Classic Hurst estimator.
    
    For each window size w, divide the series into non-overlapping windows,
    compute the range R and standard deviation S, then fit:
        log(R/S) = H * log(w) + c
    
    Parameters
    ----------
    series : 1-D array (typically log returns or log RV)
    min_window : minimum window size
    max_window : maximum window size (default: n/4)
    
    Returns
    -------
    HurstEstimate
    """
    series = _validate_series(series, "R/S")
    n = len(series)
    
    if max_window is None:
        max_window = n // 4
    
    if max_window < min_window * 2:
        return HurstEstimate(H=np.nan, r_squared=0, method="rescaled_range", n_points=n)
    
    # Log-spaced window sizes
    window_sizes = np.logspace(
        np.log10(min_window), np.log10(max_window), 15
    ).astype(int)
    window_sizes = np.unique(window_sizes)
    
    rs_values = []
    
    for w in window_sizes:
        if w >= n:
            continue
        
        n_windows = n // w
        if n_windows < 1:
            continue
        
        rs_window = []
        
        for i in range(n_windows):
            window = series[i*w:(i+1)*w]
            
            # Mean-adjusted cumulative sum
            mean_adj = window - np.mean(window)
            cumsum = np.cumsum(mean_adj)
            
            # Range
            R = np.max(cumsum) - np.min(cumsum)
            
            # Standard deviation
            S = np.std(window, ddof=1)
            
            if S > 1e-10:
                rs_window.append(R / S)
        
        if rs_window:
            rs_values.append((w, np.mean(rs_window)))
    
    if len(rs_values) < 5:
        return HurstEstimate(H=np.nan, r_squared=0, method="rescaled_range", n_points=n)
    
    windows = np.array([x[0] for x in rs_values])
    rs = np.array([x[1] for x in rs_values])
    
    # Log-log regression: log(R/S) = H * log(w) + c
    log_w = np.log(windows.astype(float))
    log_rs = np.log(rs)
    
    slope, intercept, r_value, _, std_err = linregress(log_w, log_rs)
    
    return HurstEstimate(
        H=np.clip(slope, 0.01, 0.99),  # R/S H should be in (0,1)
        r_squared=r_value ** 2,
        method="rescaled_range",
        lags_used=windows,
        log_moments=log_rs,
        slope=slope,
        intercept=intercept,
        std_err=std_err,
        n_points=n,
    )


def hurst_spectral(
    series: np.ndarray,
    freq_range: Optional[tuple[float, float]] = None,
) -> HurstEstimate:
    """
    Spectral (Periodogram-based) Hurst estimator.
    
    Computes the power spectral density and fits:
        log(PSD) = -(2H + 1) * log(f) + c
    
    This works because for a fractional Brownian motion with Hurst H:
        S(f) ~ f^{-(2H+1)}
    
    Parameters
    ----------
    series : 1-D array (log returns or log RV)
    freq_range : tuple (lower_frac, upper_frac) of frequencies to use for fitting
                 (default: use lowest 1/4 of frequencies)
    
    Returns
    -------
    HurstEstimate
    """
    from scipy import signal
    
    series = _validate_series(series, "spectral")
    n = len(series)
    
    # Compute periodogram
    freqs, psd = signal.periodogram(series, fs=1.0)
    
    # Remove zero frequency
    freqs = freqs[1:]
    psd = psd[1:]
    
    if freq_range is None:
        # Use only lowest frequencies (where scaling assumption holds best)
        n_use = max(5, len(freqs) // 4)
        freqs = freqs[:n_use]
        psd = psd[:n_use]
    else:
        lower_frac, upper_frac = freq_range
        mask = (freqs >= lower_frac) & (freqs <= upper_frac)
        freqs = freqs[mask]
        psd = psd[mask]
        
        if len(freqs) < 5:
            return HurstEstimate(H=np.nan, r_squared=0, method="spectral", n_points=n)
    
    # Log-log regression: log(PSD) = -(2H+1) * log(f) + c
    log_f = np.log(freqs + 1e-10)
    log_psd = np.log(psd + 1e-10)
    
    slope, intercept, r_value, _, std_err = linregress(log_f, log_psd)
    
    # From log(PSD) = -(2H+1) * log(f) + c, we get:
    #   slope = -(2H+1)  =>  H = -(slope + 1) / 2
    H = -(slope + 1.0) / 2.0
    
    return HurstEstimate(
        H=np.clip(H, 0.01, 0.99),
        r_squared=r_value ** 2,
        method="spectral",
        lags_used=freqs,
        log_moments=log_psd,
        slope=slope,
        intercept=intercept,
        std_err=std_err / 2.0,  # Account for the -1/2 factor
        n_points=n,
    )


# ═══════════════════════════════════════════════════════════════════
# 3. BOOTSTRAP CONFIDENCE INTERVALS
# ═══════════════════════════════════════════════════════════════════

def _block_bootstrap_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate one block-bootstrap sample of indices.

    Moving block bootstrap (non-wrapping): draw contiguous blocks of
    length `block_size` uniformly at random (without wrap-around) until
    we have at least n indices.

    Note: the circular variant can create artificial end-to-start jumps
    in non-stationary series (e.g. volatility regimes), which can heavily
    bias variogram-based H estimates.
    """
    indices = []
    while len(indices) < n:
        if block_size >= n:
            start = 0
        else:
            start = rng.integers(0, n - block_size + 1)
        block = np.arange(start, start + min(block_size, n - start))
        indices.extend(block.tolist())
    return np.array(indices[:n])


def hurst_bootstrap(
    series: np.ndarray,
    method: str = "variogram",
    n_bootstrap: int = 1000,
    block_size: Optional[int] = None,
    alpha: float = 0.05,
    max_lag: Optional[int] = None,
    fit_range: Optional[tuple[int, int]] = None,
    seed: int = 42,
    ci_method: str = "percentile",
) -> HurstBootstrapResult:
    """
    Block-bootstrap confidence intervals for the Hurst exponent.

    We use circular block bootstrap to resample the time series,
    preserving local temporal dependence.

    Parameters
    ----------
    series : 1-D array (e.g., log RV)
    method : 'variogram', 'structure_q1', 'structure_q2', 'dma', 'ratio'
    n_bootstrap : number of bootstrap replications
    block_size : block length for block bootstrap; if None, use n^{1/3}
    alpha : significance level for CI (default 0.05 → 95% CI)
    max_lag : max lag for estimators
    fit_range : optional fitting range for estimators
    seed : random seed
    ci_method : 'percentile' (simple), 'basic', or 't' (studentized bootstrap; more robust)
                - 'percentile': direct percentiles of bootstrap distribution
                - 'basic': inverted CI using reflection
                - 't': studentized (bootstrap-t): uses per-resample standard errors

    Returns
    -------
    HurstBootstrapResult with point estimate, CI, and all bootstrap H values.
    """
    series = _validate_series(series)
    n = len(series)

    if block_size is None:
        # Optimal block size ~ n^{1/3} (Hall, Horowitz & Jing 1995)
        block_size = max(2, int(np.round(n ** (1.0 / 3.0))))

    # Point estimate
    est_func = _get_estimator(method)
    kwargs = {"max_lag": max_lag}
    if fit_range is not None and method in ("variogram", "structure_q1", "structure_q2"):
        kwargs["fit_range"] = fit_range
    point_est = est_func(series, **{k: v for k, v in kwargs.items() if v is not None})

    def _se_of_H(est: HurstEstimate) -> float:
        """Approximate standard error of H from the estimator's regression SE."""
        if not np.isfinite(est.std_err):
            return np.nan

        if est.method == "variogram":
            return est.std_err / 2.0

        if est.method.startswith("structure_q"):
            # method looks like 'structure_q1.0' or 'structure_q2.0'
            try:
                q = float(est.method.split("structure_q", 1)[1])
            except Exception:
                q = 1.0
            q = q if (np.isfinite(q) and q != 0.0) else 1.0
            return est.std_err / q

        # For DMA / spectral / rescaled_range / ratio: std_err already corresponds to SE(H)
        return est.std_err

    # Bootstrap
    rng = np.random.default_rng(seed)
    h_boot = np.full(n_bootstrap, np.nan, dtype=float)
    se_boot = np.full(n_bootstrap, np.nan, dtype=float)

    for b in range(n_bootstrap):
        idx = _block_bootstrap_indices(n, block_size, rng)
        boot_series = series[idx]
        try:
            boot_est = est_func(boot_series, **{k: v for k, v in kwargs.items() if v is not None})
            h_boot[b] = boot_est.H
            se_boot[b] = _se_of_H(boot_est)
        except Exception:
            h_boot[b] = np.nan
            se_boot[b] = np.nan

    h_boot_valid = h_boot[np.isfinite(h_boot)]
    if len(h_boot_valid) < 10:
        return HurstBootstrapResult(
            n_obs=n,
            H_point=point_est.H, H_mean=point_est.H, H_std=0,
            ci_lower=point_est.H, ci_upper=point_est.H,
            alpha=alpha, method=method, n_bootstrap=0,
            r_squared=point_est.r_squared, all_H=h_boot_valid,
        )

    # Compute CI based on method
    if ci_method == "t":
        # Studentized (bootstrap-t) CI.
        # We approximate SE(H) from the regression SE returned by each estimator.
        point_se = _se_of_H(point_est)
        valid = np.isfinite(h_boot) & np.isfinite(se_boot) & (se_boot > 1e-12)
        if np.isfinite(point_se) and point_se > 1e-12 and np.sum(valid) >= 20:
            t_boot = (h_boot[valid] - point_est.H) / se_boot[valid]
            q_lo = np.percentile(t_boot, 100 * (alpha / 2))
            q_hi = np.percentile(t_boot, 100 * (1 - alpha / 2))

            # Inversion: [H_hat - q_hi*SE_hat, H_hat - q_lo*SE_hat]
            ci_lower = point_est.H - q_hi * point_se
            ci_upper = point_est.H - q_lo * point_se
        else:
            # Fallback when SE is not available/reliable
            ci_lower = np.percentile(h_boot_valid, 100 * alpha / 2)
            ci_upper = np.percentile(h_boot_valid, 100 * (1 - alpha / 2))
        
    elif ci_method == "basic":
        # Basic (inverted) percentile method
        ci_lower = 2 * point_est.H - np.percentile(h_boot_valid, 100 * (1 - alpha / 2))
        ci_upper = 2 * point_est.H - np.percentile(h_boot_valid, 100 * (alpha / 2))
        
    else:  # "percentile" (default, simple)
        ci_lower = np.percentile(h_boot_valid, 100 * alpha / 2)
        ci_upper = np.percentile(h_boot_valid, 100 * (1 - alpha / 2))

    if ci_lower > ci_upper:
        ci_lower, ci_upper = ci_upper, ci_lower

    return HurstBootstrapResult(
        n_obs=n,
        H_point=point_est.H,
        H_mean=np.mean(h_boot_valid),
        H_std=np.std(h_boot_valid),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        alpha=alpha,
        method=method,
        n_bootstrap=len(h_boot_valid),
        r_squared=point_est.r_squared,
        all_H=h_boot_valid,
    )


def _get_estimator(method: str):
    """Return the estimator function for the given method name."""
    dispatch = {
        "variogram": hurst_variogram,
        "structure_q1": lambda s, **kw: hurst_structure_function(s, q=1.0, **kw),
        "structure_q2": lambda s, **kw: hurst_structure_function(s, q=2.0, **kw),
        "dma": hurst_dma,
        "ratio": hurst_ratio_estimator,
        "rescaled_range": hurst_rescaled_range,
        "spectral": hurst_spectral,
    }
    if method not in dispatch:
        raise ValueError(f"Unknown method: {method}. Choose from {list(dispatch.keys())}")
    return dispatch[method]


# ═══════════════════════════════════════════════════════════════════
# 4. MULTI-SCALE ANALYSIS
# ═══════════════════════════════════════════════════════════════════


def _get_cfg() -> dict:
    if load_config is None:
        return {}
    try:
        return load_config() or {}
    except Exception:
        return {}


def run_multiscale_hurst(
    asset_paths: Optional[dict[str, str]] = None,
    methods: Optional[list[str]] = None,
    max_lag: int = 80,
    fit_range: Optional[tuple[int, int]] = None,
    n_bootstrap: int = 500,
    alpha: float = 0.05,
    verbose: bool = True,
) -> MultiScaleResult:
    """
    Full multi-scale Hurst analysis on one asset across all frequencies.

    Steps:
    1. Load each frequency file
    2. Compute daily log(RV) or log(TSRV)
    3. Estimate H with multiple methods
    4. Bootstrap confidence intervals
    5. Compute consensus H (inverse-variance weighted)

    Parameters
    ----------
    asset_paths : dict {freq_label: file_path}; defaults to SPX TradingView data
    methods : list of estimator names; defaults to all 4
    max_lag : maximum lag for estimators
    fit_range : optional restricted fitting range
    n_bootstrap : number of bootstrap replications
    alpha : CI significance level
    verbose : print progress

    Returns
    -------
    MultiScaleResult with all estimates, RV series, and consensus H.
    """
    cfg = _get_cfg()
    settings: RVSeriesSettings = rv_series_settings_from_config(cfg)

    if methods is None:
        methods = ["variogram", "structure_q1", "ratio", "dma"]

    rv_series: dict[str, np.ndarray] = {}
    dates_series: dict[str, pd.DatetimeIndex] = {}
    tsrv_labels: dict[str, np.ndarray] = {}
    all_estimates: dict[str, list[HurstBootstrapResult]] = {m: [] for m in methods}

    # Detect asset name from paths
    first_path = list(asset_paths.values())[0]
    asset_name = Path(first_path).stem.split("_")[0].upper()

    for freq_label, fpath in asset_paths.items():
        fpath = Path(fpath)
        if not fpath.exists():
            if verbose:
                print(f"  [{freq_label:>6s}] SKIP — file not found: {fpath}")
            continue

        try:
            log_rv, dates = compute_log_rv_series_from_file(
                fpath,
                freq_label=freq_label,
                settings=settings,
            )
        except (ValueError, FileNotFoundError) as e:
            if verbose:
                print(f"  [{freq_label:>6s}] SKIP — {e}")
            continue

        rv_series[freq_label] = log_rv
        dates_series[freq_label] = dates
        is_tsrv = freq_label in settings.tsrv_frequencies
        if is_tsrv:
            tsrv_labels[freq_label] = log_rv

        if verbose:
            rv_type = "TSRV" if is_tsrv else "RV"
            print(f"  [{freq_label:>6s}] {len(log_rv):>5d} daily log({rv_type}) values  "
                  f"({fpath.name})")

        # Adapt max_lag to series length
        effective_max_lag = min(max_lag, len(log_rv) // 4)

        for method in methods:
            if verbose:
                print(f"         → {method:15s} ... ", end="", flush=True)

            result = hurst_bootstrap(
                log_rv,
                method=method,
                n_bootstrap=n_bootstrap,
                max_lag=effective_max_lag,
                fit_range=fit_range,
                alpha=alpha,
            )
            all_estimates[method].append(result)

            if verbose:
                print(f"H = {result.H_point:+.4f}  "
                      f"[{result.ci_lower:+.4f}, {result.ci_upper:+.4f}]  "
                      f"(R²={result.r_squared:.3f})")

    # ── Consensus H (inverse-variance weighted across all scales & methods) ──
    all_H = []
    all_w = []
    for method in methods:
        for est in all_estimates[method]:
            if np.isfinite(est.H_point) and est.H_std > 1e-6:
                w = 1.0 / (est.H_std ** 2)
                all_H.append(est.H_point)
                all_w.append(w)

    if len(all_H) > 0:
        all_H = np.array(all_H)
        all_w = np.array(all_w)
        consensus_H = np.average(all_H, weights=all_w)
        # Weighted standard deviation
        consensus_std = np.sqrt(1.0 / np.sum(all_w))
    else:
        consensus_H = np.nan
        consensus_std = np.nan

    return MultiScaleResult(
        asset=asset_name,
        estimates=all_estimates,
        rv_series=rv_series,
        tsrv_series=tsrv_labels,
        dates=dates_series,
        consensus_H=consensus_H,
        consensus_std=consensus_std,
    )


# ═══════════════════════════════════════════════════════════════════
# 5. MULTIFRACTAL DIAGNOSTIC
# ═══════════════════════════════════════════════════════════════════

def multifractal_spectrum(
    series: np.ndarray,
    q_values: Optional[np.ndarray] = None,
    max_lag: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the scaling exponent ζ(q) for a range of moment orders q.

    For a self-similar process:  E[|ΔX|^q] ∝ τ^{ζ(q)}  with  ζ(q) = qH.

    If ζ(q) is linear in q → monofractal (single H).
    If ζ(q) is concave    → multifractal (scale-dependent roughness).

    Parameters
    ----------
    series : 1-D array
    q_values : array of moment orders (default: 0.5, 1, 1.5, 2, 2.5, 3)
    max_lag : maximum lag for structure function

    Returns
    -------
    (q_values, zeta_q) — moment orders and corresponding scaling exponents.
    """
    series = _validate_series(series, "multifractal")
    if q_values is None:
        q_values = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

    n = len(series)
    max_lag = min(max_lag, n // 4)
    lags = np.arange(1, max_lag + 1)

    zeta_q = np.zeros(len(q_values))

    for j, q in enumerate(q_values):
        moments = np.zeros(len(lags))
        for i, tau in enumerate(lags):
            diffs = np.abs(series[tau:] - series[:-tau])
            moments[i] = np.mean(diffs ** q)

        log_lags = np.log(lags.astype(float))
        log_mom = np.log(np.maximum(moments, 1e-30))

        slope, _, _, _, _ = linregress(log_lags, log_mom)
        zeta_q[j] = slope

    return q_values, zeta_q


def test_monofractality(
    series: np.ndarray,
    max_lag: int = 60,
) -> tuple[float, float, float]:
    """
    Test whether the process is monofractal (single H) vs multifractal.

    Fits ζ(q) = a·q + b and reports R² and the estimated H = a.
    High R² → monofractal; low R² → multifractal.

    Returns
    -------
    (H_mono, R2_mono, curvature) — estimated H, R² of linear fit,
    and quadratic curvature coefficient (≈0 for monofractal).
    """
    q_vals, zeta_q = multifractal_spectrum(series, max_lag=max_lag)

    # Linear fit: ζ(q) = H · q
    slope_lin, _, r_value, _, _ = linregress(q_vals, zeta_q)
    r2_lin = r_value ** 2

    # Quadratic fit: ζ(q) = a·q² + b·q + c
    coeffs = np.polyfit(q_vals, zeta_q, 2)
    curvature = coeffs[0]  # coefficient of q²

    return slope_lin, r2_lin, curvature


# ═══════════════════════════════════════════════════════════════════
# 6. REPORTING UTILITIES
# ═══════════════════════════════════════════════════════════════════

def format_results_table(result: MultiScaleResult) -> str:
    """Format a MultiScaleResult as a nice ASCII table."""
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"  MULTI-SCALE HURST ANALYSIS — {result.asset}")
    lines.append(f"{'='*80}")

    # Table header
    freq_labels = list(result.rv_series.keys())
    methods = list(result.estimates.keys())

    lines.append(f"\n{'Method':<18s} {'Freq':<8s} {'H':>7s}  {'95% CI':>20s}  {'R²':>6s}  {'n_obs':>6s}  {'n_boot':>6s}")
    lines.append("─" * 75)

    for method in methods:
        for i, est in enumerate(result.estimates[method]):
            freq = freq_labels[i] if i < len(freq_labels) else "?"
            n_obs = est.n_obs
            n_boot = est.n_bootstrap
            ci_str = f"[{est.ci_lower:+.4f}, {est.ci_upper:+.4f}]"
            lines.append(
                f"{method:<18s} {freq:<8s} {est.H_point:+7.4f}  {ci_str:>20s}  "
                f"{est.r_squared:6.3f}  {n_obs:>6d}  {n_boot:>6d}"
            )
        lines.append("")

    lines.append("─" * 75)
    lines.append(
        f"  CONSENSUS H = {result.consensus_H:+.4f}  "
        f"± {result.consensus_std:.4f}  "
        f"(inverse-variance weighted)"
    )

    # Interpretation
    H = result.consensus_H
    if H < 0.10:
        interp = "VERY ROUGH — consistent with Gatheral et al. (2018)"
    elif H < 0.20:
        interp = "ROUGH — within the rough volatility regime"
    elif H < 0.35:
        interp = "MODERATELY ROUGH — borderline"
    elif H < 0.50:
        interp = "SMOOTH SIDE — near Brownian motion"
    else:
        interp = "SMOOTH — H ≥ 0.5, not in the rough regime"

    lines.append(f"  Interpretation: {interp}")
    lines.append(f"{'='*80}\n")

    return "\n".join(lines)


def save_results_json(result: MultiScaleResult, output_path: str | Path) -> None:
    """Save MultiScaleResult as a JSON file for later analysis."""
    import json

    data = {
        "asset": result.asset,
        "consensus_H": float(result.consensus_H),
        "consensus_std": float(result.consensus_std),
        "scales": {},
    }

    freq_labels = list(result.rv_series.keys())
    for method, estimates in result.estimates.items():
        for i, est in enumerate(estimates):
            freq = freq_labels[i] if i < len(freq_labels) else f"scale_{i}"
            key = f"{method}_{freq}"
            data["scales"][key] = {
                "method": method,
                "frequency": freq,
                "n_obs": int(est.n_obs),
                "H_point": float(est.H_point),
                "H_mean": float(est.H_mean),
                "H_std": float(est.H_std),
                "ci_lower": float(est.ci_lower),
                "ci_upper": float(est.ci_upper),
                "r_squared": float(est.r_squared),
                "n_bootstrap": int(est.n_bootstrap),
            }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
