"""
Hurst exponent estimation for rough volatility.
Variogram and DMA methods per Gatheral et al. (2018).
"""

import numpy as np
import pandas as pd


def estimate_hurst_variogram(returns: np.ndarray, max_lag: int = 100) -> tuple:
    """
    Estimate Hurst exponent using variogram method (preferred for rough volatility).
    For fBM: E[|X(t+h) - X(t)|^2] ~ h^(2H)
    """
    max_possible_lag = min(max_lag, len(returns) // 10)
    if max_possible_lag < 5:
        return np.nan, np.array([]), np.array([])

    lags = np.arange(1, max_possible_lag)
    variogram = []
    for lag in lags:
        increments = returns[lag:] - returns[:-lag]
        variogram.append(np.mean(increments**2))
    variogram = np.array(variogram)

    log_lags = np.log(lags)
    log_var = np.log(variogram + 1e-10)
    slope, _ = np.polyfit(log_lags, log_var, 1)
    H = slope / 2
    return H, lags, variogram


def estimate_hurst_dma(returns: np.ndarray, min_scale: int = 10, max_scale: int = 500) -> tuple:
    """
    Detrended Moving Average (DMA) method for Hurst estimation.
    More robust for non-stationary financial data.
    """
    scales = np.logspace(
        np.log10(min_scale), np.log10(min(max_scale, len(returns) // 4)), 20
    ).astype(int)
    scales = np.unique(scales)
    fluctuations = []
    cumsum = np.cumsum(returns - np.mean(returns))

    for scale in scales:
        ma = pd.Series(cumsum).rolling(window=scale, center=True).mean().values
        valid = ~np.isnan(ma)
        if np.sum(valid) < scale:
            continue
        detrended = cumsum[valid] - ma[valid]
        F = np.sqrt(np.mean(detrended**2))
        fluctuations.append((scale, F))

    if len(fluctuations) < 5:
        return np.nan, [], []

    scales_used = np.array([f[0] for f in fluctuations])
    F_values = np.array([f[1] for f in fluctuations])
    slope, _ = np.polyfit(np.log(scales_used), np.log(F_values + 1e-10), 1)
    H = slope - 0.5  # Correct for cumulative sum integration
    return H, scales_used, F_values


def compute_realized_volatility(df: pd.DataFrame, window_points: int = None) -> pd.Series:
    """Compute realized volatility over rolling windows."""
    time_diff = (df.index[1] - df.index[0]).total_seconds() / 60
    if window_points is None:
        window_points = max(6, int(60 / time_diff))
    annual_factor = np.sqrt(252 * 6.5 * 60 / time_diff)
    rv = df["log_return"].rolling(window=window_points).std() * annual_factor
    return rv.dropna()
