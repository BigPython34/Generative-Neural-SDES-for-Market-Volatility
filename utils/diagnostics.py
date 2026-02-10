import numpy as np
from scipy.stats import kurtosis, linregress

def compute_acf(x, lags=10):
    """Computes Auto-Correlation Function."""
    n = len(x)
    if n <= lags + 1: 
        # Path too short, return what we can compute
        lags = max(1, n - 2)
    mean = np.mean(x)
    var = np.var(x)
    if var == 0: return np.zeros(lags)
    xp = x - mean
    corr = np.correlate(xp, xp, mode='full')[n-1:] / (var * n)
    return corr[:lags]

def estimate_hurst(data):
    """
    Estimates the Hurst Exponent (H) via the Structure Function Method (Order 1).
    Theory: E[|log(vol)_{t+tau} - log(vol)_t|] ~ tau^H
    Target for Rough Volatility: H < 0.15
    """
    # 1. Convert Variance to Log-Volatility: X_t = log(sqrt(V_t))
    # Add epsilon to avoid log(0)
    log_vol = np.log(np.sqrt(np.maximum(data, 1e-8)))
    
    # 2. Compute increments for different lags (tau = 1 to 10)
    taus = range(1, 11)
    lag_moments = []
    
    for tau in taus:
        # Compute mean absolute difference for this lag across all paths
        # Vectorized over (n_paths, n_steps)
        diffs = np.abs(log_vol[:, tau:] - log_vol[:, :-tau])
        lag_moments.append(np.mean(diffs))
        
    # 3. Log-Log Regression
    # log(Moment) = H * log(tau) + C
    x_reg = np.log(taus)
    y_reg = np.log(lag_moments)
    
    slope, _, _, _, _ = linregress(x_reg, y_reg)
    return slope

def print_distribution_stats(name: str, data: np.ndarray):
    """
    Prints comprehensive stats including Hurst and Kurtosis.
    """
    flat_data = data.flatten()
    n_steps = data.shape[1] if data.ndim > 1 else len(data)
    
    # Compute metrics
    h_est = estimate_hurst(data)
    kurt = kurtosis(flat_data)
    
    # Sample ACF on first 100 paths (use appropriate lag for path length)
    max_lag = min(10, n_steps - 2)
    acf_sample = np.mean([compute_acf(p, lags=max_lag) for p in data[:100]], axis=0)
    acf_lag1 = acf_sample[1] if len(acf_sample) > 1 else 0.0
    
    print(f"--- DIAGNOSTIC: {name} ---")
    print(f"   Mean           : {np.mean(flat_data):.5f}")
    print(f"   Median         : {np.median(flat_data):.5f}")
    print(f"   Kurtosis       : {kurt:.2f} (Target > 3.0)")
    print(f"   Hurst (Rough)  : {h_est:.3f} (Target < 0.15)")
    print(f"   ACF (Lag 1)    : {acf_lag1:.3f} (Target > 0.9)")
    print("-" * 30)
