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

def estimate_hurst(data, method='variogram'):
    """
    Estimates the Hurst Exponent (H) via the variogram method on log-variance.
    
    Theory: E[|X_{t+τ} - X_t|^2] ~ τ^{2H}
    where X_t = log(V_t).
    
    Target for Rough Volatility: H < 0.15 (on REALIZED vol, NOT VIX).
    
    IMPORTANT: If data is VIX-derived variance paths, H ≈ 0.5 is expected
    because VIX integrates over 30 days (smoothing effect).
    True roughness (H ≈ 0.1) is only visible on realized vol from returns.
    
    Args:
        data: (n_paths, n_steps) variance paths
        method: 'variogram' (order 2) or 'structure' (order 1, more robust)
    
    Returns:
        float: Estimated Hurst exponent
    """
    # 1. Convert Variance to Log-Volatility: X_t = log(sqrt(V_t))
    log_vol = np.log(np.sqrt(np.maximum(data, 1e-8)))
    
    # 2. Compute variogram for different lags
    max_lag = min(10, data.shape[1] // 3) if data.ndim > 1 else min(10, len(data) // 3)
    taus = range(1, max_lag + 1)
    lag_moments = []
    
    for tau in taus:
        if data.ndim > 1:
            # Vectorized over (n_paths, n_steps)
            diffs = log_vol[:, tau:] - log_vol[:, :-tau]
        else:
            diffs = log_vol[tau:] - log_vol[:-tau]
        
        if method == 'variogram':
            lag_moments.append(np.mean(diffs**2))
        else:  # structure function order 1
            lag_moments.append(np.mean(np.abs(diffs)))
    
    # 3. Log-Log Regression
    x_reg = np.log(list(taus))
    y_reg = np.log(lag_moments)
    
    slope, _, r_value, _, _ = linregress(x_reg, y_reg)
    
    if method == 'variogram':
        return slope / 2.0  # variogram: slope = 2H
    else:
        return slope  # structure function q=1: slope = H


def estimate_hurst_from_returns(prices_or_path, rv_window=78, max_lag=50):
    """
    Estimate the TRUE Hurst exponent from price returns via Realized Volatility.
    
    This is the CORRECT way to detect roughness (Gatheral et al. 2018).
    
    Steps:
        1. Compute log-returns from prices
        2. Compute rolling Realized Variance: RV_t = Σ r²_i
        3. Take log(RV) and estimate H via variogram
    
    Expected result: H ≈ 0.05 - 0.14 for equity markets.
    
    Args:
        prices_or_path: Either a numpy array of prices or a file path string.
        rv_window: Rolling window for RV (78 = 1 day for 5-min, 13 for 30-min)
        max_lag: Maximum lag for variogram
        
    Returns:
        dict with H estimates and diagnostics
    """
    import pandas as pd
    
    if isinstance(prices_or_path, str):
        df = pd.read_csv(prices_or_path)
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['datetime'] = pd.to_datetime(df['time'], unit='s')
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Detect overnight/weekend gaps and assign one segment per trading day
        df['time_delta'] = df['datetime'].diff()
        max_gap = pd.Timedelta(hours=4)
        df['segment_id'] = (df['time_delta'] > max_gap).cumsum()
        
        # Compute log-returns (only within segments — no overnight jumps)
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        boundaries = df['segment_id'] != df['segment_id'].shift(1)
        df.loc[boundaries, 'log_return'] = np.nan
        df = df.dropna(subset=['log_return', 'close'])
        
        # Compute DAILY Realized Variance: one RV point per trading day
        # This is the standard approach in Gatheral et al. (2018)
        daily_rv = []
        for seg_id in df['segment_id'].unique():
            seg = df[df['segment_id'] == seg_id]
            returns = seg['log_return'].values
            if len(returns) < 10:  # need at least 10 bars for meaningful RV
                continue
            rv_val = np.sum(returns**2)  # Daily RV = sum of squared intraday returns
            if rv_val > 1e-14:
                daily_rv.append(rv_val)
        
        rv = np.array(daily_rv)
        
        if len(rv) < 50:
            return {
                'H_variogram': float('nan'), 'H_structure': float('nan'),
                'R2_variogram': 0.0, 'R2_structure': 0.0,
                'n_rv_points': len(rv), 'rv_window': 'daily'
            }
    else:
        prices = np.asarray(prices_or_path, dtype=float)
        log_returns = np.diff(np.log(prices))
        rv = np.array([
            np.sum(log_returns[i:i+rv_window]**2) 
            for i in range(len(log_returns) - rv_window)
        ])
        rv = rv[rv > 1e-12]
    
    if len(rv) < 50:
        return {
            'H_variogram': float('nan'), 'H_structure': float('nan'),
            'R2_variogram': 0.0, 'R2_structure': 0.0,
            'n_rv_points': len(rv), 'rv_window': rv_window
        }
    
    # 3. Hurst on log(RV)
    log_rv = np.log(rv)
    
    lags = np.arange(1, min(max_lag, len(log_rv) // 4) + 1)
    variogram = np.zeros(len(lags))
    moments_q1 = np.zeros(len(lags))
    
    for i, lag in enumerate(lags):
        diffs = log_rv[lag:] - log_rv[:-lag]
        variogram[i] = np.mean(diffs**2)
        moments_q1[i] = np.mean(np.abs(diffs))
    
    log_lags = np.log(lags)
    
    # Variogram: slope = 2H
    slope_v, _, r_v, _, _ = linregress(log_lags, np.log(variogram + 1e-30))
    H_variogram = slope_v / 2.0
    
    # Structure function q=1: slope = H (more robust to outliers)
    slope_s, _, r_s, _, _ = linregress(log_lags, np.log(moments_q1 + 1e-30))
    H_structure = slope_s
    
    return {
        'H_variogram': H_variogram,
        'H_structure': H_structure,
        'R2_variogram': r_v**2,
        'R2_structure': r_s**2,
        'n_rv_points': len(rv),
        'rv_window': rv_window
    }

def print_distribution_stats(name: str, data: np.ndarray, is_vix_derived: bool = True):
    """
    Prints comprehensive stats including Hurst and Kurtosis.
    
    Kurtosis is computed as the MARGINAL kurtosis: for each time step t,
    compute the excess kurtosis of {V_t^(i)} across paths i, then average
    over t. This avoids the inflation caused by flattening paths at
    different VIX levels into a single array.
    
    Args:
        name: Label for the data source
        data: (n_paths, n_steps) variance paths
        is_vix_derived: If True, warns that H ≈ 0.5 is expected (VIX is smoothed).
                        Set to False when using realized vol from SPX returns.
    """
    flat_data = data.flatten()
    n_steps = data.shape[1] if data.ndim > 1 else len(data)
    
    # Compute metrics
    h_est = estimate_hurst(data, method='variogram')
    
    # Marginal kurtosis: average excess kurtosis across time steps
    if data.ndim > 1:
        marginal_kurts = [kurtosis(data[:, t]) for t in range(n_steps)]
        kurt = np.mean(marginal_kurts)
    else:
        kurt = kurtosis(data)
    
    # Sample ACF on first 100 paths
    max_lag = min(10, n_steps - 2)
    acf_sample = np.mean([compute_acf(p, lags=max_lag) for p in data[:100]], axis=0)
    acf_lag1 = acf_sample[1] if len(acf_sample) > 1 else 0.0
    
    print(f"--- DIAGNOSTIC: {name} ---")
    print(f"   Mean           : {np.mean(flat_data):.5f}")
    print(f"   Median         : {np.median(flat_data):.5f}")
    print(f"   Kurtosis (marg): {kurt:.2f} (excess, averaged over time steps)")
    print(f"   Hurst (paths)  : {h_est:.3f}", end="")
    if is_vix_derived:
        print(f"  [VIX-derived -> expect ~0.5, NOT a roughness test]")
        print(f"   ℹ️  True roughness (H~0.1) must be measured on realized vol from SPX returns")
    else:
        target = "< 0.15" 
        status = "✅" if h_est < 0.20 else "⚠️"
        print(f"  {status} (Target {target} for rough vol)")
    print(f"   ACF (Lag 1)    : {acf_lag1:.3f} (Target > 0.9)")
    print("-" * 30)
