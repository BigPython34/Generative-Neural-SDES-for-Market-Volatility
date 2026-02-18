import sys as _sys
if _sys.stdout.encoding != 'utf-8':
    _sys.stdout.reconfigure(encoding='utf-8'); _sys.stderr.reconfigure(encoding='utf-8')

"""
Hurst Diagnostic: VIX vs Realized Volatility
=============================================
Demonstrates WHY H must be measured on realized vol (from returns),
NOT on VIX (which is a 30-day integrated/smoothed quantity).

References:
- Gatheral, Jaisson, Rosenbaum (2018): "Volatility is Rough"
- Key finding: H ~ 0.05-0.14 on realized volatility
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
from scipy.stats import linregress
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------
# 1. Hurst Estimators
# ---------------------------------------------------------

def hurst_variogram(series, max_lag=None):
    """
    Variogram-based Hurst estimator (most reliable for rough vol).
    
    E[|X_{t+τ} - X_t|^2] ~ τ^{2H}
    
    => log(variogram) = 2H * log(τ) + const
    """
    series = np.asarray(series, dtype=float)
    n = len(series)
    if max_lag is None:
        max_lag = min(50, n // 4)
    
    lags = np.arange(1, max_lag + 1)
    variogram = np.zeros(len(lags))
    
    for i, lag in enumerate(lags):
        diffs = series[lag:] - series[:-lag]
        variogram[i] = np.mean(diffs ** 2)
    
    # Log-log regression
    log_lags = np.log(lags)
    log_var = np.log(variogram + 1e-30)
    
    slope, intercept, r_value, _, _ = linregress(log_lags, log_var)
    H = slope / 2.0
    
    return H, r_value**2


def hurst_structure_function(series, order=1, max_lag=None):
    """
    Structure function (order q) Hurst estimator.
    
    E[|X_{t+τ} - X_t|^q] ~ τ^{qH}
    
    Order 1 is more robust to outliers than order 2 (variogram).
    """
    series = np.asarray(series, dtype=float)
    n = len(series)
    if max_lag is None:
        max_lag = min(50, n // 4)
    
    lags = np.arange(1, max_lag + 1)
    moments = np.zeros(len(lags))
    
    for i, lag in enumerate(lags):
        diffs = np.abs(series[lag:] - series[:-lag])
        moments[i] = np.mean(diffs ** order)
    
    log_lags = np.log(lags)
    log_mom = np.log(moments + 1e-30)
    
    slope, intercept, r_value, _, _ = linregress(log_lags, log_mom)
    H = slope / order
    
    return H, r_value**2


def hurst_dma(series, max_lag=None):
    """
    Detrended Moving Average (DMA) Hurst estimator.
    More robust to trends than variogram.
    """
    series = np.asarray(series, dtype=float)
    n = len(series)
    if max_lag is None:
        max_lag = min(50, n // 4)
    
    lags = np.arange(2, max_lag + 1)
    sigma_dma = np.zeros(len(lags))
    
    cumsum = np.cumsum(series)
    
    for i, lag in enumerate(lags):
        # Moving average of cumulative sum
        ma = np.convolve(cumsum, np.ones(lag) / lag, mode='valid')
        # DMA variance
        overlap = min(len(cumsum), len(ma))
        diffs = cumsum[lag-1:lag-1+len(ma)] - ma
        sigma_dma[i] = np.sqrt(np.mean(diffs ** 2))
    
    log_lags = np.log(lags)
    log_sigma = np.log(sigma_dma + 1e-30)
    
    slope, intercept, r_value, _, _ = linregress(log_lags, log_sigma)
    H = slope - 0.5  # DMA correction
    
    return H, r_value**2


# ---------------------------------------------------------
# 2. Daily Realized Variance from High-Frequency Returns
# ---------------------------------------------------------

def compute_daily_rv(df):
    """
    Compute daily Realized Variance from intraday price data.
    
    Standard approach (Gatheral et al. 2018):
    RV_day = Σ r²_i  where r_i are intraday log-returns within that day.
    
    Returns:
        Array of daily RV values (one per trading day)
    """
    df = df.copy()
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['datetime'] = pd.to_datetime(df['time'], unit='s')
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Detect overnight/weekend gaps → one segment per trading day
    df['time_delta'] = df['datetime'].diff()
    max_gap = pd.Timedelta(hours=4)
    df['segment_id'] = (df['time_delta'] > max_gap).cumsum()
    
    # Log returns (only within segments)
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    boundaries = df['segment_id'] != df['segment_id'].shift(1)
    df.loc[boundaries, 'log_return'] = np.nan
    df = df.dropna(subset=['log_return'])
    
    # One RV per trading day
    daily_rv = []
    for seg_id in df['segment_id'].unique():
        seg = df[df['segment_id'] == seg_id]
        returns = seg['log_return'].values
        if len(returns) < 10:
            continue
        rv_val = np.sum(returns**2)
        if rv_val > 1e-14:
            daily_rv.append(rv_val)
    
    return np.array(daily_rv)


# ---------------------------------------------------------
# 3. Main Diagnostic
# ---------------------------------------------------------

def run_diagnostic():
    print("=" * 70)
    print("   HURST EXPONENT DIAGNOSTIC: VIX vs REALIZED VOLATILITY")
    print("   Gatheral, Jaisson, Rosenbaum (2018): 'Volatility is Rough'")
    print("=" * 70)
    
    results = {}
    
    # -- A. VIX (smoothed, integrated) --------------------------
    print("\n" + "-" * 60)
    print("A. VIX Index (30-day implied vol — SMOOTHED quantity)")
    print("-" * 60)
    
    for freq, fname in [("5min", "data/TVC_VIX, 5.csv"), 
                        ("15min", "data/TVC_VIX, 15.csv"),
                        ("30min", "data/TVC_VIX, 30.csv")]:
        if not Path(fname).exists():
            print(f"   [{freq}] File not found: {fname}")
            continue
            
        df = pd.read_csv(fname)
        vix_close = pd.to_numeric(df['close'], errors='coerce').dropna().values
        
        # Work on log(VIX) — standard in the literature
        log_vix = np.log(vix_close)
        
        h_var, r2_var = hurst_variogram(log_vix)
        h_sf1, r2_sf1 = hurst_structure_function(log_vix, order=1)
        h_sf2, r2_sf2 = hurst_structure_function(log_vix, order=2)
        
        print(f"\n   [{freq}] VIX — {len(vix_close)} points")
        print(f"   Variogram H     = {h_var:.4f}  (R² = {r2_var:.3f})")
        print(f"   Structure q=1 H = {h_sf1:.4f}  (R² = {r2_sf1:.3f})")
        print(f"   Structure q=2 H = {h_sf2:.4f}  (R² = {r2_sf2:.3f})")
        print(f"   [WARN] Expected: H >> 0.15 because VIX integrates over 30 days!")
        
        results[f"VIX_{freq}"] = {"H_variogram": h_var, "H_sf1": h_sf1, "H_sf2": h_sf2}
    
    # -- B. Realized Vol from SPX Returns (TRUE roughness) ------
    print("\n" + "-" * 60)
    print("B. REALIZED VOLATILITY from S&P 500 returns (TRUE roughness)")
    print("   Method: Daily RV = Σ r²_intraday per trading day")
    print("   Hurst estimated on log(RV) time series")
    print("-" * 60)
    
    for freq, fname in [
        ("5min",  "data/SP_SPX, 5.csv"),
        ("30min", "data/SP_SPX, 30.csv"),
    ]:
        if not Path(fname).exists():
            print(f"   [{freq}] File not found: {fname}")
            continue
            
        df = pd.read_csv(fname)
        print(f"\n   [{freq}] SPX data: {len(df)} bars")
        
        # Compute DAILY RV (1 point per trading day)
        rv_series = compute_daily_rv(df)
        
        if len(rv_series) < 50:
            print(f"   [{freq}] Too few daily RV points ({len(rv_series)})")
            continue
        
        # Hurst on log(RV) — this is where roughness lives
        log_rv = np.log(rv_series)
        
        h_var, r2_var = hurst_variogram(log_rv)
        h_sf1, r2_sf1 = hurst_structure_function(log_rv, order=1)
        h_sf2, r2_sf2 = hurst_structure_function(log_rv, order=2)
        
        print(f"\n   [{freq}] Daily RV — {len(rv_series)} trading days")
        print(f"   Variogram H     = {h_var:.4f}  (R² = {r2_var:.3f})")
        print(f"   Structure q=1 H = {h_sf1:.4f}  (R² = {r2_sf1:.3f})")
        print(f"   Structure q=2 H = {h_sf2:.4f}  (R² = {r2_sf2:.3f})")
        
        if h_var < 0.2:
            print(f"   [OK] ROUGH! H < 0.2 - consistent with Gatheral et al.")
        elif h_var < 0.35:
            print(f"   [WARN] Slightly rough (H < 0.35)")
        else:
            print(f"   [FAIL] NOT rough (H >= 0.35) - check data/methodology")
        
        results[f"RV_{freq}_daily"] = {
            "H_variogram": h_var, "H_sf1": h_sf1, "H_sf2": h_sf2,
            "n_days": len(rv_series)
        }
    
    # -- C. Robustness: Hurst on sub-sampled daily RV ---------
    print("\n" + "-" * 60)
    print("C. ROBUSTNESS: Different lag ranges for variogram")
    print("-" * 60)
    print("   Testing H stability across different max lags")
    
    # Use 30-min SPX data (most points)
    fname_30m = "data/SP_SPX, 30.csv"
    if Path(fname_30m).exists():
        df30 = pd.read_csv(fname_30m)
        rv_daily = compute_daily_rv(df30)
        log_rv = np.log(rv_daily)
        
        for max_lag in [10, 20, 30, 50]:
            if max_lag >= len(log_rv) // 4:
                continue
            h_var, r2 = hurst_variogram(log_rv, max_lag=max_lag)
            h_sf1, _ = hurst_structure_function(log_rv, order=1, max_lag=max_lag)
            print(f"   max_lag={max_lag:>3d}: H_var={h_var:.4f}, H_sf1={h_sf1:.4f}  (R²={r2:.3f})")
    
    # -- D. Comparison with Neural SDE output ------------------
    print("\n" + "-" * 60)
    print("D. NEURAL SDE OUTPUT — Hurst on generated paths")
    print("-" * 60)
    
    try:
        import jax
        import jax.numpy as jnp
        import equinox as eqx
        from engine.neural_sde import NeuralRoughSimulator
        from engine.signature_engine import SignatureFeatureExtractor
        import yaml
        
        with open("config/params.yaml", 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        
        model_path = Path("models/neural_sde_best.eqx")
        if model_path.exists():
            sig_dim = SignatureFeatureExtractor(truncation_order=3).get_feature_dim(1)
            key = jax.random.PRNGKey(42)
            model = NeuralRoughSimulator(sig_dim=sig_dim, key=key)
            model = eqx.tree_deserialise_leaves(model_path, model)
            
            T = cfg['simulation']['T']
            n_steps = cfg['simulation']['n_steps']
            dt = T / n_steps
            
            # Generate many paths
            n_gen = 2000
            noise = jax.random.normal(key, (n_gen, n_steps)) * jnp.sqrt(dt)
            v0 = jnp.full(n_gen, 0.035)  # ~18.7% vol
            
            gen_paths = jax.vmap(model.generate_variance_path, in_axes=(0, 0, None))(
                v0, noise, dt
            )
            gen_np = np.array(gen_paths)
            
            # Hurst on log(V) of generated paths (averaged over paths)
            log_gen = np.log(np.maximum(gen_np, 1e-8))
            
            # Average variogram across paths
            max_lag = min(8, n_steps // 3)
            lags = np.arange(1, max_lag + 1)
            variogram = np.zeros(len(lags))
            for i, lag in enumerate(lags):
                diffs = log_gen[:, lag:] - log_gen[:, :-lag]
                variogram[i] = np.mean(diffs ** 2)
            
            log_lags = np.log(lags)
            log_var = np.log(variogram + 1e-30)
            slope, _, r_value, _, _ = linregress(log_lags, log_var)
            H_neural = slope / 2
            
            print(f"   Neural SDE H (variogram on log-V) = {H_neural:.4f}  (R² = {r_value**2:.3f})")
            print(f"   Note: Only {n_steps} steps — limited resolution for Hurst estimation")
        else:
            print("   No trained model found — skipping")
    except Exception as e:
        print(f"   Error: {e}")
    
    # -- E. Summary ---------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
    THEORY (Gatheral et al. 2018):
    - Realized Volatility from high-freq returns: H ≈ 0.05 – 0.14
    - VIX = 30-day integral of risk-neutral variance
      → Integration SMOOTHS the process → H_VIX >> H_RV
    - VIX roughness ≠ true volatility roughness
    
    IMPLICATIONS FOR THIS PROJECT:
    - Training on VIX is fine for PRICING (Q-measure calibration)
    - But roughness verification must use REALIZED VOL from SPX returns
    - The Neural SDE should reproduce rough paths if fed rough RV data
    """)
    
    return results


if __name__ == "__main__":
    results = run_diagnostic()
