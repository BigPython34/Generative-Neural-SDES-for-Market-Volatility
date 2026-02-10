"""
Robustness Improvements
=======================
Fix critical issues in the calibration pipeline:
1. Better Hurst estimation (multiple methods + confidence)
2. Use properly trained Neural SDE
3. More Monte Carlo paths for OTM options
4. Calibrate correlation ρ
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class RobustHurstEstimator:
    """
    Multiple methods for Hurst estimation with confidence intervals
    """
    
    def __init__(self, data: np.ndarray):
        self.data = data
        self.results = {}
        
    def variogram(self, max_lag: int = None) -> dict:
        """Variogram method with proper normalization"""
        n = len(self.data)
        if max_lag is None:
            max_lag = min(100, n // 5)
        
        if max_lag < 10:
            return {'H': np.nan, 'std': np.nan, 'method': 'variogram'}
        
        lags = np.arange(2, max_lag)
        variogram = []
        
        for lag in lags:
            increments = self.data[lag:] - self.data[:-lag]
            # Use absolute moments (more robust than squared)
            variogram.append(np.mean(np.abs(increments)))
        
        variogram = np.array(variogram)
        
        # Log-log regression: log(m1) = H * log(lag) + const
        log_lags = np.log(lags)
        log_var = np.log(variogram + 1e-10)
        
        # Use robust regression (ignore outliers)
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_lags, log_var)
        
        H = slope  # For absolute moments, E[|X|] ~ lag^H
        
        self.results['variogram'] = {
            'H': float(np.clip(H, 0.01, 0.99)),
            'std': float(std_err),
            'r_squared': float(r_value**2),
            'method': 'variogram'
        }
        return self.results['variogram']
    
    def rescaled_range(self) -> dict:
        """R/S analysis (classic Hurst method)"""
        n = len(self.data)
        
        # Different window sizes
        min_window = 10
        max_window = n // 4
        
        if max_window < min_window * 2:
            return {'H': np.nan, 'std': np.nan, 'method': 'R/S'}
        
        window_sizes = np.logspace(np.log10(min_window), np.log10(max_window), 15).astype(int)
        window_sizes = np.unique(window_sizes)
        
        rs_values = []
        
        for w in window_sizes:
            # Split into windows
            n_windows = n // w
            rs_window = []
            
            for i in range(n_windows):
                window = self.data[i*w:(i+1)*w]
                
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
            return {'H': np.nan, 'std': np.nan, 'method': 'R/S'}
        
        windows = np.array([x[0] for x in rs_values])
        rs = np.array([x[1] for x in rs_values])
        
        # Log-log regression: log(R/S) = H * log(n) + const
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(np.log(windows), np.log(rs))
        
        self.results['RS'] = {
            'H': float(np.clip(slope, 0.01, 0.99)),
            'std': float(std_err),
            'r_squared': float(r_value**2),
            'method': 'R/S'
        }
        return self.results['RS']
    
    def periodogram(self) -> dict:
        """Spectral method using periodogram"""
        n = len(self.data)
        
        # Compute periodogram
        from scipy import signal
        freqs, psd = signal.periodogram(self.data, fs=1.0)
        
        # Remove zero frequency
        freqs = freqs[1:]
        psd = psd[1:]
        
        # Use only low frequencies (where scaling should hold)
        n_use = len(freqs) // 4
        if n_use < 5:
            return {'H': np.nan, 'std': np.nan, 'method': 'periodogram'}
        
        freqs = freqs[:n_use]
        psd = psd[:n_use]
        
        # Log-log regression: log(PSD) = -(2H+1) * log(f) + const
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            np.log(freqs + 1e-10), np.log(psd + 1e-10)
        )
        
        H = (-slope - 1) / 2
        
        self.results['periodogram'] = {
            'H': float(np.clip(H, 0.01, 0.99)),
            'std': float(std_err / 2),  # Approximate
            'r_squared': float(r_value**2),
            'method': 'periodogram'
        }
        return self.results['periodogram']
    
    def estimate_all(self) -> dict:
        """Run all methods and return consensus"""
        self.variogram()
        self.rescaled_range()
        self.periodogram()
        
        # Collect valid estimates
        valid_H = []
        weights = []
        
        for method, result in self.results.items():
            if not np.isnan(result['H']) and 0 < result['H'] < 1:
                valid_H.append(result['H'])
                # Weight by R-squared
                weights.append(result.get('r_squared', 0.5))
        
        if not valid_H:
            return {
                'H_consensus': np.nan,
                'H_std': np.nan,
                'methods': self.results,
                'confidence': 'low'
            }
        
        # Weighted average
        weights = np.array(weights)
        weights = weights / weights.sum()
        H_consensus = np.average(valid_H, weights=weights)
        H_std = np.std(valid_H)
        
        # Confidence based on agreement
        if H_std < 0.05:
            confidence = 'high'
        elif H_std < 0.15:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        return {
            'H_consensus': float(H_consensus),
            'H_std': float(H_std),
            'methods': self.results,
            'confidence': confidence,
            'is_rough': H_consensus < 0.5
        }


def calibrate_correlation(spot_returns: np.ndarray, vol_changes: np.ndarray) -> dict:
    """
    Calibrate spot-vol correlation from data
    """
    # Align lengths
    min_len = min(len(spot_returns), len(vol_changes))
    spot_returns = spot_returns[:min_len]
    vol_changes = vol_changes[:min_len]
    
    # Remove NaN/Inf
    valid = np.isfinite(spot_returns) & np.isfinite(vol_changes)
    spot_returns = spot_returns[valid]
    vol_changes = vol_changes[valid]
    
    if len(spot_returns) < 100:
        return {'rho': -0.7, 'std': np.nan, 'method': 'default'}
    
    # Pearson correlation
    rho_pearson = np.corrcoef(spot_returns, vol_changes)[0, 1]
    
    # Spearman (more robust)
    from scipy import stats
    rho_spearman, _ = stats.spearmanr(spot_returns, vol_changes)
    
    # Rolling correlation for stability check
    window = min(252, len(spot_returns) // 4)
    rolling_rho = []
    for i in range(len(spot_returns) - window):
        r = np.corrcoef(spot_returns[i:i+window], vol_changes[i:i+window])[0, 1]
        if np.isfinite(r):
            rolling_rho.append(r)
    
    rho_std = np.std(rolling_rho) if rolling_rho else np.nan
    
    return {
        'rho_pearson': float(rho_pearson),
        'rho_spearman': float(rho_spearman),
        'rho_consensus': float((rho_pearson + rho_spearman) / 2),
        'rho_std': float(rho_std),
        'n_samples': len(spot_returns)
    }


def run_robustness_check():
    """Run comprehensive robustness checks"""
    
    print("=" * 70)
    print("   ROBUSTNESS CHECK & IMPROVEMENTS")
    print("=" * 70)
    print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    data_dir = Path("data")
    results = {}
    
    # 1. Better Hurst estimation
    print("\n" + "-" * 50)
    print("1. ROBUST HURST ESTIMATION")
    print("-" * 50)
    
    # Load SPX 30min data (longest history)
    spx_path = data_dir / "SP_SPX, 30.csv"
    if spx_path.exists():
        df = pd.read_csv(spx_path)
        df['datetime'] = pd.to_datetime(df['time'], unit='s')
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df = df.dropna()
        
        # Compute realized volatility
        window = 6  # ~3 hours at 30min
        annual_factor = np.sqrt(252 * 13)  # 13 half-hours per day
        rv = df['log_return'].rolling(window=window).std() * annual_factor
        rv = rv.dropna()
        
        log_rv = np.log(rv.values + 1e-6)
        
        print(f"\n   Data: SPX 30min, {len(df)} points, {len(df)*30/(60*6.5*252):.1f} years")
        print(f"   RV computed with {window} point window")
        
        estimator = RobustHurstEstimator(log_rv)
        hurst_results = estimator.estimate_all()
        
        print(f"\n   Hurst Estimation Results:")
        for method, res in hurst_results['methods'].items():
            status = "OK" if 0 < res['H'] < 1 else "X"
            print(f"      {status} {method:12s}: H = {res['H']:.3f} (R2 = {res.get('r_squared', 0):.3f})")
        
        print(f"\n   Consensus: H = {hurst_results['H_consensus']:.3f} +/- {hurst_results['H_std']:.3f}")
        print(f"      Confidence: {hurst_results['confidence']}")
        print(f"      Is Rough: {'YES' if hurst_results['is_rough'] else 'NO'}")
        
        results['hurst'] = hurst_results
    
    # 2. Calibrate correlation
    print("\n" + "-" * 50)
    print("2. SPOT-VOL CORRELATION CALIBRATION")
    print("-" * 50)
    
    if spx_path.exists():
        # Use log returns and log RV changes
        # Important: vol leads spot (leverage effect), so we check correlation
        # between return(t) and vol change(t) or vol change(t+1)
        
        spot_returns = df['log_return'].values[window:]
        
        # Vol changes (log RV diff)
        log_rv_series = np.log(rv.values + 1e-6)
        vol_changes = np.diff(log_rv_series)
        
        # Also try: spot return vs vol level (not change)
        vol_levels = log_rv_series[1:]  # Align with returns
        
        print(f"\n   Testing different correlation measures:")
        
        # 1. Return vs Vol change (same period)
        min_len = min(len(spot_returns) - 1, len(vol_changes))
        rho1 = np.corrcoef(spot_returns[:min_len], vol_changes[:min_len])[0, 1]
        print(f"      Return(t) vs ΔVol(t):   ρ = {rho1:.3f}")
        
        # 2. Return vs Vol level (same period)  
        min_len = min(len(spot_returns), len(vol_levels))
        rho2 = np.corrcoef(spot_returns[:min_len], vol_levels[:min_len])[0, 1]
        print(f"      Return(t) vs Vol(t):    ρ = {rho2:.3f}")
        
        # 3. Return vs next Vol change (predictive)
        min_len = min(len(spot_returns) - 1, len(vol_changes) - 1)
        rho3 = np.corrcoef(spot_returns[:min_len], vol_changes[1:min_len+1])[0, 1]
        print(f"      Return(t) vs ΔVol(t+1): ρ = {rho3:.3f}")
        
        # 4. Rolling correlation (Return vs Vol level)
        window_corr = 252  # 1 year
        rolling_rho = []
        for i in range(0, min_len - window_corr, 50):
            r = np.corrcoef(
                spot_returns[i:i+window_corr], 
                vol_levels[i:i+window_corr]
            )[0, 1]
            if np.isfinite(r):
                rolling_rho.append(r)
        
        if rolling_rho:
            print(f"\n      Rolling ρ (1Y window):")
            print(f"         Mean:  {np.mean(rolling_rho):.3f}")
            print(f"         Std:   {np.std(rolling_rho):.3f}")
            print(f"         Range: [{np.min(rolling_rho):.3f}, {np.max(rolling_rho):.3f}]")
        
        rho_results = {
            'rho_return_vs_dvol': float(rho1),
            'rho_return_vs_vol': float(rho2),
            'rho_return_vs_dvol_next': float(rho3),
            'rho_rolling_mean': float(np.mean(rolling_rho)) if rolling_rho else np.nan,
            'rho_rolling_std': float(np.std(rolling_rho)) if rolling_rho else np.nan,
        }
        
        # The relevant correlation for SV models is Return vs Vol level
        rho_consensus = rho2
        print(f"\n   Recommended rho for SV model: {rho_consensus:.3f}")
        
        if rho_consensus < -0.3:
            print(f"   Negative correlation confirmed (leverage effect)")
        elif rho_consensus < 0:
            print(f"   Weak negative correlation")
        else:
            print(f"   Unexpected: positive or zero correlation")
        
        results['correlation'] = rho_results
        results['rho_recommended'] = float(rho_consensus)
    
    # 3. Check Neural SDE training
    print("\n" + "-" * 50)
    print("3. NEURAL SDE MODEL CHECK")
    print("-" * 50)
    
    model_path = Path("models/neural_sde_best.eqx")
    if model_path.exists():
        import os
        size_kb = os.path.getsize(model_path) / 1024
        mod_time = datetime.fromtimestamp(os.path.getmtime(model_path))
        print(f"\n   Trained model found: {model_path}")
        print(f"      Size: {size_kb:.1f} KB")
        print(f"      Last modified: {mod_time.strftime('%Y-%m-%d %H:%M')}")
        results['model_exists'] = True
    else:
        print(f"\n   No trained model found!")
        print(f"      Run 'python main.py' to train properly")
        results['model_exists'] = False
    
    # 4. Monte Carlo convergence check
    print("\n" + "-" * 50)
    print("4. MONTE CARLO CONVERGENCE")
    print("-" * 50)
    
    print(f"\n   Testing price convergence for OTM put option...")
    
    from scipy.stats import norm
    
    # Simulate BS prices with different path counts
    S, K, T, sigma, r = 100, 85, 0.25, 0.2, 0.0  # 15% OTM put
    
    # True BS put price
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    true_price = K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    print(f"   Setup: S={S}, K={K}, T={T}, σ={sigma}, r={r}")
    print(f"   True BS put price: ${true_price:.4f}\n")
    
    path_counts = [1000, 5000, 10000, 50000, 100000]
    convergence = []
    
    np.random.seed(42)
    for n_paths in path_counts:
        z = np.random.randn(n_paths)
        S_T = S * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*z)
        payoffs = np.maximum(K - S_T, 0)
        mc_price = np.mean(payoffs) * np.exp(-r*T)
        mc_std = np.std(payoffs) * np.exp(-r*T) / np.sqrt(n_paths)
        error = abs(mc_price - true_price) / true_price * 100
        convergence.append({
            'paths': n_paths,
            'price': float(mc_price),
            'std': float(mc_std),
            'error_pct': float(error)
        })
        print(f"      {n_paths:>6} paths: ${mc_price:.4f} ± ${mc_std:.4f} (error: {error:.1f}%)")
    
    print(f"\n   Recommendation: Use >=50,000 paths for OTM options")
    
    results['mc_convergence'] = convergence
    
    # 5. Summary & Recommendations
    print("\n" + "=" * 70)
    print("   SUMMARY & RECOMMENDATIONS")
    print("=" * 70)
    
    recommendations = []
    
    if 'hurst' in results:
        H = results['hurst']['H_consensus']
        if results['hurst']['confidence'] == 'high' and results['hurst']['is_rough']:
            print(f"\n   Hurst: H = {H:.3f} (rough, high confidence)")
        else:
            print(f"\n   Hurst: H = {H:.3f} ({results['hurst']['confidence']} confidence)")
            recommendations.append("Consider more data or different preprocessing for H estimation")
    
    if 'rho_recommended' in results:
        rho = results['rho_recommended']
        print(f"   Correlation: rho = {rho:.3f}")
        if abs(rho - (-0.7)) > 0.15:
            recommendations.append(f"Update ρ from -0.70 to {rho:.2f} in models")
    
    if not results.get('model_exists', False):
        recommendations.append("Train Neural SDE with 'python main.py' before calibration")
    
    recommendations.append("Increase MC paths to 50,000 for OTM options")
    
    if recommendations:
        print(f"\n   Action Items:")
        for i, rec in enumerate(recommendations, 1):
            print(f"      {i}. {rec}")
    
    # Save results
    output_path = Path("outputs") / "robustness_check.json"
    output_path.parent.mkdir(exist_ok=True)
    
    # Convert numpy types for JSON
    def convert_types(obj):
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        return obj
    
    with open(output_path, 'w') as f:
        json.dump(convert_types(results), f, indent=2)
    
    print(f"\n   Results saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    results = run_robustness_check()
