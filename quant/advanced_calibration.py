import sys as _sys
if _sys.stdout.encoding != 'utf-8':
    _sys.stdout.reconfigure(encoding='utf-8'); _sys.stderr.reconfigure(encoding='utf-8')

"""
Advanced Calibration using high-frequency TradingView data
Estimates Hurst exponent from multiple timeframes and calibrates forward variance curve
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
from scipy.optimize import minimize
from scipy.interpolate import interp1d

# Add parent to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import load_config


class HighFrequencyAnalyzer:
    """Analyze high-frequency data for rough volatility estimation"""
    
    def __init__(self, data_dir: str = None):
        cfg = load_config()
        if data_dir is None:
            # Extract directory from first available SPX file
            data_dir = str(Path(cfg['data']['rv_source']).parent)
        self.data_dir = Path(data_dir)
        self.cfg = cfg
        self.results = {}
        
    def load_tradingview_data(self, filename: str) -> pd.DataFrame:
        """Load TradingView CSV data"""
        filepath = Path(filename)
        if not filepath.is_absolute() and not filepath.exists():
            filepath = self.data_dir / filename
        df = pd.read_csv(filepath)
        
        # Convert Unix timestamp to datetime
        df['datetime'] = pd.to_datetime(df['time'], unit='s')
        df = df.set_index('datetime').sort_index()
        
        # Calculate log returns
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df = df.dropna()
        
        return df
    
    def estimate_hurst_variogram(self, returns: np.ndarray, max_lag: int = 100) -> tuple:
        """
        Estimate Hurst exponent using variogram method (preferred for rough volatility)
        
        For fBM: E[|X(t+h) - X(t)|^2] ~ h^(2H)
        For rough volatility: log-variance increments follow this scaling
        """
        max_possible_lag = min(max_lag, len(returns) // 10)
        if max_possible_lag < 5:
            return np.nan, np.array([]), np.array([])
            
        lags = np.arange(1, max_possible_lag)
        variogram = []
        
        for lag in lags:
            # Compute squared increments at this lag
            increments = returns[lag:] - returns[:-lag]
            variogram.append(np.mean(increments**2))
        
        variogram = np.array(variogram)
        
        # Fit log-log regression: log(variogram) = 2H * log(lag) + const
        log_lags = np.log(lags)
        log_var = np.log(variogram + 1e-10)
        
        # Use all lags — small lags capture roughness, excluding them biases H upward
        slope, intercept = np.polyfit(log_lags, log_var, 1)
        H = slope / 2
        
        return H, lags, variogram
    
    def estimate_hurst_dma(self, returns: np.ndarray, min_scale: int = 10, max_scale: int = 500) -> tuple:
        """
        Detrended Moving Average (DMA) method for Hurst estimation
        More robust for non-stationary financial data
        """
        scales = np.logspace(np.log10(min_scale), np.log10(min(max_scale, len(returns) // 4)), 20).astype(int)
        scales = np.unique(scales)
        
        fluctuations = []
        
        cumsum = np.cumsum(returns - np.mean(returns))
        
        for scale in scales:
            # Moving average
            ma = pd.Series(cumsum).rolling(window=scale, center=True).mean().values
            
            # Remove NaN edges
            valid = ~np.isnan(ma)
            if np.sum(valid) < scale:
                continue
                
            # Detrended series
            detrended = cumsum[valid] - ma[valid]
            
            # RMS fluctuation
            F = np.sqrt(np.mean(detrended**2))
            fluctuations.append((scale, F))
        
        if len(fluctuations) < 5:
            return np.nan, [], []
            
        scales_used = np.array([f[0] for f in fluctuations])
        F_values = np.array([f[1] for f in fluctuations])
        
        # Log-log fit: log(F) = alpha * log(scale) + const
        # For DMA on raw series: H = alpha - 0.5 (the cumsum adds 1 to scaling)
        # For DMA on cumsum: H = alpha
        # Since we cumsum the input, the raw slope = H + 0.5
        slope, _ = np.polyfit(np.log(scales_used), np.log(F_values + 1e-10), 1)
        H = slope - 0.5  # Correct for cumulative sum integration
        
        return H, scales_used, F_values
    
    def compute_realized_volatility(self, df: pd.DataFrame, window_points: int = None) -> pd.Series:
        """Compute realized volatility over rolling windows"""
        # Determine frequency from data
        time_diff = (df.index[1] - df.index[0]).total_seconds() / 60
        
        # Use adaptive window: ~1 hour of data or minimum 6 points
        if window_points is None:
            window_points = max(6, int(60 / time_diff))
        
        # Annualization factor (252 trading days, ~6.5h per day)
        annual_factor = np.sqrt(252 * 6.5 * 60 / time_diff)
        
        rv = df['log_return'].rolling(window=window_points).std() * annual_factor
        return rv.dropna()
    
    def analyze_spx(self):
        """Analyze SPX data at multiple frequencies"""
        print("=" * 60)
        print("SPX ANALYSIS")
        print("=" * 60)
        
        results = {}
        
        # Use config file paths for SPX data
        spx_files = self.cfg['data'].get('spx_files', {})
        
        for freq_str, filepath in spx_files.items():
            freq = str(freq_str)
            p = Path(filepath)
            if not p.exists():
                print(f"  Warning: {filepath} not found")
                continue
                
            print(f"\nSPX {freq}min data:")
            df = self.load_tradingview_data(str(p))
            
            print(f"   Period: {df.index[0].date()} → {df.index[-1].date()}")
            print(f"   Points: {len(df):,}")
            print(f"   Trading days: ~{len(df) * int(freq) / (6.5 * 60):.0f}")
            
            # Compute realized volatility
            rv = self.compute_realized_volatility(df)
            log_rv = np.log(rv.values + 1e-6)
            
            # Estimate H using variogram on log(RV)
            H_var, lags, variogram = self.estimate_hurst_variogram(log_rv)
            
            # Estimate H using DMA
            H_dma, scales, fluct = self.estimate_hurst_dma(log_rv)
            
            print(f"\n   Realized Volatility stats:")
            print(f"      Mean RV: {rv.mean()*100:.1f}%")
            print(f"      Std RV:  {rv.std()*100:.1f}%")
            print(f"      Min RV:  {rv.min()*100:.1f}%")
            print(f"      Max RV:  {rv.max()*100:.1f}%")
            
            print(f"\n   Hurst Exponent Estimation:")
            print(f"      Variogram method: H = {H_var:.3f}")
            print(f"      DMA method:       H = {H_dma:.3f}")
            
            if H_var < 0.2:
                print(f"      ✅ ROUGH VOLATILITY CONFIRMED (H_var = {H_var:.3f} < 0.2)")
            elif H_var < 0.5:
                print(f"      ⚠️  Sub-diffusive (H_var = {H_var:.3f} < 0.5) but not rough in Gatheral sense (needs H < 0.2)")
            else:
                print(f"      ❌ Not rough (H_var = {H_var:.3f} >= 0.5)")
            
            results[f'spx_{freq}min'] = {
                'H_variogram': float(H_var),
                'H_dma': float(H_dma),
                'mean_rv': float(rv.mean()),
                'std_rv': float(rv.std()),
                'n_points': len(df),
                'start_date': str(df.index[0].date()),
                'end_date': str(df.index[-1].date())
            }
        
        self.results['spx'] = results
        return results
    
    def analyze_vix(self):
        """Analyze VIX data"""
        print("\n" + "=" * 60)
        print("VIX ANALYSIS")
        print("=" * 60)
        
        results = {}
        
        vix_files = self.cfg['data'].get('vix_files', {})
        
        for freq_str, filepath in vix_files.items():
            freq = str(freq_str)
            p = Path(filepath)
            if not p.exists():
                continue
                
            print(f"\nVIX {freq}min data:")
            df = self.load_tradingview_data(str(p))
            
            print(f"   Period: {df.index[0].date()} → {df.index[-1].date()}")
            print(f"   Points: {len(df):,}")
            
            # VIX is already in volatility units (percentage)
            vix_values = df['close'].values
            log_vix = np.log(vix_values)
            
            # Estimate H on log(VIX)
            H_var, _, _ = self.estimate_hurst_variogram(log_vix)
            H_dma, _, _ = self.estimate_hurst_dma(log_vix)
            
            print(f"\n   VIX stats:")
            print(f"      Mean VIX: {vix_values.mean():.1f}")
            print(f"      Std VIX:  {vix_values.std():.1f}")
            print(f"      Min VIX:  {vix_values.min():.1f}")
            print(f"      Max VIX:  {vix_values.max():.1f}")
            
            print(f"\n   Hurst Exponent (on log-VIX):")
            print(f"      Variogram method: H = {H_var:.3f}")
            print(f"      DMA method:       H = {H_dma:.3f}")
            
            results[f'vix_{freq}min'] = {
                'H_variogram': float(H_var),
                'H_dma': float(H_dma),
                'mean_vix': float(vix_values.mean()),
                'std_vix': float(vix_values.std()),
                'n_points': len(df)
            }
        
        self.results['vix'] = results
        return results
    
    def calibrate_forward_variance(self):
        """
        Calibrate forward variance curve ξ₀(t) using VIX term structure
        VIX² ≈ (1/T) * ∫₀ᵀ ξ₀(t) dt  (under certain assumptions)
        """
        print("\n" + "=" * 60)
        print("FORWARD VARIANCE CALIBRATION")
        print("=" * 60)
        
        # Load VIX data at different frequencies (proxy for different maturities)
        # In practice, you'd use VIX futures (VX1, VX2, etc.)
        
        # Use the latest VIX value as spot variance
        vix_source = self.cfg['data']['source']  # VIX training source from config
        vix_15 = self.load_tradingview_data(vix_source)
        current_vix = vix_15['close'].iloc[-1]
        
        # VIX is in percentage, convert to decimal variance
        xi_0 = (current_vix / 100) ** 2
        
        print(f"\n   Current VIX: {current_vix:.2f}")
        print(f"   Spot variance ξ₀: {xi_0:.4f} (σ = {np.sqrt(xi_0)*100:.1f}%)")
        
        # Estimate long-term variance from historical mean
        mean_vix = vix_15['close'].mean()
        xi_long = (mean_vix / 100) ** 2
        
        print(f"\n   Mean VIX (long-term): {mean_vix:.2f}")
        print(f"   Long-term variance: {xi_long:.4f} (σ = {np.sqrt(xi_long)*100:.1f}%)")
        
        # Simple mean-reverting forward variance model
        # ξ₀(t) = ξ_long + (ξ_0 - ξ_long) * exp(-κ*t)
        # Estimate κ from VIX autocorrelation
        
        log_vix = np.log(vix_15['close'].values)
        autocorr_1 = np.corrcoef(log_vix[:-1], log_vix[1:])[0, 1]
        
        # Time step in years (15min = 15/(252*6.5*60) years)
        dt = 15 / (252 * 6.5 * 60)
        kappa = -np.log(autocorr_1) / dt
        
        print(f"\n   Autocorrelation (1 lag): {autocorr_1:.4f}")
        print(f"   Mean-reversion speed κ: {kappa:.2f} (half-life: {np.log(2)/kappa*252:.1f} days)")
        
        self.results['forward_variance'] = {
            'xi_0': float(xi_0),
            'xi_long': float(xi_long),
            'kappa': float(kappa),
            'current_vix': float(current_vix),
            'mean_vix': float(mean_vix)
        }
        
        return self.results['forward_variance']
    
    def get_optimal_bergomi_params(self):
        """Combine all analyses to get optimal Bergomi parameters"""
        print("\n" + "=" * 60)
        print("OPTIMAL BERGOMI PARAMETERS")
        print("=" * 60)
        
        # Get H from SPX realized volatility (most reliable)
        # Prefer variogram (standard in literature) over DMA
        H_variogram_estimates = []
        H_dma_estimates = []
        
        if 'spx' in self.results:
            for key, data in self.results['spx'].items():
                h_var = data['H_variogram']
                h_dma = data['H_dma']
                if 0 < h_var < 0.5:
                    H_variogram_estimates.append(h_var)
                if 0 < h_dma < 0.5:
                    H_dma_estimates.append(h_dma)
        
        if H_variogram_estimates:
            # Variogram is the gold standard (Gatheral et al. 2018)
            H_optimal = np.median(H_variogram_estimates)
        elif H_dma_estimates:
            H_optimal = np.median(H_dma_estimates)
        else:
            H_optimal = 0.1  # Literature fallback
        
        # Get variance from forward variance calibration
        if 'forward_variance' in self.results:
            xi_0 = self.results['forward_variance']['xi_0']
        else:
            xi_0 = 0.04  # 20% vol default
        
        # η (vol of vol) from VIX volatility
        if 'vix' in self.results:
            vix_data = list(self.results['vix'].values())[0]
            # η ≈ std(log VIX) * some scaling
            eta_estimate = vix_data['std_vix'] / vix_data['mean_vix'] * 5  # Heuristic
            eta_optimal = np.clip(eta_estimate, 1.0, 4.0)
        else:
            eta_optimal = 2.0
        
        # ρ (correlation) - typically -0.7 to -0.9 for equity indices
        rho_optimal = -0.75
        
        params = {
            'H': float(H_optimal),
            'eta': float(eta_optimal),
            'rho': float(rho_optimal),
            'xi_0': float(xi_0),
            'sigma_0': float(np.sqrt(xi_0))
        }
        
        print(f"\n   H (Hurst):     {params['H']:.3f}")
        print(f"   η (vol-of-vol): {params['eta']:.2f}")
        print(f"   ρ (correlation): {params['rho']:.2f}")
        print(f"   ξ₀ (spot var):  {params['xi_0']:.4f}")
        print(f"   σ₀ (spot vol):  {params['sigma_0']*100:.1f}%")
        
        if params['H'] < 0.2:
            print(f"\n   Very rough volatility regime (H < 0.2)")
        elif params['H'] < 0.5:
            print(f"\n   Rough volatility regime (0.2 < H < 0.5)")
        else:
            print(f"\n   Standard volatility regime (H >= 0.5)")
        
        self.results['optimal_params'] = params
        return params
    
    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print("\n" + "=" * 60)
        print("   ADVANCED CALIBRATION - HIGH FREQUENCY DATA ANALYSIS")
        print("=" * 60)
        print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.analyze_spx()
        self.analyze_vix()
        self.calibrate_forward_variance()
        optimal_params = self.get_optimal_bergomi_params()
        
        # Save results
        output_cfg = self.cfg.get('outputs', {})
        output_path = Path(output_cfg.get('calibration', 'outputs/advanced_calibration.json'))
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {output_path}")
        
        return optimal_params


def compare_with_previous():
    """Compare new calibration with previous grid search results"""
    data_dir = Path("data")
    
    # Load previous calibration
    prev_path = data_dir / "calibration_report.json"
    if prev_path.exists():
        with open(prev_path) as f:
            prev = json.load(f)
        
        print("\n" + "=" * 60)
        print("COMPARISON WITH PREVIOUS CALIBRATION")
        print("=" * 60)
        
        prev_bergomi = prev.get('bergomi', {})
        print(f"\n   Previous Bergomi (grid search on options):")
        print(f"      H = {prev_bergomi.get('H', 'N/A')}")
        print(f"      η = {prev_bergomi.get('eta', 'N/A')}")
        print(f"      ρ = {prev_bergomi.get('rho', 'N/A')}")
        print(f"      RMSE = {prev_bergomi.get('rmse', 'N/A')}")


if __name__ == "__main__":
    analyzer = HighFrequencyAnalyzer(data_dir="data")
    optimal = analyzer.run_full_analysis()
    compare_with_previous()
    
    print("\n" + "=" * 60)
    print("📋 SUMMARY - USE THESE PARAMETERS FOR BERGOMI")
    print("=" * 60)
    print(f"""
    H   = {optimal['H']:.3f}   # From SPX realized volatility
    η   = {optimal['eta']:.2f}    # From VIX volatility  
    ρ   = {optimal['rho']:.2f}   # Equity index typical
    ξ₀  = {optimal['xi_0']:.4f}  # From current VIX
    """)
