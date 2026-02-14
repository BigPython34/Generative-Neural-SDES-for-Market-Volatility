import sys as _sys
if _sys.stdout.encoding != 'utf-8':
    _sys.stdout.reconfigure(encoding='utf-8'); _sys.stderr.reconfigure(encoding='utf-8')

"""
Enhanced Neural SDE Calibration with VVIX and VIX Futures Term Structure

Uses:
1. VVIX → Direct observation of Vol-of-Vol (η parameter)
2. VIX Futures → Term structure for mean-reversion calibration
3. VIX Spot → Variance dynamics

Author: Enhanced calibration pipeline
"""

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
from scipy.stats import pearsonr
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))


class MarketDataEnhanced:
    """Load and process all market data sources"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.vvix = None
        self.vix = None
        self.futures = None
        self._load_all()
        
    def _load_all(self):
        """Load all data sources"""
        print("=" * 60)
        print("LOADING ENHANCED MARKET DATA")
        print("=" * 60)
        
        # 1. VVIX
        self._load_vvix()
        
        # 2. VIX Spot
        self._load_vix()
        
        # 3. VIX Futures
        self._load_futures()
        
        # 4. Align all data
        self._align_data()
        
    def _load_vvix(self):
        """Load VVIX (Vol-of-Vol)"""
        print("\nLoading VVIX...")
        df = pd.read_csv(self.data_dir / "CBOE_DLY_VVIX, 15.csv")
        df['datetime'] = pd.to_datetime(df['time'], unit='s')
        df = df.set_index('datetime').sort_index()
        df = df[['close']].rename(columns={'close': 'vvix'})
        
        # Resample to daily
        self.vvix_intraday = df.copy()
        self.vvix = df.resample('D').last().dropna()
        
        print(f"   VVIX: {len(self.vvix)} daily points")
        print(f"   Range: {self.vvix.index.min().date()} → {self.vvix.index.max().date()}")
        print(f"   Mean: {self.vvix.vvix.mean():.1f}, Std: {self.vvix.vvix.std():.1f}")
        
    def _load_vix(self):
        """Load VIX Spot"""
        print("\nLoading VIX Spot...")
        df = pd.read_csv(self.data_dir / "TVC_VIX, 15.csv")
        df['datetime'] = pd.to_datetime(df['time'], unit='s')
        df = df.set_index('datetime').sort_index()
        df = df[['close']].rename(columns={'close': 'vix'})
        
        self.vix_intraday = df.copy()
        self.vix = df.resample('D').last().dropna()
        
        print(f"   VIX: {len(self.vix)} daily points")
        print(f"   Mean: {self.vix.vix.mean():.1f}, Std: {self.vix.vix.std():.1f}")
        
    def _load_futures(self):
        """Load VIX Futures Term Structure"""
        print("\nLoading VIX Futures...")
        
        # Load all futures
        df = pd.read_csv(self.data_dir / "cboe_vix_futures_full/vix_futures_all.csv")
        df['Trade Date'] = pd.to_datetime(df['Trade Date'])
        
        # Parse futures codes to get expiry month
        def parse_expiry(code):
            """Parse 'G (Feb 2026)' -> datetime"""
            try:
                parts = code.split('(')[1].replace(')', '').strip()
                return pd.to_datetime(parts, format='%b %Y')
            except:
                return pd.NaT
        
        df['expiry'] = df['Futures'].apply(parse_expiry)
        df['days_to_expiry'] = (df['expiry'] - df['Trade Date']).dt.days
        
        # Filter valid data
        df = df[df['days_to_expiry'] > 0].copy()
        
        self.futures_raw = df
        
        # Pivot to get term structure per date
        # Keep only first 4 maturities
        df_sorted = df.sort_values(['Trade Date', 'days_to_expiry'])
        
        term_structure = []
        for date, group in df_sorted.groupby('Trade Date'):
            row = {'date': date}
            for i, (_, r) in enumerate(group.head(4).iterrows()):
                row[f'F{i+1}'] = r['Close']
                row[f'T{i+1}'] = r['days_to_expiry'] / 365
            if len(group) >= 2:
                term_structure.append(row)
        
        self.futures = pd.DataFrame(term_structure).set_index('date')
        
        print(f"   Futures: {len(self.futures)} trading days")
        print(f"   Maturities: F1, F2, F3, F4")
        
    def _align_data(self):
        """Align all data sources on common dates"""
        print("\nAligning datasets...")
        
        # Find common dates
        common_idx = self.vvix.index.intersection(self.vix.index)
        common_idx = common_idx.intersection(self.futures.index)
        
        self.aligned = pd.concat([
            self.vvix.loc[common_idx],
            self.vix.loc[common_idx],
            self.futures.loc[common_idx]
        ], axis=1)
        
        print(f"   Common dates: {len(self.aligned)}")
        print(f"   Period: {self.aligned.index.min().date()} → {self.aligned.index.max().date()}")


class VolOfVolAnalyzer:
    """
    Analyze VVIX to extract Vol-of-Vol (η) parameter
    
    Key insight: VVIX ≈ 100 × η × √(VIX)
    So: η ≈ VVIX / (100 × √VIX)
    """
    
    def __init__(self, data: MarketDataEnhanced):
        self.data = data
        
    def compute_implied_eta(self) -> pd.DataFrame:
        """
        Extract implied η from VVIX/VIX relationship
        
        In rough Bergomi: dv_t = η × v_t^α × dW^v
        The VVIX measures E[∫σ²_v dt] which scales with η²
        
        Empirical approximation: η ≈ VVIX / (100 × VIX^0.5)
        """
        df = self.data.aligned.copy()
        
        # VVIX is quoted in % (like VIX), convert to decimal
        vvix_decimal = df['vvix'] / 100
        vix_decimal = df['vix'] / 100
        
        # Implied η: vol-of-vol in variance terms
        # VVIX² ≈ η² × VIX × T where T~30 days
        # So η ≈ VVIX / √(VIX × 30/365)
        T = 30 / 365
        df['implied_eta'] = vvix_decimal / np.sqrt(vix_decimal * T)
        
        # Alternative: direct ratio approach (simpler)
        df['eta_simple'] = df['vvix'] / df['vix']
        
        return df
    
    def analyze(self) -> dict:
        """Full analysis of vol-of-vol dynamics"""
        print("\n" + "=" * 60)
        print("VOL-OF-VOL (eta) ANALYSIS FROM VVIX")
        print("=" * 60)
        
        df = self.compute_implied_eta()
        
        results = {
            'vvix_mean': float(df['vvix'].mean()),
            'vvix_std': float(df['vvix'].std()),
            'vix_mean': float(df['vix'].mean()),
            'implied_eta_mean': float(df['implied_eta'].mean()),
            'implied_eta_std': float(df['implied_eta'].std()),
            'eta_simple_mean': float(df['eta_simple'].mean()),
        }
        
        # Correlation VVIX vs VIX
        corr, pval = pearsonr(df['vvix'].dropna(), df['vix'].dropna())
        results['vvix_vix_corr'] = float(corr)
        
        print(f"\nVVIX Statistics:")
        print(f"   Mean: {results['vvix_mean']:.1f}")
        print(f"   Std:  {results['vvix_std']:.1f}")
        
        print(f"\nImplied eta (Vol-of-Vol):")
        print(f"   Mean: {results['implied_eta_mean']:.2f}")
        print(f"   Std:  {results['implied_eta_std']:.2f}")
        print(f"   Simple ratio (VVIX/VIX): {results['eta_simple_mean']:.2f}")
        
        print(f"\nVVIX-VIX Correlation: {corr:.3f} (p={pval:.2e})")
        
        print(f"\nVol-of-Vol Regimes:")
        low_vol = df[df['vix'] < df['vix'].quantile(0.25)]['implied_eta'].mean()
        high_vol = df[df['vix'] > df['vix'].quantile(0.75)]['implied_eta'].mean()
        print(f"   η in low VIX regime:  {low_vol:.2f}")
        print(f"   η in high VIX regime: {high_vol:.2f}")
        results['eta_low_vol'] = float(low_vol)
        results['eta_high_vol'] = float(high_vol)
        
        return results


class TermStructureAnalyzer:
    """
    Analyze VIX Futures Term Structure
    
    Key parameters extracted:
    - Mean reversion speed (κ)
    - Long-term mean (θ)
    - Contango/Backwardation dynamics
    """
    
    def __init__(self, data: MarketDataEnhanced):
        self.data = data
        
    def compute_term_structure_slope(self) -> pd.DataFrame:
        """Compute term structure metrics"""
        df = self.data.aligned.copy()
        
        # Spread F2-F1 (classic contango measure)
        df['spread_21'] = df['F2'] - df['F1']
        df['spread_pct'] = df['spread_21'] / df['F1'] * 100
        
        # Roll yield (annualized)
        # Roll = (F2 - F1) / F1 / (T2 - T1) × 12
        df['roll_yield'] = df['spread_21'] / df['F1'] / (df['T2'] - df['T1']) * 12 * 100
        
        return df
    
    def estimate_mean_reversion(self) -> dict:
        """
        Estimate mean reversion κ from term structure
        
        Under Heston/SV: F(T) = VIX × exp(-κT) + θ × (1 - exp(-κT))
        The slope of log(F/VIX) vs T gives κ
        """
        df = self.data.aligned.dropna().copy()
        
        # Use F1-F4 to fit exponential decay
        kappas = []
        
        for _, row in df.iterrows():
            vix = row['vix']
            Fs = [row.get(f'F{i}', np.nan) for i in range(1, 5)]
            Ts = [row.get(f'T{i}', np.nan) for i in range(1, 5)]
            
            # Filter valid
            valid = [(F, T) for F, T in zip(Fs, Ts) if not np.isnan(F) and not np.isnan(T)]
            if len(valid) < 3:
                continue
                
            Fs, Ts = zip(*valid)
            Fs, Ts = np.array(Fs), np.array(Ts)
            
            # Fit: F = VIX + (θ - VIX)(1 - e^{-κT})
            # Linearize: log((F - θ)/(VIX - θ)) = -κT
            # Approximate θ as long-term mean (~20)
            theta = 20
            
            try:
                # Simple linear regression on log scale
                y = np.log(np.abs(Fs - theta) / np.abs(vix - theta + 1e-6) + 1e-6)
                slope, _ = np.polyfit(Ts, y, 1)
                kappa = -slope
                if 0.1 < kappa < 10:  # Reasonable range
                    kappas.append(kappa)
            except:
                pass
        
        return {
            'kappa_mean': float(np.mean(kappas)) if kappas else 2.0,
            'kappa_std': float(np.std(kappas)) if kappas else 0.5,
            'kappa_median': float(np.median(kappas)) if kappas else 2.0,
        }
    
    def analyze(self) -> dict:
        """Full term structure analysis"""
        print("\n" + "=" * 60)
        print("VIX FUTURES TERM STRUCTURE ANALYSIS")
        print("=" * 60)
        
        df = self.compute_term_structure_slope()
        mr = self.estimate_mean_reversion()
        
        results = {
            'spread_mean': float(df['spread_21'].mean()),
            'spread_std': float(df['spread_21'].std()),
            'contango_pct': float((df['spread_21'] > 0).mean() * 100),
            'roll_yield_mean': float(df['roll_yield'].mean()),
            **mr
        }
        
        print(f"\nTerm Structure Shape:")
        print(f"   Contango: {results['contango_pct']:.1f}% of days")
        print(f"   Mean spread (F2-F1): {results['spread_mean']:.2f} pts")
        print(f"   Roll yield (ann.): {results['roll_yield_mean']:.1f}%")
        
        print(f"\nImplied Mean Reversion (kappa):")
        print(f"   kappa mean: {results['kappa_mean']:.2f}")
        print(f"   kappa std:  {results['kappa_std']:.2f}")
        print(f"   Half-life: {np.log(2)/results['kappa_mean']*365:.0f} days")
        
        print(f"\nRegime Analysis:")
        backwardation_days = df[df['spread_21'] < 0]
        if len(backwardation_days) > 0:
            print(f"   Backwardation VIX mean: {backwardation_days['vix'].mean():.1f}")
            print(f"   Contango VIX mean: {df[df['spread_21'] > 0]['vix'].mean():.1f}")
            results['backwardation_vix_mean'] = float(backwardation_days['vix'].mean())
        
        return results


class EnhancedCalibrator:
    """
    Calibrate Neural SDE using all market data sources
    
    Objective: Find parameters that match:
    1. VIX dynamics (variance paths)
    2. VVIX level (vol-of-vol η)
    3. Term structure (mean reversion κ)
    """
    
    def __init__(self, data: MarketDataEnhanced):
        self.data = data
        self.vol_analyzer = VolOfVolAnalyzer(data)
        self.ts_analyzer = TermStructureAnalyzer(data)
        
    def calibrate(self) -> dict:
        """Run full calibration"""
        print("\n" + "=" * 60)
        print("ENHANCED NEURAL SDE CALIBRATION")
        print("=" * 60)
        
        # Step 1: Analyze vol-of-vol
        eta_results = self.vol_analyzer.analyze()
        
        # Step 2: Analyze term structure
        ts_results = self.ts_analyzer.analyze()
        
        # Step 3: Combine into calibrated parameters
        params = {
            # Vol-of-vol from VVIX
            'eta': eta_results['implied_eta_mean'],
            'eta_low_regime': eta_results['eta_low_vol'],
            'eta_high_regime': eta_results['eta_high_vol'],
            
            # Mean reversion from futures
            'kappa': ts_results['kappa_mean'],
            'theta': self.data.vix['vix'].mean() / 100,  # Long-term variance
            
            # Current market state
            'vix_current': float(self.data.vix['vix'].iloc[-1]),
            'vvix_current': float(self.data.vvix['vvix'].iloc[-1]),
            'contango': ts_results['spread_mean'] > 0,
        }
        
        print("\n" + "=" * 60)
        print("CALIBRATED PARAMETERS")
        print("=" * 60)
        print(f"\n   eta (Vol-of-Vol):     {params['eta']:.3f}")
        print(f"   kappa (Mean Reversion): {params['kappa']:.2f}")
        print(f"   theta (Long-term var):  {params['theta']:.4f} ({np.sqrt(params['theta'])*100:.1f}% vol)")
        print(f"\n   Current VIX:  {params['vix_current']:.1f}")
        print(f"   Current VVIX: {params['vvix_current']:.1f}")
        print(f"   Market state: {'Contango' if params['contango'] else 'Backwardation'}")
        
        return {
            'parameters': params,
            'eta_analysis': eta_results,
            'term_structure': ts_results,
        }


def main():
    """Run enhanced calibration"""
    # Load all data
    data = MarketDataEnhanced()
    
    # Run calibration
    calibrator = EnhancedCalibrator(data)
    results = calibrator.calibrate()
    
    # Save results
    output_path = Path("outputs/enhanced_calibration.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_path}")
    
    # Comparison with previous assumptions
    print("\n" + "=" * 60)
    print("COMPARISON: MARKET vs PREVIOUS ASSUMPTIONS")
    print("=" * 60)
    print(f"\n{'Parameter':<25} {'Market':<15} {'Previous':<15} {'Status'}")
    print("-" * 70)
    
    eta_market = results['parameters']['eta']
    eta_prev = 0.5  # Typical assumption
    print(f"{'eta (Vol-of-Vol)':<25} {eta_market:<15.3f} {eta_prev:<15.3f} {'OK' if abs(eta_market - eta_prev) < 0.3 else 'DIFF'}")
    
    kappa_market = results['parameters']['kappa']
    kappa_prev = 2.0
    print(f"{'kappa (Mean Reversion)':<25} {kappa_market:<15.2f} {kappa_prev:<15.2f} {'OK' if abs(kappa_market - kappa_prev) < 1 else 'DIFF'}")
    
    theta_market = np.sqrt(results['parameters']['theta']) * 100
    theta_prev = 20
    print(f"{'theta (Long-term vol %)':<25} {theta_market:<15.1f} {theta_prev:<15.1f} {'OK' if abs(theta_market - theta_prev) < 5 else 'DIFF'}")
    
    return results


if __name__ == "__main__":
    results = main()
