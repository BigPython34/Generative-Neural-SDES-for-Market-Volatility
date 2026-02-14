import sys as _sys
if _sys.stdout.encoding != 'utf-8':
    _sys.stdout.reconfigure(encoding='utf-8'); _sys.stderr.reconfigure(encoding='utf-8')

"""
VIX vs Realized Volatility Comparison
=====================================
Compare the statistical properties of:
1. VIX (Implied Volatility Index) - forward-looking, smoothed
2. Realized Volatility from S&P 500 - backward-looking, rough

The key insight: Realized Vol exhibits ROUGH behavior (H ~ 0.1)
while VIX is smoother (H ~ 0.5)
"""

import jax
import jax.numpy as jnp
import numpy as np
import os
import yaml
from utils.data_loader import MarketDataLoader, RealizedVolatilityLoader
from utils.diagnostics import print_distribution_stats, compute_acf, estimate_hurst, estimate_hurst_from_returns

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def load_config():
    with open("config/params.yaml", 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    print("="*70)
    print("   VIX vs REALIZED VOLATILITY: The Roughness Battle")
    print("="*70)
    
    cfg = load_config()
    segment_length = cfg['data']['segment_length']
    
    # 1. Load VIX data
    print("\n" + "-"*70)
    print("1. VIX (Implied Volatility Index)")
    print("-"*70)
    
    vix_source = cfg['data']['source']
    vix_loader = MarketDataLoader(file_path=vix_source)
    vix_paths = vix_loader.get_realized_vol_paths(segment_length=segment_length)
    vix_np = np.array(vix_paths)
    
    vix_hurst = estimate_hurst(vix_np)
    vix_acf = np.mean([compute_acf(p, lags=10) for p in vix_np[:100]], axis=0)
    
    print(f"\n   VIX Statistics:")
    print(f"      Paths: {vix_np.shape[0]} x {vix_np.shape[1]}")
    print(f"      Mean Variance: {np.mean(vix_np):.5f} (Vol: {np.sqrt(np.mean(vix_np))*100:.1f}%)")
    print(f"      Hurst (on paths): {vix_hurst:.3f}  [VIX is smoothed → H ≈ 0.5 expected]")
    print(f"      ACF-1: {vix_acf[1]:.3f}")
    
    # 2. Load Realized Volatility from SPX returns
    print("\n" + "-"*70)
    print("2. REALIZED VOLATILITY (from S&P 500 returns)")
    print("-"*70)
    
    rv_source = cfg['data'].get('rv_source', 'data/SP_SPX, 5.csv')
    rv_loader = RealizedVolatilityLoader(file_path=rv_source)  # auto-detects frequency
    rv_paths = rv_loader.get_realized_vol_paths(segment_length=segment_length)
    rv_np = np.array(rv_paths)
    
    rv_hurst = estimate_hurst(rv_np)
    rv_acf = np.mean([compute_acf(p, lags=10) for p in rv_np[:100]], axis=0)
    
    print(f"\n   Realized Vol Statistics:")
    print(f"      Paths: {rv_np.shape[0]} x {rv_np.shape[1]}")
    print(f"      Mean Variance: {np.mean(rv_np):.5f} (Vol: {np.sqrt(np.mean(rv_np))*100:.1f}%)")
    print(f"      Hurst (on paths): {rv_hurst:.3f}  [Rolling RV paths — note: short segments]")
    print(f"      ACF-1: {rv_acf[1]:.3f}")
    
    # 3. True Hurst from daily RV (most reliable)
    print("\n" + "-"*70)
    print("3. TRUE HURST EXPONENT (Daily RV from intraday returns)")
    print("-"*70)
    
    rv_daily = estimate_hurst_from_returns(rv_source)
    print(f"      Source: {rv_source}")
    print(f"      Daily RV points: {rv_daily['n_rv_points']}")
    print(f"      H (variogram): {rv_daily['H_variogram']:.4f}  (R² = {rv_daily['R2_variogram']:.3f})")
    print(f"      H (struct q=1): {rv_daily['H_structure']:.4f}  (R² = {rv_daily['R2_structure']:.3f})")
    
    H_rv = rv_daily['H_variogram']
    
    # 4. Summary
    print("\n" + "="*70)
    print("   SUMMARY: VIX vs Realized Vol Roughness")
    print("="*70)
    
    print(f"""
    ┌───────────────────────┬────────────┬─────────────────┐
    │ Metric                │ VIX        │ Realized Vol    │
    ├───────────────────────┼────────────┼─────────────────┤
    │ Path H (segment-level)│ {vix_hurst:.3f}      │ {rv_hurst:.3f}           │
    │ True H (daily RV)     │ ~0.50      │ {H_rv:.3f} {'← ROUGH!' if H_rv < 0.15 else ''}         │
    │ ACF (Lag 1)           │ {vix_acf[1]:.3f}      │ {rv_acf[1]:.3f}           │
    │ Mean Vol              │ {np.sqrt(np.mean(vix_np))*100:.1f}%      │ {np.sqrt(np.mean(rv_np))*100:.1f}%          │
    │ # Paths               │ {vix_np.shape[0]}       │ {rv_np.shape[0]}             │
    └───────────────────────┴────────────┴─────────────────┘
    
    KEY INSIGHT (Gatheral et al. 2018):
    • VIX integrates over 30 days → smooths roughness → H ≈ 0.5
    • Realized Vol from 5-min returns → captures roughness → H ≈ 0.1
    • Training on VIX is fine for PRICING (Q-measure)
    • True roughness must be verified on REALIZED VOL (P-measure)
    """)
    
    print("="*70)


if __name__ == "__main__":
    main()
