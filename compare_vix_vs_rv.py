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
from utils.data_loader import MarketDataLoader, RealizedVolatilityLoader
from utils.diagnostics import print_distribution_stats, compute_acf, estimate_hurst

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
    print("="*70)
    print("   VIX vs REALIZED VOLATILITY: The Roughness Battle")
    print("="*70)
    
    segment_length = 20
    
    # 1. Load VIX data
    print("\n" + "-"*70)
    print("1. VIX (Implied Volatility Index)")
    print("-"*70)
    
    vix_loader = MarketDataLoader(file_path="data/TVC_VIX, 30.csv")
    vix_paths = vix_loader.get_realized_vol_paths(segment_length=segment_length)
    vix_np = np.array(vix_paths)
    
    vix_hurst = estimate_hurst(vix_np)
    vix_acf = np.mean([compute_acf(p, lags=10) for p in vix_np[:100]], axis=0)
    
    print(f"\n   VIX Statistics:")
    print(f"      Paths: {vix_np.shape[0]} x {vix_np.shape[1]}")
    print(f"      Mean Variance: {np.mean(vix_np):.5f} (Vol: {np.sqrt(np.mean(vix_np))*100:.1f}%)")
    print(f"      Hurst: {vix_hurst:.3f} {'NOT ROUGH' if vix_hurst > 0.2 else 'ROUGH'}")
    print(f"      ACF-1: {vix_acf[1]:.3f}")
    
    # 2. Load Realized Volatility
    print("\n" + "-"*70)
    print("2. REALIZED VOLATILITY (from S&P 500 returns)")
    print("-"*70)
    
    rv_loader = RealizedVolatilityLoader(file_path="data/SP_SPX, 30.csv")
    rv_paths = rv_loader.get_realized_vol_paths(segment_length=segment_length)
    rv_np = np.array(rv_paths)
    
    rv_hurst = estimate_hurst(rv_np)
    rv_acf = np.mean([compute_acf(p, lags=10) for p in rv_np[:100]], axis=0)
    
    print(f"\n   Realized Vol Statistics:")
    print(f"      Paths: {rv_np.shape[0]} x {rv_np.shape[1]}")
    print(f"      Mean Variance: {np.mean(rv_np):.5f} (Vol: {np.sqrt(np.mean(rv_np))*100:.1f}%)")
    print(f"      Hurst: {rv_hurst:.3f} {'ROUGH!' if rv_hurst < 0.2 else 'not rough'}")
    print(f"      ACF-1: {rv_acf[1]:.3f}")
    
    # 3. Summary
    print("\n" + "="*70)
    print("   SUMMARY: Why Realized Vol is Better for Rough Vol Modeling")
    print("="*70)
    
    print(f"""
    ┌─────────────────┬────────────┬─────────────────┐
    │ Metric          │ VIX        │ Realized Vol    │
    ├─────────────────┼────────────┼─────────────────┤
    │ Hurst Exponent  │ {vix_hurst:.3f}      │ {rv_hurst:.3f}  {'← ROUGH!' if rv_hurst < 0.15 else ''}         │
    │ ACF (Lag 1)     │ {vix_acf[1]:.3f}      │ {rv_acf[1]:.3f}           │
    │ Mean Vol        │ {np.sqrt(np.mean(vix_np))*100:.1f}%      │ {np.sqrt(np.mean(rv_np))*100:.1f}%          │
    │ # Paths         │ {vix_np.shape[0]}       │ {rv_np.shape[0]}             │
    └─────────────────┴────────────┴─────────────────┘
    """)
    
    if rv_hurst < 0.15:
        print("    RECOMMENDATION: Use REALIZED VOLATILITY for training!")
        print("       It exhibits the rough behavior (H < 0.15) that matches")
        print("       the theoretical rough volatility models (Gatheral et al.)")
    else:
        print("    Note: Neither shows strong roughness. Consider higher frequency data.")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
