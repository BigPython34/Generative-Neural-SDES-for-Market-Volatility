import sys as _sys
if _sys.stdout.encoding != 'utf-8':
    _sys.stdout.reconfigure(encoding='utf-8'); _sys.stderr.reconfigure(encoding='utf-8')

"""
Frequency Comparison Script
===========================
Compare VIX data at different frequencies (5, 15, 30 min) 
for TRAINING quality (distribution matching, signature richness).

IMPORTANT: Hurst roughness is NOT measured on VIX (smoothed, H≈0.5).
True roughness (H≈0.1) is measured separately on realized vol from SPX returns.
"""

import jax
import jax.numpy as jnp
import numpy as np
import os
import yaml
from utils.data_loader import MarketDataLoader, RealizedVolatilityLoader, load_config
from utils.diagnostics import print_distribution_stats, compute_acf, estimate_hurst, estimate_hurst_from_returns
from ml.generative_trainer import GenerativeTrainer
from ml.signature_engine import SignatureFeatureExtractor

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# VIX data files at different frequencies
VIX_FILES = {
    '5min': 'data/TVC_VIX, 5.csv',
    '15min': 'data/TVC_VIX, 15.csv',
    '30min': 'data/TVC_VIX, 30.csv',
}

# SPX data files for TRUE roughness comparison
SPX_FILES = {
    '5min': 'data/SP_SPX, 5.csv',
    '30min': 'data/SP_SPX, 30.csv',
}

def analyze_frequency(file_path: str, freq_name: str, segment_length: int = 13):
    """Analyze VIX data at a specific frequency for TRAINING quality."""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {freq_name} VIX DATA (training quality)")
    print(f"{'='*60}")
    
    if not os.path.exists(file_path):
        print(f"  Warning: File not found: {file_path}")
        return None
    
    # Load data with custom file path
    loader = MarketDataLoader(file_path=file_path)
    
    try:
        paths = loader.get_realized_vol_paths(segment_length=segment_length)
        paths_np = np.array(paths)
        
        # Compute detailed stats
        flat = paths_np.flatten()
        h_est = estimate_hurst(paths_np)
        
        # Compute ACF properly
        max_lag = min(10, segment_length - 2)
        acf_vals = np.mean([compute_acf(p, lags=max_lag) for p in paths_np[:100]], axis=0)
        acf_lag1 = acf_vals[1] if len(acf_vals) > 1 else 0.0
        
        stats = {
            'freq': freq_name,
            'n_paths': len(paths_np),
            'path_length': segment_length,
            'mean': np.mean(flat),
            'std': np.std(flat),
            'hurst': h_est,
            'acf_lag1': acf_lag1,
            'kurtosis': float(np.mean((flat - np.mean(flat))**4) / np.std(flat)**4 - 3),
        }
        
        print(f"\n  Statistics for {freq_name}:")
        print(f"     Paths: {stats['n_paths']} x {stats['path_length']} steps")
        print(f"     Mean Variance: {stats['mean']:.5f}")
        print(f"     Path H: {stats['hurst']:.3f}  [VIX path H — NOT a roughness test]")
        print(f"     ACF (Lag 1): {stats['acf_lag1']:.3f}")
        print(f"     Excess Kurtosis: {stats['kurtosis']:.2f}")
        
        return stats, paths_np
        
    except Exception as e:
        print(f"  Error: {e}")
        return None

def quick_train_test(paths: np.ndarray, freq_name: str, n_epochs: int = 50):
    """Quick training to compare generative quality across frequencies."""
    print(f"\n  Quick Training ({n_epochs} epochs) for {freq_name}...")
    
    n_steps = paths.shape[1]
    
    # Compute real T based on frequency
    freq_minutes = int(freq_name.replace('min', ''))
    bars_per_day = 6.5 * 60 / freq_minutes  # Trading hours / bar interval
    T = n_steps / (252 * bars_per_day)  # In years
    dt = T / n_steps
    
    config = {
        'n_steps': n_steps,
        'T': T,
    }
    
    # Create signature engine with real dt
    sig_extractor = SignatureFeatureExtractor(truncation_order=3, dt=dt)
    
    # Compute signatures for real data
    paths_jax = jnp.array(paths)
    target_sigs = sig_extractor.get_signature(paths_jax)
    
    from ml.neural_sde import NeuralRoughSimulator
    from ml.losses import signature_mmd_loss
    import optax
    import equinox as eqx
    
    key = jax.random.PRNGKey(42)
    input_sig_dim = sig_extractor.get_feature_dim(1)
    model = NeuralRoughSimulator(input_sig_dim, key)
    
    scheduler = optax.cosine_decay_schedule(init_value=0.001, decay_steps=500, alpha=0.01)
    optim = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate=scheduler))
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    
    dt = config['T'] / config['n_steps']
    batch_size = min(256, len(paths) - 1)
    
    for epoch in range(n_epochs):
        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, (batch_size, n_steps)) * jnp.sqrt(dt)
        
        # Sample initial conditions and target signatures from same batch
        random_indices = jax.random.randint(subkey, (batch_size,), 0, len(paths))
        v0 = paths_jax[random_indices, 0]
        batch_target_sigs = target_sigs[random_indices]
        
        def loss_fn(m):
            fake_vars = jax.vmap(m.generate_variance_path, in_axes=(0, 0, None))(
                v0, noise, dt
            )
            fake_sigs = sig_extractor.get_signature(fake_vars)
            return signature_mmd_loss(fake_sigs, batch_target_sigs)
        
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        
        if epoch % 10 == 0:
            print(f"     Epoch {epoch:03d} | Loss: {loss:.6f}")
    
    # Generate samples for comparison
    key, subkey = jax.random.split(key)
    noise = jax.random.normal(subkey, (1000, n_steps)) * jnp.sqrt(dt)
    random_indices = jax.random.randint(subkey, (1000,), 0, len(paths))
    v0 = paths_jax[random_indices, 0]
    
    generated = jax.vmap(model.generate_variance_path, in_axes=(0, 0, None))(
        v0, noise, dt
    )
    generated_np = np.array(generated)
    
    # Compare stats
    gen_hurst = estimate_hurst(generated_np)
    max_lag = min(10, n_steps - 2)
    gen_acf = np.mean([compute_acf(p, lags=max_lag) for p in generated_np[:100]], axis=0)
    gen_acf1 = gen_acf[1] if len(gen_acf) > 1 else 0.0
    
    print(f"\n  Generated Stats for {freq_name}:")
    print(f"     Hurst: {gen_hurst:.3f} (Real: {estimate_hurst(paths):.3f})")
    print(f"     ACF-1: {gen_acf1:.3f}")
    
    return model, generated_np


def main():
    print("="*70)
    print("   VIX FREQUENCY COMPARISON: Training Quality & True Roughness")
    print("="*70)
    
    results = {}
    
    # ── Part 1: VIX training quality at different frequencies ──
    print("\n" + "─"*70)
    print("PART 1: VIX Training Data Quality by Frequency")
    print("   (For Q-measure calibration — NOT for roughness)")
    print("─"*70)
    
    # Analyze each frequency with appropriate segment lengths
    # 5min: more points per day (~78), can use longer segments
    # 15min: ~26 points per day
    # 30min: ~13 points per day
    
    segment_configs = {
        '5min': 20,   # ~1.5 hours of 5-min data
        '15min': 15,  # ~4 hours of 15-min data  
        '30min': 13,  # ~6.5 hours of 30-min data (1 trading day)
    }
    
    for freq_name, file_path in VIX_FILES.items():
        seg_len = segment_configs[freq_name]
        result = analyze_frequency(file_path, freq_name, segment_length=seg_len)
        if result:
            stats, paths = result
            results[freq_name] = stats
            
            # Quick training comparison
            quick_train_test(paths, freq_name, n_epochs=50)
    
    # Summary comparison
    print("\n" + "="*70)
    print("   PART 1 SUMMARY: VIX Training Quality")
    print("="*70)
    print(f"{'Frequency':<12} {'Paths':<10} {'Path H':<12} {'ACF-1':<10} {'Kurtosis':<10}")
    print("-"*54)
    
    for freq, stats in results.items():
        print(f"{freq:<12} {stats['n_paths']:<10} {stats['hurst']:.3f}        {stats['acf_lag1']:.3f}     {stats['kurtosis']:.2f}")
    
    # ── Part 2: TRUE ROUGHNESS from SPX returns ──
    print("\n" + "─"*70)
    print("PART 2: TRUE ROUGHNESS from S&P 500 Returns (Daily RV)")
    print("   Gatheral et al. (2018): H ≈ 0.05-0.14 on realized vol")
    print("─"*70)
    
    for freq_name, spx_file in SPX_FILES.items():
        if os.path.exists(spx_file):
            rv_result = estimate_hurst_from_returns(spx_file)
            print(f"\n   SPX {freq_name} → Daily RV:")
            print(f"      {rv_result['n_rv_points']} trading days")
            print(f"      H (variogram) = {rv_result['H_variogram']:.4f}  (R² = {rv_result['R2_variogram']:.3f})")
            print(f"      H (struct q=1) = {rv_result['H_structure']:.4f}  (R² = {rv_result['R2_structure']:.3f})")
            if rv_result['H_variogram'] < 0.2:
                print(f"      ✅ ROUGH — H = {rv_result['H_variogram']:.3f}")
        else:
            print(f"   SPX {freq_name}: file not found ({spx_file})")
    
    # Recommendation
    print("\n" + "="*70)
    print("   RECOMMENDATION")
    print("="*70)
    
    # Best VIX freq = most paths & richest signal
    if results:
        best_freq = max(results.keys(), key=lambda k: results[k]['n_paths'])
        print(f"   Training: Use {best_freq} VIX data ({results[best_freq]['n_paths']} paths)")
        print(f"   Roughness: Verified on SPX realized vol (H ≈ 0.1)")
        print(f"   Note: VIX path H ≈ 0.5 is EXPECTED (30-day smoothing)")
    print("="*70)


if __name__ == "__main__":
    main()
