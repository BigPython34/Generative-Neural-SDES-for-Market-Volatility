"""
Frequency Comparison Script
===========================
Compare VIX data at different frequencies (5, 15, 30 min) 
to find the best resolution for rough volatility modeling.
"""

import jax
import jax.numpy as jnp
import numpy as np
import os
from utils.data_loader import MarketDataLoader, load_config
from utils.diagnostics import print_distribution_stats, compute_acf, estimate_hurst
from ml.generative_trainer import GenerativeTrainer
from ml.signature_engine import SignatureFeatureExtractor

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# VIX data files at different frequencies
VIX_FILES = {
    '5min': 'data/TVC_VIX, 5.csv',
    '15min': 'data/TVC_VIX, 15.csv',
    '30min': 'data/TVC_VIX, 30.csv',
}

def analyze_frequency(file_path: str, freq_name: str, segment_length: int = 13):
    """Analyze VIX data at a specific frequency."""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {freq_name} VIX DATA")
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
        print(f"     Hurst Exponent: {stats['hurst']:.3f} {'ROUGH' if stats['hurst'] < 0.2 else 'not rough'}")
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
    print("   VIX FREQUENCY COMPARISON: Finding Optimal Roughness Resolution")
    print("="*70)
    
    results = {}
    
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
    print("   SUMMARY COMPARISON")
    print("="*70)
    print(f"{'Frequency':<12} {'Paths':<10} {'Hurst':<12} {'ACF-1':<10} {'Kurtosis':<10}")
    print("-"*54)
    
    for freq, stats in results.items():
        hurst_status = "*" if stats['hurst'] < 0.2 else ""
        print(f"{freq:<12} {stats['n_paths']:<10} {stats['hurst']:.3f} {hurst_status:<6} {stats['acf_lag1']:.3f}     {stats['kurtosis']:.2f}")
    
    # Recommendation
    print("\n" + "-"*70)
    best_freq = min(results.keys(), key=lambda k: results[k]['hurst'])
    print(f"RECOMMENDATION: Use {best_freq} data (Hurst = {results[best_freq]['hurst']:.3f})")
    print("-"*70)


if __name__ == "__main__":
    main()
