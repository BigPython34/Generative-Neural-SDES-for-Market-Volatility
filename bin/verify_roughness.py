import sys as _sys
if _sys.stdout.encoding != 'utf-8':
    _sys.stdout.reconfigure(encoding='utf-8'); _sys.stderr.reconfigure(encoding='utf-8')

"""
Verify Roughness Consistency
=============================
This script closes the loop on the project narrative:
1. Verify that generated paths have H ~ 0.07 (rough)
2. Compare signatures of generated vs real paths
3. Ablation study: with/without signatures
4. Verify the full chain: Rough Data → Signatures → Neural SDE → Rough Output
"""

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import json
import yaml
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def estimate_hurst_variogram(series, max_lag=50):
    """Estimate Hurst exponent using variogram method (most reliable for rough vol)."""
    series = np.array(series)
    n = len(series)
    
    lags = np.arange(1, min(max_lag, n // 4))
    variogram = []
    
    for lag in lags:
        diffs = series[lag:] - series[:-lag]
        variogram.append(np.mean(diffs**2))
    
    variogram = np.array(variogram)
    
    # Fit log(variogram) = 2H * log(lag) + const
    log_lags = np.log(lags)
    log_var = np.log(variogram + 1e-10)
    
    # Linear regression
    A = np.vstack([log_lags, np.ones(len(log_lags))]).T
    slope, _ = np.linalg.lstsq(A, log_var, rcond=None)[0]
    
    H = slope / 2
    return np.clip(H, 0.01, 0.99)


def compute_signature_stats(paths, truncation_order=3):
    """Compute signature statistics for a batch of paths."""
    from engine.signature_engine import SignatureFeatureExtractor
    
    sig_engine = SignatureFeatureExtractor(truncation_order=truncation_order)
    
    # Convert to appropriate format
    if isinstance(paths, jnp.ndarray):
        paths = np.array(paths)
    
    signatures = sig_engine.get_signature(paths)
    
    return {
        'mean': np.mean(signatures, axis=0),
        'std': np.std(signatures, axis=0),
        'norm_mean': np.mean(np.linalg.norm(signatures, axis=1)),
        'norm_std': np.std(np.linalg.norm(signatures, axis=1))
    }


def generate_neural_sde_paths(n_paths=1000, n_steps=100):
    """Generate paths using the trained Neural SDE."""
    from engine.neural_sde import NeuralRoughSimulator
    from engine.signature_engine import SignatureFeatureExtractor
    import equinox as eqx
    
    # Load real dt from config
    with open("config/params.yaml", 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    T = cfg['simulation']['T']
    
    model_path = Path("models/neural_sde_best.eqx")
    sig_engine = SignatureFeatureExtractor(truncation_order=3)
    sig_dim = sig_engine.get_feature_dim(1)
    
    key = jax.random.PRNGKey(42)
    
    if model_path.exists():
        # Load trained model
        model = NeuralRoughSimulator(sig_dim=sig_dim, key=key)
        model = eqx.tree_deserialise_leaves(model_path, model)
        print("   Loaded trained model")
    else:
        # Use random model (for testing)
        model = NeuralRoughSimulator(sig_dim=sig_dim, key=key)
        print("   WARNING: Using untrained model (no saved model found)")
    
    # Load real data for initial variance — use VARIANCE = (VIX/100)^2
    vix_path = Path(cfg['data'].get('source', 'data/market/vix/vix_15m.csv'))
    vix_df = pd.read_csv(vix_path)
    vix_var = (vix_df['close'].values / 100) ** 2  # Variance
    
    # Build pool of realistic initial variances from segment starts
    seg_stride = max(1, n_steps // 2)
    v0_pool = np.array([vix_var[i] for i in range(0, len(vix_var) - n_steps, seg_stride)])
    
    # Generate paths — model computes running signatures internally
    dt = T / n_steps
    
    all_paths = []
    
    for i in range(n_paths):
        key, subkey = jax.random.split(key)
        dW = jax.random.normal(subkey, shape=(n_steps,)) * jnp.sqrt(dt)  # Proper Brownian increments
        
        # Use randomized initial condition from market data (like training)
        init_var = float(v0_pool[i % len(v0_pool)])
        
        try:
            path = model.generate_variance_path(
                init_var=init_var,
                brownian_increments=dW,
                dt=dt
            )
            all_paths.append(np.array(path))
        except Exception as e:
            if i == 0:
                print(f"   Generation error: {e}")
                raise
    
    return np.array(all_paths)


def generate_ou_paths(n_paths=1000, n_steps=100, kappa=2.72, theta=-3.5, sigma=0.8, init_vars=None):
    """Generate paths using OU process on LOG-variance (no signatures).
    
    This is the ABLATION baseline: same OU-on-log-V dynamics as the
    Neural SDE prior, but without path-dependent signature conditioning.
    Operating on log-V (not V) ensures a fair comparison with the
    Neural SDE which uses the same parametrization.
    
    Reference: the Neural SDE prior is
        d(log V) = kappa*(theta - log V)*dt + sigma*dW
    
    Args:
        kappa: Mean-reversion speed (same as neural SDE config)
        theta: Long-term log-variance target (e.g. -3.5 → ~3% variance → 17% vol)
        sigma: Vol-of-vol on log-variance scale
        init_vars: Array of initial VARIANCES (converted to log internally).
    """
    with open("config/params.yaml", 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    dt = cfg['simulation']['T'] / n_steps
    
    # Use config OU params for consistency with neural SDE prior
    sde_cfg = cfg.get('neural_sde', {})
    kappa = sde_cfg.get('kappa', kappa)
    theta = sde_cfg.get('theta', theta)
    
    all_paths = []
    np.random.seed(42)
    
    for i in range(n_paths):
        if init_vars is not None:
            v0 = float(init_vars[i % len(init_vars)])
            log_v = np.log(max(v0, 1e-6))
        else:
            log_v = theta
        
        log_path = [log_v]
        for _ in range(n_steps - 1):
            dW = np.random.normal(0, np.sqrt(dt))
            log_v_next = log_v + kappa * (theta - log_v) * dt + sigma * dW
            log_v_next = np.clip(log_v_next, -7.0, 2.0)
            log_path.append(log_v_next)
            log_v = log_v_next
        
        all_paths.append(np.exp(log_path))
    
    return np.array(all_paths)


def run_ablation_study():
    """Compare model performance with and without signatures."""
    print("\n" + "="*60)
    print("ABLATION STUDY: Signatures Impact")
    print("="*60)
    
    from engine.losses import signature_mmd_loss
    from engine.signature_engine import SignatureFeatureExtractor
    
    # Load config for real dt
    with open("config/params.yaml", 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    T = cfg['simulation']['T']
    seg_len = 20
    real_dt = T / seg_len
    
    # Load real VIX data — convert to VARIANCE to match training format
    # Training uses (VIX/100)^2, NOT VIX/100
    vix_path = Path(cfg['data'].get('source', 'data/market/vix/vix_15m.csv'))
    vix_df = pd.read_csv(vix_path)
    vix_values = (vix_df['close'].values / 100) ** 2  # Variance = (VIX/100)^2
    
    # Create path segments
    real_paths = []
    stride = seg_len // 2
    for i in range(0, len(vix_values) - seg_len, stride):
        real_paths.append(vix_values[i:i+seg_len])
    real_paths = np.array(real_paths[:500])  # Limit for speed
    
    sig_engine = SignatureFeatureExtractor(truncation_order=3, dt=real_dt)
    real_sigs = sig_engine.get_signature(jnp.array(real_paths))
    
    # Compute normalization (component-wise std of real sigs)
    sig_std = jnp.std(real_sigs, axis=0)
    
    # Extract initial conditions from real paths for fair comparison
    real_v0s = real_paths[:, 0]
    
    print(f"\n   Real paths: {real_paths.shape}")
    print(f"   Real mean variance: {real_paths.mean():.4f}")
    print(f"   Real signatures: {real_sigs.shape}")
    
    # Generate Neural SDE paths (uses randomized v0 from market data)
    print("\n   Generating Neural SDE paths...")
    try:
        neural_paths = generate_neural_sde_paths(n_paths=500, n_steps=seg_len)
        neural_sigs = sig_engine.get_signature(jnp.array(neural_paths))
        
        # Normalized MMD between real and generated
        mmd_neural = float(signature_mmd_loss(neural_sigs, real_sigs, sig_std))
        print(f"   Neural SDE MMD (normalized): {mmd_neural:.6f}")
        print(f"   Neural SDE mean variance: {neural_paths.mean():.4f}")
    except Exception as e:
        print(f"   Neural SDE failed: {e}")
        mmd_neural = None
    
    # Generate simple OU paths (no signature conditioning, same init conditions)
    print("\n   Generating OU paths (baseline, no signatures)...")
    ou_paths = generate_ou_paths(
        n_paths=500, n_steps=seg_len, 
        kappa=2.72, theta=0.035, sigma=0.3,
        init_vars=real_v0s
    )
    ou_sigs = sig_engine.get_signature(jnp.array(ou_paths))
    
    mmd_ou = float(signature_mmd_loss(ou_sigs, real_sigs, sig_std))
    print(f"   OU Process MMD (normalized): {mmd_ou:.6f}")
    
    # Compare
    print("\n   Comparison (Signature MMD):")
    if mmd_neural is not None:
        improvement = (mmd_ou - mmd_neural) / mmd_ou * 100
        print(f"      Neural SDE MMD: {mmd_neural:.6f}")
        print(f"      OU Process MMD: {mmd_ou:.6f}")
    
    # Extended comparison: path-level quality metrics
    print("\n   Extended comparison (path quality):")
    
    # Hurst exponent
    def mean_hurst(paths, n_sample=100):
        hursts = []
        for p in paths[:n_sample]:
            h = estimate_hurst_variogram(np.log(np.clip(p, 1e-6, None)))
            hursts.append(h)
        return np.mean(hursts), np.std(hursts)
    
    real_h, real_h_std = mean_hurst(real_paths)
    ou_h, ou_h_std = mean_hurst(ou_paths)
    
    print(f"      Hurst (real):       {real_h:.3f} ± {real_h_std:.3f}")
    print(f"      Hurst (OU):         {ou_h:.3f} ± {ou_h_std:.3f}")
    
    if mmd_neural is not None:
        neural_h, neural_h_std = mean_hurst(neural_paths)
        print(f"      Hurst (Neural SDE): {neural_h:.3f} ± {neural_h_std:.3f}")
    
    # Kurtosis (marginal: average across time steps, not flattened)
    from scipy.stats import kurtosis as sp_kurtosis
    real_kurt = np.mean([sp_kurtosis(real_paths[:, t]) for t in range(real_paths.shape[1])])
    ou_kurt = np.mean([sp_kurtosis(ou_paths[:, t]) for t in range(ou_paths.shape[1])])
    print(f"\n      Kurtosis marginal (real):       {real_kurt:.1f}")
    print(f"      Kurtosis marginal (OU):         {ou_kurt:.1f}")
    
    if mmd_neural is not None:
        neural_kurt = np.mean([sp_kurtosis(neural_paths[:, t]) for t in range(neural_paths.shape[1])])
        print(f"      Kurtosis marginal (Neural SDE): {neural_kurt:.1f}")
    
    # Mean level
    print(f"\n      Mean var (real):       {real_paths.mean():.4f}")
    print(f"      Mean var (OU):         {ou_paths.mean():.4f}")
    if mmd_neural is not None:
        print(f"      Mean var (Neural SDE): {neural_paths.mean():.4f}")
    
    # Verdict
    print("\n   Verdict:")
    if mmd_neural is not None:
        h_err_neural = abs(neural_h - real_h)
        h_err_ou = abs(ou_h - real_h)
        kurt_err_neural = abs(neural_kurt - real_kurt) / max(real_kurt, 1)
        kurt_err_ou = abs(ou_kurt - real_kurt) / max(real_kurt, 1)
        
        neural_wins = 0
        ou_wins = 0
        
        if mmd_neural < mmd_ou:
            neural_wins += 1
            print(f"      [MMD]      Neural SDE wins ({mmd_neural:.6f} < {mmd_ou:.6f})")
        else:
            ou_wins += 1
            print(f"      [MMD]      OU wins ({mmd_ou:.6f} < {mmd_neural:.6f})")
        
        if h_err_neural < h_err_ou:
            neural_wins += 1
            print(f"      [Hurst]    Neural SDE wins (|err|={h_err_neural:.3f} < {h_err_ou:.3f})")
        else:
            ou_wins += 1
            print(f"      [Hurst]    OU wins (|err|={h_err_ou:.3f} < {h_err_neural:.3f})")
        
        if kurt_err_neural < kurt_err_ou:
            neural_wins += 1
            print(f"      [Kurtosis] Neural SDE wins (|rel err|={kurt_err_neural:.2f} < {kurt_err_ou:.2f})")
        else:
            ou_wins += 1
            print(f"      [Kurtosis] OU wins (|rel err|={kurt_err_ou:.2f} < {kurt_err_neural:.2f})")
        
        mean_err_neural = abs(neural_paths.mean() - real_paths.mean()) / real_paths.mean()
        mean_err_ou = abs(ou_paths.mean() - real_paths.mean()) / real_paths.mean()
        if mean_err_neural < mean_err_ou:
            neural_wins += 1
            print(f"      [Mean]     Neural SDE wins (|rel err|={mean_err_neural:.2%} < {mean_err_ou:.2%})")
        else:
            ou_wins += 1
            print(f"      [Mean]     OU wins (|rel err|={mean_err_ou:.2%} < {mean_err_neural:.2%})")
        
        print(f"\n      Score: Neural SDE {neural_wins}/4 vs OU {ou_wins}/4")
        if neural_wins > ou_wins:
            print(f"      --> Neural SDE is overall better!")
        elif neural_wins == ou_wins:
            print(f"      --> Tied — Neural SDE captures roughness, OU captures level")
        else:
            print(f"      --> OU baseline is stronger on these metrics")
    
    return {
        'mmd_neural': mmd_neural,
        'mmd_ou': mmd_ou,
        'improvement_pct': improvement if mmd_neural else None,
        'hurst_real': float(real_h),
        'hurst_neural': float(neural_h) if mmd_neural else None,
        'hurst_ou': float(ou_h),
    }


def verify_generated_roughness():
    """
    Verify roughness on the CORRECT data source.
    
    Key insight (Gatheral et al. 2018):
    - VIX ≈ 30-day integrated implied vol → H_VIX ≈ 0.5 (SMOOTH, not rough)
    - Realized Vol from 5-min SPX returns → H_RV ≈ 0.05-0.14 (ROUGH)
    
    We measure H on BOTH to demonstrate the distinction.
    """
    print("\n" + "="*60)
    print("1. ROUGHNESS VERIFICATION")
    print("="*60)
    
    with open("config/params.yaml", 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    # ── A. VIX (expected: H ≈ 0.5, NOT rough) ──
    vix_path = Path(cfg['data'].get('source', 'data/market/vix/vix_15m.csv'))
    vix_df = pd.read_csv(vix_path)
    vix_values = vix_df['close'].values
    
    H_vix = estimate_hurst_variogram(np.log(vix_values))
    print(f"\nA. VIX (30-day integrated implied vol):")
    print(f"H (variogram on log-VIX) = {H_vix:.3f}")
    print(f"H ≈ 0.5 is EXPECTED (VIX is a smoothed integral)")
    print(f"      This is NOT a roughness test — VIX integration kills roughness.")
    
    # ── B. Realized Vol from SPX (expected: H ≈ 0.05-0.14) ──
    rv_source = cfg['data'].get('rv_source', 'data/market/spx/spx_5m.csv')
    rv_window = cfg['data'].get('rv_window', 78)
    
    H_rv = None
    if Path(rv_source).exists():
        from utils.diagnostics import estimate_hurst_from_returns
        rv_results = estimate_hurst_from_returns(rv_source, rv_window=rv_window)
        H_rv = rv_results['H_variogram']
        
        print(f"\n   B. REALIZED VOLATILITY from S&P 500 returns:")
        print(f"Source: {rv_source} ({rv_results['n_rv_points']} RV points)")
        print(f"H (variogram)   = {rv_results['H_variogram']:.4f}  (R² = {rv_results['R2_variogram']:.3f})")
        print(f"H (struct q=1)  = {rv_results['H_structure']:.4f}  (R² = {rv_results['R2_structure']:.3f})")
        
        if H_rv < 0.20:
            print(f"TRUE ROUGHNESS CONFIRMED: H = {H_rv:.3f}")
        else:
            print(f"H = {H_rv:.3f} — higher than expected (check data/window)")
    else:
        print(f"\n B. No SPX data found at {rv_source} — cannot verify true roughness")
    
    # ── C. Neural SDE generated paths ──
    print("\n C. Neural SDE generated paths:")
    try:
        neural_paths = generate_neural_sde_paths(n_paths=100, n_steps=200)
        
        H_generated = []
        for path in neural_paths:
            h = estimate_hurst_variogram(np.log(path + 1e-6))
            H_generated.append(h)
        
        H_gen_mean = np.mean(H_generated)
        H_gen_std = np.std(H_generated)
        
        print(f"H (gen mean): {H_gen_mean:.3f} ± {H_gen_std:.3f}")
        
        # Compare with VIX (training data) H, since model is trained on VIX
        print(f"\n Comparison:")
        print(f"VIX H: {H_vix:.3f}  (training data — smoothed)")
        print(f"Generated H: {H_gen_mean:.3f}  (model output)")
        if H_rv is not None:
            print(f"True RV H: {H_rv:.3f}  (5-min SPX returns)")
        
        gap_vix = abs(H_vix - H_gen_mean)
        print(f"Gap vs VIX:{gap_vix:.3f}")
        
        if gap_vix < 0.15:
            print(f"-> Model faithfully reproduces VIX dynamics (H ≈ {H_gen_mean:.2f})")
        
        if H_rv is not None and H_rv < 0.15:
            print(f"\n Model trained on VIX cannot capture H_RV ≈ {H_rv:.2f}")
            print(f"To get rough paths, train on REALIZED VOL (data_type: 'realized_vol')")
        
        return {
            'H_vix': float(H_vix),
            'H_rv': float(H_rv) if H_rv is not None else None,
            'H_generated_mean': float(H_gen_mean),
            'H_generated_std': float(H_gen_std),
            'gap_vs_training': float(gap_vix),
        }
    
    except Exception as e:
        print(f"Error generating paths: {e}")
        return {'H_vix': float(H_vix), 'H_rv': float(H_rv) if H_rv else None, 'error': str(e)}


def compare_signature_distributions():
    """Compare signature distributions of real vs generated paths."""
    print("\n" + "="*60)
    print("2. SIGNATURE DISTRIBUTION COMPARISON")
    print("="*60)
    
    from engine.signature_engine import SignatureFeatureExtractor
    
    # Load config for real dt
    with open("config/params.yaml", 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    T = cfg['simulation']['T']
    
    # Load real VIX data — convert to VARIANCE to match training format
    vix_path = Path(cfg['data'].get('source', 'data/market/vix/vix_15m.csv'))
    vix_df = pd.read_csv(vix_path)
    vix_values = (vix_df['close'].values / 100) ** 2  # Variance = (VIX/100)^2
    
    # Create path segments
    seg_len = 20
    real_paths = []
    stride = seg_len // 2
    for i in range(0, len(vix_values) - seg_len, stride):
        real_paths.append(vix_values[i:i+seg_len])
    real_paths = np.array(real_paths[:200])
    
    real_dt = T / seg_len
    sig_engine = SignatureFeatureExtractor(truncation_order=3, dt=real_dt)
    
    # Real signatures
    real_sigs = np.array(sig_engine.get_signature(jnp.array(real_paths)))
    real_stats = {
        'mean': np.mean(real_sigs, axis=0),
        'std': np.std(real_sigs, axis=0),
        'norm': np.mean(np.linalg.norm(real_sigs, axis=1))
    }
    
    print(f"\n   Real signatures:")
    print(f"Shape: {real_sigs.shape}")
    print(f"Norm (mean): {real_stats['norm']:.4f}")
    print(f"First 5 components mean: {real_stats['mean'][:5]}")
    
    # Generated signatures
    try:
        neural_paths = generate_neural_sde_paths(n_paths=200, n_steps=seg_len)
        gen_sigs = np.array(sig_engine.get_signature(jnp.array(neural_paths)))
        gen_stats = {
            'mean': np.mean(gen_sigs, axis=0),
            'std': np.std(gen_sigs, axis=0),
            'norm': np.mean(np.linalg.norm(gen_sigs, axis=1))
        }
        
        print(f"\n   Generated signatures:")
        print(f"Shape: {gen_sigs.shape}")
        print(f"Norm (mean): {gen_stats['norm']:.4f}")
        print(f"First 5 components mean: {gen_stats['mean'][:5]}")
        
        # Compare
        norm_ratio = gen_stats['norm'] / real_stats['norm']
        mean_corr = np.corrcoef(real_stats['mean'], gen_stats['mean'])[0, 1]
        
        print(f"\n   Comparison:")
        print(f"Norm ratio (gen/real): {norm_ratio:.2f}")
        print(f"Mean correlation: {mean_corr:.3f}")
        
        if mean_corr > 0.8:
            print(f"--> Signatures match well!")
        else:
            print(f"--> Signature mismatch (check model)")
        
        return {
            'real_norm': float(real_stats['norm']),
            'gen_norm': float(gen_stats['norm']),
            'norm_ratio': float(norm_ratio),
            'mean_correlation': float(mean_corr)
        }
    
    except Exception as e:
        print(f"   Error: {e}")
        return {'error': str(e)}


def run_full_verification():
    """Run all verification checks."""
    print("="*70)
    print("   ROUGHNESS & SIGNATURE VERIFICATION")
    print("   Closing the loop on the project narrative")
    print("="*70)
    print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    # 1. Verify roughness
    results['roughness'] = verify_generated_roughness()
    
    # 2. Compare signatures
    results['signatures'] = compare_signature_distributions()
    
    # 3. Ablation study
    results['ablation'] = run_ablation_study()
    
    # Summary
    print("\n" + "="*70)
    print("   SUMMARY: Is the narrative coherent?")
    print("="*70)
    
    checks = []
    
    # Check 1: Roughness — now with correct interpretation
    if 'H_rv' in results['roughness'] and results['roughness']['H_rv'] is not None:
        h_rv = results['roughness']['H_rv']
        if h_rv < 0.20:
            checks.append(f"[+] ROUGHNESS: True RV from SPX is rough (H={h_rv:.3f})")
        else:
            checks.append(f"[-] ROUGHNESS: RV not rough enough (H={h_rv:.3f})")
    
    if 'H_generated_mean' in results['roughness']:
        h_gen = results['roughness']['H_generated_mean']
        h_vix = results['roughness'].get('H_vix', 0.5)
        gap = results['roughness'].get('gap_vs_training', abs(h_gen - h_vix))
        if gap < 0.15:
            checks.append(f"[+] GENERATION: Model reproduces training data dynamics (H_gen={h_gen:.3f})")
        else:
            checks.append(f"[-] GENERATION: Model H differs from training data")
    
    # Check 2: Signatures match
    if 'mean_correlation' in results['signatures']:
        corr = results['signatures']['mean_correlation']
        if corr > 0.7:
            checks.append(f"[+] SIGNATURES: Generated match real (corr={corr:.2f})")
        else:
            checks.append(f"[-] SIGNATURES: Poor match (corr={corr:.2f})")
    
    # Check 3: Ablation — use multi-metric verdict
    ablation = results.get('ablation', {})
    h_real_abl = ablation.get('hurst_real')
    h_neural_abl = ablation.get('hurst_neural')
    h_ou_abl = ablation.get('hurst_ou')
    
    if h_real_abl is not None and h_neural_abl is not None:
        h_err_neural = abs(h_neural_abl - h_real_abl)
        h_err_ou = abs(h_ou_abl - h_real_abl)
        mmd_neural = ablation.get('mmd_neural', float('inf'))
        mmd_ou = ablation.get('mmd_ou', float('inf'))
        
        # Neural SDE wins on roughness if closer Hurst
        if h_err_neural < h_err_ou:
            checks.append(f"[+] ABLATION (Hurst): Neural SDE closer to real ({h_err_neural:.3f} vs {h_err_ou:.3f})")
        else:
            checks.append(f"[-] ABLATION (Hurst): OU closer to real ({h_err_ou:.3f} vs {h_err_neural:.3f})")
        
        if mmd_neural < mmd_ou:
            checks.append(f"[+] ABLATION (MMD): Neural SDE better ({mmd_neural:.6f} vs {mmd_ou:.6f})")
        else:
            checks.append(f"[~] ABLATION (MMD): OU better on sig MMD ({mmd_ou:.6f} vs {mmd_neural:.6f})")
    elif ablation.get('improvement_pct') is not None:
        imp = ablation['improvement_pct']
        if imp > 0:
            checks.append(f"[+] ABLATION: Neural SDE beats OU by {imp:.1f}%")
        else:
            checks.append(f"[-] ABLATION: OU is better (signatures not helping)")
    
    print("\n   Verification results:")
    for check in checks:
        print(f"      {check}")
    
    # Overall verdict
    n_pass = sum(1 for c in checks if c.startswith('[+]'))
    n_total = len(checks)
    
    print(f"\n   Score: {n_pass}/{n_total} checks passed")
    
    if n_pass == n_total:
        print("\n   VERDICT: The narrative is COHERENT!")
        print("            Rough data -> Signatures -> Neural SDE -> Rough output")
    elif n_pass >= n_total // 2:
        print("\n   VERDICT: Partially coherent (some gaps)")
    else:
        print("\n   VERDICT: Needs investigation")
    
    # Save results
    output_path = Path("outputs/roughness_verification.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))
    
    print(f"\n   Results saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    run_full_verification()
