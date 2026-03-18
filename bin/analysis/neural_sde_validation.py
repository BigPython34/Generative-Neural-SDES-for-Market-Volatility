import sys as _sys
if _sys.stdout.encoding != 'utf-8':
    _sys.stdout.reconfigure(encoding='utf-8'); _sys.stderr.reconfigure(encoding='utf-8')

"""
Neural SDE Validation
=====================
This script closes the loop on the project narrative:
1. Verify that generated paths have H ~ 0.07 (rough)
2. Compare signatures of generated vs real paths
3. Ablation study: with/without signatures
4. Verify the full chain: Rough Data → Signatures → Neural SDE → Rough Output
"""

import numpy as np
import pandas as pd

# Configure JAX to use CPU only to avoid DLL issues
import os as _os
_os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp
import json
import yaml
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Import from consolidated Hurst estimation module
from quant.analysis.hurst_estimation import hurst_variogram


def _get_base_dt(cfg: dict) -> float:
    """Base dt used by the simulator/training (e.g., 15m in year units)."""
    sim_cfg = cfg.get('simulation', {})
    T = float(sim_cfg['T'])
    n_steps = int(sim_cfg.get('n_steps', 120))
    return T / n_steps


def estimate_hurst_variogram(series, max_lag=None):
    """Estimate Hurst exponent via variogram and return only H."""
    estimate = hurst_variogram(series, max_lag=max_lag)
    return estimate.H


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


def generate_neural_sde_paths(
    n_paths=1000,
    n_steps=100,
    dt: float | None = None,
    init_vars: np.ndarray | None = None,
    progress_interval=100,
):
    """Generate paths using the trained Neural SDE.

    IMPORTANT: For fair validation, `dt` must match the data sampling interval.
    When generating shorter segments (smaller `n_steps`), keep `dt` fixed
    (e.g., 15m) so we compare like-with-like.

    If `init_vars` is provided, it is used as the pool of initial variances.
    """
    from engine.neural_sde import NeuralRoughSimulator
    from engine.signature_engine import SignatureFeatureExtractor
    import equinox as eqx

    with open("config/params.yaml", 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    base_dt = _get_base_dt(cfg)
    if dt is None:
        dt = base_dt
    config_path = "config/params.yaml"
    sig_engine = SignatureFeatureExtractor(truncation_order=4)  # Order 4 matches trained model
    sig_dim = sig_engine.get_feature_dim(1)
    key = jax.random.PRNGKey(42)

    # Prefer P-measure model (trained on realized vol); avoid Q (pricing)
    model_path = Path("models/neural_sde_best_p.eqx")
    if not model_path.exists():
        model_path = Path("models/neural_sde_best.eqx")
    
    if model_path.exists():
        # Detect if model was trained with jumps (check for _jump in filename or try loading with jumps)
        enable_jumps_try = "_jump" in str(model_path)
        model = NeuralRoughSimulator(sig_dim=sig_dim, key=key, config_path=config_path, enable_jumps=enable_jumps_try)
        model = eqx.tree_deserialise_leaves(model_path, model)
        if progress_interval and n_paths > progress_interval:
            jump_str = " + jumps" if enable_jumps_try else ""
            print(f"   Loaded {model_path.name}{jump_str} (P = roughness). Generating {n_paths} paths...", flush=True)
    else:
        raise FileNotFoundError(
            f"No trained model found! Expected models/neural_sde_best_p.eqx or models/neural_sde_best.eqx. "
            f"Run `python bin/training/train_multi.py --measure P` first."
        )

    v0_pool = np.asarray(init_vars, dtype=float)
    
    all_paths = []

    for i in range(n_paths):
        key, subkey = jax.random.split(key)
        dW = jax.random.normal(subkey, shape=(n_steps,)) * jnp.sqrt(dt)
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
        if progress_interval and (i + 1) % progress_interval == 0 and (i + 1) < n_paths:
            print(f"      ... {i + 1}/{n_paths} paths", flush=True)

    return np.array(all_paths)


def generate_ou_paths(
    n_paths=1000,
    n_steps=100,
    dt: float | None = None,
    kappa=2.72,
    theta=-3.5,
    sigma=0.8,
    init_vars=None,
):
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
    base_dt = _get_base_dt(cfg)
    if dt is None:
        dt = base_dt
    
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
    """Compare model performance with and without signatures. Uses P model."""
    from engine.losses import kernel_mmd_loss
    from utils.loader.RealizedVariance import RealizedVolatilityLoader

    print("   3a. Loading real paths and signatures...", flush=True)
    from engine.signature_engine import SignatureFeatureExtractor
    
    # Load config for base dt
    with open("config/params.yaml", 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    seg_len = 20  # short segments for signature-MMD (fast)
    seg_len_metrics = 200  # long segments for Hurst/kurtosis (stable)
    real_dt = _get_base_dt(cfg)
    
    # Load real REALIZED VOLATILITY data (same as training data)
    # This is crucial: must match the training distribution!
    loader = RealizedVolatilityLoader(config_path='config/params.yaml')
    real_paths = np.array(
        loader.get_realized_vol_paths(segment_length=seg_len, seed=123, shuffle=True)[:500]
    )  # Limit for speed
    
    # Use order 3 for ablation (model trained on order 3)
    sig_engine_ablation = SignatureFeatureExtractor(truncation_order=4, dt=real_dt)
    real_sigs = sig_engine_ablation.get_signature(jnp.array(real_paths))
    
    # Compute normalization (component-wise std of real sigs)
    sig_std = jnp.std(real_sigs, axis=0)
    
    # Extract initial conditions from real paths for fair comparison
    real_v0s = real_paths[:, 0]
    
    print(f"\n   Real paths: {real_paths.shape}")
    print(f"   Real mean variance: {real_paths.mean():.4f}")
    print(f"   Real signatures: {real_sigs.shape}")
    
    print("   3b. Generating Neural SDE paths (500)...", flush=True)
    try:
        neural_paths = generate_neural_sde_paths(
            n_paths=500, n_steps=seg_len, dt=real_dt, init_vars=real_v0s, progress_interval=100
        )
        neural_sigs = sig_engine_ablation.get_signature(jnp.array(neural_paths))
        
        # Kernel MMD between real and generated (same metric used in training)
        mmd_neural = float(kernel_mmd_loss(neural_sigs, real_sigs, sig_std))
        print(f"   Neural SDE MMD (kernel, normalized): {mmd_neural:.6f}")
        print(f"   Neural SDE mean variance: {neural_paths.mean():.4f}")
    except Exception as e:
        print(f"   Neural SDE failed: {e}")
        mmd_neural = None
    
    print("   3c. Generating OU baseline paths (500)...", flush=True)
    ou_paths = generate_ou_paths(
        n_paths=500, n_steps=seg_len, 
        dt=real_dt,
        kappa=2.72, theta=0.035, sigma=0.3,
        init_vars=real_v0s
    )
    ou_sigs = sig_engine_ablation.get_signature(jnp.array(ou_paths))
    
    mmd_ou = float(kernel_mmd_loss(ou_sigs, real_sigs, sig_std))
    print(f"   OU Process MMD (kernel, normalized): {mmd_ou:.6f}")
    
    improvement = None
    print("\n   Comparison (Signature MMD):")
    if mmd_neural is not None:
        improvement = (mmd_ou - mmd_neural) / mmd_ou * 100
        print(f"      Neural SDE MMD: {mmd_neural:.6f}")
        print(f"      OU Process MMD: {mmd_ou:.6f}")
    
    # Extended comparison: path-level quality metrics
    # IMPORTANT: Hurst/kurtosis on very short paths (e.g. 20 steps) is not stable.
    # We recompute these metrics on longer segments at the same dt.
    print("\n   Extended comparison (path quality, long segments):")

    real_paths_long = np.array(
        loader.get_realized_vol_paths(segment_length=seg_len_metrics, seed=123, shuffle=True)[:200]
    )
    real_v0s_long = real_paths_long[:, 0]

    neural_paths_long = None
    if mmd_neural is not None:
        neural_paths_long = generate_neural_sde_paths(
            n_paths=200, n_steps=seg_len_metrics, dt=real_dt, init_vars=real_v0s_long, progress_interval=100
        )

    ou_paths_long = generate_ou_paths(
        n_paths=200,
        n_steps=seg_len_metrics,
        dt=real_dt,
        kappa=2.72,
        theta=0.035,
        sigma=0.3,
        init_vars=real_v0s_long,
    )

    def mean_hurst(paths, n_sample=100):
        hursts = []
        for p in paths[:n_sample]:
            h = estimate_hurst_variogram(np.log(np.clip(p, 1e-12, None)))
            hursts.append(h)
        return float(np.mean(hursts)), float(np.std(hursts))

    real_h, real_h_std = mean_hurst(real_paths_long)
    ou_h, ou_h_std = mean_hurst(ou_paths_long)
    print(f"      Hurst (real):       {real_h:.3f} ± {real_h_std:.3f}")
    print(f"      Hurst (OU):         {ou_h:.3f} ± {ou_h_std:.3f}")
    if neural_paths_long is not None:
        neural_h, neural_h_std = mean_hurst(neural_paths_long)
        print(f"      Hurst (Neural SDE): {neural_h:.3f} ± {neural_h_std:.3f}")

    from scipy.stats import kurtosis as sp_kurtosis
    def marginal_log_kurt(paths):
        logp = np.log(np.clip(paths, 1e-12, None))
        return float(np.mean([sp_kurtosis(logp[:, t], fisher=True, bias=False) for t in range(logp.shape[1])]))

    real_kurt = marginal_log_kurt(real_paths_long)
    ou_kurt = marginal_log_kurt(ou_paths_long)
    print(f"\n      Kurtosis marginal (real):       {real_kurt:.1f}")
    print(f"      Kurtosis marginal (OU):         {ou_kurt:.1f}")
    if neural_paths_long is not None:
        neural_kurt = marginal_log_kurt(neural_paths_long)
        print(f"      Kurtosis marginal (Neural SDE): {neural_kurt:.1f}")

    print(f"\n      Mean var (real):       {real_paths_long.mean():.4f}")
    print(f"      Mean var (OU):         {ou_paths_long.mean():.4f}")
    if neural_paths_long is not None:
        print(f"      Mean var (Neural SDE): {neural_paths_long.mean():.4f}")
    
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
        'hurst_neural': float(neural_h) if mmd_neural and neural_paths_long is not None else None,
        'hurst_ou': float(ou_h),
    }


def verify_generated_roughness():
    """
    Verify roughness on the training distribution (realized variance proxy from SPX returns).
    Uses P-measure model (neural_sde_best_p) for generated paths.

    """
    with open("config/params.yaml", 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    base_dt = _get_base_dt(cfg)

    print("   1a. Realized variance from SPX (training distribution)...", flush=True)
    H_rv = None
    try:
        from utils.loader.RealizedVariance import RealizedVolatilityLoader
        loader = RealizedVolatilityLoader(config_path='config/params.yaml')
        # Use long-ish segments for a stable H estimate at the training sampling interval.
        rv_paths = np.array(
            loader.get_realized_vol_paths(segment_length=200, seed=123, shuffle=True)[:50]
        )
        rv_h_list = [estimate_hurst_variogram(np.log(np.clip(p, 1e-12, None))) for p in rv_paths]
        H_rv = float(np.mean(rv_h_list))
        H_rv_std = float(np.std(rv_h_list))
        print(f"       H_RV = {H_rv:.4f} ± {H_rv_std:.4f} — {len(rv_paths)} segments (200 steps, dt≈{base_dt:.2e}y)")
    except Exception as e:
        print(f"       Could not compute RV roughness from loader: {e}")

    print("   1b. Neural SDE generated paths (P model)...", flush=True)
    try:
        init_vars = None
        try:
            init_vars = rv_paths[:, 0] if 'rv_paths' in locals() else None
        except Exception:
            init_vars = None
        neural_paths = generate_neural_sde_paths(
            n_paths=100, n_steps=200, dt=base_dt, init_vars=init_vars, progress_interval=50
        )
        
        H_generated = []
        for path in neural_paths:
            h = estimate_hurst_variogram(np.log(path + 1e-6))
            H_generated.append(h)
        
        H_gen_mean = np.mean(H_generated)
        H_gen_std = np.std(H_generated)
        print(f"       H_gen = {H_gen_mean:.3f} ± {H_gen_std:.3f}")
        if H_rv is not None:
            gap_rv = abs(H_rv - H_gen_mean)
            print(f"       (RV H={H_rv:.3f})")
            print(f"       -> Generated H compared vs training distribution")
        else:
            gap_rv = float('inf')

        
        return {
            'H_rv': float(H_rv) if H_rv is not None else None,
            'H_generated_mean': float(H_gen_mean),
            'H_generated_std': float(H_gen_std),
            'gap_vs_training': float(gap_rv) if H_rv is not None else float('inf'),
        }
    
    except Exception as e:
        print(f"Error generating paths: {e}")
        return {'H_rv': float(H_rv) if H_rv else None, 'error': str(e)}


def compare_signature_distributions():
    """Compare signature distributions of real vs generated paths."""
    from engine.signature_engine import SignatureFeatureExtractor
    from utils.loader.RealizedVariance import RealizedVolatilityLoader
    from engine.losses import kernel_mmd_loss, signature_mmd_loss

    print("   2a. Loading real path segments (RV / P-measure)...", flush=True)
    with open("config/params.yaml", 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    real_dt = _get_base_dt(cfg)
    
    seg_len = 20
    loader = RealizedVolatilityLoader(config_path='config/params.yaml')
    real_paths = np.array(
        loader.get_realized_vol_paths(segment_length=seg_len, seed=123, shuffle=True)[:200]
    )
    sig_engine = SignatureFeatureExtractor(truncation_order=4, dt=real_dt)
    
    # Real signatures
    real_sigs = jnp.array(sig_engine.get_signature(jnp.array(real_paths)))
    sig_std = jnp.std(real_sigs, axis=0)
    real_stats = {
        'mean': np.array(jnp.mean(real_sigs, axis=0)),
        'std': np.array(jnp.std(real_sigs, axis=0)),
        'norm': float(jnp.mean(jnp.linalg.norm(real_sigs, axis=1)))
    }
    
    print(f"\n   Real signatures:")
    print(f"Shape: {real_sigs.shape}")
    print(f"Norm (mean): {real_stats['norm']:.4f}")
    print(f"First 5 components mean: {real_stats['mean'][:5]}")
    
    print("   2b. Generating Neural SDE paths (200)...", flush=True)
    try:
        v0s = real_paths[:, 0]
        neural_paths = generate_neural_sde_paths(
            n_paths=200, n_steps=seg_len, dt=real_dt, init_vars=v0s, progress_interval=100
        )
        gen_sigs = jnp.array(sig_engine.get_signature(jnp.array(neural_paths)))
        gen_stats = {
            'mean': np.array(jnp.mean(gen_sigs, axis=0)),
            'std': np.array(jnp.std(gen_sigs, axis=0)),
            'norm': float(jnp.mean(jnp.linalg.norm(gen_sigs, axis=1)))
        }
        
        print(f"\n   Generated signatures:")
        print(f"Shape: {gen_sigs.shape}")
        print(f"Norm (mean): {gen_stats['norm']:.4f}")
        print(f"First 5 components mean: {gen_stats['mean'][:5]}")
        
        # Compare (diagnostic) + match training objective (kernel MMD)
        norm_ratio = float(gen_stats['norm'] / max(real_stats['norm'], 1e-12))
        mean_corr = float(np.corrcoef(real_stats['mean'], gen_stats['mean'])[0, 1])

        mmd_kernel = float(kernel_mmd_loss(gen_sigs, real_sigs, sig_std))
        mmd_mean_l2 = float(signature_mmd_loss(gen_sigs, real_sigs, sig_std))

        print("\n   Training-aligned metrics:")
        print(f"Kernel MMD² (normalized): {mmd_kernel:.6f}")
        print(f"Mean-signature L2 (normalized): {mmd_mean_l2:.6f}")

        print("   2c. OU baseline (200)...", flush=True)
        ou_paths = generate_ou_paths(n_paths=200, n_steps=seg_len, dt=real_dt, init_vars=v0s)
        ou_sigs = jnp.array(sig_engine.get_signature(jnp.array(ou_paths)))
        mmd_kernel_ou = float(kernel_mmd_loss(ou_sigs, real_sigs, sig_std))
        mmd_mean_l2_ou = float(signature_mmd_loss(ou_sigs, real_sigs, sig_std))
        print(f"Kernel MMD² OU (normalized): {mmd_kernel_ou:.6f}")
        print(f"Mean-signature L2 OU (normalized): {mmd_mean_l2_ou:.6f}")
        
        print(f"\n   Comparison:")
        print(f"Norm ratio (gen/real): {norm_ratio:.2f}")
        print(f"Mean correlation: {mean_corr:.3f}")

        if mmd_kernel < mmd_kernel_ou:
            print(f"--> Neural SDE improves signature distribution vs OU (training objective)")
        else:
            print(f"--> Neural SDE does NOT beat OU on kernel MMD (investigate)")
        
        return {
            'real_norm': float(real_stats['norm']),
            'gen_norm': float(gen_stats['norm']),
            'norm_ratio': float(norm_ratio),
            'mean_correlation': float(mean_corr),
            'mmd_kernel': float(mmd_kernel),
            'mmd_mean_l2': float(mmd_mean_l2),
            'mmd_kernel_ou': float(mmd_kernel_ou),
            'mmd_mean_l2_ou': float(mmd_mean_l2_ou),
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

    print("\n[Step 1/3] Roughness verification (RV vs Neural SDE)...", flush=True)
    results['roughness'] = verify_generated_roughness()

    print("\n[Step 2/3] Signature distribution comparison...", flush=True)
    results['signatures'] = compare_signature_distributions()

    print("\n[Step 3/3] Ablation study (Neural SDE vs OU)...", flush=True)
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
        gap = results['roughness'].get('gap_vs_training', float('inf'))
        if gap < 0.15:
            checks.append(f"[+] GENERATION: Model reproduces training data dynamics (H_gen={h_gen:.3f})")
        else:
            checks.append(f"[-] GENERATION: Model H differs from training data")
    
    # Check 2: Signatures match (use training objective)
    sig_res = results.get('signatures', {})
    if 'mmd_kernel' in sig_res and 'mmd_kernel_ou' in sig_res:
        mmd_n = sig_res['mmd_kernel']
        mmd_ou = sig_res['mmd_kernel_ou']
        if mmd_n < mmd_ou:
            checks.append(f"[+] SIGNATURES: Neural beats OU on kernel MMD ({mmd_n:.3f} < {mmd_ou:.3f})")
        else:
            checks.append(f"[-] SIGNATURES: Neural worse than OU on kernel MMD ({mmd_n:.3f} >= {mmd_ou:.3f})")
    elif 'mean_correlation' in sig_res:
        corr = sig_res['mean_correlation']
        checks.append(f"[~] SIGNATURES: mean corr={corr:.2f} (diagnostic only)")
    
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
