import sys, os
# Force UTF-8 output on Windows to support Greek/math symbols in print()
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import jax
import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from engine.generative_trainer import GenerativeTrainer
from utils.diagnostics import print_distribution_stats, compute_acf, estimate_hurst_from_returns
from core.bergomi import RoughBergomiModel
from quant.pricing import DeepPricingEngine

from utils.config import load_config

def main():
    cfg = load_config()
    
    config = {
        'n_steps': cfg['simulation']['n_steps'],
        'T': cfg['simulation']['T'],
    }

    print("--- 1. Initializing Generative Quant Engine ---")
    print(f"    Horizon T = {config['T']:.5f} years (~{config['T']*252*6.5*60:.0f} min = {config['n_steps']} bars × 15 min)")
    print(f"    Time steps = {config['n_steps']}, dt = {config['T']/config['n_steps']:.6f} years")
    
    trainer = GenerativeTrainer(config)
    real_data_np = np.array(trainer.market_paths)
    print_distribution_stats("1. REAL MARKET (TARGET)", real_data_np, is_vix_derived=True)

    # Train Neural Model
    trained_model = trainer.run(
        n_epochs=cfg['training']['n_epochs'], 
        batch_size=cfg['training']['batch_size']
    )

    print("\n--- 2. Setting up the Battle: Real vs Bergomi vs AI ---")
    key = jax.random.PRNGKey(777)
    n_gen = cfg['simulation']['n_paths']

    # A. NEURAL SDE GENERATION
    dt = config['T'] / config['n_steps']
    noise = jax.random.normal(key, (n_gen, config['n_steps'])) * jnp.sqrt(dt)
    
    # Use randomized initial conditions from market history
    real_v0s = trainer.market_paths[:, 0]
    random_indices = jax.random.randint(key, (n_gen,), 0, len(real_v0s))
    v0_samples = real_v0s[random_indices]
    
    fake_vars = jax.vmap(trained_model.generate_variance_path, in_axes=(0, 0, None))(
        v0_samples, noise, dt
    )
    fake_vars_np = np.array(fake_vars)

    # B. BERGOMI GENERATION (BENCHMARK)
    # Calibrated to match Realized Volatility level
    bergomi_cfg = cfg['bergomi']
    
    # Auto-calibrate xi0 to match market mean variance
    market_mean_var = float(jnp.mean(trainer.market_paths))
    
    bergomi_params = {
        'hurst': bergomi_cfg['hurst'], 
        'eta': bergomi_cfg['eta'], 
        'rho': bergomi_cfg['rho'], 
        'xi0': market_mean_var,  # Auto-calibrated to market!
        'n_steps': config['n_steps'], 
        'T': config['T']
    }
    print(f"   Bergomi xi0 calibrated to market mean: {market_mean_var:.4f}")
    bergomi_model = RoughBergomiModel(bergomi_params)
    bergomi_vars_np = np.array(bergomi_model.simulate_variance_paths(n_gen))

    # Comparative Stats (on VIX-derived variance paths)
    print_distribution_stats("2. BERGOMI (MATHS)", bergomi_vars_np, is_vix_derived=True)
    print_distribution_stats("3. NEURAL SDE (AI)", fake_vars_np, is_vix_derived=True)

    # ── TRUE ROUGHNESS CHECK: Realized Vol from SPX Returns ──
    print("\n--- 2b. TRUE ROUGHNESS CHECK (Realized Vol from S&P 500 returns) ---")
    rv_source = cfg['data'].get('rv_source', 'data/SP_SPX, 5.csv')
    if os.path.exists(rv_source):
        rv_results = estimate_hurst_from_returns(
            rv_source,
            rv_window=cfg['data'].get('rv_window', 78)
        )
        print(f"   Source: {rv_source}")
        print(f"   RV window: {rv_results['rv_window']} bars ({rv_results['n_rv_points']} RV points)")
        print(f"   H (variogram) = {rv_results['H_variogram']:.4f}  (R² = {rv_results['R2_variogram']:.3f})")
        print(f"   H (struct q=1) = {rv_results['H_structure']:.4f}  (R² = {rv_results['R2_structure']:.3f})")
        if rv_results['H_variogram'] < 0.20:
            print(f"   [OK] TRUE ROUGHNESS CONFIRMED: H = {rv_results['H_variogram']:.3f} < 0.20")
        else:
            print(f"   [WARN] H = {rv_results['H_variogram']:.3f} - expected < 0.15 (check data/window)")
        print(f"   Note: VIX-based H ~ 0.5 is EXPECTED (30-day integration smooths roughness)")
    else:
        print(f"   [WARN] No SPX data found at {rv_source} - skipping true roughness check")
    print("-" * 30)

    print("\n--- 3. The Ultimate Pricing Test ---")
    
    pricer = DeepPricingEngine(trainer, trained_model)
    pricing_cfg = cfg['pricing']
    s0 = pricing_cfg['spot']
    strike = pricing_cfg['strike']
    barrier = pricing_cfg['barrier']
    
    # 1. Black-Scholes (Naive Benchmark)
    implied_vol_proxy = np.mean(np.sqrt(real_data_np))
    s_paths_bs = pricer.generate_bs_paths(n_gen, s0, implied_vol_proxy)
    price_bs = pricer.price_down_and_out_call(s_paths_bs, strike, barrier)
    
    # 2. Rough Bergomi (Academic Benchmark)
    s_paths_bergomi, _ = pricer.generate_bergomi_paths(bergomi_model, n_gen, s0)
    price_bergomi = pricer.price_down_and_out_call(s_paths_bergomi, strike, barrier)
    
    # 3. Neural SDE (Data-Driven Model)
    s_paths_neural, _ = pricer.generate_market_paths(n_gen, s0)
    price_neural = pricer.price_down_and_out_call(s_paths_neural, strike, barrier)

    print("\n" + "="*60)
    T_option = config['T']
    print(f"OPTION : Down-and-Out Call (K={strike}, B={barrier}, T={T_option*12:.1f} months)")
    print("="*60)
    print(f"1. Black-Scholes (Flat Vol {implied_vol_proxy:.1%}) : {price_bs:.4f} $")
    print(f"2. Rough Bergomi (Log-Normal Math)       : {price_bergomi:.4f} $")
    print(f"3. DeepRoughVol (Neural Data-Driven)     : {price_neural:.4f} $")
    print("-" * 60)
    print("SPREAD ANALYSIS (Crash Risk Premium):")
    print(f"AI vs BS      : {price_neural - price_bs:.4f} $")
    print(f"AI vs Bergomi : {price_neural - price_bergomi:.4f} $")
    print("="*60)
    
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.4, 0.3, 0.3],
        vertical_spacing=0.1,
        subplot_titles=("Volatility Paths", "Density Distribution", "Auto-Correlation (Memory)")
    )

    # PLOT 1: Paths (Show 15 of each to reduce load)
    n_show = min(15, len(real_data_np), len(bergomi_vars_np), len(fake_vars_np))
    for i in range(n_show):
        fig.add_trace(go.Scatter(y=real_data_np[i], mode='lines', line=dict(color='rgba(255, 255, 255, 0.3)', width=1), showlegend=False), row=1, col=1)
    for i in range(n_show):
        fig.add_trace(go.Scatter(y=bergomi_vars_np[i], mode='lines', line=dict(color='rgba(0, 255, 0, 0.4)', width=1), showlegend=False), row=1, col=1)
    for i in range(n_show):
        fig.add_trace(go.Scatter(y=fake_vars_np[i], mode='lines', line=dict(color='rgba(0, 255, 255, 0.4)', width=1), showlegend=False), row=1, col=1)

    # Legend Traces
    fig.add_trace(go.Scatter(x=[0], y=[0], mode='lines', line=dict(color='white'), name='1. Real (RV)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0], y=[0], mode='lines', line=dict(color='lime'), name='2. Bergomi (Math)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0], y=[0], mode='lines', line=dict(color='cyan'), name='3. Neural SDE (AI)'), row=1, col=1)

    # PLOT 2: Histogram - Auto-range based on data
    all_data = np.concatenate([real_data_np.flatten(), bergomi_vars_np.flatten(), fake_vars_np.flatten()])
    range_max = min(np.percentile(all_data, 95), 1.5)  # 95th percentile or max 1.5
    bin_size = range_max / 50
    
    fig.add_trace(go.Histogram(x=real_data_np.flatten(), xbins=dict(start=0, end=range_max, size=bin_size), marker_color='white', opacity=0.4, name='Real', histnorm='probability density'), row=2, col=1)
    fig.add_trace(go.Histogram(x=bergomi_vars_np.flatten(), xbins=dict(start=0, end=range_max, size=bin_size), marker_color='lime', opacity=0.4, name='Bergomi', histnorm='probability density'), row=2, col=1)
    fig.add_trace(go.Histogram(x=fake_vars_np.flatten(), xbins=dict(start=0, end=range_max, size=bin_size), marker_color='cyan', opacity=0.4, name='AI', histnorm='probability density'), row=2, col=1)

    # PLOT 3: Auto-Correlation
    real_acf = np.mean([compute_acf(p, lags=20) for p in real_data_np[:100]], axis=0)
    bergomi_acf = np.mean([compute_acf(p, lags=20) for p in bergomi_vars_np[:100]], axis=0)
    fake_acf = np.mean([compute_acf(p, lags=20) for p in fake_vars_np[:100]], axis=0)
    lags = np.arange(20)

    fig.add_trace(go.Scatter(x=lags, y=real_acf, mode='lines+markers', name='Real ACF', line=dict(color='white', dash='dash')), row=3, col=1)
    fig.add_trace(go.Scatter(x=lags, y=bergomi_acf, mode='lines+markers', name='Bergomi ACF', line=dict(color='lime', dash='dot')), row=3, col=1)
    fig.add_trace(go.Scatter(x=lags, y=fake_acf, mode='lines+markers', name='AI ACF', line=dict(color='cyan')), row=3, col=1)

    fig.update_layout(title='Benchmark: VIX Variance - Real vs Bergomi vs Neural SDE', template='plotly_dark', height=1200)
    fig.show()

if __name__ == "__main__":
    main()