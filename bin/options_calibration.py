"""
Options Calibration Module (Enhanced)
=====================================
Calibrates volatility models to real SPY options market data.
Features:
- Local caching for options data (no repeated API calls)
- Automatic Bergomi parameter optimization
- Extended maturities and strikes
- Calibration report export
"""

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os
import sys
import json

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Import our new modules
from quant.options_cache import OptionsDataCache, EnhancedOptionsLoader
from quant.calibration.bergomi_optimizer import BergomiOptimizer, calibrate_bergomi_to_smile
from quant.mc_pricer import MonteCarloOptionPricer
from utils.black_scholes import BlackScholes


class VolatilitySurfaceVisualizer:
    """Beautiful visualizations for volatility surfaces and smiles."""
    
    @staticmethod
    def plot_3d_surface(surface_df: pd.DataFrame, title: str = "Implied Volatility Surface"):
        """
        3D surface plot of the volatility surface.
        """
        # Pivot to grid
        pivot = surface_df.pivot_table(
            values='impliedVolatility', 
            index='moneyness', 
            columns='dte', 
            aggfunc='mean'
        )
        
        X = pivot.columns.values  # DTE
        Y = pivot.index.values    # Moneyness
        Z = pivot.values * 100    # IV in %
        
        fig = go.Figure(data=[go.Surface(
            x=X, y=Y, z=Z,
            colorscale='Viridis',
            colorbar=dict(title='IV (%)')
        )])
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            scene=dict(
                xaxis_title='Days to Expiry',
                yaxis_title='Log-Moneyness',
                zaxis_title='Implied Vol (%)',
                camera=dict(eye=dict(x=1.5, y=-1.5, z=0.8))
            ),
            template='plotly_dark',
            height=700,
            width=900
        )
        
        return fig
    
    @staticmethod
    def plot_smile_term_structure(surface_df: pd.DataFrame, title: str = "Volatility Smile by Maturity"):
        """
        2D plot showing smile evolution across maturities.
        """
        fig = go.Figure()
        
        dtes = sorted(surface_df['dte'].unique())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        
        for i, dte in enumerate(dtes):
            df_dte = surface_df[surface_df['dte'] == dte].sort_values('moneyness')
            
            fig.add_trace(go.Scatter(
                x=df_dte['moneyness'],
                y=df_dte['impliedVolatility'] * 100,
                mode='lines+markers',
                name=f'{dte} DTE',
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=6)
            ))
        
        fig.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.3)
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=18)),
            xaxis_title='Log-Moneyness (ln(K/S))',
            yaxis_title='Implied Volatility (%)',
            template='plotly_dark',
            height=500,
            legend=dict(x=1.02, y=0.98),
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def plot_atm_term_structure(surface_df: pd.DataFrame, spot: float, title: str = "ATM Term Structure"):
        """
        ATM volatility as a function of time to expiry.
        """
        # Find closest to ATM for each maturity
        atm_vols = []
        dtes = []
        
        for dte in sorted(surface_df['dte'].unique()):
            df_dte = surface_df[surface_df['dte'] == dte]
            # Closest to ATM
            atm_idx = df_dte['moneyness'].abs().idxmin()
            atm_vols.append(df_dte.loc[atm_idx, 'impliedVolatility'] * 100)
            dtes.append(dte)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dtes,
            y=atm_vols,
            mode='lines+markers',
            line=dict(color='cyan', width=3),
            marker=dict(size=10, color='white', line=dict(width=2, color='cyan')),
            name='ATM Vol'
        ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=18)),
            xaxis_title='Days to Expiry',
            yaxis_title='ATM Implied Volatility (%)',
            template='plotly_dark',
            height=400
        )
        
        return fig
    
    @staticmethod
    def plot_skew_term_structure(surface_df: pd.DataFrame, title: str = "Skew Term Structure"):
        """
        Plot the 25-delta skew (put vol - call vol) across maturities.
        """
        skews = []
        dtes = []
        
        for dte in sorted(surface_df['dte'].unique()):
            df_dte = surface_df[surface_df['dte'] == dte]
            
            # Approximate 25-delta: ~5% OTM
            put_side = df_dte[df_dte['moneyness'] < -0.03]
            call_side = df_dte[df_dte['moneyness'] > 0.03]
            
            if len(put_side) > 0 and len(call_side) > 0:
                put_vol = put_side['impliedVolatility'].mean()
                call_vol = call_side['impliedVolatility'].mean()
                skew = (put_vol - call_vol) * 100
                skews.append(skew)
                dtes.append(dte)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=dtes,
            y=skews,
            marker_color=['#FF6B6B' if s > 0 else '#4ECDC4' for s in skews],
            name='Put-Call Skew'
        ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=18)),
            xaxis_title='Days to Expiry',
            yaxis_title='Skew (Put Vol - Call Vol) %',
            template='plotly_dark',
            height=400
        )
        
        return fig
    
    @staticmethod
    def plot_model_comparison(smile_df: pd.DataFrame, model_results: dict, 
                             spot: float, title: str = "Model vs Market Smile"):
        """
        Compare model-implied smiles vs market.
        """
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=('Implied Volatility Smile', 'Pricing Error'),
            vertical_spacing=0.12
        )
        
        # Market data
        fig.add_trace(go.Scatter(
            x=smile_df['moneyness'],
            y=smile_df['impliedVolatility'] * 100,
            mode='markers',
            marker=dict(size=12, color='white', symbol='diamond',
                       line=dict(width=1, color='gray')),
            name='Market',
            legendgroup='market'
        ), row=1, col=1)
        
        # Model results
        colors = {'Black-Scholes': 'gray', 'Bergomi': 'lime', 'Neural SDE': 'cyan'}
        
        for model_name, ivs in model_results.items():
            color = colors.get(model_name, 'orange')
            
            # Smile
            fig.add_trace(go.Scatter(
                x=smile_df['moneyness'],
                y=ivs * 100,
                mode='lines+markers',
                line=dict(color=color, width=2),
                marker=dict(size=6),
                name=model_name,
                legendgroup=model_name
            ), row=1, col=1)
            
            # Error
            error = (ivs - smile_df['impliedVolatility'].values) * 100
            fig.add_trace(go.Bar(
                x=smile_df['moneyness'],
                y=error,
                marker_color=color,
                opacity=0.6,
                name=f'{model_name} Error',
                legendgroup=model_name,
                showlegend=False
            ), row=2, col=1)
        
        fig.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.3, row=1, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5, row=2, col=1)
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=18)),
            template='plotly_dark',
            height=700,
            legend=dict(x=1.02, y=0.98),
            barmode='group'
        )
        
        fig.update_xaxes(title_text='Log-Moneyness', row=2, col=1)
        fig.update_yaxes(title_text='Implied Vol (%)', row=1, col=1)
        fig.update_yaxes(title_text='Error (%)', row=2, col=1)
        
        return fig


def run_full_calibration(use_cache: bool = False, optimize_bergomi: bool = True,
                         save_report: bool = True):
    """
    Run full calibration with all models and visualizations.
    
    Args:
        use_cache: Load from local cache instead of fetching live data
        optimize_bergomi: Run automatic Bergomi parameter optimization
        save_report: Save calibration results to JSON
    """
    
    print("="*70)
    print("   ENHANCED VOLATILITY SURFACE CALIBRATION")
    print("   SPY Options - Neural SDE vs Rough Bergomi vs Black-Scholes")
    print("="*70)
    
    # Setup cache
    cache = OptionsDataCache()
    
    # 1. Load volatility surface
    print("\n" + "-"*50)
    print("STEP 1: Loading Market Data")
    print("-"*50)
    
    if use_cache:
        try:
            surface, snapshot_info = cache.load_latest("SPY")
            spot = snapshot_info['spot']
            print(f"Loaded from cache: {snapshot_info['datetime']}")
            print(f"   Spot at snapshot: ${spot:.2f}")
        except FileNotFoundError:
            print("   No cached data found, fetching live...")
            use_cache = False
    
    if not use_cache:
        loader = EnhancedOptionsLoader("SPY", cache)
        
        # Get comprehensive surface with more maturities
        surface = loader.get_full_surface(
            max_dte=120,           # Up to 4 months
            min_volume=1,          # Include less liquid for more data
            moneyness_range=0.25,  # ±25% strikes
            save_cache=True        # Auto-save to cache
        )
        spot = loader.spot
    
    # Show summary
    print(f"\nSurface Summary:")
    print(surface.groupby('dte').agg({
        'strike': 'count',
        'impliedVolatility': ['mean', 'std']
    }).rename(columns={'strike': 'count'}).round(3).head(10))
    
    # 2. Visualize the surface
    print("\n" + "-"*50)
    print("STEP 2: Generating Visualizations")
    print("-"*50)
    
    viz = VolatilitySurfaceVisualizer()
    
    # 3D Surface
    print("   Generating 3D Volatility Surface...")
    fig_3d = viz.plot_3d_surface(surface, f"SPY Implied Volatility Surface (Spot: ${spot:.2f})")
    fig_3d.show()
    
    # Smile by maturity
    print("   Generating Smile Term Structure...")
    fig_smile = viz.plot_smile_term_structure(surface)
    fig_smile.show()
    
    # ATM term structure
    print("   Generating ATM Term Structure...")
    fig_atm = viz.plot_atm_term_structure(surface, spot)
    fig_atm.show()
    
    # Skew
    print("   Generating Skew Analysis...")
    fig_skew = viz.plot_skew_term_structure(surface)
    fig_skew.show()
    
    # 3. Model calibration on 30-day smile
    print("\n" + "-"*50)
    print("STEP 3: Model Calibration (30 DTE)")
    print("-"*50)
    
    # Get ~30-day smile
    dtes = sorted(surface['dte'].unique())
    target_dte = min(dtes, key=lambda x: abs(x - 30))
    
    smile_30d = surface[surface['dte'] == target_dte].copy()
    smile_30d = smile_30d.sort_values('moneyness')
    T = smile_30d['T'].iloc[0]
    
    print(f"   Calibrating on {len(smile_30d)} options, DTE={target_dte}, T={T:.3f} years")
    
    # ATM IV
    atm_iv = smile_30d.loc[smile_30d['moneyness'].abs().idxmin(), 'impliedVolatility']
    print(f"   ATM IV: {atm_iv*100:.1f}%")
    
    # Black-Scholes (flat vol)
    bs_ivs = np.full(len(smile_30d), atm_iv)
    
    # Monte Carlo setup
    pricer = MonteCarloOptionPricer(spot)
    n_paths = 10000
    n_steps = 50
    dt = T / n_steps
    
    print("\n   Running Monte Carlo simulations...")
    
    import jax
    import jax.numpy as jnp
    key = jax.random.PRNGKey(42)
    
    # --- ROUGH BERGOMI ---
    print("\n   [1/3] Rough Bergomi...")
    from core.bergomi import RoughBergomiModel
    
    if optimize_bergomi:
        print("   Optimizing Bergomi parameters...")
        bergomi_result = calibrate_bergomi_to_smile(
            smile_30d, spot, T, n_paths=5000  # Fewer paths for speed
        )
        
        bergomi_params = {
            'hurst': bergomi_result['hurst'],
            'eta': bergomi_result['eta'],
            'rho': bergomi_result['rho'],
            'xi0': bergomi_result['xi0'],
            'n_steps': n_steps,
            'T': T
        }
        bergomi_ivs = bergomi_result['model_ivs']
    else:
        # Default params
        bergomi_params = {
            'hurst': 0.07,
            'eta': 2.5,
            'rho': -0.85,
            'xi0': atm_iv**2,
            'n_steps': n_steps,
            'T': T
        }
        bergomi = RoughBergomiModel(bergomi_params)
        s_bergomi, _ = bergomi.simulate_spot_vol_paths(n_paths, s0=spot)
        s_bergomi = np.array(s_bergomi)
        
        bergomi_ivs = pricer.compute_model_smile(
            s_bergomi, 
            smile_30d['strike'].values,
            T,
            smile_30d['type'].tolist()
        )
    
    # --- NEURAL SDE ---
    print("\n   [2/3] Neural SDE (Loading trained model)...")
    
    try:
        from engine.generative_trainer import GenerativeTrainer
        from utils.config import load_config

        cfg = load_config()
        config = {'n_steps': n_steps, 'T': T}
        
        # Quick training
        trainer = GenerativeTrainer(config)
        model = trainer.run(n_epochs=50, batch_size=128)
        
        # Generate variance paths with properly scaled Brownian increments
        key, subkey = jax.random.split(key)
        dW_vol = jax.random.normal(subkey, (n_paths, n_steps)) * jnp.sqrt(dt)
        
        v0 = jnp.full(n_paths, atm_iv**2)
        var_paths = jax.vmap(model.generate_variance_path, in_axes=(0, 0, None))(
            v0, dW_vol, dt
        )
        var_paths = np.array(var_paths)
        
        rho = cfg['bergomi']['rho']
        r = cfg['pricing']['risk_free_rate']
        key, subkey = jax.random.split(key)
        dW_perp = np.array(jax.random.normal(subkey, (n_paths, n_steps))) * np.sqrt(dt)
        dW_spot = rho * np.array(dW_vol) + np.sqrt(1 - rho**2) * dW_perp
        
        v0_col = np.full((n_paths, 1), atm_iv**2)
        var_prev = np.hstack([v0_col, var_paths[:, :-1]])
        vol_paths = np.sqrt(var_prev)
        log_ret = (r - 0.5 * var_prev) * dt + vol_paths * dW_spot
        log_s = np.cumsum(log_ret, axis=1)
        s_neural = spot * np.exp(np.hstack([np.zeros((n_paths, 1)), log_s]))
        
        neural_ivs = pricer.compute_model_smile(
            s_neural,
            smile_30d['strike'].values,
            T,
            smile_30d['type'].tolist()
        )
    except Exception as e:
        print(f"   Warning: Neural SDE failed: {e}")
        neural_ivs = bs_ivs.copy()
    
    # 4. Compare models
    print("\n" + "-"*50)
    print("STEP 4: Model Comparison")
    print("-"*50)
    
    model_results = {
        'Black-Scholes': bs_ivs,
        'Bergomi': bergomi_ivs,
        'Neural SDE': neural_ivs
    }
    
    # Compute RMSE
    market_ivs = smile_30d['impliedVolatility'].values
    rmse_results = {}
    
    for name, ivs in model_results.items():
        valid = ~np.isnan(ivs)
        if valid.sum() > 0:
            rmse = np.sqrt(np.mean((ivs[valid] - market_ivs[valid])**2)) * 100
            rmse_results[name] = rmse
            best_marker = "← BEST!" if rmse == min(rmse_results.values()) else ""
            print(f"   {name:15s} RMSE: {rmse:.2f}% {best_marker}")
    
    # Plot comparison
    print("\n   Generating comparison plot...")
    fig_comp = viz.plot_model_comparison(smile_30d, model_results, spot,
                                        f"Model Calibration - {target_dte} DTE (Spot: ${spot:.2f})")
    fig_comp.show()
    
    # 5. Save calibration report
    if save_report:
        print("\n" + "-"*50)
        print("STEP 5: Saving Calibration Report")
        print("-"*50)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'spot': float(spot),
            'target_dte': int(target_dte),
            'T': float(T),
            'atm_iv': float(atm_iv),
            'n_options': len(smile_30d),
            'surface_dtes': [int(d) for d in dtes],
            'surface_n_options': len(surface),
            'rmse_results': rmse_results,
            'bergomi_params': {
                'hurst': float(bergomi_params['hurst']),
                'eta': float(bergomi_params['eta']),
                'rho': float(bergomi_params['rho']),
                'xi0': float(bergomi_params['xi0'])
            }
        }
        
        report_path = 'outputs/calibration_report.json'
        os.makedirs('outputs', exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"   Report saved: {report_path}")
    
    # Summary
    print("\n" + "="*70)
    print("   CALIBRATION COMPLETE")
    print("="*70)
    print(f"\nFinal Rankings:")
    for i, (name, rmse) in enumerate(sorted(rmse_results.items(), key=lambda x: x[1])):
        medal = ["1.", "2.", "3."][i] if i < 3 else "  "
        print(f"   {medal} {name}: {rmse:.2f}%")
    
    if optimize_bergomi:
        print(f"\nOptimized Bergomi Parameters:")
        print(f"   H={bergomi_params['hurst']:.4f}, η={bergomi_params['eta']:.2f}, "
              f"ρ={bergomi_params['rho']:.2f}, ξ₀={bergomi_params['xi0']:.4f}")
    
    return surface, model_results, bergomi_params


def main():
    """Entry point with CLI arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Options Calibration')
    parser.add_argument('--cache', action='store_true', help='Use cached data')
    parser.add_argument('--no-optimize', action='store_true', help='Skip Bergomi optimization')
    parser.add_argument('--download-only', action='store_true', help='Just download and cache data')
    
    args = parser.parse_args()
    
    if args.download_only:
        # Just download and cache
        cache = OptionsDataCache()
        loader = EnhancedOptionsLoader("SPY", cache)
        surface = loader.get_full_surface(max_dte=180, save_cache=True)
        print(f"\nDownloaded and cached {len(surface)} options")
        return
    
    run_full_calibration(
        use_cache=args.cache,
        optimize_bergomi=not args.no_optimize,
        save_report=True
    )


if __name__ == "__main__":
    main()
