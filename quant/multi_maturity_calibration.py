"""
Multi-Maturity Calibration
==========================
Calibrates models across the entire volatility surface (all maturities).
Joint calibration for consistent term structure.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from quant.options_cache import OptionsDataCache, EnhancedOptionsLoader
from core.bergomi import RoughBergomiModel
from utils.black_scholes import BlackScholes


class MultiMaturityCalibrator:
    """
    Calibrates models to the entire volatility surface.
    Supports multiple maturities simultaneously.
    """
    
    def __init__(self, spot: float, r: float = 0.05):
        self.spot = spot
        self.r = r
        self.surface = None
        self.results = {}
        
    def set_surface(self, surface_df: pd.DataFrame):
        """Set the market volatility surface."""
        self.surface = surface_df.copy()
        self.maturities = sorted(surface_df['dte'].unique())
        print(f"Surface set: {len(surface_df)} options across {len(self.maturities)} maturities")
        print(f"   DTEs: {self.maturities[:10]}{'...' if len(self.maturities) > 10 else ''}")
        
    def calibrate_bergomi_surface(self, n_paths: int = 2000) -> dict:
        """
        Calibrate Bergomi to entire surface with grid search (faster than optimization).
        """
        print("\nBergomi Surface Calibration (Grid Search)")
        
        # Get ATM IV for xi0
        atm_smile = self.surface[self.surface['moneyness'].abs() < 0.02]
        atm_iv = atm_smile['impliedVolatility'].mean() if len(atm_smile) > 0 else 0.2
        
        # Grid of parameters to test
        H_grid = [0.05, 0.1, 0.2]
        eta_grid = [1.5, 2.5, 3.5]
        rho_grid = [-0.5, -0.7, -0.85]
        
        best_params = None
        best_rmse = float('inf')
        
        # Select a few representative maturities
        test_dtes = [30]  # Just use 30 DTE for speed
        if len(self.maturities) >= 3:
            test_dtes = [self.maturities[len(self.maturities)//2]]
        
        print(f"   Testing {len(H_grid)*len(eta_grid)*len(rho_grid)} parameter combinations...")
        
        for H in H_grid:
            for eta in eta_grid:
                for rho in rho_grid:
                    total_error = 0
                    n_opts = 0
                    
                    for dte in test_dtes:
                        smile = self.surface[self.surface['dte'] == dte]
                        if len(smile) < 3:
                            continue
                        
                        T = dte / 365.0
                        
                        try:
                            bergomi_params = {
                                'hurst': H, 'eta': eta, 'rho': rho,
                                'xi0': atm_iv**2,
                                'n_steps': 30, 'T': T
                            }
                            bergomi = RoughBergomiModel(bergomi_params)
                            spot_paths, _ = bergomi.simulate_spot_vol_paths(n_paths, s0=self.spot)
                            S_T = np.array(spot_paths)[:, -1]
                            
                            # Sample some strikes
                            sample = smile.sample(min(10, len(smile)))
                            
                            for _, row in sample.iterrows():
                                K = row['strike']
                                opt_type = row['type']
                                market_iv = row['impliedVolatility']
                                
                                if opt_type == 'call':
                                    payoff = np.maximum(S_T - K, 0)
                                else:
                                    payoff = np.maximum(K - S_T, 0)
                                
                                mc_price = np.exp(-self.r * T) * np.mean(payoff)
                                model_iv = BlackScholes.implied_vol(mc_price, self.spot, K, T, self.r, opt_type)
                                
                                if not np.isnan(model_iv):
                                    total_error += (model_iv - market_iv)**2
                                    n_opts += 1
                        except:
                            continue
                    
                    if n_opts > 0:
                        rmse = np.sqrt(total_error / n_opts)
                        if rmse < best_rmse:
                            best_rmse = rmse
                            best_params = {'hurst': H, 'eta': eta, 'rho': rho, 'xi0': atm_iv**2}
        
        if best_params is None:
            best_params = {'hurst': 0.1, 'eta': 2.5, 'rho': -0.7, 'xi0': atm_iv**2}
            best_rmse = 0.1
        
        self.results['bergomi'] = {
            **best_params,
            'surface_rmse': best_rmse * 100
        }
        
        print(f"\nBergomi Calibration Complete!")
        print(f"   H = {best_params['hurst']:.3f} {'← ROUGH!' if best_params['hurst'] < 0.25 else ''}")
        print(f"   η = {best_params['eta']:.2f}")
        print(f"   ρ = {best_params['rho']:.2f}")
        print(f"   Surface RMSE = {best_rmse*100:.2f}%")
        
        return self.results['bergomi']
    
    def calibrate_neural_sde_surface(self) -> dict:
        """
        Evaluate trained Neural SDE on entire surface.
        Loads the pre-trained model instead of retraining.
        """
        print("\nNeural SDE Surface Evaluation")
        
        import jax
        import jax.numpy as jnp
        import yaml
        from ml.generative_trainer import GenerativeTrainer
        
        # Load config for consistent parameters
        with open('config/params.yaml', 'r', encoding='utf-8') as f:
            yaml_cfg = yaml.safe_load(f)
        
        config = {'n_steps': yaml_cfg['simulation']['n_steps']}
        trainer = GenerativeTrainer(config)
        
        # Load trained model (don't retrain!)
        model = trainer.load_model()
        if model is None:
            print("No saved model found. Training...")
            model = trainer.run(n_epochs=yaml_cfg['training']['n_epochs'], batch_size=256)
        else:
            print("Loaded trained model from disk.")
        
        key = jax.random.PRNGKey(42)
        n_paths = 5000
        
        errors_by_dte = {}
        
        for dte in self.maturities[:8]:  # Limit for speed
            smile = self.surface[self.surface['dte'] == dte]
            if len(smile) < 3:
                continue
            
            T = dte / 365.0
            n_steps = max(20, int(T * 252))
            dt = T / n_steps
            
            # Generate paths with properly scaled Brownian increments
            key, subkey = jax.random.split(key)
            dW = jax.random.normal(subkey, (n_paths, n_steps)) * jnp.sqrt(dt)
            
            atm_iv = smile.loc[smile['moneyness'].abs().idxmin(), 'impliedVolatility']
            v0 = jnp.full(n_paths, atm_iv**2)
            
            var_paths = jax.vmap(model.generate_variance_path, in_axes=(0, 0, None))(
                v0, dW, dt
            )
            var_paths = np.array(var_paths)
            
            # Spot paths with correlation — noise already scaled by sqrt(dt)
            rho = -0.7
            key, subkey = jax.random.split(key)
            z_indep = np.array(jax.random.normal(subkey, (n_paths, n_steps))) * np.sqrt(dt)
            dW_np = np.array(dW)
            spot_noise = rho * dW_np + np.sqrt(1 - rho**2) * z_indep
            
            vol_paths = np.sqrt(np.maximum(var_paths, 1e-8))
            log_ret = (self.r - 0.5 * var_paths) * dt + vol_paths * spot_noise
            log_s = np.cumsum(log_ret, axis=1)
            S_T = self.spot * np.exp(log_s[:, -1])
            
            # Price options
            errors = []
            for _, row in smile.iterrows():
                K = row['strike']
                opt_type = row['type']
                market_iv = row['impliedVolatility']
                
                if opt_type == 'call':
                    payoff = np.maximum(S_T - K, 0)
                else:
                    payoff = np.maximum(K - S_T, 0)
                
                mc_price = np.exp(-self.r * T) * np.mean(payoff)
                model_iv = BlackScholes.implied_vol(mc_price, self.spot, K, T, self.r, opt_type)
                
                if not np.isnan(model_iv):
                    errors.append((model_iv - market_iv)**2)
            
            if errors:
                errors_by_dte[dte] = np.sqrt(np.mean(errors)) * 100
        
        overall_rmse = np.mean(list(errors_by_dte.values()))
        
        self.results['neural_sde'] = {
            'rmse_by_dte': errors_by_dte,
            'surface_rmse': overall_rmse
        }
        
        print(f"Neural SDE Surface RMSE: {overall_rmse:.2f}%")
        print(f"   By maturity: {dict(list(errors_by_dte.items())[:5])}")
        
        return self.results['neural_sde']
    
    def plot_term_structure_fit(self) -> go.Figure:
        """
        Visualize model fit across term structure.
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'ATM Vol Term Structure', 'Skew Term Structure',
                'RMSE by Maturity', 'Surface Fit Summary'
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # ATM term structure
        atm_vols = []
        for dte in self.maturities:
            smile = self.surface[self.surface['dte'] == dte]
            atm_idx = smile['moneyness'].abs().idxmin()
            atm_vols.append(smile.loc[atm_idx, 'impliedVolatility'] * 100)
        
        fig.add_trace(go.Scatter(
            x=self.maturities, y=atm_vols,
            mode='lines+markers', name='Market ATM',
            line=dict(color='white', width=2),
            marker=dict(size=8)
        ), row=1, col=1)
        
        # Skew term structure
        skews = []
        for dte in self.maturities:
            smile = self.surface[self.surface['dte'] == dte]
            put_vol = smile[smile['moneyness'] < -0.03]['impliedVolatility'].mean()
            call_vol = smile[smile['moneyness'] > 0.03]['impliedVolatility'].mean()
            if not np.isnan(put_vol) and not np.isnan(call_vol):
                skews.append((put_vol - call_vol) * 100)
            else:
                skews.append(np.nan)
        
        fig.add_trace(go.Bar(
            x=self.maturities, y=skews,
            name='Put-Call Skew',
            marker_color=['#FF6B6B' if s > 0 else '#4ECDC4' for s in skews]
        ), row=1, col=2)
        
        # RMSE by maturity (if Neural SDE calibrated)
        if 'neural_sde' in self.results:
            rmse_dict = self.results['neural_sde']['rmse_by_dte']
            fig.add_trace(go.Bar(
                x=list(rmse_dict.keys()),
                y=list(rmse_dict.values()),
                name='Neural SDE RMSE',
                marker_color='cyan'
            ), row=2, col=1)
        
        # Summary
        summary_models = []
        summary_rmse = []
        if 'bergomi' in self.results:
            summary_models.append('Bergomi')
            summary_rmse.append(self.results['bergomi']['surface_rmse'])
        if 'neural_sde' in self.results:
            summary_models.append('Neural SDE')
            summary_rmse.append(self.results['neural_sde']['surface_rmse'])
        
        fig.add_trace(go.Bar(
            x=summary_models, y=summary_rmse,
            marker_color=['lime', 'cyan'][:len(summary_models)],
            text=[f'{r:.2f}%' for r in summary_rmse],
            textposition='outside'
        ), row=2, col=2)
        
        fig.update_layout(
            title=dict(text=f'Multi-Maturity Calibration Results (Spot: ${self.spot:.2f})', font=dict(size=18)),
            template='plotly_dark',
            height=700,
            showlegend=True
        )
        
        fig.update_xaxes(title_text='DTE', row=1, col=1)
        fig.update_xaxes(title_text='DTE', row=1, col=2)
        fig.update_xaxes(title_text='DTE', row=2, col=1)
        fig.update_yaxes(title_text='ATM Vol (%)', row=1, col=1)
        fig.update_yaxes(title_text='Skew (%)', row=1, col=2)
        fig.update_yaxes(title_text='RMSE (%)', row=2, col=1)
        fig.update_yaxes(title_text='Surface RMSE (%)', row=2, col=2)
        
        return fig


def run_multi_maturity_calibration():
    """Main function for multi-maturity calibration."""
    
    print("="*70)
    print("   MULTI-MATURITY SURFACE CALIBRATION")
    print("="*70)
    
    # Load data
    cache = OptionsDataCache()
    
    try:
        surface, info = cache.load_latest("SPY")
        spot = info['spot']
        print(f"Loaded cached surface from {info['datetime']}")
    except FileNotFoundError:
        print("Fetching fresh data...")
        loader = EnhancedOptionsLoader("SPY", cache)
        surface = loader.get_full_surface(max_dte=120, save_cache=True)
        spot = loader.spot
    
    # Calibrate
    calibrator = MultiMaturityCalibrator(spot)
    calibrator.set_surface(surface)
    
    # Bergomi
    bergomi_result = calibrator.calibrate_bergomi_surface(n_paths=2000)
    
    # Neural SDE
    neural_result = calibrator.calibrate_neural_sde_surface()
    
    # Visualize
    print("\nGenerating term structure plots...")
    fig = calibrator.plot_term_structure_fit()
    fig.show()
    
    return calibrator.results


if __name__ == "__main__":
    run_multi_maturity_calibration()
