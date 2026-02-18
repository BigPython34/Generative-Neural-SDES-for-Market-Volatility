"""
Neural SDE Risk-Neutral Calibration
====================================
Train Neural SDE directly on options implied volatility surface
instead of realized volatility (historical measure).

This is the correct approach for option pricing:
- P-measure (real world): RV, historical patterns → good for VaR, risk
- Q-measure (risk-neutral): IV surface → good for option pricing
"""

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from pathlib import Path
from datetime import datetime
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from engine.neural_sde import NeuralRoughSimulator
from engine.signature_engine import SignatureFeatureExtractor
from utils.black_scholes import BlackScholes
import yaml


class RiskNeutralNeuralSDE:
    """
    Train Neural SDE to match implied volatility surface.
    
    Key differences from P-measure training:
    1. Target = Market IV smile (not realized variance paths)
    2. Loss = RMSE on IV (not MMD on signatures)
    3. Calibration = minimize pricing error
    """
    
    def __init__(self, n_steps: int = 20, sig_depth: int = 3, use_trained: bool = True):
        self.n_steps = n_steps
        self.sig_depth = sig_depth
        
        with open('config/params.yaml', 'r', encoding='utf-8') as f:
            self.yaml_config = yaml.safe_load(f)
        
        self.r = self.yaml_config['pricing']['risk_free_rate']
        
        if use_trained:
            from engine.generative_trainer import GenerativeTrainer
            
            config = {'n_steps': n_steps}
            trainer = GenerativeTrainer(config)
            loaded = trainer.load_model()
            if loaded is not None:
                print("Loaded trained Neural SDE from disk.")
                self.model = loaded
            else:
                print("No saved model found. Training from scratch...")
                self.model = trainer.run(n_epochs=100, batch_size=128)
            self.sig_engine = trainer.sig_extractor
        else:
            T = self.yaml_config['simulation']['T']
            dt = T / n_steps
            self.sig_engine = SignatureFeatureExtractor(truncation_order=sig_depth, dt=dt)
            self.sig_dim = self.sig_engine.get_feature_dim(1)
            
            key = jax.random.PRNGKey(0)
            self.model = NeuralRoughSimulator(sig_dim=self.sig_dim, key=key)
        
    def generate_paths(self, model, spot: float, v0: float, T: float, 
                       rho: float, n_paths: int, key) -> np.ndarray:
        """Generate spot paths using Neural SDE variance dynamics."""
        dt = T / self.n_steps
        r = self.r
        
        key, subkey = jax.random.split(key)
        dW = jax.random.normal(subkey, (n_paths, self.n_steps)) * jnp.sqrt(dt)
        
        v0_arr = jnp.full(n_paths, v0)
        var_paths = jax.vmap(model.generate_variance_path, in_axes=(0, 0, None))(
            v0_arr, dW, dt
        )
        var_paths = jnp.clip(var_paths, 1e-6, 5.0)
        
        key, subkey = jax.random.split(key)
        z_indep = jax.random.normal(subkey, (n_paths, self.n_steps)) * jnp.sqrt(dt)
        spot_dW = rho * dW + jnp.sqrt(1 - rho**2) * z_indep
        
        # Risk-neutral spot: dS/S = r*dt + sqrt(V)*dW_S
        # Previsible variance for adaptedness
        v0_col = jnp.full((n_paths, 1), v0)
        var_prev = jnp.concatenate([v0_col, var_paths[:, :-1]], axis=1)
        vol_paths = jnp.sqrt(var_prev)
        log_ret = (r - 0.5 * var_prev) * dt + vol_paths * spot_dW
        log_s = jnp.cumsum(log_ret, axis=1)
        S_T = spot * jnp.exp(log_s[:, -1])
        
        return np.array(S_T)
    
    def price_option(self, S_T: np.ndarray, K: float, T: float, opt_type: str) -> float:
        """Price option from terminal spot distribution with discounting."""
        if opt_type == 'call':
            payoffs = np.maximum(S_T - K, 0)
        else:
            payoffs = np.maximum(K - S_T, 0)
        return float(np.exp(-self.r * T) * np.mean(payoffs))
    
    def bs_iv(self, price: float, spot: float, K: float, T: float, opt_type: str) -> float:
        """Extract IV from price using shared BS utilities."""
        return BlackScholes.implied_vol(price, spot, K, T, r=self.r, opt_type=opt_type)
    
    def compute_model_ivs(self, model, spot: float, v0: float, T: float,
                         strikes: np.ndarray, types: list, rho: float,
                         n_paths: int, key) -> np.ndarray:
        """Compute model IVs for given strikes"""
        S_T = self.generate_paths(model, spot, v0, T, rho, n_paths, key)
        S_T = np.array(S_T)
        
        ivs = []
        for K, opt_type in zip(strikes, types):
            price = self.price_option(S_T, K, T, opt_type)
            iv = self.bs_iv(price, spot, K, T, opt_type)
            ivs.append(iv)
        
        return np.array(ivs)
    
    def calibrate(self, smile_df: pd.DataFrame, spot: float, T: float,
                 rho: float = -0.7, n_epochs: int = 50, n_paths: int = 10000):
        """
        Calibrate Neural SDE to match market smile.
        
        Uses gradient-free optimization since MC pricing is not differentiable.
        """
        print("\n" + "=" * 60)
        print("RISK-NEUTRAL NEURAL SDE CALIBRATION")
        print("=" * 60)
        
        market_ivs = smile_df['impliedVolatility'].values
        strikes = smile_df['strike'].values
        types = smile_df['type'].tolist()
        
        # Initial v0 from ATM IV
        atm_idx = np.argmin(np.abs(strikes - spot))
        v0 = market_ivs[atm_idx]**2
        
        print(f"\n   Spot: ${spot:.2f}")
        print(f"   T: {T:.4f} ({int(T*365)} days)")
        print(f"   ATM IV: {np.sqrt(v0)*100:.1f}%")
        print(f"   Options: {len(strikes)}")
        print(f"   MC paths: {n_paths:,}")
        
        key = jax.random.PRNGKey(42)
        
        # Initial RMSE
        key, subkey = jax.random.split(key)
        init_ivs = self.compute_model_ivs(
            self.model, spot, v0, T, strikes, types, rho, n_paths, subkey
        )
        init_rmse = np.sqrt(np.mean((init_ivs - market_ivs)**2)) * 100
        print(f"\n   Initial RMSE: {init_rmse:.2f}%")
        
        # Optimization: adjust v0 and model params
        # For simplicity, we'll just optimize v0 here
        # Full calibration would require more sophisticated methods
        
        print(f"\n   Optimizing initial variance v0...")
        
        best_rmse = init_rmse
        best_v0 = v0
        
        # Grid search on v0
        for v0_mult in np.linspace(0.5, 2.0, 10):
            test_v0 = v0 * v0_mult
            
            key, subkey = jax.random.split(key)
            test_ivs = self.compute_model_ivs(
                self.model, spot, test_v0, T, strikes, types, rho, n_paths, subkey
            )
            test_rmse = np.sqrt(np.mean((test_ivs - market_ivs)**2)) * 100
            
            if test_rmse < best_rmse:
                best_rmse = test_rmse
                best_v0 = test_v0
                print(f"      v0={test_v0:.4f} (sigma={np.sqrt(test_v0)*100:.1f}%): RMSE={test_rmse:.2f}%")
        
        print(f"\n   Best v0: {best_v0:.4f} (sigma={np.sqrt(best_v0)*100:.1f}%)")
        print(f"   Final RMSE: {best_rmse:.2f}%")
        
        # Compare with Black-Scholes
        bs_rmse = np.sqrt(np.mean((np.sqrt(best_v0) - market_ivs)**2)) * 100
        print(f"\n   Black-Scholes RMSE: {bs_rmse:.2f}%")
        
        if best_rmse < bs_rmse:
            print(f"   Neural SDE beats BS by {bs_rmse - best_rmse:.2f} vol pts")
        else:
            print(f"   BS wins by {best_rmse - bs_rmse:.2f} vol pts")
        
        # Final model IVs
        key, subkey = jax.random.split(key)
        final_ivs = self.compute_model_ivs(
            self.model, spot, best_v0, T, strikes, types, rho, n_paths, subkey
        )
        
        return {
            'model': self.model,
            'v0': best_v0,
            'rmse': best_rmse,
            'bs_rmse': bs_rmse,
            'model_ivs': final_ivs,
            'market_ivs': market_ivs,
            'strikes': strikes
        }


def main():
    """Run risk-neutral calibration"""
    
    # Load options data
    from quant.options_cache import OptionsDataCache
    
    cache = OptionsDataCache()
    
    try:
        surface, metadata = cache.load_latest("SPY")
        spot = metadata['spot']
    except FileNotFoundError:
        print("No cached options data. Run options_calibration.py first.")
        return
    
    # Filter for 30 DTE
    target_dte = 30
    surface['dte_diff'] = abs(surface['dte'] - target_dte)
    smile = surface[surface['dte_diff'] == surface['dte_diff'].min()].copy()
    
    # OTM only, tight moneyness
    smile = smile[
        ((smile['type'] == 'call') & (smile['strike'] >= spot)) |
        ((smile['type'] == 'put') & (smile['strike'] <= spot))
    ]
    smile = smile[
        (smile['strike'] / spot >= 0.92) & 
        (smile['strike'] / spot <= 1.08)
    ].sort_values('strike')
    
    T = smile['dte'].iloc[0] / 365.0
    
    print(f"\nLoaded {len(smile)} options, T={T:.4f}")
    
    # Calibrate
    calibrator = RiskNeutralNeuralSDE(n_steps=20, sig_depth=3)
    cfg = calibrator.yaml_config
    results = calibrator.calibrate(
        smile, spot, T, 
        rho=cfg['bergomi']['rho'], 
        n_epochs=50, 
        n_paths=cfg.get('pricing', {}).get('n_mc_paths', 20000)
    )
    
    # Plot
    import plotly.graph_objects as go
    
    moneyness = results['strikes'] / spot
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=moneyness, y=results['market_ivs'] * 100,
        mode='markers', name='Market',
        marker=dict(size=10, color='black')
    ))
    
    fig.add_trace(go.Scatter(
        x=moneyness, y=results['model_ivs'] * 100,
        mode='lines+markers', 
        name=f"Neural SDE (RMSE: {results['rmse']:.2f}%)",
        line=dict(color='red', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=moneyness, y=[np.sqrt(results['v0']) * 100] * len(moneyness),
        mode='lines', 
        name=f"Black-Scholes (RMSE: {results['bs_rmse']:.2f}%)",
        line=dict(color='gray', dash='dash')
    ))
    
    fig.update_layout(
        title="Risk-Neutral Neural SDE Calibration",
        xaxis_title="Moneyness (K/S)",
        yaxis_title="Implied Volatility (%)",
        template="plotly_white"
    )
    
    fig.write_html("outputs/risk_neutral_calibration.html", include_plotlyjs='cdn')
    fig.show()
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'v0': float(results['v0']),
        'sigma0': float(np.sqrt(results['v0'])),
        'neural_rmse': float(results['rmse']),
        'bs_rmse': float(results['bs_rmse']),
        'winner': 'neural' if results['rmse'] < results['bs_rmse'] else 'bs'
    }
    
    with open('outputs/risk_neutral_calibration.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to outputs/risk_neutral_calibration.json")


if __name__ == "__main__":
    main()
