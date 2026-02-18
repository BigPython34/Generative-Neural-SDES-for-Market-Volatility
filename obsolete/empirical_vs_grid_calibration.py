"""
Empirical vs Grid Search Calibration
=====================================
Compare Bergomi calibration using:
1. Empirical H from high-frequency SPX data (H ≈ 0.05)
2. Grid search H optimized on options (H ≈ 0.20)

This tests whether using the "true" roughness parameter improves pricing.
"""

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from pathlib import Path
import json
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from quant.options_cache import OptionsDataCache

# Suppress JAX warnings
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class BergomiPricer:
    """Rough Bergomi Monte Carlo pricer"""
    
    def __init__(self, H: float, eta: float, rho: float, xi0: float, n_paths: int = 5000):
        self.H = H
        self.eta = eta
        self.rho = rho
        self.xi0 = xi0
        self.n_paths = n_paths
        
    def generate_fbm(self, n_steps: int, dt: float, key) -> np.ndarray:
        """Generate fractional Brownian motion paths using Cholesky"""
        # Time grid
        t = np.arange(1, n_steps + 1) * dt
        
        # Covariance matrix for fBM
        H = self.H
        cov = np.zeros((n_steps, n_steps))
        for i in range(n_steps):
            for j in range(n_steps):
                ti, tj = t[i], t[j]
                cov[i, j] = 0.5 * (ti**(2*H) + tj**(2*H) - abs(ti - tj)**(2*H))
        
        # Regularize
        cov += np.eye(n_steps) * 1e-8
        
        # Cholesky
        try:
            L = np.linalg.cholesky(cov)
        except:
            # Fallback to eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.maximum(eigvals, 1e-8)
            L = eigvecs @ np.diag(np.sqrt(eigvals))
        
        # Generate
        z = jax.random.normal(key, (self.n_paths, n_steps))
        z = np.array(z)
        
        fbm = z @ L.T
        return fbm
    
    def simulate_paths(self, spot: float, T: float, n_steps: int, key) -> np.ndarray:
        """Simulate spot price paths under rough Bergomi"""
        dt = T / n_steps
        
        # Generate fBM for variance
        key, subkey = jax.random.split(key)
        W_H = self.generate_fbm(n_steps, dt, subkey)
        
        # Variance paths: V(t) = xi0 * exp(eta * W_H(t) - 0.5 * eta^2 * t^(2H))
        t = np.arange(1, n_steps + 1) * dt
        variance_correction = 0.5 * self.eta**2 * t**(2*self.H)
        
        log_var = self.eta * W_H - variance_correction[np.newaxis, :]
        var_paths = self.xi0 * np.exp(log_var)
        
        # Cap variance for stability
        var_paths = np.clip(var_paths, 1e-6, 10.0)
        
        # Generate correlated Brownian for spot
        key, subkey = jax.random.split(key)
        z_indep = jax.random.normal(subkey, (self.n_paths, n_steps))
        z_indep = np.array(z_indep)
        
        # Extract increments from fBM for correlation
        dW_H = np.diff(np.hstack([np.zeros((self.n_paths, 1)), W_H]), axis=1)
        dW_H_normalized = dW_H / (np.std(dW_H, axis=0, keepdims=True) + 1e-8) * np.sqrt(dt)
        
        dW_S = self.rho * dW_H_normalized + np.sqrt(1 - self.rho**2) * np.sqrt(dt) * z_indep
        
        # Simulate spot
        vol_paths = np.sqrt(var_paths)
        drift = -0.5 * var_paths * dt
        diffusion = vol_paths * dW_S
        
        log_returns = drift + diffusion
        log_spot = np.cumsum(log_returns, axis=1)
        
        spot_paths = spot * np.exp(np.hstack([np.zeros((self.n_paths, 1)), log_spot]))
        
        return spot_paths
    
    def price_option(self, spot: float, strike: float, T: float, option_type: str, key) -> float:
        """Price a single option"""
        n_steps = max(50, int(T * 252))
        paths = self.simulate_paths(spot, T, n_steps, key)
        
        S_T = paths[:, -1]
        
        if option_type == 'call':
            payoffs = np.maximum(S_T - strike, 0)
        else:
            payoffs = np.maximum(strike - S_T, 0)
        
        # Discount (assuming r=0 for simplicity)
        price = np.mean(payoffs)
        return price
    
    def implied_vol(self, price: float, spot: float, strike: float, T: float, option_type: str) -> float:
        """Extract implied volatility from price using Black-Scholes inversion."""
        from utils.black_scholes import BlackScholes
        return BlackScholes.implied_vol(price, spot, strike, T, r=0.0, opt_type=option_type)


def run_comparison():
    """Compare empirical vs grid search Bergomi calibration"""
    
    print("=" * 70)
    print("   EMPIRICAL vs GRID SEARCH BERGOMI CALIBRATION")
    print("=" * 70)
    print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load empirical calibration
    empirical_path = Path("data/advanced_calibration.json")
    if empirical_path.exists():
        with open(empirical_path) as f:
            empirical = json.load(f)
        emp_params = empirical['optimal_params']
        print(f"\nLoaded empirical calibration from high-frequency data")
    else:
        print("No empirical calibration found, using defaults")
        emp_params = {'H': 0.05, 'eta': 1.0, 'rho': -0.75, 'xi_0': 0.03}
    
    # Load grid search calibration
    grid_path = Path("outputs/calibration_report.json")
    if grid_path.exists():
        with open(grid_path) as f:
            grid = json.load(f)
        grid_params = grid.get('bergomi_params', {})
        print(f"Loaded grid search calibration from options")
    else:
        grid_params = {'hurst': 0.20, 'eta': 1.0, 'rho': -0.80, 'xi0': 0.0262}
    
    # Parameters
    print("\n" + "-" * 50)
    print("PARAMETERS COMPARISON")
    print("-" * 50)
    print(f"\n{'Parameter':<12} {'Empirical (HF)':<18} {'Grid Search':<18} {'Source'}")
    print("-" * 70)
    print(f"{'H':<12} {emp_params['H']:<18.4f} {grid_params.get('hurst', 0.20):<18.4f} SPX 30min RV")
    print(f"{'η':<12} {emp_params['eta']:<18.2f} {grid_params.get('eta', 1.0):<18.2f} VIX vol")
    print(f"{'ρ':<12} {emp_params['rho']:<18.2f} {grid_params.get('rho', -0.80):<18.2f} Options fit")
    print(f"{'ξ₀':<12} {emp_params['xi_0']:<18.4f} {grid_params.get('xi0', 0.03):<18.4f} Current VIX")
    
    # Load options data
    print("\n" + "-" * 50)
    print("LOADING OPTIONS DATA")
    print("-" * 50)
    
    cache = OptionsDataCache()
    
    try:
        surface, metadata = cache.load_latest("SPY")
        spot = metadata['spot']
        print(f"Loaded {len(surface)} options from cache")
    except FileNotFoundError:
        print("No cached options data. Run options_calibration.py first.")
        return
    print(f"   Spot: ${spot:.2f}")
    
    # Filter for ~30 DTE ATM options
    target_dte = 30
    surface['dte_diff'] = abs(surface['dte'] - target_dte)
    smile_30d = surface[surface['dte_diff'] == surface['dte_diff'].min()].copy()
    
    # Filter OTM
    smile_30d = smile_30d[
        ((smile_30d['type'] == 'call') & (smile_30d['strike'] >= spot)) |
        ((smile_30d['type'] == 'put') & (smile_30d['strike'] <= spot))
    ]
    
    # Filter by moneyness (0.85 to 1.15)
    smile_30d = smile_30d[
        (smile_30d['strike'] / spot >= 0.85) & 
        (smile_30d['strike'] / spot <= 1.15)
    ]
    
    smile_30d = smile_30d.sort_values('strike')
    
    print(f"Using {len(smile_30d)} options for 30 DTE smile")
    
    actual_dte = smile_30d['dte'].iloc[0]
    T = actual_dte / 365.0
    print(f"   T = {T:.4f} ({actual_dte} days)")
    
    # Price with both parameter sets
    print("\n" + "-" * 50)
    print("MONTE CARLO PRICING")
    print("-" * 50)
    
    n_paths = 10000
    key = jax.random.PRNGKey(42)
    
    results = {}
    
    # 1. Empirical parameters
    print(f"\nEmpirical Bergomi (H={emp_params['H']:.3f})...")
    pricer_emp = BergomiPricer(
        H=emp_params['H'],
        eta=emp_params['eta'],
        rho=emp_params['rho'],
        xi0=emp_params['xi_0'],
        n_paths=n_paths
    )
    
    emp_ivs = []
    for i, (_, row) in enumerate(smile_30d.iterrows()):
        key, subkey = jax.random.split(key)
        price = pricer_emp.price_option(spot, row['strike'], T, row['type'], subkey)
        iv = pricer_emp.implied_vol(price, spot, row['strike'], T, row['type'])
        emp_ivs.append(iv)
        
        if (i + 1) % 10 == 0:
            print(f"   Priced {i+1}/{len(smile_30d)} options...")
    
    results['Empirical Bergomi'] = np.array(emp_ivs)
    
    # 2. Grid search parameters
    print(f"\nGrid Search Bergomi (H={grid_params.get('hurst', 0.20):.3f})...")
    pricer_grid = BergomiPricer(
        H=grid_params.get('hurst', 0.20),
        eta=grid_params.get('eta', 1.0),
        rho=grid_params.get('rho', -0.80),
        xi0=grid_params.get('xi0', 0.0262),
        n_paths=n_paths
    )
    
    grid_ivs = []
    for i, (_, row) in enumerate(smile_30d.iterrows()):
        key, subkey = jax.random.split(key)
        price = pricer_grid.price_option(spot, row['strike'], T, row['type'], subkey)
        iv = pricer_grid.implied_vol(price, spot, row['strike'], T, row['type'])
        grid_ivs.append(iv)
        
        if (i + 1) % 10 == 0:
            print(f"   Priced {i+1}/{len(smile_30d)} options...")
    
    results['Grid Search Bergomi'] = np.array(grid_ivs)
    
    # 3. Black-Scholes (flat vol)
    atm_iv = smile_30d.loc[
        (smile_30d['strike'] - spot).abs().idxmin(), 'impliedVolatility'
    ]
    results['Black-Scholes'] = np.full(len(smile_30d), atm_iv)
    
    # Compute RMSE
    print("\n" + "-" * 50)
    print("RESULTS")
    print("-" * 50)
    
    market_ivs = smile_30d['impliedVolatility'].values
    
    print(f"\n{'Model':<25} {'RMSE (vol pts)':<15} {'Status'}")
    print("-" * 55)
    
    rmse_dict = {}
    for name, ivs in results.items():
        valid = ~np.isnan(ivs)
        rmse = np.sqrt(np.mean((ivs[valid] - market_ivs[valid])**2)) * 100
        rmse_dict[name] = rmse
    
    # Sort by RMSE
    sorted_results = sorted(rmse_dict.items(), key=lambda x: x[1])
    
    for i, (name, rmse) in enumerate(sorted_results):
        medal = ["1.", "2.", "3."][i] if i < 3 else "  "
        print(f"{medal} {name:<23} {rmse:<15.2f}%")
    
    # Analysis
    print("\n" + "-" * 50)
    print("ANALYSIS")
    print("-" * 50)
    
    emp_rmse = rmse_dict['Empirical Bergomi']
    grid_rmse = rmse_dict['Grid Search Bergomi']
    
    if emp_rmse < grid_rmse:
        improvement = (grid_rmse - emp_rmse) / grid_rmse * 100
        print(f"\nEmpirical H={emp_params['H']:.3f} BEATS Grid H={grid_params.get('hurst', 0.20):.3f}")
        print(f"   Improvement: {improvement:.1f}%")
        print(f"\n   This suggests the 'true' roughness from HF data gives better pricing!")
    else:
        degradation = (emp_rmse - grid_rmse) / grid_rmse * 100
        print(f"\nGrid Search H={grid_params.get('hurst', 0.20):.3f} beats Empirical H={emp_params['H']:.3f}")
        print(f"   Degradation: {degradation:.1f}%")
        print(f"\n   The market may price a different 'effective H' than the statistical one.")
        print(f"   This is a known phenomenon: risk-neutral H ≠ physical H")
    
    # Save comparison
    comparison = {
        'timestamp': datetime.now().isoformat(),
        'empirical_params': emp_params,
        'grid_params': {k: float(v) if isinstance(v, (int, float)) else v for k, v in grid_params.items()},
        'rmse_results': {k: float(v) for k, v in rmse_dict.items()},
        'n_options': len(smile_30d),
        'n_paths': n_paths,
        'winner': 'empirical' if emp_rmse < grid_rmse else 'grid'
    }
    
    with open('outputs/empirical_vs_grid.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\nResults saved to outputs/empirical_vs_grid.json")
    
    # Plot comparison
    print("\nGenerating comparison plot...")
    
    import plotly.graph_objects as go
    
    moneyness = smile_30d['strike'].values / spot
    
    fig = go.Figure()
    
    # Market
    fig.add_trace(go.Scatter(
        x=moneyness, y=market_ivs * 100,
        mode='markers', name='Market',
        marker=dict(size=10, color='black', symbol='circle')
    ))
    
    # Models
    colors = {'Empirical Bergomi': 'red', 'Grid Search Bergomi': 'blue', 'Black-Scholes': 'gray'}
    
    for name, ivs in results.items():
        fig.add_trace(go.Scatter(
            x=moneyness, y=ivs * 100,
            mode='lines+markers', name=f"{name} (RMSE: {rmse_dict[name]:.2f}%)",
            line=dict(color=colors.get(name, 'green'), width=2),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title=f"Empirical vs Grid Search Bergomi - {actual_dte} DTE",
        xaxis_title="Moneyness (K/S)",
        yaxis_title="Implied Volatility (%)",
        template="plotly_white",
        legend=dict(x=0.02, y=0.98),
        height=500
    )
    
    fig.write_html("outputs/empirical_vs_grid_plot.html", include_plotlyjs='cdn')
    fig.show()
    
    print(f"Plot saved to outputs/empirical_vs_grid_plot.html")
    
    return comparison


if __name__ == "__main__":
    comparison = run_comparison()
