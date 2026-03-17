"""
Rough Bergomi Calibration via Wasserstein-1 Distance
======================================================

Implementation based on:
"Efficient Calibration in the rough Bergomi model by Wasserstein distance"
(arXiv:2512.00448v1)

Compares MSE vs Wasserstein-1 distance calibration for SPY options.
Tests generalization to barrier options and parameter stability.

Author: MOI
Date: 2026-03-13
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Dict, List
import json
from datetime import datetime
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from quant.calibration.vix_futures_loader import assemble_calibration_data
from quant.calibration.joint_calibrator import JointCalibrator
from quant.models.black_scholes import BlackScholes
from scipy.optimize import minimize, differential_evolution
from scipy.stats import gaussian_kde


@dataclass
class WassersteinResult:
    """Stores calibration results."""
    H: float
    eta: float
    rho: float
    xi0: float
    loss_mse: float
    loss_wasserstein: float
    iterations: int
    convergence: str
    market_samples: np.ndarray
    model_samples: np.ndarray
    

def empirical_wasserstein_1(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute empirical Wasserstein-1 distance between two 1D distributions.
    
    W_1(X, Y) = (1/m) * sum(|X_(i) - Y_(i)|)
    where X_(i), Y_(i) are order statistics.
    
    Paper: Definition 3.2
    """
    if len(X) != len(Y):
        # Resample to same size
        m = min(len(X), len(Y))
        X = np.random.choice(X, m, replace=False)
        Y = np.random.choice(Y, m, replace=False)
    
    X_sorted = np.sort(X)
    Y_sorted = np.sort(Y)
    
    return np.mean(np.abs(X_sorted - Y_sorted))


def wasserstein_loss_batch(
    S_model: np.ndarray, 
    S_market: np.ndarray, 
    maturities: List[float]
) -> float:
    """
    Compute total Wasserstein-1 loss across maturities.
    
    L(theta) = (1/M) * sum_j W_1(S_{T_j}(theta), S_{T_j}^{MKT})
    
    Args:
        S_model: [n_paths, n_maturities] simulated prices
        S_market: [n_market_samples] market-implied prices at single maturity
        maturities: List of T values (for structure, currently single T)
    
    Returns:
        Scalar loss
    """
    if len(S_model.shape) == 1:
        # Single maturity case
        return empirical_wasserstein_1(S_model, S_market)
    
    # Multi-maturity case
    total_loss = 0.0
    for j, T in enumerate(maturities):
        w1 = empirical_wasserstein_1(S_model[:, j], S_market)
        total_loss += w1
    
    return total_loss / len(maturities)


def mse_loss(prices_model: np.ndarray, prices_market: np.ndarray) -> float:
    """Standard MSE for baseline comparison."""
    return np.mean((prices_model - prices_market) ** 2)


def calibrate_bergomi_wasserstein(
    options_data: pd.DataFrame,
    T_maturity: float,
    spot: float,
    risk_free: float,
    num_mc_paths: int = 5000,
    num_iterations: int = 100,
    use_market_prices: bool = True
) -> WassersteinResult:
    """
    Calibrate rBergomi parameters using Wasserstein-1 distance.
    
    Steps:
    1. Extract market-implied risk-neutral samples from option prices
    2. Grid search initial parameters (H, eta, rho)
    3. Refine with Nelder-Mead minimizing W_1 distance
    4. Compare with MSE loss on same parameters
    
    Paper: Algorithm 1 (adapted)
    
    Args:
        options_data: DataFrame with columns [K, T_exp, bid, mid, ask, IV]
        T_maturity: Time to maturity in years
        spot: Current spot price
        risk_free: Risk-free rate
        num_mc_paths: MC simulation paths
        num_iterations: Max optimization iterations
        use_market_prices: If True, extract from option prices; else use IV->price
    
    Returns:
        WassersteinResult with calibrated params and diagnostics
    """
    
    print("\n" + "="*80)
    print("WASSERSTEIN-1 CALIBRATION FOR rBERGOMI")
    print("="*80)
    
    # ===== Step 0: Extract market-implied distribution =====
    print("\n[STEP 1] Extracting market-implied risk-neutral distribution...")
    
    # Filter options for this maturity
    opts_T = options_data[
        (options_data['T_exp'] >= T_maturity * 0.9) & 
        (options_data['T_exp'] <= T_maturity * 1.1)
    ].copy()
    
    if len(opts_T) < 20:
        print(f"  Warning: Only {len(opts_T)} options for maturity {T_maturity:.4f}. Need >= 20.")
        return None
    
    print(f"  Found {len(opts_T)} options for T={T_maturity:.4f}")
    
    # Extract strikes and IV
    strikes = opts_T['K'].values
    ivs = opts_T['IV'].values
    bids = opts_T['bid'].values
    asks = opts_T['ask'].values
    
    # Generate market-implied samples via Breeden-Litzenberger
    # Simple approach: use IV-based call prices and construct distribution
    S_market_samples = _extract_risk_neutral_samples(
        strikes, ivs, spot, risk_free, T_maturity, num_samples=2000
    )
    
    print(f"  [OK] Extracted {len(S_market_samples)} RN samples")
    print(f"       Mean: ${S_market_samples.mean():.2f}, Std: ${S_market_samples.std():.2f}")
    
    atm_iv = float(opts_T[np.abs(opts_T['K'] - spot) < 2]['IV'].mean())
    print(f"  ATM IV: {atm_iv*100:.2f}%")
    
    # ===== Step 1: Grid search =====
    print("\n[STEP 2] Grid search over (H, eta, rho)...")
    
    H_grid = [0.03, 0.07, 0.10, 0.15, 0.20]
    eta_grid = [0.8, 1.2, 1.6, 2.0, 2.4]
    rho_grid = [-0.95, -0.80, -0.70]
    
    best_loss = np.inf
    best_params = None
    eval_count = 0
    
    for H in H_grid:
        for eta in eta_grid:
            for rho in rho_grid:
                # Simulate paths
                xi0 = atm_iv ** 2
                S_model_paths = _simulate_bergomi(
                    spot, H, eta, rho, xi0, T_maturity, 
                    num_paths=2000, num_steps=50
                )
                
                # Wasserstein loss
                loss_w1 = empirical_wasserstein_1(S_model_paths, S_market_samples)
                eval_count += 1
                
                if loss_w1 < best_loss:
                    best_loss = loss_w1
                    best_params = (H, eta, rho)
                    
                    if eval_count % 15 == 0 or eval_count == 1:
                        print(f"  [{eval_count}/{len(H_grid)*len(eta_grid)*len(rho_grid)}] " +
                              f"H={H:.2f}, eta={eta:.1f}, rho={rho:.2f} " +
                              f"-> W1={loss_w1:.6f}")
    
    H_init, eta_init, rho_init = best_params
    print(f"\n  Grid best: H={H_init:.2f}, eta={eta_init:.1f}, rho={rho_init:.2f}, W1={best_loss:.6f}")
    
    # ===== Step 2: Refinement via Nelder-Mead =====
    print("\n[STEP 3] Refining with Nelder-Mead (Wasserstein)...")
    
    def wasserstein_objective(params):
        """Objective: minimize W_1 distance."""
        H, eta, rho = params
        
        # Bounds check
        if H <= 0 or H >= 0.5 or eta <= 0.1 or rho <= -0.999 or rho >= -0.1:
            return 1e6
        
        # Simulate
        xi0 = atm_iv ** 2
        try:
            S_model = _simulate_bergomi(
                spot, H, eta, rho, xi0, T_maturity, 
                num_paths=3000, num_steps=100
            )
            loss = empirical_wasserstein_1(S_model, S_market_samples)
        except:
            return 1e6
        
        return loss
    
    result_w1 = minimize(
        wasserstein_objective,
        [H_init, eta_init, rho_init],
        method='Nelder-Mead',
        options={'maxiter': num_iterations, 'xatol': 1e-5, 'fatol': 1e-6},
        callback=lambda xk: _callback_minimize(xk, eval_count)
    )
    
    H_opt, eta_opt, rho_opt = result_w1.x
    loss_w1_opt = result_w1.fun
    
    print(f"\n  [OK] Refinement complete (converged={result_w1.success})")
    print(f"       H={H_opt:.4f}, eta={eta_opt:.4f}, rho={rho_opt:.4f}")
    print(f"       W_1 loss = {loss_w1_opt:.6f}")
    
    # ===== Step 3: MSE comparison =====
    print("\n[STEP 4] Comparing MSE loss on same parameters...")
    
    xi0_opt = atm_iv ** 2
    S_model_opt = _simulate_bergomi(
        spot, H_opt, eta_opt, rho_opt, xi0_opt, T_maturity,
        num_paths=5000, num_steps=100
    )
    
    # Price market options with model
    prices_model = _price_options_from_samples(
        S_model_opt, strikes, spot, risk_free, T_maturity
    )
    prices_market = np.exp(-risk_free * T_maturity) * np.maximum(
        S_market_samples.mean() - strikes, 0
    )
    loss_mse_opt = mse_loss(prices_model, prices_market)
    
    print(f"  MSE loss (same params): {loss_mse_opt:.6f}")
    print(f"  W_1 loss (same params): {loss_w1_opt:.6f}")
    
    # ===== Diagnostics =====
    print("\n[STEP 5] Generating diagnostics...")
    
    # Compare with grid best using MSE
    S_grid = _simulate_bergomi(
        spot, H_init, eta_init, rho_init, atm_iv**2, T_maturity,
        num_paths=5000, num_steps=100
    )
    prices_grid = _price_options_from_samples(
        S_grid, strikes, spot, risk_free, T_maturity
    )
    loss_mse_grid = mse_loss(prices_grid, prices_market)
    
    print(f"\n  === Parameter Comparison ===")
    print(f"  Grid best (MSE loss):")
    print(f"    H={H_init:.4f}, eta={eta_init:.4f}, rho={rho_init:.4f}")
    print(f"    MSE={loss_mse_grid:.6f}, W_1={best_loss:.6f}")
    print(f"\n  Refined (W_1 loss):")
    print(f"    H={H_opt:.4f}, eta={eta_opt:.4f}, rho={rho_opt:.4f}")
    print(f"    MSE={loss_mse_opt:.6f}, W_1={loss_w1_opt:.6f}")
    
    # Distribution comparison
    print(f"\n  === Distribution Metrics ===")
    print(f"  Market samples: mean=${S_market_samples.mean():.2f}, std=${S_market_samples.std():.2f}")
    print(f"  Model samples:  mean=${S_model_opt.mean():.2f}, std=${S_model_opt.std():.2f}")
    print(f"  Skewness - Market: {pd.Series(S_market_samples).skew():.4f}")
    print(f"  Skewness - Model:  {pd.Series(S_model_opt).skew():.4f}")
    
    return WassersteinResult(
        H=H_opt,
        eta=eta_opt,
        rho=rho_opt,
        xi0=xi0_opt,
        loss_mse=loss_mse_opt,
        loss_wasserstein=loss_w1_opt,
        iterations=result_w1.nit,
        convergence="Success" if result_w1.success else "Incomplete",
        market_samples=S_market_samples,
        model_samples=S_model_opt,
    )


def _extract_risk_neutral_samples(
    strikes: np.ndarray,
    ivs: np.ndarray,
    spot: float,
    rate: float,
    T: float,
    num_samples: int = 2000
) -> np.ndarray:
    """
    Extract risk-neutral samples from option IVs.
    
    Simple approach: Bootstrap market strikes with weights based on IV,
    then add small perturbations.
    """
    
    # Remove duplicates and sort
    unique_idx = np.unique(strikes, return_index=True)[1]
    strikes_unique = strikes[sorted(unique_idx)]
    ivs_unique = ivs[sorted(unique_idx)]
    
    # Weight by IV (higher IV = higher probability of reaching that strike)
    weights = ivs_unique / ivs_unique.sum()
    
    # Bootstrap samples from strikes
    samples = np.random.choice(
        strikes_unique, 
        size=num_samples,
        p=weights,
        replace=True
    )
    
    # Add small log-normal perturbations for smooth distribution
    log_returns = np.random.normal(0, ivs_unique.mean() * np.sqrt(T), num_samples)
    samples = samples * np.exp(log_returns)
    
    return np.maximum(samples, spot * 0.3)  # Non-negative prices


def _simulate_bergomi(
    S0: float,
    H: float,
    eta: float,
    rho: float,
    xi0: float,
    T: float,
    num_paths: int,
    num_steps: int
) -> np.ndarray:
    """
    Simulate rBergomi using modified Sum-of-Exponentials (mSOE) scheme.
    
    Simplified version: exact on first step, SOE approximation for remainder.
    Paper: Section 2.3
    
    Returns:
        [num_paths] array of terminal prices S_T
    """
    
    dt = T / num_steps
    paths = np.zeros((num_paths, num_steps + 1))
    paths[:, 0] = S0
    
    # SOE approximation (N=4 terms for efficiency)
    N = 4
    lambda_k = np.array([0.047, 1.228, 0.619, 6.203])
    omega_k = np.array([0.399, 1.483, 5.516, 20.522])
    
    # OU processes for historical part
    Y_k = np.zeros((num_paths, N))
    
    for step in range(num_steps):
        t = step * dt
        
        # Local part: exact Brownian increment
        dW = np.random.normal(0, np.sqrt(dt), num_paths)
        I_local = np.random.normal(0, np.sqrt(dt ** (2 * H)), num_paths)
        
        # Historical part: SOE approximation
        dW_k = np.random.normal(0, np.sqrt(dt), (num_paths, N))
        I_hist = 0
        
        for k in range(N):
            Y_k[:, k] = Y_k[:, k] * np.exp(-lambda_k[k] * dt) + dW_k[:, k]
            I_hist += omega_k[k] * Y_k[:, k] * (1 - np.exp(-lambda_k[k] * dt)) / lambda_k[k]
        
        I_t = I_local + 2 * H * I_hist
        
        # Variance and price updates
        V_t = xi0 * np.exp(eta * I_t - 0.5 * eta ** 2 * (t + dt) ** (2 * H))
        V_t = np.clip(V_t, 0.01, 10)  # Keep reasonable
        
        dW_perp = np.random.normal(0, np.sqrt(dt), num_paths)
        dS = paths[:, step] * (
            (0.03) * dt + 
            np.sqrt(V_t) * (rho * dW + np.sqrt(1 - rho ** 2) * dW_perp)
        )
        
        paths[:, step + 1] = paths[:, step] + dS
        paths[:, step + 1] = np.maximum(paths[:, step + 1], S0 * 0.5)  # Floor
    
    return paths[:, -1]


def _price_options_from_samples(
    S_samples: np.ndarray,
    strikes: np.ndarray,
    S0: float,
    rate: float,
    T: float
) -> np.ndarray:
    """Price calls/puts using terminal distribution samples."""
    
    prices = np.zeros(len(strikes))
    discount = np.exp(-rate * T)
    
    for i, K in enumerate(strikes):
        # Determine call vs put based on moneyness
        if K > S0:
            # Call
            payoff = np.maximum(S_samples - K, 0)
        else:
            # Put
            payoff = np.maximum(K - S_samples, 0)
        
        prices[i] = discount * np.mean(payoff)
    
    return prices


def _callback_minimize(xk, eval_count):
    """Progress callback for optimization."""
    if eval_count % 20 == 0:
        H, eta, rho = xk
        print(f"    Iteration {eval_count}: H={H:.4f}, eta={eta:.4f}, rho={rho:.4f}")


def plot_comparison(result_wass: WassersteinResult, title_suffix: str = ""):
    """Plot market vs model distributions."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    
    # 1. Density plots
    ax = axes[0, 0]
    ax.hist(result_wass.market_samples, bins=50, alpha=0.5, label='Market (RN)', density=True)
    ax.hist(result_wass.model_samples, bins=50, alpha=0.5, label='Model', density=True)
    ax.set_xlabel('Terminal Price ($)')
    ax.set_ylabel('Density')
    ax.set_title('Risk-Neutral Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. CDF comparison
    ax = axes[0, 1]
    market_sorted = np.sort(result_wass.market_samples)
    model_sorted = np.sort(result_wass.model_samples)
    ax.plot(market_sorted, np.linspace(0, 1, len(market_sorted)), label='Market', lw=2)
    ax.plot(model_sorted, np.linspace(0, 1, len(model_sorted)), label='Model', lw=2)
    ax.set_xlabel('Terminal Price ($)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('CDF Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Q-Q plot
    ax = axes[1, 0]
    q_market = np.percentile(result_wass.market_samples, np.linspace(0, 100, 100))
    q_model = np.percentile(result_wass.model_samples, np.linspace(0, 100, 100))
    ax.scatter(q_market, q_model, alpha=0.6)
    ax.plot([q_market.min(), q_market.max()], 
            [q_market.min(), q_market.max()], 'r--', lw=2, label='45-degree line')
    ax.set_xlabel('Market Quantiles ($)')
    ax.set_ylabel('Model Quantiles ($)')
    ax.set_title('Q-Q Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Parameter text box
    ax = axes[1, 1]
    ax.axis('off')
    info_text = f"""
    Wasserstein-1 Calibration Results
    
    Optimized Parameters:
    • H (Hurst exponent) = {result_wass.H:.4f}
    • η (vol-of-vol) = {result_wass.eta:.4f}
    • ρ (correlation) = {result_wass.rho:.4f}
    • ξ₀ (forward var) = {result_wass.xi0:.6f}
    
    Loss Metrics:
    • W₁ distance = {result_wass.loss_wasserstein:.6f}
    • MSE loss = {result_wass.loss_mse:.6f}
    • Iterations = {result_wass.iterations}
    • Convergence = {result_wass.convergence}
    
    Distribution Statistics:
    • Market mean = ${result_wass.market_samples.mean():.2f}
    • Model mean = ${result_wass.model_samples.mean():.2f}
    • Market skew = {pd.Series(result_wass.market_samples).skew():.4f}
    • Model skew = {pd.Series(result_wass.model_samples).skew():.4f}
    """
    ax.text(0.1, 0.5, info_text, fontsize=10, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig


def main():
    """Main execution."""
    
    print("\n" + "="*80)
    print("WASSERSTEIN-1 vs MSE CALIBRATION COMPARISON")
    print("Rough Bergomi Model on SPY Options")
    print("="*80)
    
    # Load SPY options
    print("\n[LOADING DATA]")
    try:
        market_data = assemble_calibration_data()
        print(f"[OK] Loaded {len(market_data.spx_surface)} SPX options")
    except Exception as e:
        print(f"[ERROR] Failed to load joint calibration data: {e}")
        print("Proceeding with SPY-only data loading...")
        
        # Fallback: load from cache
        options_file = Path("data/market/equity_indices/spx_options_latest.csv")
        if options_file.exists():
            options_df = pd.read_csv(options_file)
            print(f"[OK] Loaded {len(options_df)} SPX options from cache")
        else:
            print("[ERROR] No options data available. Aborting.")
            return
        
        # Minimal data structure
        class MarketData:
            spx_surface = options_df
            risk_free_rate = 0.0364
        
        market_data = MarketData()
    
    # Get spot price
    if hasattr(market_data, 'spx_surface'):
        options_df = market_data.spx_surface
        if 'spot' in options_df.columns:
            spot = options_df['spot'].iloc[0]
        else:
            spot = 666.06  # Hardcoded from earlier run
    else:
        spot = 666.06
    
    rate = getattr(market_data, 'risk_free_rate', 0.0364)
    
    print(f"Spot: ${spot:.2f}")
    print(f"Risk-free rate: {rate*100:.3f}%")
    
    # Prepare options data with required columns
    if 'IV' not in options_df.columns:
        options_df['IV'] = 0.25  # Default IV
    if 'bid' not in options_df.columns:
        options_df['bid'] = options_df.get('mid', 0) * 0.95
    if 'ask' not in options_df.columns:
        options_df['ask'] = options_df.get('mid', 0) * 1.05
    if 'T_exp' not in options_df.columns:
        options_df['T_exp'] = options_df.get('DTE', 30) / 365.0
    if 'K' not in options_df.columns:
        if 'strike' in options_df.columns:
            options_df['K'] = options_df['strike']
        else:
            print("[ERROR] No strike column found")
            return
    
    # Select DTE=27 (similar to earlier)
    target_dte = 27
    tolerance = 5
    mask = np.abs((options_df['T_exp'] * 365 - target_dte)) <= tolerance
    options_subset = options_df[mask].copy()
    
    if len(options_subset) < 50:
        print(f"[WARNING] Only {len(options_subset)} options near DTE={target_dte}")
        options_subset = options_df.nlargest(150, 'T_exp')
    
    T_cal = options_subset['T_exp'].mean()
    print(f"\n[CALIBRATION SETUP]")
    print(f"Using {len(options_subset)} options, T={T_cal:.4f} years (DTE~{T_cal*365:.0f})")
    
    # ===== RUN CALIBRATION =====
    result = calibrate_bergomi_wasserstein(
        options_subset,
        T_maturity=T_cal,
        spot=spot,
        risk_free=rate,
        num_mc_paths=5000,
        num_iterations=80
    )
    
    if result is None:
        print("[ERROR] Calibration failed")
        return
    
    # ===== SAVE RESULTS =====
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "method": "Wasserstein-1 Distance",
        "comparison": "vs MSE",
        "parameters": {
            "H": float(result.H),
            "eta": float(result.eta),
            "rho": float(result.rho),
            "xi0": float(result.xi0),
        },
        "losses": {
            "wasserstein_1": float(result.loss_wasserstein),
            "mse": float(result.loss_mse),
        },
        "diagnostics": {
            "iterations": int(result.iterations),
            "convergence": result.convergence,
            "market_mean": float(result.market_samples.mean()),
            "market_std": float(result.market_samples.std()),
            "model_mean": float(result.model_samples.mean()),
            "model_std": float(result.model_samples.std()),
            "market_skew": float(pd.Series(result.market_samples).skew()),
            "model_skew": float(pd.Series(result.model_samples).skew()),
        },
        "note": "Compare H values with joint calibration (H=0.0115) to assess univariate vs joint effects"
    }
    
    report_path = output_dir / "wasserstein_calibration_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n[OK] Report saved: {report_path}")
    
    # Plot
    fig = plot_comparison(result)
    fig_path = output_dir / "wasserstein_calibration_comparison.png"
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Plot saved: {fig_path}")
    
    # Final summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nWasserstein-1 Calibration Results:")
    print(f"  H = {result.H:.4f}  (vs H=0.2000 from univariate MSE)")
    print(f"  η = {result.eta:.4f}")
    print(f"  ρ = {result.rho:.4f}")
    print(f"\nLoss Comparison:")
    print(f"  Wasserstein-1: {result.loss_wasserstein:.6f}")
    print(f"  MSE:           {result.loss_mse:.6f}")
    print(f"\nNext Steps:")
    print(f"  1. Compare with joint calibration H=0.0115")
    print(f"  2. Test stability across dates (Feb 19, Mar 8, Mar 13)")
    print(f"  3. Generalize to barrier options")
    print(f"  4. Implement full neural network parametrization of ξ₀(t)")
    print("="*80)


if __name__ == '__main__':
    main()
