import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import  List
import json
from datetime import datetime
from scipy.optimize import minimize
from quant.calibration.market_data_vix import assemble_calibration_data

@dataclass
class WassersteinResult:
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
    if len(X) != len(Y):
        m = min(len(X), len(Y))
        X = np.random.choice(X, m, replace=False)
        Y = np.random.choice(Y, m, replace=False)

    X_sorted = np.sort(X)
    Y_sorted = np.sort(Y)
    return np.mean(np.abs(X_sorted - Y_sorted))

def wasserstein_loss_batch(S_model: np.ndarray, S_market: np.ndarray, maturities: List[float]) -> float:
    if len(S_model.shape) == 1:
        return empirical_wasserstein_1(S_model, S_market)
    total_loss = 0.0
    for j, T in enumerate(maturities):
        w1 = empirical_wasserstein_1(S_model[:, j], S_market)
        total_loss += w1
    return total_loss / len(maturities)

def mse_loss(prices_model: np.ndarray, prices_market: np.ndarray) -> float:
    return np.mean((prices_model - prices_market) ** 2)

def calibrate_bergomi_wasserstein(options_data: pd.DataFrame, T_maturity: float, spot: float, risk_free: float, num_mc_paths: int = 5000, num_iterations: int = 100, use_market_prices: bool = True) -> WassersteinResult:
    print("\n" + "="*80)
    print("WASSERSTEIN-1 CALIBRATION FOR rBERGOMI")
    print("="*80)

    print("\n[STEP 1] Extracting market-implied risk-neutral distribution...")
    opts_T = options_data[(options_data['T_exp'] >= T_maturity * 0.9) & (options_data['T_exp'] <= T_maturity * 1.1)].copy()

    if len(opts_T) < 20:
        print(f"  Warning: Only {len(opts_T)} options for maturity {T_maturity:.4f}. Need >= 20.")
        return None

    print(f"  Found {len(opts_T)} options for T={T_maturity:.4f}")
    strikes = opts_T['K'].values
    ivs = opts_T['IV'].values
    bids = opts_T['bid'].values
    asks = opts_T['ask'].values

    S_market_samples = _extract_risk_neutral_samples(strikes, ivs, spot, risk_free, T_maturity, num_samples=2000)

    print(f"  [OK] Extracted {len(S_market_samples)} RN samples")
    print(f"       Mean: , Std: ")
    atm_iv = float(opts_T[np.abs(opts_T['K'] - spot) < 2]['IV'].mean())
    print(f"  ATM IV: {atm_iv*100:.2f}%")

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
                xi0 = atm_iv ** 2
                S_model_paths = _simulate_bergomi(spot, H, eta, rho, xi0, T_maturity, num_paths=2000, num_steps=50)
                loss_w1 = empirical_wasserstein_1(S_model_paths, S_market_samples)
                eval_count += 1

                if loss_w1 < best_loss:
                    best_loss = loss_w1
                    best_params = (H, eta, rho)
                    if eval_count % 15 == 0 or eval_count == 1:
                        print(f"  [{eval_count}/{len(H_grid)*len(eta_grid)*len(rho_grid)}] H={H:.2f}, eta={eta:.1f}, rho={rho:.2f} -> W1={loss_w1:.6f}")

    H_init, eta_init, rho_init = best_params
    print(f"\n  Grid best: H={H_init:.2f}, eta={eta_init:.1f}, rho={rho_init:.2f}, W1={best_loss:.6f}")

    print("\n[STEP 3] Refining with Nelder-Mead (Wasserstein)...")
    def wasserstein_objective(params):
        H, eta, rho = params
        if H <= 0 or H >= 0.5 or eta <= 0.1 or rho <= -0.999 or rho >= -0.1:
            return 1e6
        xi0 = atm_iv ** 2
        try:
            S_model = _simulate_bergomi(spot, H, eta, rho, xi0, T_maturity, num_paths=3000, num_steps=100)
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

    print("\n[STEP 4] Comparing MSE loss on same parameters...")
    xi0_opt = atm_iv ** 2
    S_model_opt = _simulate_bergomi(spot, H_opt, eta_opt, rho_opt, xi0_opt, T_maturity, num_paths=5000, num_steps=100)

    prices_model = _price_options_from_samples(S_model_opt, strikes, spot, risk_free, T_maturity)
    prices_market = np.exp(-risk_free * T_maturity) * np.maximum(S_market_samples.mean() - strikes, 0)
    loss_mse_opt = mse_loss(prices_model, prices_market)

    print(f"  MSE loss (same params): {loss_mse_opt:.6f}")
    print(f"  W_1 loss (same params): {loss_w1_opt:.6f}")

    print("\n[STEP 5] Generating diagnostics...")
    S_grid = _simulate_bergomi(spot, H_init, eta_init, rho_init, atm_iv**2, T_maturity, num_paths=5000, num_steps=100)
    prices_grid = _price_options_from_samples(S_grid, strikes, spot, risk_free, T_maturity)
    loss_mse_grid = mse_loss(prices_grid, prices_market)

    print(f"\n  === Parameter Comparison ===")
    print(f"  Grid best (MSE loss):")
    print(f"    H={H_init:.4f}, eta={eta_init:.4f}, rho={rho_init:.4f}")
    print(f"    MSE={loss_mse_grid:.6f}, W_1={best_loss:.6f}")
    print(f"\n  Refined (W_1 loss):")
    print(f"    H={H_opt:.4f}, eta={eta_opt:.4f}, rho={rho_opt:.4f}")
    print(f"    MSE={loss_mse_opt:.6f}, W_1={loss_w1_opt:.6f}")

    print(f"\n  === Distribution Metrics ===")
    print(f"  Market samples: mean=, std=")
    print(f"  Model samples:  mean=, std=")
    print(f"  Skewness - Market: {pd.Series(S_market_samples).skew():.4f}")
    print(f"  Skewness - Model:  {pd.Series(S_model_opt).skew():.4f}")

    return WassersteinResult(
        H=H_opt, eta=eta_opt, rho=rho_opt, xi0=xi0_opt,
        loss_mse=loss_mse_opt, loss_wasserstein=loss_w1_opt,
        iterations=result_w1.nit,
        convergence="Success" if result_w1.success else "Incomplete",
        market_samples=S_market_samples, model_samples=S_model_opt,
    )

def _extract_risk_neutral_samples(strikes, ivs, spot, rate, T, num_samples=2000):
    from scipy.interpolate import CubicSpline
    from quant.models.black_scholes import BlackScholes

    # 1. Prix calls sur grille fine
    K_grid = np.linspace(strikes.min() * 0.85, strikes.max() * 1.15, 300)
    call_prices = np.array([
        BlackScholes.price(spot, K, T, rate, iv, 'call')
        for K, iv in zip(strikes, ivs)
    ])

    # 2. Spline monotone sur les prix
    cs = CubicSpline(strikes, call_prices, bc_type='natural')

    # 3. Densité RN = e^{rT} · C''(K)  (Breeden-Litzenberger)
    d2C = cs(K_grid, 2)  # dérivée seconde
    density = np.exp(rate * T) * np.maximum(d2C, 0)

    if density.sum() < 1e-12:
        # Fallback log-normal avec IV ATM
        atm_iv = ivs[np.argmin(np.abs(strikes - spot))]
        return spot * np.exp(
            np.random.normal(-0.5 * atm_iv**2 * T, atm_iv * np.sqrt(T), num_samples)
        )

    # 4. Normaliser et échantillonner
    density /= density.sum()
    return np.random.choice(K_grid, size=num_samples, p=density, replace=True)

def _simulate_bergomi(S0: float, H: float, eta: float, rho: float, xi0: float, T: float, num_paths: int, num_steps: int) -> np.ndarray:
    dt = T / num_steps
    paths = np.zeros((num_paths, num_steps + 1))
    paths[:, 0] = S0

    N = 4
    lambda_k = np.array([0.047, 1.228, 0.619, 6.203])
    omega_k = np.array([0.399, 1.483, 5.516, 20.522])
    Y_k = np.zeros((num_paths, N))

    for step in range(num_steps):
        t = step * dt
        dW = np.random.normal(0, np.sqrt(dt), num_paths)
        I_local = np.random.normal(0, np.sqrt(dt ** (2 * H)), num_paths)
        dW_k = np.random.normal(0, np.sqrt(dt), (num_paths, N))
        I_hist = 0

        for k in range(N):
            Y_k[:, k] = Y_k[:, k] * np.exp(-lambda_k[k] * dt) + dW_k[:, k]
            I_hist += omega_k[k] * Y_k[:, k] * (1 - np.exp(-lambda_k[k] * dt)) / lambda_k[k]

        I_t = I_local + 2 * H * I_hist
        V_t = xi0 * np.exp(eta * I_t - 0.5 * eta ** 2 * (t + dt) ** (2 * H))
        V_t = np.clip(V_t, 0.01, 10)
        dW_perp = np.random.normal(0, np.sqrt(dt), num_paths)
        dS = paths[:, step] * ((0.03) * dt + np.sqrt(V_t) * (rho * dW + np.sqrt(1 - rho ** 2) * dW_perp))

        paths[:, step + 1] = paths[:, step] + dS
        paths[:, step + 1] = np.maximum(paths[:, step + 1], S0 * 0.5)
    return paths[:, -1]

def _price_options_from_samples(S_samples: np.ndarray, strikes: np.ndarray, S0: float, rate: float, T: float) -> np.ndarray:
    prices = np.zeros(len(strikes))
    discount = np.exp(-rate * T)
    for i, K in enumerate(strikes):
        if K > S0:
            payoff = np.maximum(S_samples - K, 0)
        else:
            payoff = np.maximum(K - S_samples, 0)
        prices[i] = discount * np.mean(payoff)
    return prices

def _callback_minimize(xk, eval_count):
    if eval_count % 20 == 0:
        H, eta, rho = xk
        print(f"    Iteration {eval_count}: H={H:.4f}, eta={eta:.4f}, rho={rho:.4f}")

def plot_comparison(result_wass: WassersteinResult, title_suffix: str = ""):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax = axes[0, 0]
    ax.hist(result_wass.market_samples, bins=50, alpha=0.5, label='Market (RN)', density=True)
    ax.hist(result_wass.model_samples, bins=50, alpha=0.5, label='Model', density=True)
    ax.set_xlabel('Terminal Price ($)')
    ax.set_ylabel('Density')
    ax.set_title('Risk-Neutral Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)

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

    ax = axes[1, 0]
    q_market = np.percentile(result_wass.market_samples, np.linspace(0, 100, 100))
    q_model = np.percentile(result_wass.model_samples, np.linspace(0, 100, 100))
    ax.scatter(q_market, q_model, alpha=0.6)
    ax.plot([q_market.min(), q_market.max()], [q_market.min(), q_market.max()], 'r--', lw=2, label='45-degree line')
    ax.set_xlabel('Market Quantiles ($)')
    ax.set_ylabel('Model Quantiles ($)')
    ax.set_title('Q-Q Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.axis('off')
    info_text = f'''
    Wasserstein-1 Calibration Results

    Optimized Parameters:
    - H (Hurst exponent) = {result_wass.H:.4f}
    - eta (vol-of-vol) = {result_wass.eta:.4f}
    - rho (correlation) = {result_wass.rho:.4f}
    - xi0 (forward var) = {result_wass.xi0:.6f}

    Loss Metrics:
    - W_1 distance = {result_wass.loss_wasserstein:.6f}
    - MSE loss = {result_wass.loss_mse:.6f}
    - Iterations = {result_wass.iterations}
    - Convergence = {result_wass.convergence}

    Distribution Statistics:
    - Market mean = 
    - Model mean = 
    - Market skew = {pd.Series(result_wass.market_samples).skew():.4f}
    - Model skew = {pd.Series(result_wass.model_samples).skew():.4f}'''
    ax.text(0.1, 0.5, info_text, fontsize=10, family='monospace', verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    return fig

def run_wasserstein_calibration_pipeline():
    print("\n" + "="*80)
    print("WASSERSTEIN-1 vs MSE CALIBRATION COMPARISON")
    print("Rough Bergomi Model on SPY Options")
    print("="*80)

    print("\n[LOADING DATA]")
    try:
        market_data = assemble_calibration_data()
        print(f"[OK] Loaded {len(market_data.spx_surface)} SPX options")
    except Exception as e:
        raise ValueError(f"Failed to load market data: {e}")

    options_df = market_data.spx_surface
    spot = market_data.spx_spot
    if spot is None:
        raise ValueError("Missing SPY spot in assembled market data.")

    rate = getattr(market_data, 'risk_free_rate', 0.0364)
    print(f"Spot: ")
    print(f"Risk-free rate: {rate*100:.3f}%")

    vix_futures_path = Path('data/market/cboe_vix_futures_full/vix_futures_all.csv')
    if not vix_futures_path.exists():
        raise FileNotFoundError(f"VIX futures file not found: {vix_futures_path}")

    if 'IV' not in options_df.columns:
        if 'impliedVolatility' in options_df.columns:
            options_df = options_df.copy()
            options_df['IV'] = options_df['impliedVolatility']
        else:
            raise KeyError("Missing implied volatility column: expected 'IV' or 'impliedVolatility'.")

    if 'T_exp' not in options_df.columns:
        if 'T' in options_df.columns:
            options_df = options_df.copy()
            options_df['T_exp'] = options_df['T']
        elif 'dte' in options_df.columns:
            options_df = options_df.copy()
            options_df['T_exp'] = options_df['dte'] / 365.0
        else:
            raise KeyError("Missing maturity column: expected 'T', 'dte', or 'T_exp'.")

    if 'K' not in options_df.columns:
        if 'strike' in options_df.columns:
            options_df['K'] = options_df['strike']
        else:
            raise KeyError("Missing strike column: expected 'K' or 'strike'.")

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

    result = calibrate_bergomi_wasserstein(
        options_subset, T_maturity=T_cal, spot=spot, risk_free=rate,
        num_mc_paths=5000, num_iterations=80
    )

    if result is None:
        print("[ERROR] Calibration failed")
        return

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    report = {
        "timestamp": datetime.now().isoformat(),
        "method": "Wasserstein-1 Distance",
        "comparison": "vs MSE",
        "parameters": {
            "H": float(result.H), "eta": float(result.eta),
            "rho": float(result.rho), "xi0": float(result.xi0),
        },
        "losses": {
            "wasserstein_1": float(result.loss_wasserstein), "mse": float(result.loss_mse),
        },
        "diagnostics": {
            "iterations": int(result.iterations), "convergence": result.convergence,
            "market_mean": float(result.market_samples.mean()), "market_std": float(result.market_samples.std()),
            "model_mean": float(result.model_samples.mean()), "model_std": float(result.model_samples.std()),
            "market_skew": float(pd.Series(result.market_samples).skew()), "model_skew": float(pd.Series(result.model_samples).skew()),
        },
        "note": "Compare H values with joint calibration (H=0.0115) to assess univariate vs joint effects"
    }

    report_path = output_dir / "wasserstein_calibration_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n[OK] Report saved: {report_path}")

    fig = plot_comparison(result)
    fig_path = output_dir / "wasserstein_calibration_comparison.png"
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Plot saved: {fig_path}")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nWasserstein-1 Calibration Results:")
    print(f"  H = {result.H:.4f}  (vs H=0.2000 from univariate MSE)")
    print(f"  eta = {result.eta:.4f}")
    print(f"  rho = {result.rho:.4f}")
    print(f"\nLoss Comparison:")
    print(f"  Wasserstein-1: {result.loss_wasserstein:.6f}")
    print(f"  MSE:           {result.loss_mse:.6f}")
    print(f"\nNext Steps:")
    print(f"  1. Compare with joint calibration H=0.0115")
    print(f"  2. Test stability across dates (Feb 19, Mar 8, Mar 13)")
    print(f"  3. Generalize to barrier options")
    print(f"  4. Implement full neural network parametrization of xi0(t)")
    print("="*80)
