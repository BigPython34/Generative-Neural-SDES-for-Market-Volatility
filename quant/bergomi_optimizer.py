"""
Bergomi Parameter Optimizer
===========================
Automatic calibration of Rough Bergomi parameters to market smile.
Uses differential evolution and gradient-based optimization.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.bergomi import RoughBergomiModel


class BlackScholes:
    """BS utilities for IV extraction."""
    
    @staticmethod
    def price(S, K, T, r, sigma, opt_type='call'):
        if T <= 0:
            return max(S - K, 0) if opt_type == 'call' else max(K - S, 0)
        
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if opt_type == 'call':
            return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
        else:
            return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    @staticmethod
    def implied_vol(price, S, K, T, r, opt_type='call'):
        """Newton-Raphson for fast IV extraction."""
        if price <= 0 or T <= 0:
            return np.nan
        
        # Initial guess from approximation
        sigma = np.sqrt(2 * np.pi / T) * price / S
        sigma = np.clip(sigma, 0.05, 2.0)
        
        for _ in range(50):
            bs_price = BlackScholes.price(S, K, T, r, sigma, opt_type)
            
            # Vega
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
            vega = S * np.sqrt(T) * norm.pdf(d1)
            
            if vega < 1e-10:
                break
            
            diff = bs_price - price
            if abs(diff) < 1e-8:
                break
            
            sigma -= diff / vega
            sigma = np.clip(sigma, 0.01, 3.0)
        
        return sigma


class BergomiOptimizer:
    """
    Calibrate Rough Bergomi parameters to market smile.
    
    Parameters to calibrate:
    - H (Hurst): Controls term structure steepness (0.01-0.5)
    - eta: Vol-of-vol amplitude
    - rho: Spot-vol correlation (skew controller)
    - xi0: Forward variance (ATM level)
    """
    
    def __init__(self, spot: float, r: float = 0.05, n_paths: int = 5000):
        self.spot = spot
        self.r = r
        self.n_paths = n_paths
        self.market_data = None
        
    def set_market_data(self, strikes: np.ndarray, market_ivs: np.ndarray,
                       T: float, option_types: list):
        """Set market data for calibration."""
        self.market_data = {
            'strikes': strikes,
            'market_ivs': market_ivs,
            'T': T,
            'option_types': option_types
        }
    
    def price_options_bergomi(self, H: float, eta: float, rho: float, xi0: float) -> np.ndarray:
        """
        Price all options with given Bergomi params.
        Returns model implied volatilities.
        """
        T = self.market_data['T']
        n_steps = max(30, int(T * 252))
        
        params = {
            'hurst': H,
            'eta': eta,
            'rho': rho,
            'xi0': xi0,
            'n_steps': n_steps,
            'T': T
        }
        
        try:
            bergomi = RoughBergomiModel(params)
            spot_paths, _ = bergomi.simulate_spot_vol_paths(self.n_paths, s0=self.spot)
            spot_paths = np.array(spot_paths)
            S_T = spot_paths[:, -1]
        except Exception as e:
            return np.full(len(self.market_data['strikes']), np.nan)
        
        model_ivs = []
        
        for K, opt_type in zip(self.market_data['strikes'], self.market_data['option_types']):
            # Monte Carlo price
            if opt_type == 'call':
                payoff = np.maximum(S_T - K, 0)
            else:
                payoff = np.maximum(K - S_T, 0)
            
            mc_price = np.exp(-self.r * T) * np.mean(payoff)
            
            # Extract IV
            iv = BlackScholes.implied_vol(mc_price, self.spot, K, T, self.r, opt_type)
            model_ivs.append(iv)
        
        return np.array(model_ivs)
    
    def objective(self, params: np.ndarray) -> float:
        """
        Calibration objective: RMSE of implied vols.
        params = [H, eta, rho, xi0]
        """
        H, eta, rho, xi0 = params
        
        # Constraints
        if H <= 0 or H >= 0.5:
            return 1e6
        if eta <= 0 or eta > 10:
            return 1e6
        if rho <= -1 or rho >= 0:
            return 1e6
        if xi0 <= 0 or xi0 > 2:
            return 1e6
        
        model_ivs = self.price_options_bergomi(H, eta, rho, xi0)
        
        valid = ~np.isnan(model_ivs)
        if valid.sum() < 3:
            return 1e6
        
        market_ivs = self.market_data['market_ivs'][valid]
        model_ivs = model_ivs[valid]
        
        # RMSE + small penalty for extreme params
        rmse = np.sqrt(np.mean((model_ivs - market_ivs)**2))
        
        # Regularization (favor rough regime)
        penalty = 0.01 * (H - 0.1)**2 + 0.001 * (eta - 2)**2
        
        return rmse + penalty
    
    def calibrate_grid_search(self) -> dict:
        """
        Fast grid search calibration with refined grid.
        """
        if self.market_data is None:
            raise ValueError("Set market data first with set_market_data()")
        
        # ATM IV for xi0 guess
        strikes = self.market_data['strikes']
        moneyness = np.log(strikes / self.spot)
        atm_idx = np.argmin(np.abs(moneyness))
        atm_iv = self.market_data['market_ivs'][atm_idx]
        
        print(f"Bergomi Grid Search Calibration")
        print(f"   ATM IV: {atm_iv*100:.1f}%")
        
        # Fine grid - more values for H and rho which are most important
        H_grid = [0.03, 0.05, 0.07, 0.10, 0.15, 0.20]
        eta_grid = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
        rho_grid = [-0.9, -0.8, -0.7, -0.6, -0.5]
        
        best_params = None
        best_rmse = float('inf')
        total_combos = len(H_grid) * len(eta_grid) * len(rho_grid)
        
        print(f"   Testing {total_combos} combinations...")
        
        tested = 0
        for H in H_grid:
            for eta in eta_grid:
                for rho in rho_grid:
                    tested += 1
                    
                    model_ivs = self.price_options_bergomi(H, eta, rho, atm_iv**2)
                    valid = ~np.isnan(model_ivs)
                    
                    if valid.sum() >= 3:
                        rmse = np.sqrt(np.mean((model_ivs[valid] - self.market_data['market_ivs'][valid])**2))
                        
                        if rmse < best_rmse:
                            best_rmse = rmse
                            best_params = {'hurst': H, 'eta': eta, 'rho': rho, 'xi0': atm_iv**2}
                            print(f"   [{tested}/{total_combos}] New best: H={H:.2f}, η={eta:.1f}, ρ={rho:.1f} → RMSE={rmse*100:.2f}%")
        
        if best_params is None:
            best_params = {'hurst': 0.1, 'eta': 2.5, 'rho': -0.7, 'xi0': atm_iv**2}
            best_rmse = 0.1
        
        # Final evaluation with more paths
        print(f"\n   Refining with {self.n_paths*2} paths...")
        self.n_paths *= 2
        final_ivs = self.price_options_bergomi(
            best_params['hurst'], best_params['eta'], 
            best_params['rho'], best_params['xi0']
        )
        self.n_paths //= 2
        
        valid = ~np.isnan(final_ivs)
        final_rmse = np.sqrt(np.mean((final_ivs[valid] - self.market_data['market_ivs'][valid])**2)) * 100
        
        calibration_result = {
            **best_params,
            'rmse': final_rmse,
            'model_ivs': final_ivs,
            'success': True
        }
        
        print(f"\nGrid Search Complete!")
        print(f"   H (Hurst)  = {best_params['hurst']:.4f} {'← ROUGH!' if best_params['hurst'] < 0.25 else ''}")
        print(f"   η (vol-vol)= {best_params['eta']:.3f}")
        print(f"   ρ (correl) = {best_params['rho']:.3f}")
        print(f"   ξ₀ (ATM var)= {best_params['xi0']:.4f} (σ={np.sqrt(best_params['xi0'])*100:.1f}%)")
        print(f"   RMSE       = {final_rmse:.2f}%")
        
        return calibration_result
    
    def calibrate(self, method: str = 'grid',
                 initial_guess: dict = None) -> dict:
        """
        Run calibration.
        
        Args:
            method: 'grid' (fast grid search) or 'local' (L-BFGS-B)
            initial_guess: Dict with initial H, eta, rho, xi0
        
        Returns:
            Calibrated parameters and diagnostics
        """
        if method == 'grid':
            return self.calibrate_grid_search()
        
        if self.market_data is None:
            raise ValueError("Set market data first with set_market_data()")
        
        # ATM IV for xi0 initial guess
        strikes = self.market_data['strikes']
        moneyness = np.log(strikes / self.spot)
        atm_idx = np.argmin(np.abs(moneyness))
        atm_iv = self.market_data['market_ivs'][atm_idx]
        
        print(f"Bergomi Calibration")
        print(f"   ATM IV: {atm_iv*100:.1f}%, ATM variance: {atm_iv**2:.4f}")
        
        # Bounds
        bounds = [
            (0.01, 0.45),   # H (Hurst)
            (0.5, 5.0),     # eta
            (-0.99, -0.1),  # rho
            (0.01, 1.0)     # xi0
        ]
        
        # Default initial guess
        if initial_guess is None:
            x0 = [0.07, 2.5, -0.7, atm_iv**2]
        else:
            x0 = [
                initial_guess.get('hurst', 0.07),
                initial_guess.get('eta', 2.5),
                initial_guess.get('rho', -0.7),
                initial_guess.get('xi0', atm_iv**2)
            ]
        
        print(f"   Initial: H={x0[0]:.3f}, η={x0[1]:.2f}, ρ={x0[2]:.2f}, ξ₀={x0[3]:.4f}")
        print("   Running L-BFGS-B optimization (fast)...")
        
        result = minimize(
            self.objective,
            x0=x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 30, 'disp': False}
        )
        
        H_opt, eta_opt, rho_opt, xi0_opt = result.x
        
        # Final evaluation
        final_ivs = self.price_options_bergomi(H_opt, eta_opt, rho_opt, xi0_opt)
        valid = ~np.isnan(final_ivs)
        rmse = np.sqrt(np.mean((final_ivs[valid] - self.market_data['market_ivs'][valid])**2)) * 100
        
        calibration_result = {
            'hurst': H_opt,
            'eta': eta_opt,
            'rho': rho_opt,
            'xi0': xi0_opt,
            'rmse': rmse,
            'model_ivs': final_ivs,
            'success': result.success,
            'n_iterations': result.nit if hasattr(result, 'nit') else result.nfev
        }
        
        print(f"\nCalibration complete!")
        print(f"   H (Hurst)  = {H_opt:.4f} {'← ROUGH!' if H_opt < 0.25 else ''}")
        print(f"   η (vol-vol)= {eta_opt:.3f}")
        print(f"   ρ (correl) = {rho_opt:.3f}")
        print(f"   ξ₀ (ATM var)= {xi0_opt:.4f} (σ={np.sqrt(xi0_opt)*100:.1f}%)")
        print(f"   RMSE       = {rmse:.2f}%")
        
        return calibration_result


def calibrate_bergomi_to_smile(smile_df, spot: float, T: float, n_paths: int = 5000) -> dict:
    """
    Convenience function to calibrate Bergomi to a smile DataFrame.
    
    Args:
        smile_df: DataFrame with 'strike', 'impliedVolatility', 'type' columns
        spot: Current spot price
        T: Time to maturity in years
        n_paths: Number of MC paths (default 5000 for balance of speed/accuracy)
    
    Returns:
        Calibration results dict
    """
    optimizer = BergomiOptimizer(spot=spot, n_paths=n_paths)
    
    optimizer.set_market_data(
        strikes=smile_df['strike'].values,
        market_ivs=smile_df['impliedVolatility'].values,
        T=T,
        option_types=smile_df['type'].tolist()
    )
    
    result = optimizer.calibrate(method='grid')  # Grid search for better coverage
    return result


if __name__ == "__main__":
    # Test with synthetic data
    print("="*60)
    print("   BERGOMI OPTIMIZER TEST")
    print("="*60)
    
    spot = 100
    T = 0.1  # ~36 days
    
    # Generate synthetic smile (with skew)
    strikes = np.linspace(90, 110, 15)
    moneyness = np.log(strikes / spot)
    
    # Synthetic smile with negative skew
    atm_vol = 0.20
    synthetic_ivs = atm_vol - 0.5 * moneyness + 0.8 * moneyness**2
    
    import pandas as pd
    smile = pd.DataFrame({
        'strike': strikes,
        'impliedVolatility': synthetic_ivs,
        'type': ['put' if k < spot else 'call' for k in strikes]
    })
    
    print(f"\nSynthetic smile: ATM={atm_vol*100:.0f}%, range [{synthetic_ivs.min()*100:.1f}%, {synthetic_ivs.max()*100:.1f}%]")
    
    result = calibrate_bergomi_to_smile(smile, spot, T, n_paths=5000)
    
    print(f"\nResults summary:")
    print(f"   Calibrated RMSE: {result['rmse']:.2f}%")
