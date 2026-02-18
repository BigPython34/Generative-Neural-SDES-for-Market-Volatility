"""
Monte Carlo Option Pricer
=========================
Prices European options from simulated spot paths using Monte Carlo.
"""

import numpy as np
from utils.black_scholes import BlackScholes


class MonteCarloOptionPricer:
    """Monte Carlo pricing for options using stochastic vol models."""
    
    def __init__(self, spot: float, r: float = None):
        self.spot = spot
        if r is None:
            from utils.config import load_config
            r = load_config()['pricing']['risk_free_rate']
        self.r = r
        
    def price_european(self, spot_paths: np.ndarray, strike: float, T: float, 
                       option_type: str = 'call') -> float:
        """
        Price European option from spot paths.
        
        Args:
            spot_paths: (n_paths, n_steps+1) array of spot prices
            strike: Option strike
            T: Time to maturity
            option_type: 'call' or 'put'
        """
        S_T = spot_paths[:, -1]
        
        if option_type == 'call':
            payoff = np.maximum(S_T - strike, 0)
        else:
            payoff = np.maximum(strike - S_T, 0)
        
        return np.exp(-self.r * T) * np.mean(payoff)
    
    def compute_model_smile(self, spot_paths: np.ndarray, strikes: np.ndarray,
                           T: float, option_types: list) -> np.ndarray:
        """Compute implied vols for a set of strikes."""
        ivs = []
        
        for K, opt_type in zip(strikes, option_types):
            price = self.price_european(spot_paths, K, T, opt_type)
            iv = BlackScholes.implied_vol(price, self.spot, K, T, self.r, opt_type)
            ivs.append(iv)
            
        return np.array(ivs)
