import jax
import jax.numpy as jnp
import numpy as np
from engine.generative_trainer import GenerativeTrainer
from utils.config import load_config

class DeepPricingEngine:
    """
    Monte Carlo Pricing Engine.
    Compares: Neural SDE vs Rough Bergomi vs Black-Scholes.
    """
    def __init__(self, trainer: GenerativeTrainer, model):
        self.trainer = trainer
        self.model = model
        self.config = trainer.config
        self._pricing_cfg = load_config()['pricing']

    def generate_market_paths(self, n_paths, s0=None, mu=None, rho=None):
        """
        Generates Spot-Vol paths using the trained Neural SDE.
        Implements leverage effect (rho) between the generated vol and spot noise.
        """
        if s0 is None:
            s0 = self._pricing_cfg['spot']
        if mu is None:
            mu = self._pricing_cfg['risk_free_rate']
        if rho is None:
            rho = load_config()['bergomi']['rho']
        key = jax.random.PRNGKey(42)
        key_vol, key_spot = jax.random.split(key)
        
        # 1. Generate Neural Volatility
        dt = self.config['T'] / self.config['n_steps']
        noise_vol = jax.random.normal(key_vol, (n_paths, self.config['n_steps'])) * jnp.sqrt(dt)
        
        # Sample initial variance from empirical distribution
        real_v0s = self.trainer.market_paths[:, 0]
        random_indices = jax.random.randint(key, (n_paths,), 0, len(real_v0s))
        v0_samples = real_v0s[random_indices]
        
        var_paths = jax.vmap(self.model.generate_variance_path, in_axes=(0, 0, None))(
            v0_samples, noise_vol, dt
        )
        
        # 2. Generate Spot with Leverage
        z_spot_indep = jax.random.normal(key_spot, (n_paths, self.config['n_steps']))
        
        # Correlate spot noise with the volatility driver
        # noise_vol is already dW_vol = sqrt(dt)*Z, so spot_driver has correct scaling
        spot_driver = rho * noise_vol + jnp.sqrt(1 - rho**2) * z_spot_indep * jnp.sqrt(dt)
        
        # Use *previsible* variance V_{k-1} at each step to avoid
        # adaptedness bias (V_k depends on same noise as spot driver).
        # Per-path v0 preserves the heterogeneity of initial conditions.
        v0_col = v0_samples.reshape(-1, 1)
        var_prev = jnp.concatenate([v0_col, var_paths[:, :-1]], axis=1)

        vol_paths = jnp.sqrt(var_prev)
        # spot_driver is already dW (scaled by sqrt(dt)), so no extra sqrt(dt) here
        log_ret = (mu - 0.5 * var_prev) * dt + vol_paths * spot_driver
        log_s = jnp.cumsum(log_ret, axis=1)
        s_paths = s0 * jnp.exp(log_s)
        
        s0_col = jnp.full((n_paths, 1), s0)
        return np.hstack([s0_col, s_paths]), np.array(var_paths)

    def generate_bs_paths(self, n_paths, s0, vol, mu=None):
        """Generates standard Geometric Brownian Motion paths (Black-Scholes)."""
        if mu is None:
            mu = self._pricing_cfg['risk_free_rate']
        dt = self.config['T'] / self.config['n_steps']
        key = jax.random.PRNGKey(123)
        z = jax.random.normal(key, (n_paths, self.config['n_steps']))
        z = np.array(z)
        
        drift = (mu - 0.5 * vol**2) * dt
        diffusion = vol * np.sqrt(dt) * z
        
        log_ret = drift + diffusion
        log_s = np.cumsum(log_ret, axis=1)
        s_paths = s0 * np.exp(log_s)
        
        s0_col = np.full((n_paths, 1), s0)
        return np.hstack([s0_col, s_paths])

    def generate_bergomi_paths(self, bergomi_model, n_paths, s0=None):
        """Wrapper to extract numpy arrays from JAX-based Bergomi model."""
        if s0 is None:
            s0 = self._pricing_cfg['spot']
        st, vt = bergomi_model.simulate_spot_vol_paths(n_paths, s0=s0)
        return np.array(st), np.array(vt)

    def price_down_and_out_call(self, s_paths, strike, barrier, r=None, T=None):
        """
        Prices a Barrier Option (Down-and-Out Call).
        Payoff = max(S_T - K, 0) if min(S_t) > Barrier, else 0.
        Sensitive to 'Fat Tails' (Crash Risk).
        
        Args:
            T: Maturity. If None, uses the model's horizon for consistency.
        """
        if r is None:
            r = self._pricing_cfg['risk_free_rate']
        if T is None:
            T = self.config['T']  # Use consistent horizon
        min_s = np.min(s_paths, axis=1)
        alive = (min_s > barrier).astype(float)
        
        s_T = s_paths[:, -1]
        payoff = np.maximum(s_T - strike, 0)
        
        # Discounted expected payoff
        return np.mean(payoff * alive) * np.exp(-r * T)