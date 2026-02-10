import jax
import jax.numpy as jnp
import numpy as np
from ml.generative_trainer import GenerativeTrainer

class DeepPricingEngine:
    """
    Monte Carlo Pricing Engine.
    Compares: Neural SDE vs Rough Bergomi vs Black-Scholes.
    """
    def __init__(self, trainer: GenerativeTrainer, model):
        self.trainer = trainer
        self.model = model
        self.config = trainer.config

    def generate_market_paths(self, n_paths, s0=100.0, mu=0.05, rho=-0.7):
        """
        Generates Spot-Vol paths using the trained Neural SDE.
        Implements leverage effect (rho) between the generated vol and spot noise.
        """
        key = jax.random.PRNGKey(42)
        key_vol, key_spot = jax.random.split(key)
        
        # 1. Generate Neural Volatility
        noise_vol = jax.random.normal(key_vol, (n_paths, self.config['n_steps']))
        noise_sigs = self.trainer.sig_extractor.get_signature(noise_vol)
        
        # Sample initial variance from empirical distribution
        real_v0s = self.trainer.market_paths[:, 0]
        random_indices = jax.random.randint(key, (n_paths,), 0, len(real_v0s))
        v0_samples = real_v0s[random_indices]
        
        dt = self.config['T'] / self.config['n_steps']
        
        var_paths = jax.vmap(self.model.generate_variance_path, in_axes=(0, 0, 0, None))(
            v0_samples, noise_sigs, noise_vol, dt
        )
        
        # 2. Generate Spot with Leverage
        z_spot_indep = jax.random.normal(key_spot, (n_paths, self.config['n_steps']))
        
        # Correlate spot noise with the volatility driver
        # noise_vol represents the Brownian increments dW_vol (up to scaling)
        spot_driver = rho * noise_vol + jnp.sqrt(1 - rho**2) * z_spot_indep
        
        vol_paths = jnp.sqrt(var_paths)
        log_ret = (mu - 0.5 * var_paths) * dt + vol_paths * jnp.sqrt(dt) * spot_driver
        log_s = jnp.cumsum(log_ret, axis=1)
        s_paths = s0 * jnp.exp(log_s)
        
        s0_col = jnp.full((n_paths, 1), s0)
        return np.hstack([s0_col, s_paths]), np.array(var_paths)

    def generate_bs_paths(self, n_paths, s0, vol, mu=0.05):
        """Generates standard Geometric Brownian Motion paths (Black-Scholes)."""
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

    def generate_bergomi_paths(self, bergomi_model, n_paths, s0=100.0):
        """Wrapper to extract numpy arrays from JAX-based Bergomi model."""
        st, vt = bergomi_model.simulate_spot_vol_paths(n_paths, s0=s0)
        return np.array(st), np.array(vt)

    def price_down_and_out_call(self, s_paths, strike, barrier, r=0.05, T=None):
        """
        Prices a Barrier Option (Down-and-Out Call).
        Payoff = max(S_T - K, 0) if min(S_t) > Barrier, else 0.
        Sensitive to 'Fat Tails' (Crash Risk).
        
        Args:
            T: Maturity. If None, uses the model's horizon for consistency.
        """
        if T is None:
            T = self.config['T']  # Use consistent horizon
        min_s = np.min(s_paths, axis=1)
        alive = (min_s > barrier).astype(float)
        
        s_T = s_paths[:, -1]
        payoff = np.maximum(s_T - strike, 0)
        
        # Discounted expected payoff
        return np.mean(payoff * alive) * np.exp(-r * T)