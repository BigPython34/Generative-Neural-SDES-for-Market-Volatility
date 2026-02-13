import jax
import jax.numpy as jnp
from core.stochastic_process import JAXFractionalBrownianMotion
from functools import partial

class RoughBergomiModel:
    """
    Implements the Rough Bergomi (rBergomi) stochastic volatility model.
    Dynamics:
      V_t = xi_0 * exp(eta * W^H_t - 0.5 * eta^2 * t^(2H))
      dS_t / S_t = (mu - 0.5*V_t)dt + sqrt(V_t) * dW_S
      dW_S . dW^H = rho * dt
    """
    def __init__(self, params: dict):
        self.h = params['hurst']
        self.eta = params['eta']      # Vol-of-vol
        self.rho = params['rho']      # Spot-Vol correlation (Leverage)
        self.xi0 = params['xi0']      # Initial forward variance curve (flat)
        self.n_steps = params['n_steps']
        self.T = params['T']
        self.dt = self.T / self.n_steps
        self.fbm_gen = JAXFractionalBrownianMotion(self.n_steps, self.T, self.h)

    def simulate_variance_paths(self, n_paths: int, key=None):
        """Generates variance paths only (for statistical comparison)."""
        if key is None: key = jax.random.PRNGKey(42)
        wh = self.fbm_gen.generate_paths(key, n_paths)
        return self._compute_variance(wh)

    def simulate_spot_vol_paths(self, n_paths: int, s0=100.0, mu=0.05, key=None):
        """
        Generates coupled Spot (S_t) and Variance (V_t) paths for pricing.
        """
        if key is None: key = jax.random.PRNGKey(42)
        key_fbm, key_spot = jax.random.split(key)

        # 1. Generate Variance Process (Driven by Fractional Brownian Motion Wh)
        wh = self.fbm_gen.generate_paths(key_fbm, n_paths)
        vt = self._compute_variance(wh)

        # 2. Construct Correlated Noise for Spot
        # We correlate the spot noise Z_s with the increments of Wh
        dwh = jnp.diff(wh, axis=1, prepend=0)
        
        # Normalize dWh to act as a standard Gaussian driver for correlation
        scale = jnp.std(dwh) + 1e-8
        dwh_norm = dwh / scale

        # Z_independent is the idiosyncratic component of the spot noise
        z_indep = jax.random.normal(key_spot, (n_paths, self.n_steps))
        
        # dZ_correlated = rho * dWh + sqrt(1 - rho^2) * dZ_indep
        dz_correlated = self.rho * dwh_norm * jnp.sqrt(self.dt) + jnp.sqrt(1 - self.rho**2) * z_indep * jnp.sqrt(self.dt)

        # 3. Integrate Spot Process (Euler-Maruyama)
        vol_path = jnp.sqrt(vt)
        
        # Geometric Brownian Motion with Stochastic Volatility
        # Log-Euler scheme for stability: d(log S) = (mu - 0.5*V)dt + sqrt(V)dW
        drift_term = (mu - 0.5 * vt) * self.dt
        diff_term = vol_path * dz_correlated
        
        log_s = jnp.cumsum(drift_term + diff_term, axis=1)
        st = s0 * jnp.exp(log_s)
        
        # Prepend S0 at t=0
        s0_col = jnp.full((n_paths, 1), s0)
        st = jnp.hstack([s0_col, st])
        
        return st, vt

    @partial(jax.jit, static_argnums=(0,))
    def _compute_variance(self, wh):
        """Computes V_t ensuring the exponential martingale property."""
        time_grid = jnp.linspace(0, self.T, self.n_steps)
        # Drift correction term ensures E[V_t] = xi0
        drift_correction = 0.5 * (self.eta**2) * (time_grid**(2*self.h))
        vt = self.xi0 * jnp.exp(self.eta * wh - drift_correction)
        return vt
