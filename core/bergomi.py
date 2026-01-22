import jax
import jax.numpy as jnp
from core.stochastic_process import JAXFractionalBrownianMotion
from functools import partial

class RoughBergomiModel:
    """
    Non-Markovian Stochastic Volatility model with rough kernel.
    Simulates price paths using a hybrid discretization scheme.
    """
    def __init__(self, params: dict):
        self.h = params['hurst']
        self.eta = params['eta']      # Vol-of-vol
        self.rho = params['rho']      # Leverage effect
        self.xi0 = params['xi0']      # Initial forward variance
        self.n_steps = params['n_steps']
        self.T = params['T']
        self.dt = self.T / self.n_steps
        self.fbm_gen = JAXFractionalBrownianMotion(self.n_steps, self.T, self.h)

    def simulate_paths(self, n_paths: int, key=None):
        """Simulates price paths using fully vectorized JAX logic."""
        if key is None:
            key = jax.random.PRNGKey(42)
        
        k1, k2 = jax.random.split(key)
        wh = self.fbm_gen.generate_paths(k1, n_paths) # Rough driver
        z = jax.random.normal(k2, (n_paths, self.n_steps)) # Spot driver
        
        return self._compute_paths_jit(wh, z)
    
    def simulate_variance_paths(self, n_paths: int, key=None):
        """Helper to return only the Variance process V_t."""
        if key is None: key = jax.random.PRNGKey(1)
        wh = self.fbm_gen.generate_paths(key, n_paths)
        time_grid = jnp.linspace(0, self.T, self.n_steps)
        # Formula: V_t = xi0 * exp(eta * Wh - 0.5 * eta^2 * t^2H)
        vt = self.xi0 * jnp.exp(self.eta * wh - 0.5 * (self.eta**2) * (time_grid**(2*self.h)))
        return vt
    
    @partial(jax.jit, static_argnums=(0,))
    def _compute_paths_jit(self, wh, z):
        time_grid = jnp.linspace(0, self.T, self.n_steps)
        # Variance process V_t with rough kernel compensation
        vt = self.xi0 * jnp.exp(self.eta * wh - 0.5 * (self.eta**2) * (time_grid**(2*self.h)))
        
        # Correlate spot and volatility drivers
        dw_h = jnp.diff(wh, axis=1, prepend=0)
        dw_spot = self.rho * dw_h + jnp.sqrt(1 - self.rho**2) * z * jnp.sqrt(self.dt)
        
        # Price dynamics: S_t = S_0 * exp(int(-0.5*V dt + sqrt(V)*dW))
        log_increment = -0.5 * vt * self.dt + jnp.sqrt(vt) * dw_spot
        return jnp.exp(jnp.cumsum(log_increment, axis=1))