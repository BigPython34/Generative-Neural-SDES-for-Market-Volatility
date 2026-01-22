import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial

class JAXFractionalBrownianMotion:
    """
    Exact simulation of Fractional Brownian Motion using the Davies-Harte algorithm.
    Optimized for JAX with XLA compilation.
    """
    def __init__(self, n_steps: int, T: float, hurst: float):
        self.n = n_steps
        self.T = T
        self.h = hurst
        self.dt = T / n_steps

    @partial(jit, static_argnums=(0, 2))
    def generate_paths(self, key, n_paths: int):
        """Vectorized path generation using JAX vmap."""
        keys = jax.random.split(key, n_paths)
        return vmap(self._single_path)(keys)

    def _single_path(self, key):
        """Circulant matrix embedding for O(N log N) simulation."""
        k = jnp.arange(self.n)
        # Autocovariance function for fBm
        gamma = 0.5 * (jnp.abs(k-1)**(2*self.h) - 2*jnp.abs(k)**(2*self.h) + jnp.abs(k+1)**(2*self.h))
        c = jnp.concatenate([gamma, jnp.zeros(1), gamma[:0:-1]])
        
        # FFT of the first row of the circulant matrix
        w = jnp.fft.fft(c).real
        w = jnp.maximum(w, 0) # Ensure positivity for numerical stability
        
        z = jax.random.normal(key, (2 * self.n,))
        # Map Gaussian noise to the circulant spectrum
        f = jnp.fft.ifft(jnp.sqrt(w) * z).real[:self.n]
        return jnp.cumsum(f * (self.n**(-self.h))) * (self.T**self.h)