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
        """
        Circulant matrix embedding for O(N log N) fBM simulation.
        
        Correct Davies-Harte algorithm (Dieker 2004):
          1. Eigenvalues lambda = FFT(first row of 2N circulant)
          2. V_k = sqrt(lambda_k / M) * (N1 + i*N2)   [complex Gaussian]
          3. Z = M * IFFT(V)   [undo the 1/M in IFFT]
          4. fGn samples X = Re(Z[0:N])
          5. fBM path = cumsum(X) * dt^H
        
        This ensures Var(B^H(t)) = t^{2H} exactly.
        """
        k = jnp.arange(self.n)
        # Autocovariance function for fGn
        gamma = 0.5 * (jnp.abs(k-1)**(2*self.h) - 2*jnp.abs(k)**(2*self.h) + jnp.abs(k+1)**(2*self.h))
        c = jnp.concatenate([gamma, jnp.zeros(1), gamma[:0:-1]])
        M = 2 * self.n
        
        # Eigenvalues of circulant matrix
        lam = jnp.fft.fft(c).real
        lam = jnp.maximum(lam, 0)
        
        # Complex Gaussian noise
        key1, key2 = jax.random.split(key)
        n1 = jax.random.normal(key1, (M,))
        n2 = jax.random.normal(key2, (M,))
        
        # Scale and transform
        v = jnp.sqrt(lam / M) * (n1 + 1j * n2)
        z = (jnp.fft.ifft(v) * M).real[:self.n]   # fGn samples
        
        # Cumulative sum → fBM, scaled by dt^H
        return jnp.cumsum(z) * (self.dt ** self.h)