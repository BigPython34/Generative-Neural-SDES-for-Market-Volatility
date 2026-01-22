import equinox as eqx
import jax.nn as jnn
import jax.random as jrandom
import jax.numpy as jnp

class SignatureCompensator(eqx.Module):
    """
    Neural network that predicts the drift correction term based on Path Signatures.
    """
    mlp: eqx.nn.MLP

    def __init__(self, input_dim: int, key: jrandom.PRNGKey):
        # input_dim will be the size of the truncated signature
        self.mlp = eqx.nn.MLP(
            in_size=input_dim,
            out_size=1, # Outputs a single scalar: the drift correction mu_t
            width_size=32,
            depth=2,
            activation=jnn.relu,
            key=key
        )

    def __call__(self, sig_vector: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass to get the drift compensation.
        """
        return self.mlp(sig_vector)