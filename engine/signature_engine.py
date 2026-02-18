import jax
import jax.numpy as jnp
import numpy as np
import esig

class SignatureFeatureExtractor:
    """
    Dual-Engine Signature Extractor.
    - Uses 'esig' (C++) for static real data (Pre-processing).
    - Uses 'JAX' (Custom implementation) for training (Differentiable).

    Time augmentation uses the REAL time scale (in years) to ensure
    temporal consistency between the signature and the SDE dynamics.
    """
    def __init__(self, truncation_order: int = 3, dt: float = None):
        self.order = truncation_order
        self.dt = dt  # Real time step in years (e.g. 0.000153 for 15-min bars)

    def get_signature(self, paths):
        """
        Smart dispatcher: detects if input is JAX (Tracer) or NumPy.
        """
        if isinstance(paths, (np.ndarray, list)):
            return self._get_signature_numpy(paths)

        return self._get_signature_jax(paths)

    def _get_signature_numpy(self, paths):
        np_paths = np.array(paths)
        if np_paths.ndim == 2:
            n_paths, n_steps = np_paths.shape
            if self.dt is not None:
                time_axis = np.arange(n_steps) * self.dt
            else:
                time_axis = np.linspace(0, 1, n_steps)
            augmented_paths = np.array([np.column_stack([time_axis, p]) for p in np_paths])
        else:
            augmented_paths = np_paths

        func = getattr(esig, 'stream2sig', None) or getattr(esig.tosig, 'stream2sig')
        return np.array([func(p, self.order) for p in augmented_paths])

    def _get_signature_jax(self, paths):
        """
        Pure JAX implementation of Path Signatures (Order 1, 2, 3).
        Differentiable & GPU compatible.
        Assumes paths are 1D (Variance process).
        """
        batch_size, n_steps = paths.shape
        if self.dt is not None:
            time_axis = jnp.arange(n_steps) * self.dt
        else:
            time_axis = jnp.linspace(0, 1, n_steps)
        time_axis = jnp.tile(time_axis, (batch_size, 1))

        path_augmented = jnp.stack([time_axis, paths], axis=-1)

        dX = path_augmented[:, 1:] - path_augmented[:, :-1]

        def scan_step(carry, dx):
            s0, s1, s2, s3 = carry

            new_s1 = s1 + dx
            term2_a = jnp.outer(s1, dx).flatten()
            term2_b = 0.5 * jnp.outer(dx, dx).flatten()
            new_s2 = s2 + term2_a + term2_b
            term3_a = jnp.kron(s2, dx)
            term3_b = jnp.kron(s1, 0.5 * jnp.outer(dx, dx).flatten())
            term3_c = (1/6.0) * jnp.kron(dx, jnp.kron(dx, dx))
            new_s3 = s3 + term3_a + term3_b + term3_c

            return (s0, new_s1, new_s2, new_s3), None

        def compute_single_path_sig(dx_seq):
            d = 2
            s0 = jnp.array([1.0])
            s1 = jnp.zeros(d)
            s2 = jnp.zeros(d**2)
            s3 = jnp.zeros(d**3)

            final_carry, _ = jax.lax.scan(scan_step, (s0, s1, s2, s3), dx_seq)

            if self.order == 2:
                return jnp.concatenate([final_carry[1], final_carry[2]])
            else:
                return jnp.concatenate([final_carry[1], final_carry[2], final_carry[3]])

        sigs = jax.vmap(compute_single_path_sig)(dX)
        return sigs

    def get_feature_dim(self, input_dim: int) -> int:
        d = input_dim + 1
        total = 0
        for k in range(1, self.order + 1):
            total += d**k
        return total
