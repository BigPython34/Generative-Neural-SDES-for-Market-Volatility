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

    The JAX engine implements the EXACT Chen's identity for signature updates
    up to order 4, ensuring mathematical correctness.

    Config: neural_sde.sig_truncation_order (default 3, supports 2, 3, 4)
    """
    def __init__(self, truncation_order: int = 3, dt: float = None):
        self.order = truncation_order
        self.dt = dt  # Real time step in years (e.g. 0.000153 for 15-min bars)
        if self.order > 4:
            raise ValueError(f"Truncation order {self.order} not supported (max 4). "
                             f"Use esig for higher orders.")

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
        Pure JAX implementation of Path Signatures with exact Chen's identity.
        Differentiable & GPU compatible.
        Supports orders 2, 3, and 4.
        """
        batch_size, n_steps = paths.shape
        if self.dt is not None:
            time_axis = jnp.arange(n_steps) * self.dt
        else:
            time_axis = jnp.linspace(0, 1, n_steps)
        time_axis = jnp.tile(time_axis, (batch_size, 1))

        path_augmented = jnp.stack([time_axis, paths], axis=-1)

        dX = path_augmented[:, 1:] - path_augmented[:, :-1]

        if self.order == 2:
            scan_step = self._make_scan_step_order2()
        elif self.order == 3:
            scan_step = self._make_scan_step_order3()
        else:  # order 4
            scan_step = self._make_scan_step_order4()

        def compute_single_path_sig(dx_seq):
            d = 2
            init_state = self._make_init_state(d)
            final_carry, _ = jax.lax.scan(scan_step, init_state, dx_seq)
            return self._extract_signature(final_carry)

        sigs = jax.vmap(compute_single_path_sig)(dX)
        return sigs

    def _make_init_state(self, d: int):
        """Create initial signature state (all zeros except s0=1)."""
        s0 = jnp.array([1.0])
        s1 = jnp.zeros(d)
        s2 = jnp.zeros(d ** 2)
        if self.order >= 3:
            s3 = jnp.zeros(d ** 3)
        else:
            s3 = jnp.zeros(0)
        if self.order >= 4:
            s4 = jnp.zeros(d ** 4)
        else:
            s4 = jnp.zeros(0)
        return (s0, s1, s2, s3, s4)

    def _extract_signature(self, carry):
        """Extract signature vector from carry state."""
        _, s1, s2, s3, s4 = carry
        if self.order == 2:
            return jnp.concatenate([s1, s2])
        elif self.order == 3:
            return jnp.concatenate([s1, s2, s3])
        else:  # order 4
            return jnp.concatenate([s1, s2, s3, s4])

    @staticmethod
    def _make_scan_step_order2():
        """Chen's identity, exact, order 2."""
        def scan_step(carry, dx):
            s0, s1, s2, s3, s4 = carry
            # Level 1: S^1 += dx
            new_s1 = s1 + dx
            # Level 2: S^2 += S^1 ⊗ dx + ½ dx ⊗ dx  (exact Chen)
            new_s2 = s2 + jnp.outer(s1, dx).flatten() + 0.5 * jnp.outer(dx, dx).flatten()
            return (s0, new_s1, new_s2, s3, s4), None
        return scan_step

    @staticmethod
    def _make_scan_step_order3():
        """
        Chen's identity, exact, order 3.

        For a path X_t with increment dx = X_{t+1} - X_t:
          S^1 += dx
          S^2 += S^1 ⊗ dx + ½ dx ⊗ dx
          S^3 += S^2 ⊗ dx + S^1 ⊗ (½ dx⊗dx) + (1/6) dx⊗dx⊗dx

        This is mathematically exact (Chen 1957).
        """
        def scan_step(carry, dx):
            s0, s1, s2, s3, s4 = carry
            dx_outer = jnp.outer(dx, dx).flatten()

            new_s1 = s1 + dx
            new_s2 = s2 + jnp.outer(s1, dx).flatten() + 0.5 * dx_outer
            new_s3 = (s3
                      + jnp.kron(s2, dx)
                      + jnp.kron(s1, 0.5 * dx_outer)
                      + (1.0 / 6.0) * jnp.kron(dx, jnp.kron(dx, dx)))
            return (s0, new_s1, new_s2, new_s3, s4), None
        return scan_step

    @staticmethod
    def _make_scan_step_order4():
        """
        Chen's identity, exact, order 4.

        S^4 += S^3 ⊗ dx + S^2 ⊗ (½dx⊗dx) + S^1 ⊗ (1/6 dx⊗dx⊗dx) + (1/24) dx⊗⁴
        """
        def scan_step(carry, dx):
            s0, s1, s2, s3, s4 = carry
            dx_outer = jnp.outer(dx, dx).flatten()
            dx3 = jnp.kron(dx, jnp.kron(dx, dx))

            new_s1 = s1 + dx
            new_s2 = s2 + jnp.outer(s1, dx).flatten() + 0.5 * dx_outer
            new_s3 = (s3
                      + jnp.kron(s2, dx)
                      + jnp.kron(s1, 0.5 * dx_outer)
                      + (1.0 / 6.0) * dx3)
            new_s4 = (s4
                      + jnp.kron(s3, dx)
                      + jnp.kron(s2, 0.5 * dx_outer)
                      + jnp.kron(s1, (1.0 / 6.0) * dx3)
                      + (1.0 / 24.0) * jnp.kron(dx, dx3))
            return (s0, new_s1, new_s2, new_s3, new_s4), None
        return scan_step

    def get_feature_dim(self, input_dim: int) -> int:
        d = input_dim + 1
        total = 0
        for k in range(1, self.order + 1):
            total += d**k
        return total
