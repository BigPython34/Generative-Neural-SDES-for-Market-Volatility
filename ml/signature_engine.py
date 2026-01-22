import jax
import jax.numpy as jnp
import numpy as np
import esig

class SignatureFeatureExtractor:
    """
    Dual-Engine Signature Extractor.
    - Uses 'esig' (C++) for static real data (Pre-processing).
    - Uses 'JAX' (Custom implementation) for training (Differentiable).
    """
    def __init__(self, truncation_order: int = 3):
        self.order = truncation_order

    def get_signature(self, paths):
        """
        Smart dispatcher: detects if input is JAX (Tracer) or NumPy.
        """
        # Si c'est un tableau NumPy (Données réelles), on utilise esig (rapide)
        if isinstance(paths, (np.ndarray, list)):
            return self._get_signature_numpy(paths)
        
        # Si c'est un JAX Array (Entraînement), on utilise le moteur JAX différentiable
        return self._get_signature_jax(paths)

    def _get_signature_numpy(self, paths):
        np_paths = np.array(paths)
        if np_paths.ndim == 2:
            n_paths, n_steps = np_paths.shape
            time_axis = np.linspace(0, 1, n_steps)
            augmented_paths = np.array([np.column_stack([time_axis, p]) for p in np_paths])
        else:
            augmented_paths = np_paths
            
        # Utilisation robuste de esig
        func = getattr(esig, 'stream2sig', None) or getattr(esig.tosig, 'stream2sig')
        return np.array([func(p, self.order) for p in augmented_paths])

    def _get_signature_jax(self, paths):
        """
        Pure JAX implementation of Path Signatures (Order 1, 2, 3).
        Differentiable & GPU compatible.
        Assumes paths are 1D (Variance process).
        """
        # 1. Augment with Time: (Batch, Time) -> (Batch, Time, 2)
        batch_size, n_steps = paths.shape
        time_axis = jnp.linspace(0, 1, n_steps)
        # Broadcast time across batch
        time_axis = jnp.tile(time_axis, (batch_size, 1))
        
        # Stack: (Batch, Time, 2) where dim 0 is time, dim 1 is value
        path_augmented = jnp.stack([time_axis, paths], axis=-1)
        
        # 2. Compute Increments: dX_t
        dX = path_augmented[:, 1:] - path_augmented[:, :-1] # (Batch, Time-1, 2)
        
        # 3. Compute Signature via Tensor Algebra (Chen's Identity)
        # We define a scan function to accumulate iterated integrals
        
        def scan_step(carry, dx):
            # carry contains [Sig_Order_0, Sig_Order_1, Sig_Order_2, ...]
            # dx is the increment at current step
            
            # Current signature terms
            s0, s1, s2, s3 = carry
            
            # Update rules (Tensor product integration)
            # S^1_{new} = S^1 + dx
            new_s1 = s1 + dx
            
            # S^2_{new} = S^2 + S^1 (outer) dx + dx^2/2
            term2_a = jnp.outer(s1, dx).flatten()
            term2_b = 0.5 * jnp.outer(dx, dx).flatten()
            new_s2 = s2 + term2_a + term2_b
            
            # S^3_{new} = S^3 + S^2 (outer) dx + S^1 (outer) dx^2/2 + dx^3/6
            # For simplicity & speed in JAX, we implement the recursive "Chen's Identity" approximation
            # S_{t+1} = S_t (tensor) exp(dx)
            # This is complex to write generically, so we stick to the explicit update for Order 3
            
            term3_a = jnp.kron(s2, dx) # (d^2 * d) = d^3
            term3_b = jnp.kron(s1, 0.5 * jnp.outer(dx, dx).flatten())
            term3_c = (1/6.0) * jnp.kron(dx, jnp.kron(dx, dx)) # dx^3
            new_s3 = s3 + term3_a + term3_b + term3_c

            return (s0, new_s1, new_s2, new_s3), None

        # Initialize carriers (Batch processing via vmap is implied by input shape if we structure correctly)
        # But here 'paths' is a batch. We need to vmap the scan.
        
        def compute_single_path_sig(dx_seq):
            d = 2 # Time + Value
            s0 = jnp.array([1.0])
            s1 = jnp.zeros(d)
            s2 = jnp.zeros(d**2)
            s3 = jnp.zeros(d**3)
            
            final_carry, _ = jax.lax.scan(scan_step, (s0, s1, s2, s3), dx_seq)
            
            # Concatenate all orders (excluding order 0 which is always 1)
            # Result size: 2 + 4 + 8 = 14 features
            if self.order == 2:
                return jnp.concatenate([final_carry[1], final_carry[2]])
            else: # order 3
                return jnp.concatenate([final_carry[1], final_carry[2], final_carry[3]])

        # Vectorize over the batch
        sigs = jax.vmap(compute_single_path_sig)(dX)
        return sigs

    def get_feature_dim(self, input_dim: int) -> int:
        """
        Returns the dimension of the signature vector.
        d = input_dim + 1 (Time augmentation)
        Size = d + d^2 + d^3 ...
        """
        d = input_dim + 1
        total = 0
        for k in range(1, self.order + 1):
            total += d**k
        return total