import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from ml.model import SignatureCompensator
from ml.losses import martingale_violation_loss
from ml.signature_engine import SignatureFeatureExtractor
from core.bergomi import RoughBergomiModel

class Trainer:
    """
    Orchestrates the training of the Signature Compensator using JAX and Equinox.
    """
    def __init__(self, model_params: dict, learning_rate: float = 1e-3):
        self.params = model_params
        self.sig_extractor = SignatureFeatureExtractor(truncation_order=2)
        self.optim = optax.adam(learning_rate)

    def train_step(self, model, opt_state, paths, signatures, dt):
        """
        Single JIT-compiled update step. 
        Adjusts model weights to minimize Martingale violation.
        """
        def loss_fn(m):
            # Model predicts the drift correction lambda for each path
            lambdas = jax.vmap(m)(signatures) # Shape: (n_paths, 1)
            
            time_grid = jnp.linspace(0, self.params['T'], paths.shape[1])
            
            # Apply neural drift compensation: S_corrected = S * exp(-lambda * t)
            correction_factor = jnp.exp(-lambdas * time_grid)
            corrected_paths = paths * correction_factor
            
            return martingale_violation_loss(corrected_paths, dt)

        # Functional gradient calculation using Equinox filters
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = self.optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    def run(self, n_epochs: int = 200, n_paths: int = 4000):
        """Executes the training loop."""
        key = jax.random.PRNGKey(42)
        
        # Calculate signature dimension (d=2: time & price, order=2)
        input_dim = self.sig_extractor.get_feature_dim(1) 
        model_key, _ = jax.random.split(key)
        
        model = SignatureCompensator(input_dim, model_key)
        opt_state = self.optim.init(eqx.filter(model, eqx.is_array))
        
        simulator = RoughBergomiModel(self.params)
        dt = self.params['T'] / self.params['n_steps']

        print(f"Starting training for {n_epochs} epochs...")
        for epoch in range(n_epochs):
            # Monte Carlo simulation for fresh training data
            paths = simulator.simulate_paths(n_paths)
            sigs = self.sig_extractor.get_signature(paths)
            
            model, opt_state, loss = self.train_step(
                model, opt_state, jnp.array(paths), jnp.array(sigs), dt
            )
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch:03d} | Loss: {loss:.10f}")

        return model