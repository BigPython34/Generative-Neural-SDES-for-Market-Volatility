import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from ml.neural_sde import NeuralRoughSimulator
from ml.losses import signature_mmd_loss
from ml.signature_engine import SignatureFeatureExtractor
from utils.data_loader import MarketDataLoader

class GenerativeTrainer:
    """
    Trains a Neural SDE to generate synthetic volatility paths 
    that are statistically indistinguishable from the S&P 500.
    """
    def __init__(self, config: dict):
        self.config = config
        # Higher order signature (3 or 4) needed to capture complex market features
        self.sig_extractor = SignatureFeatureExtractor(truncation_order=3)
        self.optim = optax.chain(
            optax.clip_by_global_norm(1.0),  
            optax.adam(learning_rate=5e-4)
        )
        # Input dimension for the Neural SDE (Noise driver)
        self.input_sig_dim = self.sig_extractor.get_feature_dim(1)
        
        # Load Real Market Data
        loader = MarketDataLoader()
        self.market_paths = loader.get_realized_vol_paths(segment_length=config['n_steps'])
        
        # Pre-compute signatures of real data (Target Distribution)
        print("Computing Signatures of Real Market Data...")
        # We process in chunks if data is too large, here it fits in RAM
        self.target_sigs = self.sig_extractor.get_signature(self.market_paths)
        self.target_sigs = jax.device_put(self.target_sigs) # Move to GPU if available

    def train_step(self, model, opt_state, noise_driver, noise_sigs, dt):
        """
        Unsupervised training step against Market Data.
        """
        def loss_fn(m):
            # 1. Generate Fake Paths (Neural SDE)
            # Initial condition v0 sampled from real data distribution
            # We pick random starting points from real market paths
            random_indices = jax.random.randint(
                jax.random.PRNGKey(0), (noise_driver.shape[0],), 0, self.market_paths.shape[0]
            )
            v0 = self.market_paths[random_indices, 0]
            
            fake_vars = jax.vmap(m.generate_variance_path, in_axes=(0, 0, 0, None))(
                v0, noise_sigs, noise_driver, dt
            )
            
            # 2. Compute Signatures of Generated Paths
            fake_sigs = self.sig_extractor.get_signature(fake_vars)
            
            # 3. MMD Loss: Fake Signatures vs Real Market Signatures
            return signature_mmd_loss(fake_sigs, self.target_sigs)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = self.optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    def run(self, n_epochs=500, batch_size=256):
        key = jax.random.PRNGKey(42)
        model_key, _ = jax.random.split(key)
        
        model = NeuralRoughSimulator(self.input_sig_dim, model_key)
        opt_state = self.optim.init(eqx.filter(model, eqx.is_array))
        
        # Assuming 1 year of daily data normalized to T=1
        dt = 1.0 / self.config['n_steps'] 
        
        print("Starting Market-Driven Training...")
        for epoch in range(n_epochs):
            key, subkey = jax.random.split(key)
            
            # Sample random noise batch
            noise = jax.random.normal(subkey, (batch_size, self.config['n_steps']))
            noise_sigs = self.sig_extractor.get_signature(noise)
            
            # Train against pre-computed real signatures
            model, opt_state, loss = self.train_step(
                model, opt_state, noise, noise_sigs, dt
            )
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:04d} | MMD Loss (vs SPX): {loss:.8f}")
                
        return model