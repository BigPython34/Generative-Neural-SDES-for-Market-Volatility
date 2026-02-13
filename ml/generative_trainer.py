import jax
import jax.numpy as jnp
import numpy as np
import optax
import equinox as eqx
import yaml
from ml.neural_sde import NeuralRoughSimulator
from ml.losses import signature_mmd_loss, mean_penalty_loss
from ml.signature_engine import SignatureFeatureExtractor
from utils.data_loader import MarketDataLoader, RealizedVolatilityLoader

def load_config(config_path: str = "config/params.yaml") -> dict:
    """Loads configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

class GenerativeTrainer:
    """
    Trains a Neural SDE to generate synthetic volatility paths.
    Supports both VIX and Realized Volatility data sources.
    """
    
    def __init__(self, config: dict, config_path: str = "config/params.yaml"):
        self.config = config
        self.yaml_config = load_config(config_path)
        
        # Load signature config
        sig_order = self.yaml_config['neural_sde']['sig_truncation_order']
        
        # Compute real dt for temporal consistency
        T = self.yaml_config['simulation']['T']
        n_steps = config['n_steps']
        self.dt = T / n_steps
        
        self.sig_extractor = SignatureFeatureExtractor(truncation_order=sig_order, dt=self.dt)
        self.input_sig_dim = self.sig_extractor.get_feature_dim(1)
        
        # Load market data based on config
        data_type = self.yaml_config['data'].get('data_type', 'vix')
        
        if data_type == 'realized_vol':
            print("Using REALIZED VOLATILITY (S&P 500)")
            loader = RealizedVolatilityLoader(config_path=config_path)
        else:
            print("Using VIX data")
            loader = MarketDataLoader(config_path=config_path)
            
        self.market_paths = loader.get_realized_vol_paths(segment_length=config['n_steps'])
        
        self.target_sigs = self.sig_extractor.get_signature(self.market_paths)
        self.target_sigs = jax.device_put(self.target_sigs)
        
        # Precompute signature normalization (component-wise std of real signatures)
        self.sig_std = jnp.std(self.target_sigs, axis=0)
        print(f"Signature normalization: std range [{float(jnp.min(self.sig_std)):.2e}, {float(jnp.max(self.sig_std)):.2e}]")
        
        # Precompute real mean variance for mean penalty
        self.real_mean = float(jnp.mean(self.market_paths))
        print(f"Real mean variance: {self.real_mean:.6f} (vol: {np.sqrt(self.real_mean)*100:.1f}%)")
        
        # Mean penalty weight
        self.lambda_mean = self.yaml_config.get('training', {}).get('lambda_mean', 10.0)

        # Learning Rate Scheduler from config
        train_cfg = self.yaml_config['training']
        scheduler = optax.cosine_decay_schedule(
            init_value=train_cfg['learning_rate_init'], 
            decay_steps=train_cfg['decay_steps'], 
            alpha=train_cfg['learning_rate_final'] / train_cfg['learning_rate_init']
        )
        
        self.optim = optax.chain(
            optax.clip_by_global_norm(train_cfg['gradient_clip']),
            optax.adam(learning_rate=scheduler)
        )    
    def train_step(self, model, opt_state, noise_driver, dt):
        """Single training step using normalized MMD loss + mean penalty."""
        def loss_fn(m):
            random_indices = jax.random.randint(
                jax.random.PRNGKey(0), (noise_driver.shape[0],), 0, self.market_paths.shape[0]
            )
            v0 = self.market_paths[random_indices, 0]
            
            # Model now computes running signatures internally
            fake_vars = jax.vmap(m.generate_variance_path, in_axes=(0, 0, None))(
                v0, noise_driver, dt
            )
            fake_sigs = self.sig_extractor.get_signature(fake_vars)
            
            # Normalized signature MMD (all components equally weighted)
            mmd = signature_mmd_loss(fake_sigs, self.target_sigs, self.sig_std)
            
            # Mean penalty (prevents Jensen bias from exp(log_v))
            mean_pen = mean_penalty_loss(fake_vars, self.real_mean)
            
            return mmd + self.lambda_mean * mean_pen

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = self.optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    def run(self, n_epochs=500, batch_size=256):
        key = jax.random.PRNGKey(42)
        model_key, _ = jax.random.split(key)
        
        model = NeuralRoughSimulator(self.input_sig_dim, model_key)
        opt_state = self.optim.init(eqx.filter(model, eqx.is_array))
        
        # Real temporal scale: T (in years) / n_steps
        # For VIX 15-min: T ≈ 0.00305 years for 20 steps
        T = self.yaml_config['simulation']['T']
        dt = T / self.config['n_steps']
        
        print("Starting Market-Driven Training...")
        best_loss = float('inf')
        
        for epoch in range(n_epochs):
            key, subkey = jax.random.split(key)
            
            # Sample random Brownian increments: dW_t = sqrt(dt) * Z
            noise = jax.random.normal(subkey, (batch_size, self.config['n_steps'])) * jnp.sqrt(dt)
            
            # Train: model computes running signatures internally
            model, opt_state, loss = self.train_step(
                model, opt_state, noise, dt
            )
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:04d} | MMD Loss: {loss:.8f}")
            
            # Save best model
            if loss < best_loss:
                best_loss = loss
                self._save_model(model)
                
        print(f"Training complete. Best loss: {best_loss:.8f}")
        return model
    
    def _save_model(self, model):
        """Save model to disk."""
        import os
        from pathlib import Path
        
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        model_path = models_dir / "neural_sde_best.eqx"
        eqx.tree_serialise_leaves(model_path, model)
    
    def load_model(self):
        """Load saved model from disk."""
        from pathlib import Path
        
        model_path = Path("models/neural_sde_best.eqx")
        if model_path.exists():
            key = jax.random.PRNGKey(0)
            model = NeuralRoughSimulator(self.input_sig_dim, key)
            return eqx.tree_deserialise_leaves(model_path, model)
        return None