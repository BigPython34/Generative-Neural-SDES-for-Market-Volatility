import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import yaml
from ml.neural_sde import NeuralRoughSimulator
from ml.losses import signature_mmd_loss
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
        self.sig_extractor = SignatureFeatureExtractor(truncation_order=sig_order)
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
    def train_step(self, model, opt_state, noise_driver, noise_sigs, dt):
        """Single training step using MMD loss."""
        def loss_fn(m):
            random_indices = jax.random.randint(
                jax.random.PRNGKey(0), (noise_driver.shape[0],), 0, self.market_paths.shape[0]
            )
            v0 = self.market_paths[random_indices, 0]
            
            fake_vars = jax.vmap(m.generate_variance_path, in_axes=(0, 0, 0, None))(
                v0, noise_sigs, noise_driver, dt
            )
            fake_sigs = self.sig_extractor.get_signature(fake_vars)
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
                print(f"Epoch {epoch:04d} | MMD Loss: {loss:.8f}")
                
        return model