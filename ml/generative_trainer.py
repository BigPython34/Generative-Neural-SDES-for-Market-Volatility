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
    
    Features:
    - LR warmup + cosine decay
    - Validation split with early stopping
    - Normalized MMD + mean penalty loss
    - Learnable OU parameters (κ, θ)
    """
    
    def __init__(self, config: dict, config_path: str = "config/params.yaml"):
        self.config = config
        self.yaml_config = load_config(config_path)
        self.config_path = config_path
        
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
            
        all_paths = loader.get_realized_vol_paths(segment_length=config['n_steps'])
        all_paths = jax.device_put(all_paths)
        
        # --- Validation split ---
        train_cfg = self.yaml_config.get('training', {})
        val_split = train_cfg.get('validation_split', 0.15)
        n_total = len(all_paths)
        n_val = max(1, int(n_total * val_split))
        n_train = n_total - n_val
        
        # Deterministic split (last fraction as validation)
        self.market_paths = all_paths[:n_train]
        self.val_paths = all_paths[n_train:]
        print(f"Data split: {n_train} train / {n_val} validation ({val_split:.0%})")
        
        # Compute signatures for train and validation
        self.target_sigs = self.sig_extractor.get_signature(self.market_paths)
        self.target_sigs = jax.device_put(self.target_sigs)
        
        self.val_sigs = self.sig_extractor.get_signature(self.val_paths)
        self.val_sigs = jax.device_put(self.val_sigs)
        
        # Precompute signature normalization (from training data only)
        self.sig_std = jnp.std(self.target_sigs, axis=0)
        print(f"Signature normalization: std range [{float(jnp.min(self.sig_std)):.2e}, {float(jnp.max(self.sig_std)):.2e}]")
        
        # Precompute real mean variance for mean penalty (training data only)
        self.real_mean = float(jnp.mean(self.market_paths))
        self.val_mean = float(jnp.mean(self.val_paths))
        print(f"Real mean variance: train={self.real_mean:.6f}, val={self.val_mean:.6f}")
        
        # Mean penalty weight
        self.lambda_mean = train_cfg.get('lambda_mean', 10.0)
        
        # Early stopping config
        self.patience = train_cfg.get('early_stopping_patience', 50)

        # --- LR Schedule: warmup + cosine decay ---
        warmup_steps = train_cfg.get('warmup_steps', 50)
        lr_init = train_cfg.get('learning_rate_init', 0.001)
        lr_final = train_cfg.get('learning_rate_final', 0.00001)
        decay_steps = train_cfg.get('decay_steps', 2000)
        
        # Warmup: linear ramp from 0 to lr_init over warmup_steps
        warmup_schedule = optax.linear_schedule(
            init_value=0.0,
            end_value=lr_init,
            transition_steps=warmup_steps
        )
        # Cosine decay after warmup
        cosine_schedule = optax.cosine_decay_schedule(
            init_value=lr_init,
            decay_steps=decay_steps,
            alpha=lr_final / lr_init
        )
        # Combine: warmup then decay
        scheduler = optax.join_schedules(
            schedules=[warmup_schedule, cosine_schedule],
            boundaries=[warmup_steps]
        )
        
        self.optim = optax.chain(
            optax.clip_by_global_norm(train_cfg.get('gradient_clip', 1.0)),
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
        
        # Pass config_path so model reads κ, θ from YAML
        learn_ou = self.yaml_config.get('neural_sde', {}).get('learn_ou_params', True)
        model = NeuralRoughSimulator(
            self.input_sig_dim, model_key,
            config_path=self.config_path,
            learn_ou_params=learn_ou
        )
        opt_state = self.optim.init(eqx.filter(model, eqx.is_array))
        
        # Real temporal scale: T (in years) / n_steps
        T = self.yaml_config['simulation']['T']
        dt = T / self.config['n_steps']
        
        print("Starting Market-Driven Training...")
        print(f"   LR warmup: {self.yaml_config.get('training',{}).get('warmup_steps',50)} steps")
        print(f"   Early stopping patience: {self.patience} epochs")
        if learn_ou:
            print(f"   Learnable OU params: κ₀={float(model.kappa):.3f}, θ₀={float(model.theta):.3f}")
        
        best_val_loss = float('inf')
        best_train_loss = float('inf')
        patience_counter = 0
        best_model = model
        
        for epoch in range(n_epochs):
            key, subkey = jax.random.split(key)
            
            # Sample random Brownian increments: dW_t = sqrt(dt) * Z
            noise = jax.random.normal(subkey, (batch_size, self.config['n_steps'])) * jnp.sqrt(dt)
            
            # Train step
            model, opt_state, loss = self.train_step(
                model, opt_state, noise, dt
            )
            
            # Validation every 10 epochs
            if epoch % 10 == 0:
                key, val_key = jax.random.split(key)
                val_loss = self._compute_val_loss(model, val_key, dt, batch_size)
                
                # Log with OU params if learnable
                kappa_str = f" | κ={float(model.kappa):.3f}" if learn_ou else ""
                theta_str = f" θ={float(model.theta):.2f}" if learn_ou else ""
                print(f"Epoch {epoch:04d} | Train: {loss:.6f} | Val: {val_loss:.6f}{kappa_str}{theta_str}")
                
                # Early stopping on validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model = model
                    self._save_model(model)
                else:
                    patience_counter += 10  # We check every 10 epochs
                
                if patience_counter >= self.patience:
                    print(f"\nEarly stopping at epoch {epoch} (patience={self.patience})")
                    print(f"   Best val loss: {best_val_loss:.6f}")
                    break
            
            # Track best training loss (for backward compat)
            if loss < best_train_loss:
                best_train_loss = loss
                
        print(f"Training complete. Best train: {best_train_loss:.6f}, Best val: {best_val_loss:.6f}")
        if learn_ou:
            print(f"   Final OU params: κ={float(best_model.kappa):.3f}, θ={float(best_model.theta):.2f}")
        return best_model
    
    def _compute_val_loss(self, model, key, dt, batch_size):
        """Compute loss on validation set."""
        noise = jax.random.normal(key, (batch_size, self.config['n_steps'])) * jnp.sqrt(dt)
        
        # Sample initial conditions from validation paths
        random_indices = jax.random.randint(
            key, (batch_size,), 0, self.val_paths.shape[0]
        )
        v0 = self.val_paths[random_indices, 0]
        
        fake_vars = jax.vmap(model.generate_variance_path, in_axes=(0, 0, None))(
            v0, noise, dt
        )
        fake_sigs = self.sig_extractor.get_signature(fake_vars)
        
        # Use same normalization (from training data) for fair comparison
        mmd = signature_mmd_loss(fake_sigs, self.val_sigs, self.sig_std)
        mean_pen = mean_penalty_loss(fake_vars, self.val_mean)
        
        return float(mmd + self.lambda_mean * mean_pen)
    
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