import jax
import jax.numpy as jnp
import numpy as np
import optax
import equinox as eqx
from engine.neural_sde import NeuralRoughSimulator
from engine.losses import kernel_mmd_loss, signature_mmd_loss, mean_penalty_loss, marginal_mean_penalty_loss
from engine.signature_engine import SignatureFeatureExtractor
from utils.data_loader import MarketDataLoader, RealizedVolatilityLoader
from utils.config import load_config


class GenerativeTrainer:
    """
    Trains a Neural SDE to generate synthetic volatility paths.
    Supports both VIX and Realized Volatility data sources.

    Features:
    - LR warmup + cosine decay
    - Validation split with early stopping
    - Normalized MMD + mean penalty loss
    - Learnable OU parameters (kappa, theta)
    """

    def __init__(self, config: dict, config_path: str = "config/params.yaml"):
        self.config = config
        self.yaml_config = load_config(config_path)
        self.config_path = config_path

        sig_order = self.yaml_config['neural_sde']['sig_truncation_order']
        T = self.yaml_config['simulation']['T']
        n_steps = config['n_steps']
        self.dt = T / n_steps

        self.sig_extractor = SignatureFeatureExtractor(truncation_order=sig_order, dt=self.dt)
        self.input_sig_dim = self.sig_extractor.get_feature_dim(1)

        data_type = self.yaml_config['data'].get('data_type', 'vix')
        if data_type == 'realized_vol':
            print("Using REALIZED VOLATILITY (S&P 500)")
            loader = RealizedVolatilityLoader(config_path=config_path)
        else:
            print("Using VIX data")
            loader = MarketDataLoader(config_path=config_path)

        all_paths = loader.get_realized_vol_paths(segment_length=config['n_steps'])
        all_paths = jax.device_put(all_paths)

        train_cfg = self.yaml_config.get('training', {})
        val_split = train_cfg.get('validation_split', 0.15)
        n_total = len(all_paths)
        n_val = max(1, int(n_total * val_split))
        n_train = n_total - n_val

        self.market_paths = all_paths[:n_train]
        self.val_paths = all_paths[n_train:]
        print(f"Data split: {n_train} train / {n_val} validation ({val_split:.0%})")

        self.target_sigs = self.sig_extractor.get_signature(self.market_paths)
        self.target_sigs = jax.device_put(self.target_sigs)

        self.val_sigs = self.sig_extractor.get_signature(self.val_paths)
        self.val_sigs = jax.device_put(self.val_sigs)

        self.sig_std = jnp.std(self.target_sigs, axis=0)
        print(f"Signature normalization: std range [{float(jnp.min(self.sig_std)):.2e}, {float(jnp.max(self.sig_std)):.2e}]")

        self.real_mean = float(jnp.mean(self.market_paths))
        self.val_mean = float(jnp.mean(self.val_paths))
        print(f"Real mean variance: train={self.real_mean:.6f}, val={self.val_mean:.6f}")

        self.lambda_mean = train_cfg.get('lambda_mean', 10.0)
        self.patience = train_cfg.get('early_stopping_patience', 50)

        # "global" (default): single scalar E[V] matching
        # "marginal": per-step E[V_t] matching (tighter, Bayer & Stemper 2018)
        self.mean_penalty_mode = train_cfg.get('mean_penalty_mode', 'global')

        warmup_steps = train_cfg.get('warmup_steps', 50)
        lr_init = train_cfg.get('learning_rate_init', 0.001)
        lr_final = train_cfg.get('learning_rate_final', 0.00001)
        decay_steps = train_cfg.get('decay_steps', 2000)

        warmup_schedule = optax.linear_schedule(
            init_value=0.0, end_value=lr_init, transition_steps=warmup_steps
        )
        cosine_schedule = optax.cosine_decay_schedule(
            init_value=lr_init, decay_steps=decay_steps, alpha=lr_final / lr_init
        )
        scheduler = optax.join_schedules(
            schedules=[warmup_schedule, cosine_schedule], boundaries=[warmup_steps]
        )

        self.optim = optax.chain(
            optax.clip_by_global_norm(train_cfg.get('gradient_clip', 1.0)),
            optax.adam(learning_rate=scheduler)
        )

    def train_step(self, model, opt_state, noise_driver, dt, key):
        def loss_fn(diff_model, static_model):
            m = eqx.combine(diff_model, static_model)
            random_indices = jax.random.randint(
                key, (noise_driver.shape[0],), 0, self.market_paths.shape[0]
            )
            v0 = self.market_paths[random_indices, 0]

            fake_vars = jax.vmap(m.generate_variance_path, in_axes=(0, 0, None))(
                v0, noise_driver, dt
            )
            fake_sigs = self.sig_extractor.get_signature(fake_vars)

            mmd = kernel_mmd_loss(fake_sigs, self.target_sigs, self.sig_std)

            if self.mean_penalty_mode == 'marginal':
                real_batch = self.market_paths[random_indices]
                mean_pen = marginal_mean_penalty_loss(fake_vars, real_batch)
            else:
                mean_pen = mean_penalty_loss(fake_vars, self.real_mean)

            return mmd + self.lambda_mean * mean_pen

        diff_model, static_model = eqx.partition(model, self._filter_spec)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(diff_model, static_model)
        updates, opt_state = self.optim.update(grads, opt_state)
        diff_model = eqx.apply_updates(diff_model, updates)
        model = eqx.combine(diff_model, static_model)
        return model, opt_state, loss

    def run(self, n_epochs=500, batch_size=256):
        key = jax.random.PRNGKey(42)
        model_key, _ = jax.random.split(key)

        learn_ou = self.yaml_config.get('neural_sde', {}).get('learn_ou_params', True)
        model = NeuralRoughSimulator(
            self.input_sig_dim, model_key,
            config_path=self.config_path,
            learn_ou_params=learn_ou
        )

        # Partition model into trainable / frozen parameters.
        # When learn_ou_params=False, kappa and theta are frozen.
        # Uses eqx.partition for clean gradient handling.
        filter_spec = jax.tree_util.tree_map(lambda _: True, model)
        if not learn_ou:
            filter_spec = eqx.tree_at(lambda m: m.kappa, filter_spec, False)
            filter_spec = eqx.tree_at(lambda m: m.theta, filter_spec, False)
        self._filter_spec = filter_spec
        self._learn_ou = learn_ou

        diff_model, _ = eqx.partition(model, filter_spec)
        opt_state = self.optim.init(eqx.filter(diff_model, eqx.is_array))

        T = self.yaml_config['simulation']['T']
        dt = T / self.config['n_steps']

        # Pre-generate fixed validation noise for deterministic early stopping
        # (Bergmeir & Benítez, 2012: validation loss must be deterministic)
        val_key_fixed = jax.random.PRNGKey(999)
        self._val_noise_fixed = jax.random.normal(
            val_key_fixed, (batch_size, self.config['n_steps'])
        ) * jnp.sqrt(dt)
        vk1, vk2 = jax.random.split(val_key_fixed)
        self._val_indices_fixed = jax.random.randint(
            vk1, (batch_size,), 0, self.val_paths.shape[0]
        )

        print("Starting Market-Driven Training...")
        print(f"   LR warmup: {self.yaml_config.get('training',{}).get('warmup_steps',50)} steps")
        print(f"   Early stopping patience: {self.patience} epochs")
        if learn_ou:
            print(f"   Learnable OU params: k0={float(model.kappa):.3f}, theta0={float(model.theta):.3f}")

        best_val_loss = float('inf')
        best_train_loss = float('inf')
        patience_counter = 0
        best_model = model

        for epoch in range(n_epochs):
            key, subkey, v0_key = jax.random.split(key, 3)
            noise = jax.random.normal(subkey, (batch_size, self.config['n_steps'])) * jnp.sqrt(dt)

            model, opt_state, loss = self.train_step(model, opt_state, noise, dt, v0_key)

            if epoch % 10 == 0:
                val_loss = self._compute_val_loss(model, dt)

                kappa_str = f" | k={float(model.kappa):.3f}" if learn_ou else ""
                theta_str = f" theta={float(model.theta):.2f}" if learn_ou else ""
                print(f"Epoch {epoch:04d} | Train: {loss:.6f} | Val: {val_loss:.6f}{kappa_str}{theta_str}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model = model
                    self._save_model(model)
                else:
                    patience_counter += 10

                if patience_counter >= self.patience:
                    print(f"\nEarly stopping at epoch {epoch} (patience={self.patience})")
                    print(f"   Best val loss: {best_val_loss:.6f}")
                    break

            if loss < best_train_loss:
                best_train_loss = loss

        print(f"Training complete. Best train: {best_train_loss:.6f}, Best val: {best_val_loss:.6f}")
        if learn_ou:
            print(f"   Final OU params: k={float(best_model.kappa):.3f}, theta={float(best_model.theta):.2f}")
        return best_model

    def _compute_val_loss(self, model, dt):
        v0 = self.val_paths[self._val_indices_fixed, 0]

        fake_vars = jax.vmap(model.generate_variance_path, in_axes=(0, 0, None))(
            v0, self._val_noise_fixed, dt
        )
        fake_sigs = self.sig_extractor.get_signature(fake_vars)

        mmd = kernel_mmd_loss(fake_sigs, self.val_sigs, self.sig_std)

        if self.mean_penalty_mode == 'marginal':
            real_batch = self.val_paths[self._val_indices_fixed]
            mean_pen = marginal_mean_penalty_loss(fake_vars, real_batch)
        else:
            mean_pen = mean_penalty_loss(fake_vars, self.val_mean)

        return float(mmd + self.lambda_mean * mean_pen)

    def _save_model(self, model):
        import os
        from pathlib import Path
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / "neural_sde_best.eqx"
        eqx.tree_serialise_leaves(model_path, model)

    def load_model(self):
        from pathlib import Path
        model_path = Path("models/neural_sde_best.eqx")
        if model_path.exists():
            key = jax.random.PRNGKey(0)
            model = NeuralRoughSimulator(self.input_sig_dim, key, config_path=self.config_path)
            return eqx.tree_deserialise_leaves(model_path, model)
        return None
