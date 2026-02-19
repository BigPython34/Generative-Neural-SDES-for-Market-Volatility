"""
Multi-Measure Generative Trainer
=================================
Trains Neural SDE models under different probability measures:

  P-measure (physical / real-world):
    - Data: Realized volatility from SPX high-frequency returns
    - Loss: MMD on signatures + mean penalty
    - Use for: VaR, CVaR, stress testing, vol forecasting

  Q-measure (risk-neutral):
    - Data: VIX (implied vol, already risk-neutral)
    - Loss: MMD on signatures + mean penalty + martingale constraint
    - Use for: Option pricing, delta hedging, calibration

Usage:
    trainer = GenerativeTrainer(config, measure='Q')
    model = trainer.run(n_epochs=500)
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import equinox as eqx
from engine.neural_sde import NeuralRoughSimulator
from engine.losses import (
    kernel_mmd_loss, mean_penalty_loss, marginal_mean_penalty_loss,
    martingale_violation_loss, jump_regularization_loss,
)
from engine.signature_engine import SignatureFeatureExtractor
from utils.data_loader import MarketDataLoader, RealizedVolatilityLoader
from utils.config import load_config


class GenerativeTrainer:
    """
    Trains a Neural SDE to generate synthetic volatility paths.

    Supports:
    - P-measure and Q-measure training
    - Optional jump component
    - LR warmup + cosine decay
    - Validation split with early stopping
    - Learnable OU parameters (kappa, theta)
    """

    def __init__(self, config: dict, config_path: str = "config/params.yaml",
                 measure: str = "auto"):
        """
        Args:
            config: dict with 'n_steps', 'T'
            config_path: path to YAML config
            measure: 'P' (physical), 'Q' (risk-neutral), or 'auto' (from config)
        """
        self.config = config
        self.yaml_config = load_config(config_path)
        self.config_path = config_path

        # Determine measure
        if measure == "auto":
            data_type = self.yaml_config['data'].get('data_type', 'vix')
            self.measure = 'P' if data_type == 'realized_vol' else 'Q'
        else:
            self.measure = measure.upper()

        sig_order = self.yaml_config['neural_sde']['sig_truncation_order']
        T = self.yaml_config['simulation']['T']
        n_steps = config['n_steps']
        self.dt = T / n_steps

        self.sig_extractor = SignatureFeatureExtractor(truncation_order=sig_order, dt=self.dt)
        self.input_sig_dim = self.sig_extractor.get_feature_dim(1)

        # Data loading depends on measure
        if self.measure == 'P':
            print(f"[{self.measure}-measure] Using REALIZED VOLATILITY (S&P 500)")
            loader = RealizedVolatilityLoader(config_path=config_path)
        else:
            data_type = self.yaml_config['data'].get('data_type', 'vix')
            if data_type == 'realized_vol':
                print(f"[{self.measure}-measure] Using REALIZED VOLATILITY (override)")
                loader = RealizedVolatilityLoader(config_path=config_path)
            else:
                print(f"[{self.measure}-measure] Using VIX data")
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
        self.mean_penalty_mode = train_cfg.get('mean_penalty_mode', 'global')

        # Q-measure specific parameters
        q_cfg = train_cfg.get('q_measure', {})
        self.lambda_martingale = q_cfg.get('lambda_martingale', 5.0)
        self.lambda_jump_reg = q_cfg.get('lambda_jump_reg', 0.1)

        # SOFR integration for Q-measure
        self._risk_free_rate = self.yaml_config['pricing']['risk_free_rate']
        if self.measure == 'Q':
            try:
                from utils.sofr_loader import get_sofr
                sofr = get_sofr()
                if sofr.is_available:
                    self._risk_free_rate = sofr.get_rate()
                    print(f"[Q-measure] Using SOFR rate: {self._risk_free_rate:.4f}")
            except Exception:
                pass

        # Learning rate schedule
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

            # MMD loss
            mmd = kernel_mmd_loss(fake_sigs, self.target_sigs, self.sig_std)

            # Mean penalty
            if self.mean_penalty_mode == 'marginal':
                real_batch = self.market_paths[random_indices]
                mean_pen = marginal_mean_penalty_loss(fake_vars, real_batch)
            else:
                mean_pen = mean_penalty_loss(fake_vars, self.real_mean)

            total = mmd + self.lambda_mean * mean_pen

            # Q-measure: add martingale constraint
            if self.measure == 'Q':
                rho = self.yaml_config['bergomi']['rho']
                n_mc, n_steps_gen = fake_vars.shape

                k_spot = jax.random.fold_in(key, 999)
                z_indep = jax.random.normal(k_spot, (n_mc, n_steps_gen))

                v0_col = v0.reshape(-1, 1)
                var_prev = jnp.concatenate([v0_col, fake_vars[:, :-1]], axis=1)
                vol_prev = jnp.sqrt(var_prev)

                dw_spot = rho * noise_driver + jnp.sqrt(1 - rho**2) * z_indep * jnp.sqrt(dt)
                log_ret = (self._risk_free_rate - 0.5 * var_prev) * dt + vol_prev * dw_spot
                spot_normalized = jnp.exp(jnp.cumsum(log_ret, axis=1))

                mart_loss = martingale_violation_loss(
                    spot_normalized, dt, self._risk_free_rate
                )
                total = total + self.lambda_martingale * mart_loss

            # Jump regularization
            if m.enable_jumps:
                j_reg = jump_regularization_loss(m.jump_params.log_lambda)
                total = total + self.lambda_jump_reg * j_reg

            return total

        diff_model, static_model = eqx.partition(model, self._filter_spec)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(diff_model, static_model)
        updates, opt_state = self.optim.update(grads, opt_state)
        diff_model = eqx.apply_updates(diff_model, updates)
        model = eqx.combine(diff_model, static_model)
        return model, opt_state, loss

    def run(self, n_epochs=500, batch_size=256, enable_jumps=False):
        key = jax.random.PRNGKey(42)
        model_key, _ = jax.random.split(key)

        learn_ou = self.yaml_config.get('neural_sde', {}).get('learn_ou_params', True)
        model = NeuralRoughSimulator(
            self.input_sig_dim, model_key,
            config_path=self.config_path,
            learn_ou_params=learn_ou,
            enable_jumps=enable_jumps,
        )

        # Partition trainable / frozen parameters
        filter_spec = jax.tree_util.tree_map(lambda _: True, model)
        if not learn_ou:
            filter_spec = eqx.tree_at(lambda m: m.kappa, filter_spec, False)
            filter_spec = eqx.tree_at(lambda m: m.theta, filter_spec, False)
        if not enable_jumps:
            filter_spec = eqx.tree_at(lambda m: m.jump_params, filter_spec,
                                      jax.tree_util.tree_map(lambda _: False, model.jump_params))
        self._filter_spec = filter_spec
        self._learn_ou = learn_ou

        diff_model, _ = eqx.partition(model, filter_spec)
        opt_state = self.optim.init(eqx.filter(diff_model, eqx.is_array))

        T = self.yaml_config['simulation']['T']
        dt = T / self.config['n_steps']

        val_key_fixed = jax.random.PRNGKey(999)
        self._val_noise_fixed = jax.random.normal(
            val_key_fixed, (batch_size, self.config['n_steps'])
        ) * jnp.sqrt(dt)
        vk1, _ = jax.random.split(val_key_fixed)
        self._val_indices_fixed = jax.random.randint(
            vk1, (batch_size,), 0, self.val_paths.shape[0]
        )

        measure_str = f"[{self.measure}-measure]"
        jump_str = " + JUMPS" if enable_jumps else ""
        print(f"\nStarting {measure_str} Training{jump_str}...")
        print(f"   LR warmup: {self.yaml_config.get('training',{}).get('warmup_steps',50)} steps")
        print(f"   Early stopping patience: {self.patience} epochs")
        if self.measure == 'Q':
            print(f"   Martingale penalty: λ={self.lambda_martingale}")
            print(f"   Risk-free rate: {self._risk_free_rate:.4f}")
        if learn_ou:
            print(f"   Learnable OU: κ₀={float(model.kappa):.3f}, θ₀={float(model.theta):.3f}")
        if enable_jumps:
            jp = model.jump_params
            print(f"   Jump params: λ={float(jp.intensity):.2f}, "
                  f"μ_J={float(jp.mu_j):.3f}, σ_J={float(jp.sigma_j):.3f}")

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

                kappa_str = f" | κ={float(model.kappa):.3f}" if learn_ou else ""
                theta_str = f" θ={float(model.theta):.2f}" if learn_ou else ""
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
            print(f"   Final OU: κ={float(best_model.kappa):.3f}, θ={float(best_model.theta):.2f}")
        if enable_jumps:
            jp = best_model.jump_params
            print(f"   Final jumps: λ={float(jp.intensity):.2f}, "
                  f"μ_J={float(jp.mu_j):.3f}, σ_J={float(jp.sigma_j):.3f}")
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
        from pathlib import Path
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)

        suffix = f"_{self.measure.lower()}"
        if model.enable_jumps:
            suffix += "_jump"
        model_path = models_dir / f"neural_sde_best{suffix}.eqx"
        eqx.tree_serialise_leaves(model_path, model)

        # Also save as generic "best" for backward compatibility
        generic_path = models_dir / "neural_sde_best.eqx"
        eqx.tree_serialise_leaves(generic_path, model)

    def load_model(self, model_path=None, enable_jumps=False):
        from pathlib import Path
        if model_path is None:
            model_path = Path("models/neural_sde_best.eqx")
        else:
            model_path = Path(model_path)

        if not model_path.exists():
            return None

        key = jax.random.PRNGKey(0)

        # Try loading with requested architecture first
        try:
            model = NeuralRoughSimulator(
                self.input_sig_dim, key,
                config_path=self.config_path,
                enable_jumps=enable_jumps,
            )
            return eqx.tree_deserialise_leaves(model_path, model)
        except Exception:
            pass

        # Backward compatibility: model saved with old architecture (no jumps).
        # Load into a jump-disabled skeleton, then wrap into the new structure.
        try:
            from engine._legacy_loader import load_legacy_model
            return load_legacy_model(
                model_path, self.input_sig_dim, self.config_path
            )
        except Exception:
            pass

        print("   [WARN] Could not load model - architecture mismatch. Retrain with bin/train_multi.py")
        return None
