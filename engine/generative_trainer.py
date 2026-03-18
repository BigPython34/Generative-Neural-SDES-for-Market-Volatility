"""
Multi-Measure Generative Trainer
=================================
Trains Neural SDE models under P-measure (physical / real-world):
    - Data: Realized volatility from SPX high-frequency returns
    - Loss: MMD on signatures + mean penalty (PRIMARY: distribution matching)
    - Use for: VaR, CVaR, stress testing, vol forecasting

Usage:
    trainer = GenerativeTrainer(config)
    model = trainer.run(n_epochs=500)
"""

import jax
import jax.numpy as jnp
import os
import optax
import equinox as eqx
from engine.neural_sde import NeuralRoughSimulator
from engine.losses import (
    kernel_mmd_loss, mean_penalty_loss, marginal_mean_penalty_loss,
    martingale_violation_loss, jump_regularization_loss, smile_fit_loss,
)
from engine.signature_engine import SignatureFeatureExtractor
from utils.loader.RealizedVariance import RealizedVolatilityLoader
from utils.config import load_config


class GenerativeTrainer:
    """
    Trains a Neural SDE to generate synthetic volatility paths.

    Supports:
    - Historical measure 
    - Optional jump component
    - LR warmup + cosine decay
    - Validation split with early stopping
    - Learnable OU parameters (kappa, theta)
    """

    def __init__(self, config: dict, config_path: str = "config/params.yaml"
               ):
        """
        Args:
            config: dict with 'n_steps', 'T'
            config_path: path to YAML config
        """
        self.config = config
        self.yaml_config = load_config(config_path)
        self.config_path = config_path

        sig_order = self.yaml_config['neural_sde']['sig_truncation_order']
        T = self.yaml_config['simulation']['T']
        n_steps = config['n_steps']
        self.dt = T / n_steps

        self.sig_extractor = SignatureFeatureExtractor(truncation_order=sig_order, dt=self.dt)
        self.input_sig_dim = self.sig_extractor.get_feature_dim(1)
        
        print(f"[Config] Signature order={sig_order}, sig_dim={self.input_sig_dim}")

       
        loader = RealizedVolatilityLoader(config_path=config_path)
        
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
        self.mean_penalty_space = train_cfg.get('mean_penalty_space', 'log_v')

        
        # η auto-calibration from VVIX (when configured)
        bergomi_cfg = self.yaml_config.get('bergomi', {})
        if bergomi_cfg.get('eta_source', 'config') == 'vvix':
            self._auto_calibrate_eta(bergomi_cfg)

        
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

            mmd = kernel_mmd_loss(fake_sigs, self.target_sigs, self.sig_std)

        
            return mmd

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

        # Fractional backbone: freeze/unfreeze H and eta based on config
        backbone = self.yaml_config.get('neural_sde', {}).get('backbone', 'ou')
        frac_cfg = self.yaml_config.get('neural_sde', {}).get('fractional', {})
        if backbone != 'fractional':
            # Freeze fractional params entirely when using OU backbone
            filter_spec = eqx.tree_at(lambda m: m.fractional_params, filter_spec,
                                      jax.tree_util.tree_map(lambda _: False, model.fractional_params))
        else:
            # Fractional backbone: control individual param learning
            if not frac_cfg.get('learn_hurst', True):
                filter_spec = eqx.tree_at(lambda m: m.fractional_params.log_H, filter_spec, False)
            if not frac_cfg.get('learn_eta', True):
                filter_spec = eqx.tree_at(lambda m: m.fractional_params.log_eta, filter_spec, False)
            # Freeze OU params when using fractional backbone
            filter_spec = eqx.tree_at(lambda m: m.kappa, filter_spec, False)
            filter_spec = eqx.tree_at(lambda m: m.theta, filter_spec, False)

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
        backbone_str = f" ({model.backbone} backbone)"
        print(f"\nStarting {measure_str} Training{jump_str}{backbone_str}...")
        print(f"   LR warmup: {self.yaml_config.get('training',{}).get('warmup_steps',50)} steps")
        print(f"   Early stopping patience: {self.patience} epochs")
        if self.measure == 'Q':
            print(f"   Martingale penalty: λ={self.lambda_martingale}")
            print(f"   Risk-free rate: {self._risk_free_rate:.4f}")
        if model.backbone == 'fractional':
            fp = model.fractional_params
            print(f"   Fractional backbone: H₀={float(fp.H):.4f}, η₀={float(fp.eta):.3f}")
        elif learn_ou:
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

                if model.backbone == 'fractional':
                    fp = model.fractional_params
                    extra_str = f" | H={float(fp.H):.4f} η={float(fp.eta):.3f}"
                elif learn_ou:
                    extra_str = f" | κ={float(model.kappa):.3f} θ={float(model.theta):.2f}"
                else:
                    extra_str = ""
                print(f"Epoch {epoch:04d} | Train: {loss:.6f} | Val: {val_loss:.6f}{extra_str}")

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
        if best_model.backbone == 'fractional':
            fp = best_model.fractional_params
            print(f"   Final fractional: H={float(fp.H):.4f}, η={float(fp.eta):.3f}")
        elif learn_ou:
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
        elif self.mean_penalty_space == 'log_v':
            # Mean penalty in log-V space to avoid Jensen bias
            fake_log = jnp.log(jnp.maximum(fake_vars, 1e-10))
            val_log_mean = float(jnp.mean(jnp.log(jnp.maximum(self.val_paths, 1e-10))))
            mean_pen = jnp.square(jnp.mean(fake_log) - val_log_mean)
        else:
            mean_pen = mean_penalty_loss(fake_vars, self.val_mean)

        return float(mmd + self.lambda_mean * mean_pen)

    def _auto_calibrate_eta(self, bergomi_cfg: dict):
        """
        Auto-calibrate η from VVIX when bergomi.eta_source == 'vvix'.

        Overwrites:
          - bergomi.eta in yaml_config (for downstream use)
          - neural_sde.fractional.eta_init (to initialize fractional backbone)
        """
        try:
            from quant.calibration.vvix_calibrator import VVIXCalibrator
            cal = VVIXCalibrator()
            if not cal.is_available:
                print("   [η AUTO] VVIX data not available — keeping config η")
                return
            H = bergomi_cfg.get('hurst', 0.07)
            result = cal.estimate_eta(H=H)
            eta_new = result['eta_recommended']
            eta_old = bergomi_cfg.get('eta', 1.9)
            print(f"   [η AUTO] VVIX calibration: η = {eta_new:.3f} "
                  f"(was {eta_old:.3f}, VVIX = {result.get('vvix_current', '?')})")
            self.yaml_config['bergomi']['eta'] = eta_new
            # Also update fractional backbone init (if used)
            frac_cfg = self.yaml_config.get('neural_sde', {}).get('fractional', {})
            if frac_cfg:
                frac_cfg['eta_init'] = eta_new
        except Exception as e:
            print(f"   [η AUTO] Failed: {e}")

    def _save_model(self, model):
        from pathlib import Path
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)

        suffix = f"_{self.measure.lower()}"
        if model.enable_jumps:
            suffix += "_jump"
        model_path = models_dir / f"neural_sde_best{suffix}.eqx"
        eqx.tree_serialise_leaves(model_path, model)

    def load_model(self, model_path=None, enable_jumps=False):
        from pathlib import Path
        if model_path is None:
            suffix = f"_{self.measure.lower()}"
            if enable_jumps:
                suffix += "_jump"
            model_path = Path(f"models/neural_sde_best{suffix}.eqx")
        else:
            model_path = Path(model_path)

        if not model_path.exists():
            return None

        key = jax.random.PRNGKey(0)

        try:
            model = NeuralRoughSimulator(
                self.input_sig_dim, key,
                config_path=self.config_path,
                enable_jumps=enable_jumps,
            )
            return eqx.tree_deserialise_leaves(model_path, model)
        except Exception:
            pass

        print("   [WARN] Could not load model - architecture mismatch. Retrain with bin/train_multi.py")
        return None
