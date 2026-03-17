"""
Multi-Measure Generative Trainer
=================================
Trains Neural SDE models under different probability measures:

  P-measure (physical / real-world):
    - Data: Realized volatility from SPX high-frequency returns
    - Loss: MMD on signatures + mean penalty (PRIMARY: distribution matching)
    - Use for: VaR, CVaR, stress testing, vol forecasting

  Q-measure (risk-neutral):
    - Data: Market option prices (IV surface from cached SPY options)
    - Loss: smile_fit + martingale + MMD regularizer
      → PRIMARY: match observed option prices (Bayer & Stemper 2018)
      → CONSTRAINT: E[e^{-rT}S_T] = S_0 (no-arbitrage)
      → REGULARIZER: MMD on signatures (path realism, small weight)
    - Fallback (no options data): P-loss + martingale (weaker)
    - Use for: Option pricing, delta hedging, calibration

  Theoretical justification (v3.0):
    Under Q, prices are expectations: C(K,T) = E^Q[e^{-rT}(S_T-K)^+].
    The vol dynamics under Q ≠ under P (Buehler et al. 2021).
    Matching historical path distributions (MMD on P-data) does NOT learn
    Q-dynamics — it learns P with a first-moment correction. The correct
    Q-loss is driven by market option prices (Gierjatowicz et al. 2022).

Usage:
    trainer = GenerativeTrainer(config, measure='Q')
    model = trainer.run(n_epochs=500)
"""

import jax
import jax.numpy as jnp
import numpy as np
import os
import optax
import equinox as eqx
from engine.neural_sde import NeuralRoughSimulator
from engine.losses import (
    kernel_mmd_loss, mean_penalty_loss, marginal_mean_penalty_loss,
    martingale_violation_loss, jump_regularization_loss, smile_fit_loss,
)
from engine.signature_engine import SignatureFeatureExtractor
from utils.loader.data_loader import MarketDataLoader, RealizedVolatilityLoader
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
        
        print(f"[Config] Signature order={sig_order}, sig_dim={self.input_sig_dim}")

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
        self.mean_penalty_space = train_cfg.get('mean_penalty_space', 'log_v')

        # Training mode: "general", "pricing", "stress_test"
        self.training_mode = train_cfg.get('training_mode', 'general')

        # Q-measure specific parameters
        q_cfg = train_cfg.get('q_measure', {})
        self.lambda_martingale = q_cfg.get('lambda_martingale', 5.0)
        self.lambda_smile = q_cfg.get('lambda_smile', 1.0)
        self.lambda_jump_reg = q_cfg.get('lambda_jump_reg', 0.1)

        # Load smile target for Q-measure (PRIMARY loss under Q)
        # Under Q, option prices are the ground truth — not historical paths.
        self._smile_target = None
        if self.measure == 'Q':
            self._smile_target = self._load_smile_target(q_cfg)
            if self._smile_target is None:
                print(f"   [Q-measure] No options data -> falling back to "
                      f"P+martingale loss. This is theoretically weaker.")
                print(f"     Run `python bin/data/fetch_options.py` to cache SPY options.")
            else:
                print(f"   [Q-measure] IV surface loaded -> smile_fit is PRIMARY loss")

        # η auto-calibration from VVIX (when configured)
        bergomi_cfg = self.yaml_config.get('bergomi', {})
        if bergomi_cfg.get('eta_source', 'config') == 'vvix':
            self._auto_calibrate_eta(bergomi_cfg)

        # SOFR integration for Q-measure
        self._risk_free_rate = self.yaml_config['pricing']['risk_free_rate']
        if self.measure == 'Q':
            try:
                from utils.loader.sofr_loader import get_sofr
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

            # --- P-MEASURE LOSS (primary for P, regularizer for Q) ---
            # MMD on signatures: matches distribution of generated paths
            # to historical data. This is a P-measure objective.
            mmd = kernel_mmd_loss(fake_sigs, self.target_sigs, self.sig_std)

            # Mean penalty
            if self.mean_penalty_mode == 'marginal':
                real_batch = self.market_paths[random_indices]
                mean_pen = marginal_mean_penalty_loss(fake_vars, real_batch)
            elif self.mean_penalty_space == 'log_v':
                fake_log = jnp.log(jnp.maximum(fake_vars, 1e-10))
                real_log_mean = jnp.mean(jnp.log(jnp.maximum(self.market_paths, 1e-10)))
                mean_pen = jnp.square(jnp.mean(fake_log) - real_log_mean)
            else:
                mean_pen = mean_penalty_loss(fake_vars, self.real_mean)

            if self.measure == 'P':
                # ────────────────────────────────────────────────────
                # P-MEASURE: MMD is the primary objective
                # ────────────────────────────────────────────────────
                total = mmd + self.lambda_mean * mean_pen

            else:
                # ────────────────────────────────────────────────────
                # Q-MEASURE: IV surface matching is the primary objective
                # MMD becomes a regularizer (prevents degenerate Q-dynamics)
                #
                # Architecture (Bayer & Stemper 2018, Buehler et al. 2021):
                #   L_Q = λ_smile · L_smile    [PRIMARY: market prices]
                #       + λ_mart  · L_mart     [CONSTRAINT: no-arbitrage]
                #       + λ_mmd   · L_MMD      [REGULARIZER: path realism]
                #       + λ_mean  · L_mean     [REGULARIZER: level]
                # ────────────────────────────────────────────────────
                rho = self.yaml_config['bergomi']['rho']
                n_mc, n_steps_gen = fake_vars.shape

                # Build correlated spot paths (leverage effect)
                k_spot = jax.random.fold_in(key, 999)
                z_indep = jax.random.normal(k_spot, (n_mc, n_steps_gen))

                v0_col = v0.reshape(-1, 1)
                var_prev = jnp.concatenate([v0_col, fake_vars[:, :-1]], axis=1)
                vol_prev = jnp.sqrt(var_prev)

                dw_spot = rho * noise_driver + jnp.sqrt(1 - rho**2) * z_indep * jnp.sqrt(dt)
                log_ret = (self._risk_free_rate - 0.5 * var_prev) * dt + vol_prev * dw_spot
                spot_normalized = jnp.exp(jnp.cumsum(log_ret, axis=1))

                # Martingale constraint: E[e^{-rT}S_T] = S_0
                mart_loss = martingale_violation_loss(
                    spot_normalized, dt, self._risk_free_rate
                )

                # Smile fitting loss: match market option prices
                smile_loss = jnp.float32(0.0)
                if self._smile_target is not None:
                    target = self._smile_target
                    spot_normalized_final = spot_normalized[:, -1]
                    s0 = self.yaml_config['pricing']['spot']
                    S_T = s0 * spot_normalized_final
                    T_smile = target['T']
                    r_smile = self._risk_free_rate

                    # Vectorized MC pricing at all target strikes
                    model_prices_list = []
                    for ki, K in enumerate(target['strikes']):
                        is_call = K >= s0
                        if is_call:
                            payoff = jnp.maximum(S_T - K, 0)
                        else:
                            payoff = jnp.maximum(K - S_T, 0)
                        mc_price = jnp.exp(-r_smile * T_smile) * jnp.mean(payoff)
                        model_prices_list.append(mc_price)

                    model_prices = jnp.array(model_prices_list)
                    market_prices = target.get('prices', model_prices)
                    if 'prices' in target:
                        smile_loss = smile_fit_loss(
                            model_prices, market_prices, target.get('vega_weights')
                        )

                # When we have smile data: smile is primary, MMD is regularizer.
                # When no smile data: fallback to P+martingale.
                if self._smile_target is not None:
                    lambda_mmd_reg = self.yaml_config.get('training', {}).get(
                        'q_measure', {}
                    ).get('lambda_mmd_regularizer', 0.1)
                    total = (
                        self.lambda_smile * smile_loss
                        + self.lambda_martingale * mart_loss
                        + lambda_mmd_reg * mmd
                        + 0.1 * mean_pen  # light mean regularizer
                    )
                else:
                    # Fallback: no smile data → P+martingale
                    # (with warning logged at init time)
                    total = (
                        mmd + self.lambda_mean * mean_pen
                        + self.lambda_martingale * mart_loss
                    )

            # Jump regularization (both measures)
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

    def _load_smile_target(self, q_cfg: dict):
        """
        Load a market smile target for smile_fit_loss in pricing training mode.
        Uses the most recent cached options surface.
        """
        import pandas as pd
        cache_dir = q_cfg.get('options_data_source', 'data/options_cache')

        if not os.path.exists(cache_dir):
            print(f"   [SMILE] No options cache at {cache_dir} — smile_fit disabled")
            return None

        # Find most recent surface CSV
        csvs = sorted([f for f in os.listdir(cache_dir) if f.endswith('.csv') and 'SPY_surface' in f])
        if not csvs:
            print("   [SMILE] No surface files found — smile_fit disabled")
            return None

        latest = os.path.join(cache_dir, csvs[-1])
        try:
            df = pd.read_csv(latest)
        except Exception as e:
            print(f"   [SMILE] Error loading {latest}: {e}")
            return None

        # Extract 30-day ATM smile (moneyness -15% to +15%)
        target_dte = 30
        available_dtes = df['dte'].unique() if 'dte' in df.columns else []
        if len(available_dtes) == 0:
            return None

        closest_dte = min(available_dtes, key=lambda x: abs(x - target_dte))
        smile = df[df['dte'] == closest_dte].copy()

        # OTM filter
        otm_mask = (
            ((smile['type'] == 'call') & (smile['moneyness'] >= 0)) |
            ((smile['type'] == 'put') & (smile['moneyness'] <= 0))
        )
        smile = smile[otm_mask]
        smile = smile[
            (smile['impliedVolatility'] > 0.05) &
            (smile['impliedVolatility'] < 1.5) &
            (smile['moneyness'] >= -0.15) &
            (smile['moneyness'] <= 0.15)
        ]

        if len(smile) < 5:
            print(f"   [SMILE] Only {len(smile)} OTM options for DTE={closest_dte} — disabled")
            return None

        strikes = jax.device_put(jnp.array(smile['moneyness'].values))
        market_ivs = jax.device_put(jnp.array(smile['impliedVolatility'].values))

        # Vega weights: approximate with BS vega (higher near ATM)
        vega_w = jnp.exp(-0.5 * (strikes / 0.10) ** 2)  # Gaussian weighting
        vega_w = jax.device_put(vega_w)

        print(f"   [SMILE] Loaded {len(smile)} OTM options, DTE={closest_dte}d")

        # Compute market prices from IVs for differentiable loss comparison
        spot = (float(df.get('spot', pd.Series([100.0])).iloc[0]) if 'spot' in df.columns
                else 100.0)
        T_smile = float(closest_dte / 365.0)
        r_smile = self._risk_free_rate if hasattr(self, '_risk_free_rate') else 0.05
        abs_strikes = spot * (1.0 + smile['moneyness'].values)

        from quant.models.black_scholes import BlackScholes
        market_prices = []
        for i, (K, iv, m) in enumerate(zip(abs_strikes, smile['impliedVolatility'].values,
                                             smile['moneyness'].values)):
            opt_type = 'call' if m >= 0 else 'put'
            p = BlackScholes.price(spot, K, T_smile, r_smile, iv, opt_type)
            market_prices.append(p)
        market_prices_arr = jax.device_put(jnp.array(market_prices))

        return {
            'moneyness': strikes,
            'strikes': jax.device_put(jnp.array(abs_strikes)),
            'market_ivs': market_ivs,
            'prices': market_prices_arr,
            'vega_weights': vega_w,
            'T': T_smile,
            'spot': spot,
        }

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
