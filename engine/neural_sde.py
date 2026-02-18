import jax
import jax.numpy as jnp
import equinox as eqx
import jax.nn as jnn
import yaml


def _load_neural_sde_config(config_path: str = "config/params.yaml") -> dict:
    """Load neural SDE parameters from YAML config."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        return cfg.get('neural_sde', {})
    except FileNotFoundError:
        return {}


class NeuralSDEFunc(eqx.Module):
    """Neural network for drift and diffusion, conditioned on running signature."""
    drift_net: eqx.nn.MLP
    diff_net: eqx.nn.MLP
    drift_scale: float
    diffusion_min: float
    diffusion_max: float
    log_v_center: float
    log_v_scale: float

    def __init__(self, sig_dim: int, key: jax.random.PRNGKey,
                 mlp_width: int = 64, mlp_depth: int = 3,
                 drift_scale: float = 0.5,
                 diffusion_min: float = 0.1, diffusion_max: float = 1.6,
                 log_v_center: float = -3.5, log_v_scale: float = 1.5):
        k1, k2 = jax.random.split(key)
        input_dim = sig_dim + 1  # signature + log_variance

        self.drift_net = eqx.nn.MLP(
            in_size=input_dim, out_size=1, width_size=mlp_width, depth=mlp_depth,
            activation=jnn.tanh, key=k1
        )
        self.diff_net = eqx.nn.MLP(
            in_size=input_dim, out_size=1, width_size=mlp_width, depth=mlp_depth,
            activation=jnn.tanh, key=k2
        )
        self.drift_scale = drift_scale
        self.diffusion_min = diffusion_min
        self.diffusion_max = diffusion_max
        # Proper centering/scaling so that typical log-variance maps to ~[-1, 1]
        # for better tanh activation utilization (LeCun et al., 1998)
        self.log_v_center = log_v_center
        self.log_v_scale = log_v_scale

    def __call__(self, signature_t, log_v_prev):
        norm_log_v = (log_v_prev - self.log_v_center) / self.log_v_scale
        v_in = jnp.array([norm_log_v])

        net_input = jnp.concatenate([signature_t, v_in])

        drift = self.drift_scale * jnn.tanh(self.drift_net(net_input))

        raw_diff = self.diff_net(net_input)
        diff_range = self.diffusion_max - self.diffusion_min
        diffusion = diff_range * jnn.sigmoid(raw_diff) + self.diffusion_min

        return drift, diffusion


class NeuralRoughSimulator(eqx.Module):
    """
    Neural SDE with running path signature conditioning.

    At each time step t, the drift and diffusion networks receive:
      - The running signature Sig(X_{0:t}) of the generated path so far
      - The current log-variance log(V_t)

    This makes the model genuinely path-dependent (non-Markovian),
    which is essential for capturing rough volatility dynamics
    (Kidger et al., 2021; Cuchiero et al., 2023).

    The signature is updated incrementally via Chen's identity:
      S_{0,t+dt} = S_{0,t} ⊗ S_{t,t+dt}

    Parameters (κ, θ) are read from config/params.yaml and can optionally
    be made learnable via learn_ou_params=True.
    """
    func: NeuralSDEFunc

    # OU prior parameters — read from config, optionally learnable.
    # When learn_ou_params=False, these are frozen at init values
    # by filtering them out of the optimizer tree (see GenerativeTrainer).
    kappa: jnp.ndarray
    theta: jnp.ndarray

    # Clipping bounds for log-variance (configurable)
    log_v_min: float
    log_v_max: float

    def __init__(self, sig_dim: int, key: jax.random.PRNGKey,
                 config_path: str = "config/params.yaml",
                 learn_ou_params: bool = True):
        cfg = _load_neural_sde_config(config_path)

        kappa_init = cfg.get('kappa', 2.72)
        theta_init = cfg.get('theta', -3.5)
        mlp_width = cfg.get('mlp_width', 64)
        mlp_depth = cfg.get('mlp_depth', 3)
        drift_scale = cfg.get('drift_scale', 0.5)
        diffusion_min = cfg.get('diffusion_min', 0.1)
        diffusion_max = cfg.get('diffusion_max', 1.6)

        # log-variance clipping & normalization from config
        # Default: [-7, 2] → variance in [~0.001, ~7.4] (vol ~3% to ~272%)
        self.log_v_min = cfg.get('log_v_clip_min', -7.0)
        self.log_v_max = cfg.get('log_v_clip_max', 2.0)
        log_v_center = (self.log_v_min + self.log_v_max) / 2.0
        log_v_scale = (self.log_v_max - self.log_v_min) / 4.0  # maps ±2σ to ±1

        self.func = NeuralSDEFunc(
            sig_dim, key,
            mlp_width=mlp_width, mlp_depth=mlp_depth,
            drift_scale=drift_scale,
            diffusion_min=diffusion_min, diffusion_max=diffusion_max,
            log_v_center=log_v_center, log_v_scale=log_v_scale,
        )

        # Always store as jnp arrays for serialization compatibility.
        # Freezing is handled by the optimizer filter (eqx.partition).
        self.kappa = jnp.array(float(kappa_init))
        self.theta = jnp.array(float(theta_init))

    def generate_variance_path(self, init_var, brownian_increments, dt,
                               init_sig_state=None):
        """
        Generate a variance path using the Neural SDE with running signatures.

        Args:
            init_var: Initial variance (scalar, e.g. 0.03 for ~17% vol)
            brownian_increments: Brownian increments dW, shape (n_steps,)
            dt: Time step size
            init_sig_state: Optional (s1, s2, s3) tuple to continue from a
                           previous block. If None, signature starts at zero.

        Returns:
            Variance path, shape (n_steps,)
        """
        safe_init = jnp.clip(init_var, jnp.exp(self.log_v_min), jnp.exp(self.log_v_max))
        init_log_var = jnp.log(safe_init)

        d = 2
        if init_sig_state is not None:
            s1_init, s2_init, s3_init = init_sig_state
        else:
            s1_init = jnp.zeros(d)
            s2_init = jnp.zeros(d ** 2)
            s3_init = jnp.zeros(d ** 3)

        log_v_lo = self.log_v_min
        log_v_hi = self.log_v_max

        def scan_fn(carry, dw_t):
            log_v_prev, s1, s2, s3 = carry

            sig_t = jnp.concatenate([s1, s2, s3])

            drift_nn, diff_nn = self.func(sig_t, log_v_prev)
            drift_nn = jnp.squeeze(drift_nn)
            diff_nn = jnp.squeeze(diff_nn)

            drift_ou = self.kappa * (self.theta - log_v_prev) * dt

            total_drift = drift_ou + drift_nn * dt
            log_v_next = log_v_prev + total_drift + diff_nn * dw_t
            log_v_next = jnp.clip(log_v_next, log_v_lo, log_v_hi)

            d_log_v = log_v_next - log_v_prev
            dx = jnp.array([dt, d_log_v])

            new_s1 = s1 + dx
            new_s2 = s2 + jnp.outer(s1, dx).flatten() + 0.5 * jnp.outer(dx, dx).flatten()
            new_s3 = (s3
                      + jnp.kron(s2, dx)
                      + jnp.kron(s1, 0.5 * jnp.outer(dx, dx).flatten())
                      + (1.0 / 6.0) * jnp.kron(dx, jnp.kron(dx, dx)))

            return (log_v_next, new_s1, new_s2, new_s3), log_v_next

        init_carry = (init_log_var, s1_init, s2_init, s3_init)
        final_carry, log_variance_path = jax.lax.scan(scan_fn, init_carry, brownian_increments)

        return jnp.exp(log_variance_path)

    def generate_variance_path_with_state(self, init_var, brownian_increments, dt,
                                          init_sig_state=None):
        """
        Like generate_variance_path, but also returns the final signature state
        for chaining across blocks (Chevyrev & Kormilitzin, 2016).

        Returns:
            (variance_path, final_sig_state) where final_sig_state = (s1, s2, s3)
        """
        safe_init = jnp.clip(init_var, jnp.exp(self.log_v_min), jnp.exp(self.log_v_max))
        init_log_var = jnp.log(safe_init)

        d = 2
        if init_sig_state is not None:
            s1_init, s2_init, s3_init = init_sig_state
        else:
            s1_init = jnp.zeros(d)
            s2_init = jnp.zeros(d ** 2)
            s3_init = jnp.zeros(d ** 3)

        log_v_lo = self.log_v_min
        log_v_hi = self.log_v_max

        def scan_fn(carry, dw_t):
            log_v_prev, s1, s2, s3 = carry
            sig_t = jnp.concatenate([s1, s2, s3])
            drift_nn, diff_nn = self.func(sig_t, log_v_prev)
            drift_nn = jnp.squeeze(drift_nn)
            diff_nn = jnp.squeeze(diff_nn)
            drift_ou = self.kappa * (self.theta - log_v_prev) * dt
            total_drift = drift_ou + drift_nn * dt
            log_v_next = log_v_prev + total_drift + diff_nn * dw_t
            log_v_next = jnp.clip(log_v_next, log_v_lo, log_v_hi)
            d_log_v = log_v_next - log_v_prev
            dx = jnp.array([dt, d_log_v])
            new_s1 = s1 + dx
            new_s2 = s2 + jnp.outer(s1, dx).flatten() + 0.5 * jnp.outer(dx, dx).flatten()
            new_s3 = (s3
                      + jnp.kron(s2, dx)
                      + jnp.kron(s1, 0.5 * jnp.outer(dx, dx).flatten())
                      + (1.0 / 6.0) * jnp.kron(dx, jnp.kron(dx, dx)))
            return (log_v_next, new_s1, new_s2, new_s3), log_v_next

        init_carry = (init_log_var, s1_init, s2_init, s3_init)
        final_carry, log_variance_path = jax.lax.scan(scan_fn, init_carry, brownian_increments)

        _, final_s1, final_s2, final_s3 = final_carry
        return jnp.exp(log_variance_path), (final_s1, final_s2, final_s3)
