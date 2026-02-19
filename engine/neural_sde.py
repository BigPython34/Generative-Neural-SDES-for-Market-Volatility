"""
Neural SDE with Path Signature Conditioning
============================================
Signature-conditioned Neural SDE for rough volatility modeling.

Architecture:
    dlog(V) = [κ(θ - log V) + f_nn(Sig, log V)] dt
            + g_nn(Sig, log V) dW
            + J dN                              (optional jump component)

    where Sig = running path signature of (t, log V) up to order M,
    f_nn and g_nn are MLPs, and J·dN is a compound Poisson jump process.

Two model variants via `enable_jumps`:
  - False (default): Pure diffusion — suitable for normal / calm regimes
  - True: Jump-diffusion — captures sudden vol spikes (crisis / flash crash)

References:
  - Kidger et al. (2021): Neural SDEs as Infinite-Dimensional GANs
  - Cuchiero et al. (2023): Signature SDEs from an affine and polynomial perspective
  - Merton (1976): Option pricing when underlying stock returns are discontinuous
"""

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
        input_dim = sig_dim + 1

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


class JumpParams(eqx.Module):
    """Learnable jump process parameters (Merton 1976)."""
    log_lambda: jnp.ndarray   # log(jump intensity) — log-space for positivity
    mu_j: jnp.ndarray         # mean jump size in log-variance
    log_sigma_j: jnp.ndarray  # log(jump vol) — log-space for positivity

    def __init__(self, key: jax.random.PRNGKey,
                 lambda_init: float = 2.0,
                 mu_j_init: float = 0.5,
                 sigma_j_init: float = 0.3):
        self.log_lambda = jnp.array(jnp.log(lambda_init))
        self.mu_j = jnp.array(mu_j_init)
        self.log_sigma_j = jnp.array(jnp.log(sigma_j_init))

    @property
    def intensity(self) -> jnp.ndarray:
        return jnp.exp(self.log_lambda)

    @property
    def sigma_j(self) -> jnp.ndarray:
        return jnp.exp(self.log_sigma_j)


class NeuralRoughSimulator(eqx.Module):
    """
    Neural SDE with running path signature conditioning.

    At each time step, drift and diffusion networks receive the running
    signature Sig(X_{0:t}) and current log-variance, making the model
    genuinely path-dependent (non-Markovian).

    Optionally includes a compound Poisson jump component for
    modeling sudden volatility spikes (flash crashes, geopolitical events).
    """
    func: NeuralSDEFunc

    kappa: jnp.ndarray
    theta: jnp.ndarray

    log_v_min: float
    log_v_max: float

    enable_jumps: bool
    jump_params: JumpParams

    def __init__(self, sig_dim: int, key: jax.random.PRNGKey,
                 config_path: str = "config/params.yaml",
                 learn_ou_params: bool = True,
                 enable_jumps: bool = False):
        cfg = _load_neural_sde_config(config_path)

        kappa_init = cfg.get('kappa', 2.72)
        theta_init = cfg.get('theta', -3.5)
        mlp_width = cfg.get('mlp_width', 64)
        mlp_depth = cfg.get('mlp_depth', 3)
        drift_scale = cfg.get('drift_scale', 0.5)
        diffusion_min = cfg.get('diffusion_min', 0.1)
        diffusion_max = cfg.get('diffusion_max', 1.6)

        self.log_v_min = cfg.get('log_v_clip_min', -7.0)
        self.log_v_max = cfg.get('log_v_clip_max', 2.0)
        log_v_center = (self.log_v_min + self.log_v_max) / 2.0
        log_v_scale = (self.log_v_max - self.log_v_min) / 4.0

        k1, k2 = jax.random.split(key)

        self.func = NeuralSDEFunc(
            sig_dim, k1,
            mlp_width=mlp_width, mlp_depth=mlp_depth,
            drift_scale=drift_scale,
            diffusion_min=diffusion_min, diffusion_max=diffusion_max,
            log_v_center=log_v_center, log_v_scale=log_v_scale,
        )

        self.kappa = jnp.array(float(kappa_init))
        self.theta = jnp.array(float(theta_init))

        # Jump component
        self.enable_jumps = enable_jumps
        jump_cfg = cfg.get('jumps', {})
        self.jump_params = JumpParams(
            k2,
            lambda_init=jump_cfg.get('lambda_init', 2.0),
            mu_j_init=jump_cfg.get('mu_j_init', 0.5),
            sigma_j_init=jump_cfg.get('sigma_j_init', 0.3),
        )

    def generate_variance_path(self, init_var, brownian_increments, dt,
                               init_sig_state=None):
        """
        Generate a variance path using the Neural SDE.
        Pure diffusion version (no jumps) for backward compatibility.
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
        _, log_variance_path = jax.lax.scan(scan_fn, init_carry, brownian_increments)

        return jnp.exp(log_variance_path)

    def generate_variance_path_with_jumps(self, init_var, brownian_increments, dt,
                                          jump_key: jax.random.PRNGKey,
                                          init_sig_state=None):
        """
        Generate a variance path with compound Poisson jumps.

        Jump component (Merton 1976):
            At each dt: N ~ Bernoulli(λ·dt), J ~ N(μ_J, σ_J²)
            log V_{t+dt} = log V_t + drift + diffusion + J·N

        The jump compensator (-λ·E[e^J - 1]·dt) is included to maintain
        the correct drift under the chosen measure.
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

        n_steps = brownian_increments.shape[0]
        lam = self.jump_params.intensity
        mu_j = self.jump_params.mu_j
        sig_j = self.jump_params.sigma_j

        k1, k2 = jax.random.split(jump_key)
        jump_occurs = jax.random.bernoulli(k1, lam * dt, (n_steps,)).astype(jnp.float32)
        jump_sizes = mu_j + sig_j * jax.random.normal(k2, (n_steps,))

        jump_compensator = lam * (jnp.exp(mu_j + 0.5 * sig_j ** 2) - 1.0) * dt

        log_v_lo = self.log_v_min
        log_v_hi = self.log_v_max

        def scan_fn(carry, inputs):
            log_v_prev, s1, s2, s3 = carry
            dw_t, j_occur, j_size = inputs

            sig_t = jnp.concatenate([s1, s2, s3])
            drift_nn, diff_nn = self.func(sig_t, log_v_prev)
            drift_nn = jnp.squeeze(drift_nn)
            diff_nn = jnp.squeeze(diff_nn)

            drift_ou = self.kappa * (self.theta - log_v_prev) * dt
            jump_term = j_occur * j_size - jump_compensator
            total_drift = drift_ou + drift_nn * dt + jump_term
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
        inputs = (brownian_increments, jump_occurs, jump_sizes)
        _, log_variance_path = jax.lax.scan(scan_fn, init_carry, inputs)

        return jnp.exp(log_variance_path)

    def generate_variance_path_with_state(self, init_var, brownian_increments, dt,
                                          init_sig_state=None):
        """
        Like generate_variance_path, but also returns the final signature state
        for chaining across blocks (Chevyrev & Kormilitzin, 2016).
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
