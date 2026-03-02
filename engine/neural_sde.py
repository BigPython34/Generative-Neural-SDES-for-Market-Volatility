"""
Neural SDE with Path Signature Conditioning
============================================
Signature-conditioned Neural SDE for rough volatility modeling.

Two backbone architectures (configurable via neural_sde.backbone):

  1. "ou" (default): OU mean-reversion prior + neural corrections
    dlog(V) = [κ(θ - log V) + f_nn(Sig, log V)] dt
            + g_nn(Sig, log V) dW
            + J dN                              (optional jump component)

  2. "fractional": Volterra (rough) kernel prior + neural corrections
    log(V_t) = η·Ŵ^H_t - ½η²·Var[Ŵ^H_t] + f_nn(Sig, log V)·dt + g_nn(Sig, log V)·dW
    where Ŵ^H is Riemann-Liouville fBM with learnable (H, η).
    This nests rBergomi as a special case (when f_nn = g_nn = 0).

    where Sig = running path signature of (t, log V) up to order M,
    f_nn and g_nn are MLPs, and J·dN is a compound Poisson jump process.

Two model variants via `enable_jumps`:
  - False (default): Pure diffusion — suitable for normal / calm regimes
  - True: Jump-diffusion — captures sudden vol spikes (crisis / flash crash)

References:
  - Kidger et al. (2021): Neural SDEs as Infinite-Dimensional GANs
  - Cuchiero et al. (2023): Signature SDEs from an affine and polynomial perspective
  - Bayer, Friz & Gatheral (2016): Pricing under rough volatility
  - Merton (1976): Option pricing when underlying stock returns are discontinuous
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import jax.nn as jnn
import numpy as np
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


class FractionalParams(eqx.Module):
    """
    Learnable parameters for the fractional (Volterra) backbone.
    Allows the Neural SDE to nest rBergomi as a special case.

    The variance follows:
        log V_t = η · Ŵ^H_t - ½η² · Var[Ŵ^H_t]  +  neural corrections

    where Ŵ^H_t = sqrt(2H) ∫₀ᵗ (t-s)^{H-0.5} dW(s)  (RL-fBM).

    When neural corrections → 0, this exactly recovers rBergomi.
    """
    log_H: jnp.ndarray         # log(H) → H ∈ (0, 0.5) via sigmoid
    log_eta: jnp.ndarray       # log(eta) → eta > 0

    def __init__(self, key: jax.random.PRNGKey,
                 H_init: float = 0.10,
                 eta_init: float = 1.9):
        # Store in logit/log space for unconstrained optimization
        # H ∈ (0, 0.5): use logit(H / 0.5)
        self.log_H = jnp.array(jnp.log(H_init / (0.5 - H_init)))  # logit
        self.log_eta = jnp.array(jnp.log(eta_init))

    @property
    def H(self) -> jnp.ndarray:
        """Hurst exponent, constrained to (0, 0.5)."""
        return 0.5 * jax.nn.sigmoid(self.log_H)

    @property
    def eta(self) -> jnp.ndarray:
        """Vol-of-vol, constrained > 0."""
        return jnp.exp(self.log_eta)


def _build_volterra_kernel_jax(n_steps: int, dt: float, H: jnp.ndarray):
    """
    Build Volterra kernel for RL-fBM — differentiable w.r.t. H.

    A[j,k] = sqrt(2H) · dt^H / (H + 0.5) · [(j-k+1)^{H+0.5} - (j-k)^{H+0.5}]

    Returns (A, var_wh) where var_wh[j] = Σ_k A[j,k]².

    Note: This is O(N²) in memory. For training with n_steps=120 this is fine.
    """
    C = jnp.sqrt(2 * H) * dt ** H / (H + 0.5)
    alpha = H + 0.5

    # Build indices: j from 0..n-1, k from 0..n-1
    j_idx = jnp.arange(n_steps)[:, None]  # (n, 1)
    k_idx = jnp.arange(n_steps)[None, :]  # (1, n)
    lag = j_idx - k_idx  # (n, n)

    # Lower-triangular mask: A[j,k] = 0 for k > j
    mask = (lag >= 0).astype(jnp.float32)

    # Kernel values (safe: use max(lag, 0) to avoid negative lags)
    safe_lag = jnp.maximum(lag, 0)
    A = C * ((safe_lag + 1) ** alpha - safe_lag ** alpha) * mask

    # Variance of W^H at each step
    var_wh = jnp.sum(A ** 2, axis=1)

    return A, var_wh


class NeuralRoughSimulator(eqx.Module):
    """
    Neural SDE with running path signature conditioning.

    Supports two backbone architectures:

    1. "ou" backbone (default):
       dlog(V) = [κ(θ - log V) + f_nn(Sig)] dt + g_nn(Sig) dW
       Good for: fast training, P-measure / stress testing.

    2. "fractional" backbone:
       log(V_t) = η·Ŵ^H_t - ½η²·Var[Ŵ^H_t] + Σ f_nn·dt + Σ g_nn·dW
       The Volterra kernel makes this a strict generalization of rBergomi.
       Good for: Q-measure pricing, smile calibration.

    In both cases, the neural networks receive the running signature
    Sig(X_{0:t}) and current log-variance, making the model
    genuinely path-dependent (non-Markovian).
    """
    func: NeuralSDEFunc

    # OU backbone params
    kappa: jnp.ndarray
    theta: jnp.ndarray

    log_v_min: float
    log_v_max: float

    # Backbone selection
    backbone: str

    # Fractional backbone params
    fractional_params: FractionalParams

    enable_jumps: bool
    jump_params: JumpParams

    def __init__(self, sig_dim: int, key: jax.random.PRNGKey,
                 config_path: str = "config/params.yaml",
                 learn_ou_params: bool = True,
                 enable_jumps: bool = False):
        cfg = _load_neural_sde_config(config_path)

        self.backbone = cfg.get('backbone', 'ou')

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

        k1, k2, k3 = jax.random.split(key, 3)

        self.func = NeuralSDEFunc(
            sig_dim, k1,
            mlp_width=mlp_width, mlp_depth=mlp_depth,
            drift_scale=drift_scale,
            diffusion_min=diffusion_min, diffusion_max=diffusion_max,
            log_v_center=log_v_center, log_v_scale=log_v_scale,
        )

        self.kappa = jnp.array(float(kappa_init))
        self.theta = jnp.array(float(theta_init))

        # Fractional backbone params
        frac_cfg = cfg.get('fractional', {})
        self.fractional_params = FractionalParams(
            k3,
            H_init=frac_cfg.get('hurst_init', 0.10),
            eta_init=frac_cfg.get('eta_init', 1.9),
        )

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
        Dispatches to OU or fractional backbone based on self.backbone.
        """
        if self.backbone == 'fractional':
            return self._generate_fractional_path(init_var, brownian_increments, dt, init_sig_state)
        return self._generate_ou_path(init_var, brownian_increments, dt, init_sig_state)

    def _generate_fractional_path(self, init_var, brownian_increments, dt,
                                   init_sig_state=None):
        """
        Fractional backbone: rBergomi + neural corrections.

        log V_t = η·Ŵ^H_t - ½η²·Var[Ŵ^H_t] + Σ_s<t [f_nn·ds + g_nn·dW_s]

        The first two terms are the exact rBergomi variance process.
        The neural correction terms allow departure from rBergomi when
        the data justifies it. When f_nn = g_nn = 0, this IS rBergomi.
        """
        n_steps = brownian_increments.shape[0]
        H = self.fractional_params.H
        eta = self.fractional_params.eta

        # Build differentiable Volterra kernel
        A, var_wh = _build_volterra_kernel_jax(n_steps, dt, H)

        # Compute RL-fBM from the SAME Brownian increments
        # dW = brownian_increments (already scaled by sqrt(dt))
        Z = brownian_increments / jnp.sqrt(dt)  # standardize
        Wh = A @ Z  # (n_steps,) — RL-fBM values

        # rBergomi log-variance baseline
        log_v_bergomi = jnp.log(jnp.maximum(init_var, 1e-10)) + eta * Wh - 0.5 * eta ** 2 * var_wh

        # Neural corrections via scan (captures non-Markovian effects beyond fBM)
        d = 2
        if init_sig_state is not None:
            s1_init, s2_init, s3_init = init_sig_state
        else:
            s1_init = jnp.zeros(d)
            s2_init = jnp.zeros(d ** 2)
            s3_init = jnp.zeros(d ** 3)

        log_v_lo = self.log_v_min
        log_v_hi = self.log_v_max

        def scan_fn(carry, inputs):
            correction_accum, s1, s2, s3 = carry
            dw_t, log_v_base = inputs

            # Current log-V = base (Bergomi) + accumulated neural correction
            log_v_current = jnp.clip(log_v_base + correction_accum, log_v_lo, log_v_hi)

            sig_t = jnp.concatenate([s1, s2, s3])
            drift_nn, diff_nn = self.func(sig_t, log_v_current)
            drift_nn = jnp.squeeze(drift_nn)
            diff_nn = jnp.squeeze(diff_nn)

            # Neural correction increment
            d_correction = drift_nn * dt + (diff_nn - eta) * dw_t
            new_correction = correction_accum + d_correction

            log_v_next = jnp.clip(log_v_base + new_correction, log_v_lo, log_v_hi)

            # Signature update
            d_log_v = log_v_next - log_v_current
            dx = jnp.array([dt, d_log_v])
            new_s1 = s1 + dx
            new_s2 = s2 + jnp.outer(s1, dx).flatten() + 0.5 * jnp.outer(dx, dx).flatten()
            new_s3 = (s3
                      + jnp.kron(s2, dx)
                      + jnp.kron(s1, 0.5 * jnp.outer(dx, dx).flatten())
                      + (1.0 / 6.0) * jnp.kron(dx, jnp.kron(dx, dx)))

            return (new_correction, new_s1, new_s2, new_s3), log_v_next

        init_carry = (jnp.float32(0.0), s1_init, s2_init, s3_init)
        _, log_variance_path = jax.lax.scan(
            scan_fn, init_carry, (brownian_increments, log_v_bergomi)
        )

        return jnp.exp(log_variance_path)

    def _generate_ou_path(self, init_var, brownian_increments, dt,
                           init_sig_state=None):
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
        Supports both OU and fractional backbones.
        """
        if self.backbone == 'fractional':
            return self._generate_fractional_path_with_state(
                init_var, brownian_increments, dt, init_sig_state)
        return self._generate_ou_path_with_state(
            init_var, brownian_increments, dt, init_sig_state)

    def _generate_fractional_path_with_state(self, init_var, brownian_increments, dt,
                                              init_sig_state=None):
        """Fractional backbone variant that returns final signature state."""
        n_steps = brownian_increments.shape[0]
        H = self.fractional_params.H
        eta = self.fractional_params.eta

        A, var_wh = _build_volterra_kernel_jax(n_steps, dt, H)
        Z = brownian_increments / jnp.sqrt(dt)
        Wh = A @ Z
        log_v_bergomi = jnp.log(jnp.maximum(init_var, 1e-10)) + eta * Wh - 0.5 * eta ** 2 * var_wh

        d = 2
        if init_sig_state is not None:
            s1_init, s2_init, s3_init = init_sig_state
        else:
            s1_init = jnp.zeros(d)
            s2_init = jnp.zeros(d ** 2)
            s3_init = jnp.zeros(d ** 3)

        log_v_lo = self.log_v_min
        log_v_hi = self.log_v_max

        def scan_fn(carry, inputs):
            correction_accum, s1, s2, s3 = carry
            dw_t, log_v_base = inputs
            log_v_current = jnp.clip(log_v_base + correction_accum, log_v_lo, log_v_hi)
            sig_t = jnp.concatenate([s1, s2, s3])
            drift_nn, diff_nn = self.func(sig_t, log_v_current)
            drift_nn = jnp.squeeze(drift_nn)
            diff_nn = jnp.squeeze(diff_nn)
            d_correction = drift_nn * dt + (diff_nn - eta) * dw_t
            new_correction = correction_accum + d_correction
            log_v_next = jnp.clip(log_v_base + new_correction, log_v_lo, log_v_hi)
            d_log_v = log_v_next - log_v_current
            dx = jnp.array([dt, d_log_v])
            new_s1 = s1 + dx
            new_s2 = s2 + jnp.outer(s1, dx).flatten() + 0.5 * jnp.outer(dx, dx).flatten()
            new_s3 = (s3
                      + jnp.kron(s2, dx)
                      + jnp.kron(s1, 0.5 * jnp.outer(dx, dx).flatten())
                      + (1.0 / 6.0) * jnp.kron(dx, jnp.kron(dx, dx)))
            return (new_correction, new_s1, new_s2, new_s3), log_v_next

        init_carry = (jnp.float32(0.0), s1_init, s2_init, s3_init)
        final_carry, log_variance_path = jax.lax.scan(
            scan_fn, init_carry, (brownian_increments, log_v_bergomi)
        )
        _, final_s1, final_s2, final_s3 = final_carry
        return jnp.exp(log_variance_path), (final_s1, final_s2, final_s3)

    def _generate_ou_path_with_state(self, init_var, brownian_increments, dt,
                                      init_sig_state=None):
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
