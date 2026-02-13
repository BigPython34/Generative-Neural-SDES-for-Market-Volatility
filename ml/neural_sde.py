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

    def __init__(self, sig_dim: int, key: jax.random.PRNGKey, 
                 mlp_width: int = 64, mlp_depth: int = 3,
                 drift_scale: float = 0.5, 
                 diffusion_min: float = 0.1, diffusion_max: float = 1.6):
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

    def __call__(self, signature_t, log_v_prev):
        # Normalize log_v to reasonable range for neural net
        norm_log_v = (log_v_prev + 4.0)
        v_in = jnp.array([norm_log_v])

        net_input = jnp.concatenate([signature_t, v_in])

        # Drift: allow meaningful correction over OU prior
        drift = self.drift_scale * jnn.tanh(self.drift_net(net_input))

        # Diffusion: strictly positive, learned vol-of-vol
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
    which is essential for capturing rough volatility dynamics.
    
    The signature is updated incrementally via Chen's identity:
      S_{0,t+dt} = S_{0,t} ⊗ S_{t,t+dt}
    
    Parameters (κ, θ) are read from config/params.yaml and can optionally
    be made learnable via learn_ou_params=True.
    """
    func: NeuralSDEFunc
    
    # OU prior parameters — read from config, optionally learnable
    kappa: jnp.ndarray   # Mean-reversion speed
    theta: jnp.ndarray   # Long-term log-variance target

    def __init__(self, sig_dim: int, key: jax.random.PRNGKey, 
                 config_path: str = "config/params.yaml",
                 learn_ou_params: bool = True):
        # Load config from YAML
        cfg = _load_neural_sde_config(config_path)
        
        kappa_init = cfg.get('kappa', 2.72)
        theta_init = cfg.get('theta', -3.5)
        mlp_width = cfg.get('mlp_width', 64)
        mlp_depth = cfg.get('mlp_depth', 3)
        drift_scale = cfg.get('drift_scale', 0.5)
        diffusion_min = cfg.get('diffusion_min', 0.1)
        diffusion_max = cfg.get('diffusion_max', 1.6)
        
        self.func = NeuralSDEFunc(
            sig_dim, key, 
            mlp_width=mlp_width, mlp_depth=mlp_depth,
            drift_scale=drift_scale,
            diffusion_min=diffusion_min, diffusion_max=diffusion_max
        )
        
        # Store as arrays so Equinox can track them as learnable parameters
        if learn_ou_params:
            self.kappa = jnp.array(float(kappa_init))
            self.theta = jnp.array(float(theta_init))
        else:
            # Use Python floats — Equinox won't differentiate through these
            self.kappa = jnp.array(float(kappa_init))
            self.theta = jnp.array(float(theta_init))

    def generate_variance_path(self, init_var, brownian_increments, dt):
        """
        Generate a variance path using the Neural SDE with running signatures.
        
        The running signature Sig(X_{0:t}) is computed incrementally at each step
        via Chen's identity and fed to the neural drift/diffusion networks.
        This makes the SDE genuinely path-dependent (non-Markovian).
        
        Args:
            init_var: Initial variance (scalar, e.g. 0.03 for ~17% vol)
            brownian_increments: Brownian increments dW, shape (n_steps,)
            dt: Time step size
            
        Returns:
            Variance path, shape (n_steps,)
        """
        safe_init = jnp.clip(init_var, 0.01, 1.5)
        init_log_var = jnp.log(safe_init)

        # Initialize running signature state for 2D path (time, log_variance)
        # Order 3 with d=2: s1 in R^2, s2 in R^4, s3 in R^8 -> total 14
        d = 2
        s1_init = jnp.zeros(d)
        s2_init = jnp.zeros(d ** 2)
        s3_init = jnp.zeros(d ** 3)

        def scan_fn(carry, dw_t):
            log_v_prev, s1, s2, s3 = carry

            # Current running signature as conditioning vector (14-dim)
            sig_t = jnp.concatenate([s1, s2, s3])

            # Neural drift and diffusion, conditioned on path history
            drift_nn, diff_nn = self.func(sig_t, log_v_prev)
            drift_nn = jnp.squeeze(drift_nn)
            diff_nn = jnp.squeeze(diff_nn)

            # OU mean-reversion prior
            drift_ou = self.kappa * (self.theta - log_v_prev) * dt

            # Total update: OU prior + learned path-dependent correction
            total_drift = drift_ou + drift_nn * dt
            log_v_next = log_v_prev + total_drift + diff_nn * dw_t
            log_v_next = jnp.clip(log_v_next, -5.0, 1.0)

            # --- Update running signature via Chen's identity ---
            # Path increment in (time, log_variance) space
            d_log_v = log_v_next - log_v_prev
            dx = jnp.array([dt, d_log_v])

            # Order 1: S^1 += dx
            new_s1 = s1 + dx

            # Order 2: S^2 += S^1 (x) dx + dx(x)dx / 2
            new_s2 = s2 + jnp.outer(s1, dx).flatten() + 0.5 * jnp.outer(dx, dx).flatten()

            # Order 3: S^3 += S^2 (x) dx + S^1 (x) (dx(x)dx/2) + dx^(x)3 / 6
            new_s3 = (s3
                      + jnp.kron(s2, dx)
                      + jnp.kron(s1, 0.5 * jnp.outer(dx, dx).flatten())
                      + (1.0 / 6.0) * jnp.kron(dx, jnp.kron(dx, dx)))

            return (log_v_next, new_s1, new_s2, new_s3), log_v_next

        init_carry = (init_log_var, s1_init, s2_init, s3_init)
        _, log_variance_path = jax.lax.scan(scan_fn, init_carry, brownian_increments)

        return jnp.exp(log_variance_path)