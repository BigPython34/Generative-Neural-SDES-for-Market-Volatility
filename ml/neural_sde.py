import jax
import jax.numpy as jnp
import equinox as eqx
import jax.nn as jnn

class NeuralSDEFunc(eqx.Module):
    drift_net: eqx.nn.MLP
    diff_net: eqx.nn.MLP

    def __init__(self, sig_dim: int, key: jax.random.PRNGKey):
        k1, k2 = jax.random.split(key)
        input_dim = sig_dim + 1 
        
        self.drift_net = eqx.nn.MLP(
            in_size=input_dim, out_size=1, width_size=32, depth=2, 
            activation=jnn.tanh, key=k1
        )
        self.diff_net = eqx.nn.MLP(
            in_size=input_dim, out_size=1, width_size=32, depth=2, 
            activation=jnn.tanh, key=k2
        )

    def __call__(self, signature_t, log_v_prev):
        # Center the input around 0 (Log-Var -4.0 corresponds to ~1.8% variance)
        norm_log_v = (log_v_prev + 4.0)
        v_in = jnp.array([norm_log_v])
        
        net_input = jnp.concatenate([signature_t, v_in])
        
        # Constrain drift to [-5.0, 5.0] to prevent exponential explosion
        drift = 3.0 * jnn.tanh(self.drift_net(net_input))
        
        # Ensure positive diffusion with a non-zero floor to avoid deterministic collapse
        raw_diff = self.diff_net(net_input)
        diffusion = 1.0 * jnn.sigmoid(raw_diff) + 0.1
        
        return drift, diffusion

class NeuralRoughSimulator(eqx.Module):
    func: NeuralSDEFunc
    
    def __init__(self, sig_dim: int, key: jax.random.PRNGKey):
        self.func = NeuralSDEFunc(sig_dim, key)

    def generate_variance_path(self, init_var, signatures, brownian_increments, dt):
        # Clip initial condition to a realistic range [0.5%, 20%]
        safe_init = jnp.clip(init_var, 0.005, 0.2)
        init_log_var = jnp.log(safe_init)

        def scan_fn(log_v_prev, inputs):
            sig_t, dw_t = inputs
            
            drift, diff = self.func(sig_t, log_v_prev)
            
            drift = jnp.squeeze(drift)
            diff = jnp.squeeze(diff)
            
            # Euler-Maruyama step in Log-Space
            log_v_next = log_v_prev + drift * dt + diff * dw_t
            
            # Hard constraints:
            # -7.0 corresponds to ~0.0009 variance (floor)
            # -0.5 corresponds to ~0.60 variance (ceiling)
            log_v_next = jnp.clip(log_v_next, -7.0, -0.5)
            
            return log_v_next, log_v_next

        n_steps = brownian_increments.shape[0]
        # Broadcast global signature context to every time step
        signatures_repeated = jnp.tile(signatures, (n_steps, 1))
        
        _, log_variance_path = jax.lax.scan(scan_fn, init_log_var, (signatures_repeated, brownian_increments))
        
        return jnp.exp(log_variance_path)