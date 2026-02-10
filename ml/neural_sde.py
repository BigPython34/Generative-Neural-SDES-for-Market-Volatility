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
            in_size=input_dim, out_size=1, width_size=64, depth=3, 
            activation=jnn.tanh, key=k1
        )
        self.diff_net = eqx.nn.MLP(
            in_size=input_dim, out_size=1, width_size=64, depth=3, 
            activation=jnn.tanh, key=k2
        )

    def __call__(self, signature_t, log_v_prev):
        norm_log_v = (log_v_prev + 4.0)
        v_in = jnp.array([norm_log_v])
        
        net_input = jnp.concatenate([signature_t, v_in])
        
        drift = 0.2 * jnn.tanh(self.drift_net(net_input))
        raw_diff = self.diff_net(net_input)
        diffusion = 1.5 * jnn.sigmoid(raw_diff) + 0.1
        
        return drift, diffusion

class NeuralRoughSimulator(eqx.Module):
    func: NeuralSDEFunc
    
    def __init__(self, sig_dim: int, key: jax.random.PRNGKey):
        self.func = NeuralSDEFunc(sig_dim, key)

    def generate_variance_path(self, init_var, signatures, brownian_increments, dt):
        safe_init = jnp.clip(init_var, 0.01, 1.5)
        init_log_var = jnp.log(safe_init)

        def scan_fn(log_v_prev, inputs):
            sig_t, dw_t = inputs
            
            drift_nn, diff_nn = self.func(sig_t, log_v_prev)
            
            drift_nn = jnp.squeeze(drift_nn)
            diff_nn = jnp.squeeze(diff_nn)

            # Mean-reversion prior
            theta = -1.6
            kappa = 0.15
            drift_phy = kappa * (theta - log_v_prev)
            
            total_drift = drift_phy + drift_nn
            log_v_next = log_v_prev + total_drift * dt + diff_nn * dw_t
            log_v_next = jnp.clip(log_v_next, -5.0, 1.0)
            
            return log_v_next, log_v_next

        n_steps = brownian_increments.shape[0]
        signatures_repeated = jnp.tile(signatures, (n_steps, 1))
        
        _, log_variance_path = jax.lax.scan(scan_fn, init_log_var, (signatures_repeated, brownian_increments))
        
        return jnp.exp(log_variance_path)