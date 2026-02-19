"""
Legacy Model Loader
===================
Loads models saved with the old NeuralRoughSimulator architecture
(before enable_jumps / JumpParams were added) into the new structure.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from pathlib import Path


def load_legacy_model(model_path, sig_dim: int, config_path: str = "config/params.yaml"):
    """
    Load a model saved with the old architecture (no jumps fields)
    by reconstructing the old pytree shape, deserializing, then
    copying weights into a new-style model.
    """
    from engine.neural_sde import NeuralRoughSimulator, NeuralSDEFunc, JumpParams, _load_neural_sde_config

    model_path = Path(model_path)
    key = jax.random.PRNGKey(0)

    # Build a new-architecture model (with jumps disabled)
    new_model = NeuralRoughSimulator(
        sig_dim, key, config_path=config_path, enable_jumps=False
    )

    # Build an old-architecture skeleton manually:
    # Old model = NeuralSDEFunc + kappa + theta + log_v_min + log_v_max
    # (no enable_jumps, no jump_params)
    cfg = _load_neural_sde_config(config_path)

    old_func = NeuralSDEFunc(
        sig_dim, key,
        mlp_width=cfg.get('mlp_width', 64),
        mlp_depth=cfg.get('mlp_depth', 3),
        drift_scale=cfg.get('drift_scale', 0.5),
        diffusion_min=cfg.get('diffusion_min', 0.1),
        diffusion_max=cfg.get('diffusion_max', 1.6),
        log_v_center=(cfg.get('log_v_clip_min', -7.0) + cfg.get('log_v_clip_max', 2.0)) / 2.0,
        log_v_scale=(cfg.get('log_v_clip_max', 2.0) - cfg.get('log_v_clip_min', -7.0)) / 4.0,
    )

    # The old model was a simple Module with: func, kappa, theta, log_v_min, log_v_max
    class _OldNeuralRoughSimulator(eqx.Module):
        func: NeuralSDEFunc
        kappa: jnp.ndarray
        theta: jnp.ndarray
        log_v_min: float
        log_v_max: float

    old_model = _OldNeuralRoughSimulator(
        func=old_func,
        kappa=jnp.array(float(cfg.get('kappa', 2.72))),
        theta=jnp.array(float(cfg.get('theta', -3.5))),
        log_v_min=cfg.get('log_v_clip_min', -7.0),
        log_v_max=cfg.get('log_v_clip_max', 2.0),
    )

    try:
        loaded_old = eqx.tree_deserialise_leaves(model_path, old_model)
    except Exception:
        return None

    # Transfer weights: replace func, kappa, theta in the new model
    new_model = eqx.tree_at(lambda m: m.func, new_model, loaded_old.func)
    new_model = eqx.tree_at(lambda m: m.kappa, new_model, loaded_old.kappa)
    new_model = eqx.tree_at(lambda m: m.theta, new_model, loaded_old.theta)

    return new_model
