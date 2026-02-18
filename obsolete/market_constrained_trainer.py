"""
Neural SDE Training with Market-Calibrated Parameters

Uses VVIX and VIX Futures data to:
1. Initialize network with market-implied parameters
2. Constrain training to respect term structure
3. Validate against VVIX observations

Author: Market-enhanced pipeline
"""

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.neural_sde import NeuralRoughSimulator
from ml.signature_engine import SignatureFeatureExtractor
from ml.generative_trainer import GenerativeTrainer
from utils.config import load_config


class MarketConstrainedTrainer:
    """
    Train Neural SDE with constraints from VVIX and Futures
    
    Key innovations:
    1. Loss = MMD(signatures) + λ₁|η_model - η_vvix| + λ₂|κ_model - κ_futures|
    2. Initialize network weights to produce market-implied η
    3. Validate term structure consistency
    """
    
    def __init__(self, config_path: str = "config/params.yaml"):
        self.config = load_config(config_path)
        
        # Load market-calibrated parameters
        calib_path = Path("outputs/enhanced_calibration.json")
        if not calib_path.exists():
            calib_path = Path("data/enhanced_calibration.json")
        if calib_path.exists():
            with open(calib_path) as f:
                self.market_params = json.load(f)
            print("Loaded market-calibrated parameters")
        else:
            raise FileNotFoundError("Run enhanced_calibration.py first!")
        
        # Extract key parameters
        params = self.market_params['parameters']
        self.eta_target = self._compute_correct_eta()
        self.kappa_target = params['kappa']
        self.theta_target = params['theta']
        
        print(f"\nTarget Parameters from Market:")
        print(f"   eta (Vol-of-Vol):     {self.eta_target:.2f}")
        print(f"   kappa (Mean Reversion): {self.kappa_target:.2f}")
        print(f"   theta (Long-term VIX):  {np.sqrt(self.theta_target)*100:.1f}%")
        
    def _compute_correct_eta(self) -> float:
        """
        Compute η correctly from VVIX
        
        VVIX = 100 × σ_VIX (annualized vol of VIX in %)
        In our model: dV = η V^α dW → σ_V = η V^(α-1)
        For α=0.5 (rough): σ_V = η / sqrt(V)
        
        So: η = VVIX/100 × sqrt(VIX/100)
        """
        vvix = self.market_params['eta_analysis']['vvix_mean']  # ~96
        vix = self.market_params['eta_analysis']['vix_mean']    # ~18.6
        
        # Convert to decimal
        vvix_dec = vvix / 100  # 0.96
        vix_dec = vix / 100    # 0.186
        
        # η such that vol-of-vol ≈ VVIX
        # η × sqrt(VIX variance) ≈ VVIX
        eta = vvix_dec * np.sqrt(vix_dec)
        
        print(f"\n   VVIX={vvix:.1f}%, VIX={vix:.1f}%")
        print(f"   Computed η = {eta:.3f}")
        
        return eta
    
    def load_vix_data(self) -> np.ndarray:
        """Load VIX paths for training"""
        df = pd.read_csv("data/TVC_VIX, 15.csv")
        df['datetime'] = pd.to_datetime(df['time'], unit='s')
        df = df.set_index('datetime').sort_index()
        
        # Convert to variance (VIX^2 / 100^2)
        variance = (df['close'] / 100) ** 2
        
        # Segment into paths
        n_steps = self.config.get('n_steps', 20)
        n_paths = len(variance) // n_steps
        
        paths = variance.values[:n_paths * n_steps].reshape(n_paths, n_steps)
        
        print(f"\nLoaded {n_paths} VIX variance paths of length {n_steps}")
        return paths
    
    def load_vvix_data(self) -> np.ndarray:
        """Load VVIX for validation"""
        df = pd.read_csv("data/CBOE_DLY_VVIX, 15.csv")
        df['datetime'] = pd.to_datetime(df['time'], unit='s')
        df = df.set_index('datetime').sort_index()
        return df['close'].values / 100  # Convert to decimal
    
    def compute_model_volvol(self, model, paths: jnp.ndarray, dt: float) -> float:
        """
        Compute implied vol-of-vol from model-generated paths
        
        σ_V = std(dV/V) / sqrt(dt) (annualized)
        """
        key = jax.random.PRNGKey(42)
        n_paths, n_steps = paths.shape
        
        # Sample initial conditions from real data
        v0 = paths[:, 0]
        
        # Generate noise
        noise = jax.random.normal(key, (n_paths, n_steps))
        
        # Generate variance paths — model computes running signatures internally
        gen_paths = jax.vmap(model.generate_variance_path, in_axes=(0, 0, None))(
            v0, noise, dt
        )
        
        # Compute vol of vol
        log_var = jnp.log(jnp.clip(gen_paths, 1e-6, 10))
        log_returns = jnp.diff(log_var, axis=1)
        vol_of_vol = jnp.std(log_returns) / jnp.sqrt(dt)
        
        return float(vol_of_vol)
    
    def create_constrained_loss(self, target_sigs, lambda_eta: float = 0.1):
        """
        Create loss function with market constraints
        
        Loss = MMD(sigs) + λ × |η_model - η_target|²
        """
        from ml.losses import signature_mmd_loss
        
        def loss_fn(model, noise, v0, dt, sig_engine):
            # Generate fake paths — model computes running signatures internally
            fake_vars = jax.vmap(model.generate_variance_path, in_axes=(0, 0, None))(
                v0, noise, dt
            )
            
            # Signature MMD loss
            fake_sigs = sig_engine.get_signature(fake_vars)
            mmd_loss = signature_mmd_loss(fake_sigs, target_sigs)
            
            # Vol-of-vol penalty
            log_var = jnp.log(jnp.clip(fake_vars, 1e-6, 10))
            log_returns = jnp.diff(log_var, axis=1)
            model_volvol = jnp.std(log_returns) / jnp.sqrt(dt)
            
            eta_penalty = lambda_eta * (model_volvol - self.eta_target) ** 2
            
            return mmd_loss + eta_penalty, (mmd_loss, model_volvol)
        
        return loss_fn
    
    def train(self, n_epochs: int = 200, batch_size: int = 128) -> tuple:
        """Train with market constraints"""
        print("\n" + "=" * 60)
        print("MARKET-CONSTRAINED NEURAL SDE TRAINING")
        print("=" * 60)
        
        # Load data
        paths = self.load_vix_data()
        paths_jax = jnp.array(paths)
        
        # Signature engine
        sig_order = self.config['neural_sde']['sig_truncation_order']
        sig_engine = SignatureFeatureExtractor(truncation_order=sig_order)
        sig_dim = sig_engine.get_feature_dim(1)
        
        # Compute target signatures (use JAX paths to get 14-dim sigs)
        target_sigs = sig_engine.get_signature(paths_jax)  # JAX array → JAX engine (14 dims)
        
        # Initialize model
        key = jax.random.PRNGKey(0)
        model = NeuralRoughSimulator(sig_dim=sig_dim, key=key)
        
        # Optimizer
        train_cfg = self.config['training']
        scheduler = optax.cosine_decay_schedule(
            init_value=train_cfg['learning_rate_init'],
            decay_steps=train_cfg['decay_steps'],
            alpha=train_cfg['learning_rate_final'] / train_cfg['learning_rate_init']
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(train_cfg['gradient_clip']),
            optax.adam(learning_rate=scheduler)
        )
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
        
        # Training parameters
        dt = (1/12) / paths.shape[1]  # Monthly horizon
        n_paths = paths.shape[0]
        
        # Loss function with constraints
        loss_fn = self.create_constrained_loss(target_sigs, lambda_eta=0.5)
        
        @eqx.filter_jit
        def train_step(model, opt_state, noise, v0):
            (loss, (mmd, volvol)), grads = eqx.filter_value_and_grad(
                lambda m: loss_fn(m, noise, v0, dt, sig_engine),
                has_aux=True
            )(model)
            
            updates, opt_state_new = optimizer.update(
                grads, opt_state, eqx.filter(model, eqx.is_array)
            )
            model_new = eqx.apply_updates(model, updates)
            
            return model_new, opt_state_new, loss, mmd, volvol
        
        # Training loop
        print(f"\nTraining with η target = {self.eta_target:.3f}")
        print("-" * 50)
        
        history = {'loss': [], 'mmd': [], 'volvol': []}
        
        for epoch in range(n_epochs):
            # Sample batch
            key, subkey = jax.random.split(key)
            batch_idx = jax.random.randint(subkey, (batch_size,), 0, n_paths)
            
            # Generate noise
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, (batch_size, paths.shape[1]))
            
            # Initial conditions from data
            v0 = paths_jax[batch_idx, 0]
            
            # Train step
            model, opt_state, loss, mmd, volvol = train_step(
                model, opt_state, noise, v0
            )
            
            history['loss'].append(float(loss))
            history['mmd'].append(float(mmd))
            history['volvol'].append(float(volvol))
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch:04d} | Loss: {loss:.6f} | MMD: {mmd:.6f} | eta: {volvol:.3f} (target: {self.eta_target:.3f})")
        
        # Final validation
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"\n   Final MMD Loss:  {history['mmd'][-1]:.6f}")
        print(f"   Final eta (model): {history['volvol'][-1]:.3f}")
        print(f"   Target eta (VVIX): {self.eta_target:.3f}")
        print(f"   eta Error: {abs(history['volvol'][-1] - self.eta_target):.3f}")
        
        return model, history


def main():
    """Run market-constrained training"""
    trainer = MarketConstrainedTrainer()
    model, history = trainer.train(n_epochs=200, batch_size=128)
    
    # Save training history
    output = {
        'final_mmd': history['mmd'][-1],
        'final_volvol': history['volvol'][-1],
        'target_eta': trainer.eta_target,
        'history': {
            'loss': [float(x) for x in history['loss'][-10:]],
            'volvol': [float(x) for x in history['volvol'][-10:]]
        }
    }
    
    with open("outputs/market_constrained_training.json", 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to outputs/market_constrained_training.json")
    
    return model, history


if __name__ == "__main__":
    model, history = main()
