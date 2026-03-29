"""
Multi-Measure Training Script
==============================
Trains Neural SDE models under P

Usage:
    python bin/train_multi.py                    # Train P
    python bin/train_multi.py --jumps            # Enable jump component
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

from utils.config import load_config


def train_model(enable_jumps: bool = False):
    """Train a single model under the specified measure.
    
    Note: GenerativeTrainer import is deferred to avoid JAX DLL issues
    on Windows systems with strict security policies.
    """
    # Deferred import to avoid JAX DLL blocking at module load time
    from engine.generative_trainer import GenerativeTrainer
    
    cfg = load_config()
    config = {
        'n_steps': cfg['simulation']['n_steps'],
        'T': cfg['simulation']['T'],
    }

    jump_str = " + jumps" if enable_jumps else ""
    print(f"\n{'='*60}")
    print(f"{'='*60}")

    cfg['data']['data_type'] = 'realized_vol'

    trainer = GenerativeTrainer(config)
    model = trainer.run(
        n_epochs=cfg['training']['n_epochs'],
        batch_size=cfg['training']['batch_size'],
        enable_jumps=enable_jumps,
    )

    print(f"   Model saved to models/neural_sde_best.eqx")

    return model


def main():
    parser = argparse.ArgumentParser(description="Multi-measure Neural SDE training")
    
    parser.add_argument("--jumps", action="store_true",
                        help="Enable jump-diffusion component")
    args = parser.parse_args()


        
    model = train_model( enable_jumps=args.jumps)

    print(f"\n{'='*60}")
    print(f"   ALL TRAINING COMPLETE")
    print(f"{'='*60}")

    print(f"   P-measure model → VaR, stress testing, vol forecasting")
    
    return model


if __name__ == "__main__":
    main()
