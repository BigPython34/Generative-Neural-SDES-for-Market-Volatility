"""
Multi-Measure Training Script
==============================
Trains Neural SDE models under both P and Q measures.

Usage:
    python bin/train_multi.py                    # Train both P and Q
    python bin/train_multi.py --measure P        # P-measure only
    python bin/train_multi.py --measure Q        # Q-measure only
    python bin/train_multi.py --jumps            # Enable jump component
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

from utils.config import load_config
from engine.generative_trainer import GenerativeTrainer


def train_model(measure: str, enable_jumps: bool = False):
    """Train a single model under the specified measure."""
    cfg = load_config()
    config = {
        'n_steps': cfg['simulation']['n_steps'],
        'T': cfg['simulation']['T'],
    }

    jump_str = " + jumps" if enable_jumps else ""
    print(f"\n{'='*60}")
    print(f"   Training Neural SDE [{measure}-measure]{jump_str}")
    print(f"{'='*60}")

    # Override data_type for P-measure
    if measure == 'P':
        cfg['data']['data_type'] = 'realized_vol'

    trainer = GenerativeTrainer(config, measure=measure)
    model = trainer.run(
        n_epochs=cfg['training']['n_epochs'],
        batch_size=cfg['training']['batch_size'],
        enable_jumps=enable_jumps,
    )

    print(f"\n[{measure}-measure] Training complete.")
    print(f"   Model saved to models/neural_sde_best_{measure.lower()}.eqx")

    return model


def main():
    parser = argparse.ArgumentParser(description="Multi-measure Neural SDE training")
    parser.add_argument("--measure", choices=["P", "Q", "both"], default="both",
                        help="Probability measure (P=physical, Q=risk-neutral, both)")
    parser.add_argument("--jumps", action="store_true",
                        help="Enable jump-diffusion component")
    args = parser.parse_args()

    measures = ["P", "Q"] if args.measure == "both" else [args.measure]

    models = {}
    for m in measures:
        models[m] = train_model(m, enable_jumps=args.jumps)

    print(f"\n{'='*60}")
    print(f"   ALL TRAINING COMPLETE")
    print(f"{'='*60}")
    for m in measures:
        suffix = f"_{m.lower()}"
        if args.jumps:
            suffix += "_jump"
        print(f"   [{m}-measure]: models/neural_sde_best{suffix}.eqx")

    print(f"\nUsage guide:")
    if 'P' in measures:
        print(f"   P-measure model → VaR, stress testing, vol forecasting")
    if 'Q' in measures:
        print(f"   Q-measure model → Option pricing, hedging, calibration")

    return models


if __name__ == "__main__":
    main()
