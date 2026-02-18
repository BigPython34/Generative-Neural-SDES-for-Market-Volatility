"""
Neural SDE Training Entrypoint
==============================
Trains the signature-conditioned Neural SDE on market variance paths.
Uses config/params.yaml for all hyperparameters.
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from utils.config import load_config
from engine.generative_trainer import GenerativeTrainer


def main():
    cfg = load_config()
    config = {
        'n_steps': cfg['simulation']['n_steps'],
        'T': cfg['simulation']['T'],
    }
    trainer = GenerativeTrainer(config)
    model = trainer.run(
        n_epochs=cfg['training']['n_epochs'],
        batch_size=cfg['training']['batch_size'],
    )
    print("Training complete. Model saved to models/neural_sde_best.eqx")
    return model


if __name__ == "__main__":
    main()
