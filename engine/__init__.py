"""
ML Engine: Neural SDE, signatures, training, losses.

Note: Imports are lazily loaded to avoid JAX DLL issues on Windows with
strict security policies. Use explicit imports instead of this __init__.
"""

import sys

def __getattr__(name):
    """Lazy loader for JAX-dependent modules."""
    if name == 'NeuralRoughSimulator' or name == 'NeuralSDEFunc' or name == 'JumpParams':
        from engine.neural_sde import NeuralRoughSimulator, NeuralSDEFunc, JumpParams
        return globals()[name]
    elif name == 'SignatureFeatureExtractor':
        from engine.signature_engine import SignatureFeatureExtractor
        return SignatureFeatureExtractor
    elif name == 'GenerativeTrainer':
        from engine.generative_trainer import GenerativeTrainer
        return GenerativeTrainer
    elif name == 'losses':
        from engine import losses
        return losses
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "NeuralRoughSimulator",
    "NeuralSDEFunc",
    "JumpParams",
    "SignatureFeatureExtractor",
    "GenerativeTrainer",
    "losses",
]
