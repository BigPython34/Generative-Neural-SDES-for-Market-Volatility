"""
LossStrategy / Modular Loss Architecture (Research Phase)
==========================================================
Interface for composable loss strategies.
Config-driven activation and weighting via params.yaml.

Usage:
    from engine.loss_strategy import LossStrategy, CombinedLoss, KernelMMDLoss
    strategy = CombinedLoss([
        (KernelMMDLoss(), 1.0),
        (MeanPenaltyLoss(), 10.0),
    ])
    loss = strategy.compute(fake_sigs, real_sigs, fake_paths, real_mean)
"""

import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

from engine.losses import kernel_mmd_loss, signature_mmd_loss, mean_penalty_loss, marginal_mean_penalty_loss


class LossStrategy(ABC):
    """Base interface for loss computation."""

    @abstractmethod
    def compute(self, fake_signatures: jnp.ndarray = None, real_signatures: jnp.ndarray = None,
                fake_paths: jnp.ndarray = None, real_mean: float = None,
                sig_std: jnp.ndarray = None, **kwargs) -> jnp.ndarray:
        """Compute loss. Override with required inputs."""
        pass


class KernelMMDLoss(LossStrategy):
    """Kernel MMD on signatures (full distribution)."""

    def __init__(self, n_bandwidths: int = 5):
        self.n_bandwidths = n_bandwidths

    def compute(self, fake_signatures=None, real_signatures=None, sig_std=None, **kwargs):
        return kernel_mmd_loss(
            fake_signatures, real_signatures,
            sig_std=sig_std, n_bandwidths=self.n_bandwidths
        )


class SignatureMMDLoss(LossStrategy):
    """Legacy mean-signature L2."""

    def compute(self, fake_signatures=None, real_signatures=None, sig_std=None, **kwargs):
        return signature_mmd_loss(fake_signatures, real_signatures, sig_std=sig_std)


class MeanPenaltyLoss(LossStrategy):
    """Jensen correction: penalize deviation of E[exp(log_v)] from target."""

    def compute(self, fake_paths=None, real_mean=None, **kwargs):
        return mean_penalty_loss(fake_paths, real_mean)


class MarginalMeanPenaltyLoss(LossStrategy):
    """Per-step marginal mean matching (Bayer & Stemper, 2018 style)."""

    def compute(self, fake_paths=None, real_paths=None, **kwargs):
        return marginal_mean_penalty_loss(fake_paths, real_paths)


class CombinedLoss(LossStrategy):
    """Weighted combination of strategies."""

    def __init__(self, strategies: List[Tuple[LossStrategy, float]]):
        self.strategies = strategies

    def compute(self, **kwargs) -> jnp.ndarray:
        total = jnp.float32(0.0)
        for strat, weight in self.strategies:
            total = total + weight * strat.compute(**kwargs)
        return total


def from_config(config: dict) -> CombinedLoss:
    """
    Build CombinedLoss from YAML config.
    Example:
      loss:
        kernel_mmd: { weight: 1.0, n_bandwidths: 5 }
        mean_penalty: { weight: 10.0 }
    """
    loss_cfg = config.get("loss", config.get("training", {}))
    strategies = []

    if "kernel_mmd" in loss_cfg:
        cfg = loss_cfg["kernel_mmd"]
        strategies.append((KernelMMDLoss(n_bandwidths=cfg.get("n_bandwidths", 5)),
                          cfg.get("weight", 1.0)))
    if "signature_mmd" in loss_cfg:
        cfg = loss_cfg["signature_mmd"]
        strategies.append((SignatureMMDLoss(), cfg.get("weight", 1.0)))
    if "marginal_mean_penalty" in loss_cfg:
        cfg = loss_cfg["marginal_mean_penalty"]
        strategies.append((MarginalMeanPenaltyLoss(), cfg.get("weight", 10.0)))
    elif "mean_penalty" in loss_cfg:
        cfg = loss_cfg["mean_penalty"]
        strategies.append((MeanPenaltyLoss(), cfg.get("weight", 10.0)))

    if not strategies:
        strategies = [(KernelMMDLoss(), 1.0), (MeanPenaltyLoss(), 10.0)]

    return CombinedLoss(strategies)
