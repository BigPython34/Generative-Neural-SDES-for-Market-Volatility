"""
Advanced Signature Engine (Research Phase)
==========================================
- Logsignature mode (dimension reduction via Lyndon basis)
- Lead-lag augmentation: (X_t, X_{t-1}) for momentum
- Optional basepoint

Usage (future):
    from engine.signature_advanced import AdvancedSignatureExtractor
    ext = AdvancedSignatureExtractor(
        truncation_order=3,
        representation="logsignature",  # or "signature"
        augmentation="leadlag",         # or "time", "basepoint"
    )
    sigs = ext.get_signature(paths)
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional, Literal

# esig for logsignature (if available)
try:
    import esig
    _HAS_ESIG = True
except ImportError:
    esig = None
    _HAS_ESIG = False


class AdvancedSignatureExtractor:
    """
    Research-grade signature extractor with:
    - representation: "signature" | "logsignature"
    - augmentation: "time" | "leadlag" | "basepoint"
    """

    def __init__(
        self,
        truncation_order: int = 3,
        dt: Optional[float] = None,
        representation: Literal["signature", "logsignature"] = "signature",
        augmentation: Literal["time", "leadlag", "basepoint"] = "time",
        basepoint: Optional[float] = None,
    ):
        self.order = truncation_order
        self.dt = dt
        self.representation = representation
        self.augmentation = augmentation
        self.basepoint = basepoint if basepoint is not None else 0.0

        if representation == "logsignature" and not _HAS_ESIG:
            raise ImportError("logsignature requires esig: pip install esig")

    def _augment_leadlag(self, paths: np.ndarray) -> np.ndarray:
        """Lead-lag: (X_t, X_{t-1}) for 2D path. Shape (n_paths, n_steps, 2*dim)."""
        if paths.ndim == 2:
            paths = paths[:, :, np.newaxis]
        n_paths, n_steps, dim = paths.shape
        lagged = np.roll(paths, 1, axis=1)
        lagged[:, 0] = paths[:, 0]  # first step: no lag
        return np.concatenate([paths, lagged], axis=-1)  # (n_paths, n_steps, 2*dim)

    def _augment_time(self, paths: np.ndarray) -> np.ndarray:
        """Time augmentation: (t, X_t)."""
        n_paths, n_steps = paths.shape[:2]
        if self.dt is not None:
            time_axis = np.arange(n_steps) * self.dt
        else:
            time_axis = np.linspace(0, 1, n_steps)
        time_axis = np.tile(time_axis, (n_paths, 1))
        if paths.ndim == 2:
            return np.stack([time_axis, paths], axis=-1)
        return np.concatenate([time_axis[:, :, np.newaxis], paths], axis=-1)

    def get_signature(self, paths) -> np.ndarray:
        """
        Compute signatures. Uses esig for numpy (logsignature if available).
        JAX path: falls back to standard signature (logsignature TBD).
        """
        if isinstance(paths, (list, np.ndarray)):
            return self._get_signature_numpy(np.asarray(paths))

        # JAX: use standard time-augmented signature (no lead-lag in JAX yet)
        from engine.signature_engine import SignatureFeatureExtractor
        std_ext = SignatureFeatureExtractor(truncation_order=self.order, dt=self.dt)
        return std_ext.get_signature(paths)

    def _get_signature_numpy(self, paths: np.ndarray) -> np.ndarray:
        if paths.ndim == 2:
            paths = paths[:, :, np.newaxis]

        if self.augmentation == "leadlag":
            paths = self._augment_leadlag(paths)
        else:
            paths = self._augment_time(paths.squeeze(-1) if paths.shape[-1] == 1 else paths)

        if self.representation == "logsignature" and _HAS_ESIG:
            stream2logsig = getattr(esig, "stream2logsig", None) or getattr(
                getattr(esig, "tosig", esig), "stream2logsig", None
            )
            if stream2logsig:
                return np.array([stream2logsig(p, self.order) for p in paths])

        # Fallback: standard signature
        stream2sig = None
        if _HAS_ESIG:
            stream2sig = getattr(esig, "stream2sig", None) or getattr(
                getattr(esig, "tosig", esig), "stream2sig", None
            )
        if stream2sig:
            return np.array([stream2sig(p, self.order) for p in paths])

        raise RuntimeError("esig not available for signature computation")

    def get_feature_dim(self, input_dim: int) -> int:
        """Approximate output dimension. Logsignature is smaller than signature."""
        d = input_dim + (1 if self.augmentation == "time" else 0)
        if self.augmentation == "leadlag":
            d = 2 * input_dim
        total = 0
        for k in range(1, self.order + 1):
            total += d ** k
        if self.representation == "logsignature":
            # Logsignature has fewer components (Lyndon basis)
            total = total // 2  # rough upper bound
        return total
