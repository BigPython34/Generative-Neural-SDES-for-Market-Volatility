from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class PDVParams:
    beta_0: float
    beta_1: float
    beta_2: float


class PDVModel:
    """Lightweight PDV simulator for calibration loops."""

    def __init__(self, params: PDVParams, r: float = 0.0):
        self.params = params
        self.r = float(r)

    def simulate(self, spot: float, T: float, n_paths: int, n_steps: int, seed: int = 42):
        dt = float(T) / max(int(n_steps), 1)
        rng = np.random.default_rng(int(seed))
        z = rng.standard_normal((int(n_paths), int(n_steps)))
        x = np.zeros((int(n_paths), int(n_steps)))
        v = np.zeros((int(n_paths), int(n_steps)))
        s = np.zeros((int(n_paths), int(n_steps) + 1))
        s[:, 0] = float(spot)

        for k in range(int(n_steps)):
            prev = x[:, k - 1] if k > 0 else 0.0
            x[:, k] = prev + np.sqrt(dt) * z[:, k]
            raw_v = self.params.beta_0 + self.params.beta_1 * x[:, k] + self.params.beta_2 * (x[:, k] ** 2)
            v[:, k] = np.clip(raw_v, 1e-8, 4.0)
            drift = (self.r - 0.5 * v[:, k]) * dt
            diff = np.sqrt(v[:, k] * dt) * z[:, k]
            s[:, k + 1] = s[:, k] * np.exp(drift + diff)

        return {"spot_paths": s, "var_paths": v}

