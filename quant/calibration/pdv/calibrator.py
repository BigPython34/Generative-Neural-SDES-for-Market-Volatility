from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np
from scipy.optimize import minimize
from quant.models.black_scholes import BlackScholes
from .model import PDVModel, PDVParams


@dataclass
class PDVResult:
    params: dict
    total_loss: float
    smile_loss: float
    vix_loss: float
    martingale_loss: float
    model_ivs: dict
    elapsed_seconds: float
    method: str


class PDVCalibrator:
    def __init__(self, spx_slices, vix_targets, spot, risk_free_rate, cfg):
        self.spx_slices = spx_slices
        self.vix_targets = vix_targets or {}
        self.spot = float(spot)
        self.r = float(risk_free_rate)
        self.cfg = cfg
        self.w_smile = float(cfg.get("w_smile", 1.0))
        self.w_vix = float(cfg.get("w_vix", 1.0))
        self.w_vix_smooth = float(cfg.get("w_vix_smooth", 0.0))
        self.w_martingale = float(cfg.get("w_martingale", 0.0))
        self.n_steps_per_year = int(cfg.get("n_steps_per_year", 365))
        self.n_mc_paths = int(cfg.get("n_mc_paths", 100_000))
        self.smile_loss_mode = str(cfg.get("smile_loss_mode", "iv")).lower()
        self.min_option_price = float(cfg.get("min_option_price", 0.5))
        self.max_spread_pct = float(cfg.get("max_spread_pct", 0.5))
        self.max_rel_error_clip = float(cfg.get("max_rel_error_clip", 2.0))
        self._eval_count = 0

    def _evaluate(self, x, n_paths, seed=42, full_iv=False):
        self._eval_count += 1
        model = PDVModel(PDVParams(*x), r=self.r)
        smile_losses = []
        iv_by_mat = {}

        for sl in self.spx_slices:
            T = float(sl["T"])
            sim = model.simulate(self.spot, T, n_paths=n_paths, n_steps=max(2, int(self.n_steps_per_year * T)), seed=seed)
            ST = sim["spot_paths"][:, -1]
            disc = np.exp(-self.r * T)
            strikes = np.asarray(sl["strikes"], dtype=float)
            market_ivs = np.asarray(sl["market_ivs"], dtype=float)
            option_types = sl["option_types"]

            prices = []
            model_ivs = []
            for i, K in enumerate(strikes):
                if option_types[i] == "call":
                    payoff = np.maximum(ST - K, 0.0)
                else:
                    payoff = np.maximum(K - ST, 0.0)
                p = float(disc * np.mean(payoff))
                prices.append(p)
                iv = BlackScholes.implied_vol(p, self.spot, float(K), T, self.r, option_types[i])
                model_ivs.append(np.nan if not np.isfinite(iv) else iv)
            prices = np.asarray(prices, dtype=float)
            model_ivs = np.asarray(model_ivs, dtype=float)

            valid = np.isfinite(prices) & np.isfinite(market_ivs) & (prices > 1e-8)
            mids = np.asarray(sl.get("market_mids", []), dtype=float) if "market_mids" in sl else None
            spd = np.asarray(sl.get("market_spread_pct", []), dtype=float) if "market_spread_pct" in sl else None
            if mids is not None and mids.size == prices.size:
                valid &= np.isfinite(mids) & (mids > self.min_option_price)
            if spd is not None and spd.size == prices.size:
                valid &= (~np.isfinite(spd)) | (spd <= self.max_spread_pct)

            if valid.sum() < 2:
                continue

            if self.smile_loss_mode == "iv":
                mask = valid & np.isfinite(model_ivs)
                if mask.sum() >= 2:
                    rel = (model_ivs[mask] - market_ivs[mask]) / np.maximum(market_ivs[mask], 1e-4)
                    smile_losses.append(float(np.mean(np.clip(rel, -self.max_rel_error_clip, self.max_rel_error_clip) ** 2)))
            else:
                rel = prices[valid] / np.maximum(mids[valid], 1e-8) - 1.0 if mids is not None and mids.size == prices.size else (prices[valid] / np.maximum(prices[valid], 1e-8) - 1.0)
                smile_losses.append(float(np.mean(np.clip(rel, -self.max_rel_error_clip, self.max_rel_error_clip) ** 2)))
            if full_iv:
                iv_by_mat[int(sl.get("dte", round(T * 365)))] = model_ivs.tolist()

        smile_loss = float(np.mean(smile_losses)) if smile_losses else 1e3
        vix_loss = self._vix_loss(model, x, n_paths=max(2000, n_paths // 8), seed=seed)
        mart = self._martingale_loss(model, n_paths=max(5000, n_paths // 4), seed=seed)
        total = self.w_smile * smile_loss + self.w_vix * vix_loss + self.w_martingale * mart
        print(f"    eval {self._eval_count:>4d}  loss={total:.6f}")
        return total, smile_loss, vix_loss, mart, iv_by_mat

    def _vix_loss(self, model, x, n_paths, seed):
        if not self.vix_targets:
            return 0.0
        errs = []
        tenor_vals = []
        for days, vix_mkt in sorted(self.vix_targets.items()):
            T = float(days) / 365.0
            sim = model.simulate(self.spot, T, n_paths=n_paths, n_steps=max(2, int(self.n_steps_per_year * T)), seed=seed + int(days))
            v = sim["var_paths"]
            vix_model = 100.0 * float(np.sqrt(np.mean(v)))
            tenor_vals.append(vix_model)
            errs.append((vix_model - float(vix_mkt)) ** 2)
        smooth = 0.0
        if len(tenor_vals) >= 3 and self.w_vix_smooth > 0:
            t = np.asarray(tenor_vals, dtype=float)
            smooth = float(np.mean((t[2:] - 2 * t[1:-1] + t[:-2]) ** 2))
        return float(np.mean(errs)) + self.w_vix_smooth * smooth

    def _martingale_loss(self, model, n_paths, seed):
        T = max(float(s["T"]) for s in self.spx_slices) if self.spx_slices else 30.0 / 365.0
        sim = model.simulate(self.spot, T, n_paths=n_paths, n_steps=max(2, int(self.n_steps_per_year * T)), seed=seed + 777)
        lhs = np.exp(-self.r * T) * np.mean(sim["spot_paths"][:, -1]) / self.spot
        return float((lhs - 1.0) ** 2)

    def calibrate(self, quick=False):
        t0 = time.time()
        grid = self.cfg.get("grid", {})
        beta0 = grid.get("beta_0", [0.02, 0.03, 0.04])
        beta1 = grid.get("beta_1", [-0.10, -0.07, -0.04])
        beta2 = grid.get("beta_2", [0.70, 0.85, 1.00])

        best = None
        best_loss = np.inf
        print("[PDV] coarse search...")
        for b0 in beta0:
            for b1 in beta1:
                for b2 in beta2:
                    loss, *_ = self._evaluate(np.array([b0, b1, b2], dtype=float), n_paths=max(4000, self.n_mc_paths // 10), seed=42)
                    if loss < best_loss:
                        best_loss = loss
                        best = np.array([b0, b1, b2], dtype=float)

        nm = self.cfg.get("nelder_mead", {})
        maxiter = int(nm.get("maxiter", 500 if not quick else 120))
        print("[PDV] Nelder-Mead refine...")
        res = minimize(
            lambda z: self._evaluate(np.asarray(z, dtype=float), n_paths=self.n_mc_paths, seed=123)[0],
            best,
            method="Nelder-Mead",
            options={"maxiter": maxiter, "xatol": float(nm.get("xatol", 5e-4)), "fatol": float(nm.get("fatol", 1e-7))},
        )

        total, smile, vix, mart, ivs = self._evaluate(np.asarray(res.x, dtype=float), n_paths=int(self.cfg.get("n_mc_paths_final", self.n_mc_paths)), seed=2026, full_iv=True)
        return PDVResult(
            params={"beta_0": float(res.x[0]), "beta_1": float(res.x[1]), "beta_2": float(res.x[2])},
            total_loss=float(total),
            smile_loss=float(smile),
            vix_loss=float(vix),
            martingale_loss=float(mart),
            model_ivs=ivs,
            elapsed_seconds=float(time.time() - t0),
            method="nelder-mead",
        )

