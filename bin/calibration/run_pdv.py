#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from quant.calibration.cli_support import bootstrap_project_root, prepare_spx_targets
from quant.calibration.market_data_vix import assemble_calibration_data
from quant.calibration.pdv.calibrator import PDVCalibrator
from utils.config import load_config


def _get_risk_free_rate(root: Path, cfg: dict) -> float:
    p0 = cfg.get("data", {}).get("sofr_source", "data/market/rates/sofr_daily_nyfed.csv")
    candidates = [
        root / p0,
        root / "data/market/rates/sofr_daily_nyfed.csv",
        root / "data/market/rates/sofr_daily.csv",
    ]
    sofr = None
    for p in candidates:
        if p.exists():
            sofr = pd.read_csv(p)
            break
    if sofr is None:
        raise FileNotFoundError("SOFR file not found. Run data refresh (NY Fed) before calibration.")

    date_col = "date" if "date" in sofr.columns else ("Date" if "Date" in sofr.columns else None)
    rate_col = "rate" if "rate" in sofr.columns else ("Close" if "Close" in sofr.columns else None)
    if date_col is None or rate_col is None:
        raise KeyError("SOFR file must contain date/rate (or Date/Close) columns.")
    sofr[date_col] = pd.to_datetime(sofr[date_col], errors="coerce")
    val = float(pd.to_numeric(sofr[rate_col], errors="coerce").dropna().iloc[-1])
    if not np.isfinite(val):
        raise ValueError("SOFR latest value is NaN/invalid.")
    return val / 100.0


def main():
    root = bootstrap_project_root()
    parser = argparse.ArgumentParser(description="Run PDV calibration on SPX surface")
    parser.add_argument("--as-of", type=str, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vix-mode", type=str, default=None, choices=["indices", "futures", "merged"])
    parser.add_argument("--hard", action="store_true")
    parser.add_argument("--hard-lite", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    cal = cfg.get("calibration", {})
    pdv_cfg = dict(cal.get("pdv", {}))
    if args.vix_mode:
        pdv_cfg.setdefault("vix_targets", {})["mode"] = args.vix_mode

    if args.hard:
        pdv_cfg["n_mc_paths"] = 150_000
        pdv_cfg["n_mc_paths_final"] = 1_000_000
        pdv_cfg["smile_loss_mode"] = "iv"
        pdv_cfg.setdefault("nelder_mead", {}).update({"maxiter": 2800, "xatol": 2e-4, "fatol": 5e-8})
        print("[HARD MODE] Long NM preset enabled.")
    elif args.hard_lite:
        pdv_cfg["n_mc_paths"] = 80_000
        pdv_cfg["n_mc_paths_final"] = 700_000
        pdv_cfg["smile_loss_mode"] = "iv"
        pdv_cfg.setdefault("nelder_mead", {}).update({"maxiter": 1600, "xatol": 3e-4, "fatol": 8e-8})
        print("[HARD-LITE MODE] Long NM preset enabled.")

    mkt = assemble_calibration_data(as_of_date=args.as_of, verbose=True)
    if mkt.spx_surface is None or len(mkt.spx_surface) == 0:
        raise RuntimeError("SPX options surface not available in cache.")
    spot = float(mkt.spx_spot)
    r = _get_risk_free_rate(root, cfg)
    print(f"Risk-free rate (SOFR): {r:.4f}")
    print(f"Smile loss mode: {pdv_cfg.get('smile_loss_mode', 'iv')}")

    slices = prepare_spx_targets(
        mkt.spx_surface,
        spot,
        quick=args.quick,
        max_strikes=int(pdv_cfg.get("max_strikes_per_maturity", cal.get("max_strikes_per_maturity", 12))),
        include_itm=bool(pdv_cfg.get("include_itm", cal.get("include_itm", True))),
        moneyness_basis=str(cal.get("moneyness_basis", "spot")),
        moneyness_tenor_widen_alpha=float(cal.get("moneyness_tenor_widen_alpha", 0.0)),
        dividend_yield=float(cal.get("dividend_yield", 0.0)),
    )
    vix_targets = {}
    if mkt.vix_term_structure is not None:
        for t, v in zip(mkt.vix_term_structure.tenors_days, mkt.vix_term_structure.vix_levels):
            vix_targets[int(t)] = float(v)

    calib = PDVCalibrator(
        spx_slices=slices,
        vix_targets=vix_targets,
        spot=spot,
        risk_free_rate=r,
        cfg=pdv_cfg,
    )
    result = calib.calibrate(quick=args.quick)
    out = {
        "calibrated_params": result.params,
        "losses": {
            "total": result.total_loss,
            "smile": result.smile_loss,
            "vix": result.vix_loss,
            "martingale": result.martingale_loss,
        },
        "diagnostics": {"elapsed_seconds": result.elapsed_seconds, "method": result.method},
    }
    out_dir = root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pdv_calibration.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved: {out_dir / 'pdv_calibration.json'}")


if __name__ == "__main__":
    main()

