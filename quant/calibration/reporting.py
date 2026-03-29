"""Shared calibration reporting helpers."""

from __future__ import annotations

import json
from pathlib import Path


def save_json_report(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str, ensure_ascii=False)
    return path


def update_joint_config(result, config_path: Path) -> None:
    """Write joint calibration outputs back to config/params.yaml."""
    import yaml

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["bergomi"]["hurst"] = round(float(result.H), 4)
    config["bergomi"]["eta"] = round(float(result.eta), 3)
    config["bergomi"]["rho"] = round(float(result.rho), 3)
    config["bergomi"]["xi0"] = round(float(result.xi0_values[0]), 6)

    if "neural_sde" in config and "fractional" in config["neural_sde"]:
        config["neural_sde"]["fractional"]["hurst_init"] = round(float(result.H), 4)
        config["neural_sde"]["fractional"]["eta_init"] = round(float(result.eta), 3)

    with config_path.open("w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def build_joint_report(result) -> dict:
    return {
        "calibrated_params": {
            "H": float(result.H),
            "eta": float(result.eta),
            "rho": float(result.rho),
            "xi0_maturities": result.xi0_maturities.tolist(),
            "xi0_values": result.xi0_values.tolist(),
        },
        "losses": {
            "total": float(result.total_loss),
            "spx": float(result.spx_loss),
            "vix": float(result.vix_loss),
            "martingale": float(result.martingale_loss),
            "spx_rmse_bps": float(result.spx_rmse_bps),
            "spx_shape_rmse_bps": float(result.spx_shape_rmse_bps),
            "spx_atm_bias_bps": float(result.spx_atm_bias_bps),
        },
        "vix_term_structure": {
            "model": {str(k): v for k, v in result.model_vix_ts.items()},
            "market": {str(k): v for k, v in result.market_vix_ts.items()},
        },
        "diagnostics": {
            "n_mc_paths": result.n_mc_paths,
            "n_iterations": result.n_iterations,
            "elapsed_seconds": result.elapsed_seconds,
            "grid_evaluated": result.grid_evaluated,
            "method": result.method,
        },
    }


def build_neural_q_report(result, seed_params: dict, market_summary: dict, vix_futures: dict) -> dict:
    return {
        "method": "neural_sde_q_girsanov",
        "seed_params": {
            "H": float(seed_params["H"]),
            "eta": float(seed_params["eta"]),
            "rho": float(seed_params["rho"]),
        },
        "market_data": market_summary,
        "training": {
            "n_epochs": result.n_epochs,
            "n_paths": result.n_paths,
            "n_girsanov_params": result.n_girsanov_params,
            "elapsed_seconds": round(result.elapsed_seconds, 1),
        },
        "loss_components": {k: float(v) for k, v in result.loss_components.items()},
        "loss_history_last10": [float(x) for x in result.loss_history[-10:]],
        "vix_fit": {
            str(k): {
                "market": result.market_vix_ts.get(k, None),
                "model": result.model_vix_ts.get(k, None),
            }
            for k in sorted(set(result.market_vix_ts.keys()) | set(result.model_vix_ts.keys()))
        },
        "martingale_error": float(result.martingale_error),
        "vix_futures": {str(k): float(v) for k, v in vix_futures.items()},
    }
