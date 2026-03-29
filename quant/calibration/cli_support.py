"""Shared CLI helpers for calibration launchers."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def bootstrap_project_root() -> Path:
    """Ensure the repository root is importable and return it."""
    root = Path(__file__).resolve().parents[2]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    try:
        import sitecustomize  # noqa: F401
    except Exception:
        pass
    return root


def load_previous_joint_result(root: Path | None = None) -> dict[str, float]:
    """Load seed parameters from outputs/joint_calibration.json if available."""
    root = root or bootstrap_project_root()
    result_path = root / "outputs" / "joint_calibration.json"
    if not result_path.exists():
        return {}

    with result_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    params = data.get("calibrated_params", {})
    return {
        "H": float(params.get("H", 0.08)),
        "eta": float(params.get("eta", 1.0)),
        "rho": float(params.get("rho", -0.90)),
    }


def vix_snapshot_to_dict(snapshot: Any) -> dict[int, float]:
    """Convert a VIX term-structure snapshot into {tenor_days: level}."""
    if snapshot is None:
        return {}

    tenors = getattr(snapshot, "tenors_days", None)
    levels = getattr(snapshot, "vix_levels", None)
    labels = getattr(snapshot, "labels", None)
    if tenors is None or levels is None:
        return {}

    if labels is None:
        return {int(t): float(v) for t, v in zip(tenors, levels)}

    out: dict[int, float] = {}
    from quant.calibration.market_data_vix import VIX_INDEX_TENORS

    for i, label in enumerate(labels):
        if label in VIX_INDEX_TENORS:
            tau_days = int(VIX_INDEX_TENORS[label]["tau_days"])
        elif label.startswith("fut_"):
            try:
                tau_days = int(label.replace("fut_", "").replace("d", ""))
            except ValueError:
                tau_days = int(tenors[i])
        else:
            tau_days = int(tenors[i])
        out[tau_days] = float(levels[i])
    return out


def vix_futures_to_dict(term: Any) -> dict[int, float]:
    """Convert a VIX futures term-structure snapshot into {days_to_expiry: price}."""
    if term is None:
        return {}
    days = getattr(term, "days_to_exp", None)
    prices = getattr(term, "prices", None)
    if days is None or prices is None:
        return {}
    return {int(d): float(p) for d, p in zip(days, prices)}


def prepare_spx_targets(
    surface_df,
    spot: float,
    quick: bool = False,
    max_strikes: int = 8,
    max_maturities: int | None = None,
    include_itm: bool = True,
    moneyness_basis: str = "spot",
    moneyness_tenor_widen_alpha: float = 0.0,
    dividend_yield: float = 0.0,
):
    """Prepare SPX slices for calibration launchers."""
    from quant.calibration.market_targets import prepare_spx_slices

    if quick:
        max_strikes = min(max_strikes, 6)
        if max_maturities is None:
            max_maturities = 6

    return prepare_spx_slices(
        surface_df,
        spot,
        max_strikes=max_strikes,
        max_maturities=max_maturities,
        include_itm=include_itm,
        moneyness_basis=moneyness_basis,
        moneyness_tenor_widen_alpha=moneyness_tenor_widen_alpha,
        dividend_yield=dividend_yield,
    )
