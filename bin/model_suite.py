"""
Model Suite Pipeline
====================
Trains multiple Neural SDE profiles and produces practical use-case reports:
  1) Volatility scenario generation + conditional VaR/ES
  2) Pricing complement (dynamic prior + local ATM recalibration)
  3) Regime analysis (contango/backwardation, panic states)

Usage examples:
  python bin/model_suite.py --train-suite --run-usecases
  python bin/model_suite.py --train-suite --profiles pricing,scenarios
  python bin/model_suite.py --run-usecases
"""

import sys
import os
import json
import copy
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import equinox as eqx
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from utils.config import load_config
from engine.generative_trainer import GenerativeTrainer
from engine.signature_engine import SignatureFeatureExtractor
from engine.neural_sde import NeuralRoughSimulator
from quant.options_cache import OptionsDataCache
from quant.backtesting import HistoricalBacktester


def _deep_update(base: dict, updates: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _write_yaml(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def _train_profile(profile_name: str, overrides: dict) -> dict:
    base_cfg = load_config()
    cfg = _deep_update(base_cfg, overrides)

    cfg_dir = Path("outputs/model_suite/configs")
    cfg_path = cfg_dir / f"params_{profile_name}.yaml"
    _write_yaml(cfg_path, cfg)

    # Reset cache for config loader before each profile.
    try:
        load_config.cache_clear()
    except Exception:
        pass

    train_config = {
        "n_steps": cfg["simulation"]["n_steps"],
        "T": cfg["simulation"]["T"],
    }

    print("\n" + "=" * 70)
    print(f"TRAIN PROFILE: {profile_name}")
    print("=" * 70)
    print(f"Config: {cfg_path}")

    trainer = GenerativeTrainer(train_config, config_path=str(cfg_path))
    trainer.run(
        n_epochs=cfg["training"]["n_epochs"],
        batch_size=cfg["training"]["batch_size"],
    )

    src = Path("models/neural_sde_best.eqx")
    if not src.exists():
        raise FileNotFoundError("models/neural_sde_best.eqx not found after training")

    dst = Path("models") / f"neural_sde_{profile_name}.eqx"
    dst.write_bytes(src.read_bytes())

    return {
        "profile": profile_name,
        "config_path": str(cfg_path),
        "model_path": str(dst),
        "trained_at": datetime.now().isoformat(),
    }


def train_suite(profiles: list[str]) -> dict:
    profile_overrides = {
        # Better distributional realism / risk scenarios
        "scenarios": {
            "training": {
                "mean_penalty_mode": "global",
                "lambda_mean": 12.0,
                "n_epochs": 500,
            },
            "data": {
                "data_type": "vix",
            },
        },
        # Better local pricing fit (marginals)
        "pricing": {
            "training": {
                "mean_penalty_mode": "marginal",
                "lambda_mean": 8.0,
                "n_epochs": 350,
            },
            "data": {
                "data_type": "vix",
            },
        },
        # Regime-sensitive dynamics (slightly longer context)
        "regimes": {
            "training": {
                "mean_penalty_mode": "global",
                "lambda_mean": 10.0,
                "n_epochs": 450,
            },
            "data": {
                "data_type": "vix",
                "segment_length": 120,
            },
        },
    }

    chosen = []
    for p in profiles:
        p = p.strip().lower()
        if p in profile_overrides:
            chosen.append(p)

    if not chosen:
        raise ValueError("No valid profiles selected. Use any of: scenarios, pricing, regimes")

    entries = []
    for profile in chosen:
        entries.append(_train_profile(profile, profile_overrides[profile]))

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "profiles": entries,
        "default_model": entries[-1]["model_path"],
    }

    out_path = Path("models/model_suite_manifest.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Keep last profile as default model for existing scripts.
    default_src = Path(entries[-1]["model_path"])
    default_dst = Path("models/neural_sde_best.eqx")
    default_dst.write_bytes(default_src.read_bytes())

    print(f"\nModel suite manifest saved: {out_path}")
    print(f"Default model set to: {default_src.name}")
    return manifest


def _load_profile_model(entry: dict):
    cfg_path = entry["config_path"]
    model_path = Path(entry["model_path"])

    cfg = load_config(cfg_path)
    n_steps = int(cfg["simulation"]["n_steps"])
    T = float(cfg["simulation"]["T"])
    dt = T / n_steps

    sig_order = cfg["neural_sde"]["sig_truncation_order"]
    sig_extractor = SignatureFeatureExtractor(truncation_order=sig_order, dt=dt)
    input_sig_dim = sig_extractor.get_feature_dim(1)

    key = jax.random.PRNGKey(0)

    # Try current architecture, then legacy fallback
    try:
        model_template = NeuralRoughSimulator(input_sig_dim, key, config_path=cfg_path)
        model = eqx.tree_deserialise_leaves(model_path, model_template)
        return model, cfg
    except Exception:
        pass

    try:
        from engine._legacy_loader import load_legacy_model
        model = load_legacy_model(model_path, input_sig_dim, cfg_path)
        if model is not None:
            return model, cfg
    except Exception:
        pass

    raise RuntimeError(f"Cannot load model: {model_path}")


def _simulate_risk_metrics(model, cfg: dict, n_paths: int = 5000, seed: int = 123) -> dict:
    n_steps = int(cfg["simulation"]["n_steps"])
    T = float(cfg["simulation"]["T"])
    dt = T / n_steps

    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)

    dW_var = jax.random.normal(k1, (n_paths, n_steps)) * jnp.sqrt(dt)

    theta = float(getattr(model, "theta", -3.5))
    v0 = float(np.clip(np.exp(theta), 1e-4, 0.2))
    v0_arr = jnp.full((n_paths,), v0)

    var_paths = jax.vmap(model.generate_variance_path, in_axes=(0, 0, None))(v0_arr, dW_var, dt)
    var_paths = jnp.clip(var_paths, 1e-6, 5.0)

    z = jax.random.normal(k2, (n_paths, n_steps))
    log_ret = -0.5 * var_paths * dt + jnp.sqrt(var_paths) * jnp.sqrt(dt) * z
    horizon_ret = np.array(jnp.sum(log_ret, axis=1))

    q05 = float(np.quantile(horizon_ret, 0.05))
    q01 = float(np.quantile(horizon_ret, 0.01))
    es05 = float(horizon_ret[horizon_ret <= q05].mean()) if np.any(horizon_ret <= q05) else q05
    es01 = float(horizon_ret[horizon_ret <= q01].mean()) if np.any(horizon_ret <= q01) else q01

    term_vol = np.sqrt(np.array(var_paths[:, -1]))
    panic_prob = float(np.mean(term_vol > 0.40))

    return {
        "horizon_days_equiv": float(T * 252),
        "VaR_95": float(-q05),
        "ES_95": float(-es05),
        "VaR_99": float(-q01),
        "ES_99": float(-es01),
        "terminal_vol_mean": float(term_vol.mean()),
        "terminal_vol_p95": float(np.quantile(term_vol, 0.95)),
        "panic_prob_vol_gt_40pct": panic_prob,
    }


def _pricing_prior_recalibration(model, cfg: dict) -> dict:
    n_paths = 4000
    n_steps = int(cfg["simulation"]["n_steps"])
    T_model = float(cfg["simulation"]["T"])
    dt = T_model / n_steps

    cache = OptionsDataCache()
    surface, info = cache.load_latest("SPY")

    bt = HistoricalBacktester(fair_mode=True)
    smile, actual_dte = bt._filter_smile(surface, target_dte=30)
    if len(smile) < 5:
        return {"status": "insufficient_smile_data"}

    m = smile["moneyness"].to_numpy(dtype=float)
    iv = smile["impliedVolatility"].to_numpy(dtype=float)
    atm_vol = float(iv[np.argmin(np.abs(m))])

    key = jax.random.PRNGKey(321)
    dW = jax.random.normal(key, (n_paths, n_steps)) * jnp.sqrt(dt)

    theta = float(getattr(model, "theta", -3.5))
    v0 = float(np.clip(np.exp(theta), 1e-4, 0.2))
    v0_arr = jnp.full((n_paths,), v0)

    var_paths = jax.vmap(model.generate_variance_path, in_axes=(0, 0, None))(v0_arr, dW, dt)
    var_paths = np.array(jnp.clip(var_paths, 1e-6, 5.0))

    # Prior ATM proxy from integrated variance
    iv_prior = float(np.sqrt(np.maximum(var_paths.mean(), 1e-8)))
    scale = float(atm_vol / max(iv_prior, 1e-6))
    iv_recalibrated = float(iv_prior * scale)

    return {
        "snapshot_time": str(info.get("datetime", "")),
        "target_dte": int(actual_dte),
        "market_atm_iv": atm_vol,
        "prior_atm_iv": iv_prior,
        "local_scale_factor": scale,
        "recalibrated_atm_iv": iv_recalibrated,
    }


def _regime_analysis() -> dict:
    path = Path("data/cboe_vix_futures_full/vix_futures_all.csv")
    if not path.exists():
        return {"status": "missing_vix_futures_all.csv"}

    df = pd.read_csv(path)
    if "Trade Date" not in df.columns:
        return {"status": "invalid_vix_futures_all.csv"}

    df["trade_date"] = pd.to_datetime(df["Trade Date"], errors="coerce")
    if "expiration_date" in df.columns:
        df["expiration_date"] = pd.to_datetime(df["expiration_date"], errors="coerce")
    else:
        df["expiration_date"] = pd.to_datetime(
            df["Futures"].astype(str).str.extract(r"\((.*?)\)")[0], format="%b %Y", errors="coerce"
        )

    price_col = next((c for c in ["Close", "close", "Settle", "settle"] if c in df.columns), None)
    if price_col is None:
        return {"status": "missing_price_column"}

    df["price"] = pd.to_numeric(df[price_col], errors="coerce")
    df["dte"] = (df["expiration_date"] - df["trade_date"]).dt.days
    df = df.dropna(subset=["trade_date", "expiration_date", "price", "dte"])
    df = df[(df["dte"] > 0) & (df["price"] > 0)]

    rows = []
    for d, g in df.groupby("trade_date"):
        g = g.sort_values("dte")
        if len(g) < 2:
            continue
        f1 = float(g.iloc[0]["price"])
        f2 = float(g.iloc[1]["price"])
        rows.append({
            "trade_date": d,
            "f1": f1,
            "f2": f2,
            "contango": f2 > f1,
            "backwardation": f2 < f1,
            "panic": f1 >= 30.0,
        })

    if not rows:
        return {"status": "no_valid_regime_rows"}

    rg = pd.DataFrame(rows)
    return {
        "n_days": int(len(rg)),
        "contango_freq": float(rg["contango"].mean()),
        "backwardation_freq": float(rg["backwardation"].mean()),
        "panic_freq_f1_ge_30": float(rg["panic"].mean()),
        "front_second_spread_mean": float((rg["f2"] - rg["f1"]).mean()),
    }


def _build_default_manifest() -> dict:
    """
    Build a manifest from existing trained models (P/Q/Q+jump)
    so that --run-usecases works without --train-suite.
    """
    entries = []
    default_cfg = "config/params.yaml"

    model_map = [
        ("P-measure (risk)", "models/neural_sde_best_p.eqx", "risk"),
        ("Q-measure (pricing)", "models/neural_sde_best_q.eqx", "pricing"),
        ("Q+jump (crisis)", "models/neural_sde_best_q_jump.eqx", "crisis"),
        ("default", "models/neural_sde_best.eqx", "scenarios"),
    ]

    for label, path, profile in model_map:
        if Path(path).exists():
            entries.append({
                "profile": profile,
                "model_path": path,
                "config_path": default_cfg,
                "label": label,
            })

    if not entries:
        raise FileNotFoundError(
            "No trained models found in models/. "
            "Run 'python bin/train_multi.py' first."
        )

    print(f"   Using {len(entries)} existing model(s): "
          + ", ".join(e['label'] for e in entries))
    return {"profiles": entries}


def run_usecases(n_paths: int = 5000, seed: int = 123) -> dict:
    manifest_path = Path("models/model_suite_manifest.json")
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        print("   No model suite manifest found - using existing trained models.")
        manifest = _build_default_manifest()

    report = {
        "timestamp": datetime.now().isoformat(),
        "profiles": {},
        "regime_analysis": _regime_analysis(),
    }

    for entry in manifest.get("profiles", []):
        profile = entry["profile"]
        print(f"\n--- Use-cases for profile: {profile} ---")

        try:
            load_config.cache_clear()
        except Exception:
            pass

        try:
            model, cfg = _load_profile_model(entry)
        except Exception as e:
            print(f"   [SKIP] Could not load model: {e}")
            continue

        risk = _simulate_risk_metrics(model, cfg, n_paths=n_paths, seed=seed)
        pricing = _pricing_prior_recalibration(model, cfg)

        report["profiles"][profile] = {
            "model_path": entry["model_path"],
            "config_path": entry["config_path"],
            "risk_scenarios": risk,
            "pricing_prior_local_recalibration": pricing,
        }

    out_path = Path("outputs/model_usecases_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nUse-cases report saved: {out_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Train a Neural SDE model suite and run use-case reports.")
    parser.add_argument("--train-suite", action="store_true", help="Train multiple profiles.")
    parser.add_argument("--run-usecases", action="store_true", help="Run scenario/pricing/regime reports.")
    parser.add_argument(
        "--profiles",
        type=str,
        default="scenarios,pricing,regimes",
        help="Comma-separated profile list among: scenarios,pricing,regimes",
    )
    parser.add_argument("--n-paths", type=int, default=5000, help="MC paths for risk report.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reports.")

    args = parser.parse_args()

    do_train = args.train_suite
    do_usecases = args.run_usecases
    if not do_train and not do_usecases:
        do_train = True
        do_usecases = True

    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]

    if do_train:
        train_suite(profiles)

    if do_usecases:
        run_usecases(n_paths=args.n_paths, seed=args.seed)


if __name__ == "__main__":
    main()
