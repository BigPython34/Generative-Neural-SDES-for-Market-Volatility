"""
DeepRoughVol REST API + Interactive Web UI
==========================================
FastAPI server with:
  - Full pricing/risk/hedging/P&L API
  - Interactive web dashboard with animated visualizations
  - Script runner for training, backtesting, etc.

Launch:
    python bin/api_server.py
    → open http://localhost:8000 for the UI
    → open http://localhost:8000/docs for Swagger
"""

import sys
import os
import json
import glob
import asyncio
import subprocess
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

app = FastAPI(
    title="DeepRoughVol API",
    description="Neural SDE volatility modeling, pricing, risk & interactive dashboard",
    version="3.1.0",
)


# =====================================================================
#  Input validation mixin
# =====================================================================

def _validate_positive(v, name):
    if v is not None and v <= 0:
        raise ValueError(f"{name} must be positive, got {v}")
    return v


def _validate_nonneg(v, name):
    if v is not None and v < 0:
        raise ValueError(f"{name} must be non-negative, got {v}")
    return v


# =====================================================================
#  Request / Response models
# =====================================================================

class VanillaPriceRequest(BaseModel):
    spot: float = 100.0
    strike: float = 100.0
    T: float = 0.25
    r: Optional[float] = None
    sigma: float = 0.20
    opt_type: str = "call"

    @field_validator("spot", "strike", "T", "sigma")
    @classmethod
    def must_be_positive(cls, v, info):
        return _validate_positive(v, info.field_name)

    @field_validator("opt_type")
    @classmethod
    def valid_opt_type(cls, v):
        if v not in ("call", "put"):
            raise ValueError(f"opt_type must be 'call' or 'put', got '{v}'")
        return v

class ExoticPriceRequest(BaseModel):
    product: str = "asian_call"
    spot: float = 100.0
    strike: float = 100.0
    T: float = 0.25
    r: Optional[float] = None
    n_mc_paths: int = 10000
    model: str = "neural_sde"
    extra_params: Dict = Field(default_factory=dict)

    @field_validator("spot", "strike", "T")
    @classmethod
    def must_be_positive(cls, v, info):
        return _validate_positive(v, info.field_name)

    @field_validator("n_mc_paths")
    @classmethod
    def reasonable_paths(cls, v):
        if v < 100 or v > 500_000:
            raise ValueError(f"n_mc_paths must be in [100, 500000], got {v}")
        return v

    @field_validator("model")
    @classmethod
    def valid_model(cls, v):
        if v not in ("neural_sde", "bergomi", "bs"):
            raise ValueError(f"model must be 'neural_sde', 'bergomi' or 'bs', got '{v}'")
        return v

class VaRRequest(BaseModel):
    spot: float = 100.0
    positions: List[Dict] = Field(default_factory=list)
    n_mc_paths: int = 10000
    horizon_days: int = 1
    model: str = "neural_sde"

    @field_validator("spot")
    @classmethod
    def must_be_positive(cls, v, info):
        return _validate_positive(v, info.field_name)

    @field_validator("horizon_days")
    @classmethod
    def valid_horizon(cls, v):
        if v < 1 or v > 252:
            raise ValueError(f"horizon_days must be in [1, 252], got {v}")
        return v

class StressRequest(BaseModel):
    spot: float = 100.0
    positions: List[Dict] = Field(default_factory=list)
    scenarios: Optional[Dict] = None
    model: str = "bs"

    @field_validator("spot")
    @classmethod
    def must_be_positive(cls, v, info):
        return _validate_positive(v, info.field_name)

    @field_validator("model")
    @classmethod
    def valid_model(cls, v):
        if v not in ("neural_sde", "bergomi", "bs"):
            raise ValueError(f"model must be 'neural_sde', 'bergomi' or 'bs', got '{v}'")
        return v


class NeuralStressRequest(BaseModel):
    """Request for Neural SDE-driven stress testing (model-driven, not deterministic)."""
    spot: float = 100.0
    positions: List[Dict] = Field(default_factory=list)
    n_mc_paths: int = 5000
    stress_regimes: Optional[Dict] = None
    seed: int = 42

    @field_validator("spot")
    @classmethod
    def must_be_positive(cls, v, info):
        return _validate_positive(v, info.field_name)

    @field_validator("n_mc_paths")
    @classmethod
    def reasonable_paths(cls, v):
        if v < 100 or v > 500_000:
            raise ValueError(f"n_mc_paths must be in [100, 500000], got {v}")
        return v


class HedgeRequest(BaseModel):
    spot: float = 100.0
    strike: float = 100.0
    T: float = 0.25
    sigma: float = 0.20
    opt_type: str = "call"
    n_mc_paths: int = 5000
    hedge_freq: str = "daily"
    model: str = "neural_sde"

    @field_validator("spot", "strike", "T", "sigma")
    @classmethod
    def must_be_positive(cls, v, info):
        return _validate_positive(v, info.field_name)

    @field_validator("opt_type")
    @classmethod
    def valid_opt_type(cls, v):
        if v not in ("call", "put"):
            raise ValueError(f"opt_type must be 'call' or 'put', got '{v}'")
        return v

class PnLRequest(BaseModel):
    spot: float = 100.0
    strike: float = 100.0
    T: float = 0.25
    r: float = 0.045
    sigma: float = 0.20
    opt_type: str = "call"
    spot_new: float = 99.0
    sigma_new: Optional[float] = None
    dt: float = 0.003968
    r_new: Optional[float] = None

class EtaCalibRequest(BaseModel):
    H: float = 0.1
    window_days: int = 252

class MCAnimRequest(BaseModel):
    spot: float = 100.0
    sigma: float = 0.20
    T: float = 0.25
    n_paths: int = 200
    n_frames: int = 60
    model: str = "bs"

    @field_validator("model")
    @classmethod
    def valid_model(cls, v):
        if v not in ("neural_sde", "bergomi", "bs"):
            raise ValueError(f"model must be 'neural_sde', 'bergomi' or 'bs', got '{v}'")
        return v

class ScriptRequest(BaseModel):
    script: str = "bin/backtest.py"
    args: List[str] = Field(default_factory=list)


# =====================================================================
#  Singletons
# =====================================================================

_sofr = None
_vvix_cal = None
_regime_detector = None


def _get_sofr():
    global _sofr
    if _sofr is None:
        from quant.loader.sofr_loader import SOFRRateLoader
        _sofr = SOFRRateLoader()
    return _sofr


def _get_vvix():
    global _vvix_cal
    if _vvix_cal is None:
        from quant.calibration.vvix_calibrator import VVIXCalibrator
        _vvix_cal = VVIXCalibrator()
    return _vvix_cal


def _get_regime():
    global _regime_detector
    if _regime_detector is None:
        from quant.regimes.regime_detector import RegimeDetector
        _regime_detector = RegimeDetector()
    return _regime_detector


def _get_r(r_override=None):
    if r_override is not None:
        return r_override
    sofr = _get_sofr()
    return sofr.get_rate() if sofr.is_available else 0.045


# =====================================================================
#  Shared path generator — respects the model parameter
# =====================================================================

_pricing_engine = None
_bergomi_model = None


def _get_pricing_engine():
    """Lazy-load the Neural SDE pricing engine (heavy, done once)."""
    global _pricing_engine
    if _pricing_engine is not None:
        return _pricing_engine
    try:
        from engine.generative_trainer import GenerativeTrainer
        from utils.config import load_config
        cfg = load_config()
        # GenerativeTrainer expects {'n_steps': ..., 'T': ...}, NOT the full YAML
        trainer_config = {
            'n_steps': cfg['simulation']['n_steps'],
            'T': cfg['simulation']['T'],
        }
        trainer = GenerativeTrainer(trainer_config)
        # Load best Q-measure model, fall back to generic best
        for model_path in ["models/neural_sde_pricing.eqx",
                           "models/neural_sde_best_q.eqx",
                           "models/neural_sde_best.eqx"]:
            if os.path.exists(model_path):
                model = trainer.load_model(model_path)
                if model is not None:
                    from quant.pricers.pricing import DeepPricingEngine
                    _pricing_engine = DeepPricingEngine(trainer, model)
                    print(f"[OK] Neural SDE loaded from {model_path}")
                    return _pricing_engine
    except Exception as e:
        print(f"[WARN] Cannot load Neural SDE: {e}")
    return None


def _get_bergomi():
    """Lazy-load the Rough Bergomi model."""
    global _bergomi_model
    if _bergomi_model is not None:
        return _bergomi_model
    try:
        from quant.models.bergomi import RoughBergomiModel
        from utils.config import load_config
        cfg = load_config()
        bergomi_cfg = dict(cfg['bergomi'])
        bergomi_cfg.update({'n_steps': 252, 'T': 0.25})
        _bergomi_model = RoughBergomiModel(bergomi_cfg)
        return _bergomi_model
    except Exception as e:
        print(f"[WARN] Cannot load Bergomi: {e}")
    return None


def _generate_paths(model: str, n_paths: int, spot: float, T: float,
                    sigma: float = 0.20, r: float = 0.045):
    """
    Generate spot paths (and optionally var paths) using the requested model.

    Returns: (spot_paths, var_paths_or_None)
    """
    n_steps = max(60, int(T * 252))

    # ---- Neural SDE ----
    if model == "neural_sde":
        engine = _get_pricing_engine()
        if engine is not None:
            try:
                s_paths, v_paths = engine.generate_market_paths(
                    n_paths, s0=spot, mu=r)
                return np.array(s_paths), np.array(v_paths)
            except Exception as e:
                print(f"[WARN] Neural SDE path gen failed ({e}), falling back to BS")

    # ---- Rough Bergomi ----
    if model == "bergomi":
        bergomi = _get_bergomi()
        if bergomi is not None:
            try:
                st, vt = bergomi.simulate_spot_vol_paths(n_paths, s0=spot)
                return np.array(st), np.array(vt)
            except Exception as e:
                print(f"[WARN] Bergomi path gen failed ({e}), falling back to BS")

    # ---- Black-Scholes (fallback) ----
    dt_sim = T / n_steps
    z = np.random.randn(n_paths, n_steps)
    log_ret = (r - 0.5 * sigma**2) * dt_sim + sigma * np.sqrt(dt_sim) * z
    s_paths = spot * np.exp(np.cumsum(log_ret, axis=1))
    s_paths = np.column_stack([np.full(n_paths, spot), s_paths])
    return s_paths, None


# =====================================================================
#  Core API endpoints
# =====================================================================

@app.get("/health")
def health():
    return {"status": "ok", "model": "DeepRoughVol", "version": "3.2.0"}


@app.get("/regime")
def get_regime():
    try:
        detector = _get_regime()
        return detector.detect()
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.get("/rates/sofr")
def get_sofr_rate(as_of_date: Optional[str] = None):
    sofr = _get_sofr()
    return {
        "rate": sofr.get_rate(as_of_date),
        "available": sofr.is_available,
        "as_of_date": as_of_date or "latest",
    }


@app.post("/price/vanilla")
def price_vanilla(req: VanillaPriceRequest):
    from quant.models.black_scholes import BlackScholes
    r = _get_r(req.r)
    price = BlackScholes.price(req.spot, req.strike, req.T, r, req.sigma, req.opt_type)
    delta = BlackScholes.delta(req.spot, req.strike, req.T, r, req.sigma, req.opt_type)
    gamma = BlackScholes.gamma(req.spot, req.strike, req.T, r, req.sigma)
    vega = BlackScholes.vega(req.spot, req.strike, req.T, r, req.sigma)
    theta = BlackScholes.theta(req.spot, req.strike, req.T, r, req.sigma, req.opt_type)
    return {
        "price": float(price),
        "greeks": {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta},
        "r_used": r,
    }


@app.post("/price/exotic")
def price_exotic(req: ExoticPriceRequest):
    from quant.pricers.exotic_pricer import ExoticPricer
    r = _get_r(req.r)
    pricer = ExoticPricer(spot=req.spot, r=r, T=req.T)
    sigma = req.extra_params.get("sigma", 0.20)

    # Use the requested model for path generation
    s_paths, _var = _generate_paths(
        req.model, req.n_mc_paths, req.spot, req.T, sigma, r)

    strike = req.strike
    # Only pass allowed kwargs to each pricer (autocallable/cliquet don't take sigma)
    autocall_kw = {k: req.extra_params[k] for k in ("coupon_rate", "autocall_barrier", "ki_barrier", "observation_freq") if k in req.extra_params}
    cliquet_kw = {k: req.extra_params[k] for k in ("local_cap", "local_floor", "global_cap", "global_floor") if k in req.extra_params}
    method_map = {
        "asian_call": lambda: pricer.asian_call(s_paths, strike),
        "asian_put": lambda: pricer.asian_put(s_paths, strike),
        "lookback_call": lambda: pricer.lookback_call(s_paths, strike),
        "lookback_put": lambda: pricer.lookback_put(s_paths, strike),
        "autocallable": lambda: pricer.autocallable(s_paths, **autocall_kw),
        "cliquet": lambda: pricer.cliquet(s_paths, **cliquet_kw),
        "variance_swap": lambda: pricer.variance_swap(s_paths, sigma**2),
        "volatility_swap": lambda: pricer.volatility_swap(s_paths, sigma),
    }
    product = req.product
    if product not in method_map:
        raise HTTPException(400, f"Unknown product: {product}. Available: {list(method_map.keys())}")
    result = method_map[product]()
    result["model_used"] = req.model
    return result


@app.post("/risk/var")
def compute_var(req: VaRRequest):
    from quant.risk.risk_engine import RiskEngine
    r = _get_r()
    engine = RiskEngine(spot=req.spot, r=r)
    for pos in req.positions:
        p = dict(pos)
        if 'opt_type' in p:
            p['instrument'] = p.pop('opt_type')
        engine.add_position(**p)
    if not engine.positions:
        engine.add_position('call', strike=req.spot, T=0.25, quantity=1)

    T_sim = req.horizon_days / 252.0
    s_paths, _var = _generate_paths(
        req.model, req.n_mc_paths, req.spot, T_sim, 0.20, r)

    report = engine.compute_var(s_paths)
    return {
        "var_95": report.var_95, "var_99": report.var_99,
        "cvar_95": report.cvar_95, "cvar_99": report.cvar_99,
        "stressed_var_99": report.stressed_var_99,
        "expected_pnl": report.expected_pnl, "pnl_std": report.pnl_std,
        "skew": report.pnl_skew, "kurtosis": report.pnl_kurtosis,
        "n_scenarios": report.n_scenarios,
        "model_used": req.model,
    }


@app.post("/risk/stress")
def run_stress(req: StressRequest):
    """Deterministic stress scenarios (fast, no MC). Use /risk/stress/neural for model-driven MC stress."""
    from quant.risk.risk_engine import RiskEngine
    r = _get_r()
    engine = RiskEngine(spot=req.spot, r=r)
    for pos in req.positions:
        p = dict(pos)
        if 'opt_type' in p:
            p['instrument'] = p.pop('opt_type')
        engine.add_position(**p)
    if not engine.positions:
        engine.add_position('call', strike=req.spot, T=0.25, quantity=1)
    result = engine.stress_test(req.scenarios)
    result["_info"] = {
        "type": "deterministic",
        "note": "These are sensitivity shocks, not model-driven. Use POST /risk/stress/neural for Neural SDE MC stress testing.",
    }
    return result


@app.post("/risk/stress/neural")
def run_neural_stress(req: NeuralStressRequest):
    """
    Neural SDE-driven stress testing.

    Generates MC paths from the Neural SDE conditioned on crisis initial states
    (high vol, VIX~45%, VIX~30%, etc.). Returns full P&L distributions, VaR/CVaR,
    and tail statistics per scenario.

    Unlike /risk/stress (deterministic), this captures:
      - Path-dependent effects (barrier breaching, Asian averaging)
      - Joint spot-vol dynamics with leverage effect (ρ)
      - Full distributional output (not point estimates)
    """
    from quant.risk.risk_engine import RiskEngine
    from utils.config import load_config
    r = _get_r()
    engine = RiskEngine(spot=req.spot, r=r)
    for pos in req.positions:
        p = dict(pos)
        if 'opt_type' in p:
            p['instrument'] = p.pop('opt_type')
        engine.add_position(**p)
    if not engine.positions:
        engine.add_position('call', strike=req.spot, T=0.25, quantity=1)

    # We need the Neural SDE model directly (not the pricing engine wrapper)
    pe = _get_pricing_engine()
    if pe is None:
        raise HTTPException(503, detail="Neural SDE model not available. Train first: python bin/train_multi.py")

    cfg = load_config()
    sim_config = {
        'n_steps': cfg['simulation']['n_steps'],
        'T': cfg['simulation']['T'],
    }

    result = engine.neural_stress_test(
        model=pe.model,
        config=sim_config,
        n_paths=req.n_mc_paths,
        stress_regimes=req.stress_regimes,
        seed=req.seed,
    )
    return result


@app.post("/pnl/attribute")
def pnl_attribute(req: PnLRequest):
    from quant.hedging.pnl_attribution import PnLAttributor
    attr = PnLAttributor(req.spot, req.strike, req.T, req.r, req.sigma, req.opt_type)
    result = attr.attribute(req.spot_new, req.sigma_new, req.dt, req.r_new)
    return {
        "total_pnl": result.total_pnl, "delta_pnl": result.delta_pnl,
        "gamma_pnl": result.gamma_pnl, "vega_pnl": result.vega_pnl,
        "theta_pnl": result.theta_pnl, "vanna_pnl": result.vanna_pnl,
        "volga_pnl": result.volga_pnl, "rho_pnl": result.rho_pnl,
        "residual": result.residual, "explained_pct": result.explained_pct,
    }


@app.post("/hedge/simulate")
def simulate_hedge(req: HedgeRequest):
    from quant.hedging.hedging_simulator import HedgingSimulator
    r = _get_r()
    sim = HedgingSimulator(
        spot=req.spot, strike=req.strike, T=req.T,
        r=r, iv=req.sigma, opt_type=req.opt_type,
    )

    # Use the requested model for path generation
    s_paths, v_paths = _generate_paths(
        req.model, req.n_mc_paths, req.spot, req.T, req.sigma, r)

    results = sim.run(s_paths, var_paths=v_paths, hedge_freq=req.hedge_freq)
    out = {}
    for name, hr in results.items():
        out[name] = {
            "mean_pnl": float(hr.mean_pnl), "std_pnl": float(hr.std_pnl),
            "mean_abs_pnl": float(hr.mean_abs_pnl), "max_loss": float(hr.max_loss),
            "sharpe": float(hr.sharpe), "tracking_error": float(hr.tracking_error),
            "hedge_cost": float(hr.hedge_cost), "n_rebalances": hr.n_rebalances,
        }
    return out


@app.post("/calibrate/eta")
def calibrate_eta(req: EtaCalibRequest):
    cal = _get_vvix()
    return cal.estimate_eta(H=req.H, window_days=req.window_days)


@app.get("/models/info")
def models_info():
    """Show which models are loaded and their status."""
    import glob as _glob
    available_files = sorted(_glob.glob("models/neural_sde_*.eqx"))

    neural_ok = _get_pricing_engine() is not None
    bergomi_ok = _get_bergomi() is not None

    info = {
        "neural_sde": {
            "loaded": neural_ok,
            "status": "ready" if neural_ok else "not loaded (train first: python bin/train_multi.py)",
        },
        "bergomi": {
            "loaded": bergomi_ok,
            "status": "ready" if bergomi_ok else "not loaded",
        },
        "bs": {
            "loaded": True,
            "status": "always available (analytical fallback)",
        },
        "model_files": available_files,
    }

    # Add config info if available
    try:
        from utils.config import load_config
        cfg = load_config()
        info["config"] = {
            "H": cfg['bergomi']['hurst'],
            "eta": cfg['bergomi']['eta'],
            "rho": cfg['bergomi']['rho'],
            "n_steps": cfg['simulation']['n_steps'],
            "T": cfg['simulation']['T'],
        }
    except Exception:
        pass

    return info


# =====================================================================
#  Animation data endpoints
# =====================================================================

@app.post("/animate/mc-paths")
def animate_mc_paths(req: MCAnimRequest):
    """Generate MC path data for animated visualization. Supports neural_sde/bergomi/bs."""
    r = _get_r()
    n = min(req.n_paths, 500)  # cap for animation performance

    s_paths, _ = _generate_paths(req.model, n, req.spot, req.T, req.sigma, r)
    n_steps_actual = s_paths.shape[1] - 1  # exclude s0 column
    t_grid = np.linspace(0, req.T * 252, n_steps_actual + 1).tolist()

    return {
        "t": t_grid,
        "paths": s_paths.tolist(),
        "spot": req.spot,
        "sigma": req.sigma,
        "T": req.T,
        "model_used": req.model,
    }


@app.get("/animate/vol-surface")
def animate_vol_surface():
    """Load options surface: dense interpolated grid + raw quotes (robust spikes)."""
    cache_dir = PROJECT_ROOT / "data" / "options_cache"
    files = sorted(cache_dir.glob("SPX_surface_*.csv"))
    if not files:
        raise HTTPException(404, "No options surface cached. Run: python bin/data/fetch_options.py")

    import pandas as pd
    from scipy.interpolate import griddata

    df = pd.read_csv(files[-1])
    df = df[(df['impliedVolatility'] > 0.01) & (df['impliedVolatility'] < 3.0)]
    df = df[df['dte'] > 0]
    if "spread_pct" in df.columns:
        df = df[(~df["spread_pct"].notna()) | (df["spread_pct"] <= 0.50)]
    if "mid" in df.columns:
        df = df[df["mid"] > 0.1]

    strikes = df['strike'].values.tolist()
    dtes = df['dte'].values.tolist()
    ivs = (df['impliedVolatility'] * 100).values.tolist()
    types = df['type'].values.tolist()

    out = {
        "strikes": strikes,
        "dtes": dtes,
        "ivs": ivs,
        "types": types,
        "n_points": len(strikes),
        "grid_strikes": [],
        "grid_dtes": [],
        "grid_ivs": [],
    }

    if len(df) >= 6:
        pts = df[["strike", "dte"]].to_numpy(dtype=float)
        vals = (df["impliedVolatility"].to_numpy(dtype=float) * 100.0)
        kmin, kmax = float(df["strike"].min()), float(df["strike"].max())
        tmin, tmax = float(df["dte"].min()), float(df["dte"].max())
        n = len(df)
        nk = int(min(70, max(20, round(np.sqrt(n) * 2))))
        nt = int(min(50, max(15, round(np.sqrt(n)))))
        grid_k = np.linspace(kmin, kmax, nk)
        grid_t = np.linspace(tmin, tmax, nt)
        K, T = np.meshgrid(grid_k, grid_t)
        Z = griddata(pts, vals, (K, T), method="linear")
        if np.isnan(Z).any():
            Z_near = griddata(pts, vals, (K, T), method="nearest")
            Z = np.where(np.isnan(Z), Z_near, Z)
        out["grid_strikes"] = grid_k.tolist()
        out["grid_dtes"] = grid_t.tolist()
        out["grid_ivs"] = Z.tolist()

    return out


@app.post("/animate/hedging")
def animate_hedging(req: HedgeRequest):
    """Generate step-by-step hedging data for animated chart."""
    from scipy.stats import norm
    r = _get_r()
    n = min(req.n_mc_paths, 500)
    n_steps = int(req.T * 252)
    if n_steps < 10:
        n_steps = 63
    dt = req.T / n_steps
    sigma = req.sigma

    z = np.random.randn(n, n_steps)
    log_ret = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    s_paths = req.spot * np.exp(np.cumsum(log_ret, axis=1))
    s_paths = np.column_stack([np.full(n, req.spot), s_paths])

    from quant.models.black_scholes import BlackScholes
    option_p0 = BlackScholes.price(req.spot, req.strike, req.T, r, sigma, req.opt_type)

    cash = np.full(n, option_p0)
    delta_prev = np.zeros(n)
    cum_pnl = np.zeros((n, n_steps + 1))

    for step in range(n_steps):
        S = s_paths[:, step]
        T_rem = max(req.T - step * dt, 1e-8)
        sqrt_T = np.sqrt(T_rem)
        d1 = (np.log(S / req.strike) + (r + 0.5 * sigma**2) * T_rem) / (sigma * sqrt_T)
        if req.opt_type == 'call':
            delta_new = norm.cdf(d1)
        else:
            delta_new = norm.cdf(d1) - 1.0

        trade = delta_new - delta_prev
        cash -= trade * S
        delta_prev = delta_new
        cash *= np.exp(r * dt)

        S_next = s_paths[:, step + 1]
        port_val = cash + delta_prev * S_next
        if req.opt_type == 'call':
            intrinsic = np.maximum(S_next - req.strike, 0)
        else:
            intrinsic = np.maximum(req.strike - S_next, 0)
        cum_pnl[:, step + 1] = port_val - intrinsic

    t_days = np.linspace(0, req.T * 252, n_steps + 1).tolist()
    mean_pnl = np.mean(cum_pnl, axis=0).tolist()
    p5 = np.percentile(cum_pnl, 5, axis=0).tolist()
    p95 = np.percentile(cum_pnl, 95, axis=0).tolist()
    sample_paths = cum_pnl[:5].tolist()

    return {
        "t_days": t_days,
        "mean_pnl": mean_pnl,
        "p5": p5,
        "p95": p95,
        "sample_paths": sample_paths,
        "option_price": float(option_p0),
    }


@app.get("/data/vix-history")
def get_vix_history(limit_days: int = 400):
    """Load historical VIX for the regime/history chart. Handles unix timestamps."""
    import pandas as pd
    vix_dir = PROJECT_ROOT / "data" / "market" / "vix"
    candidates = ["vix_30m.csv", "vix_5m.csv", "vix_daily.csv"]
    for c in candidates:
        fp = vix_dir / c
        if fp.exists():
            df = pd.read_csv(fp)
            col = "datetime" if "datetime" in df.columns else df.columns[0]
            # Unix timestamp (e.g. time column with values > 1e9)
            sample = pd.to_numeric(df[col].iloc[0] if len(df) else 0, errors="coerce")
            if pd.notna(sample) and sample > 1e9:
                df[col] = pd.to_datetime(df[col], unit="s", errors="coerce")
            else:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            df = df.dropna(subset=[col])
            if "daily" not in c:
                df = df.set_index(col).resample("1D").last().dropna().reset_index()
            close_col = "close" if "close" in df.columns else "Close"
            if close_col not in df.columns:
                close_col = df.columns[1]
            df[close_col] = pd.to_numeric(df[close_col], errors="coerce")
            df = df.dropna(subset=[close_col])
            df = df.tail(int(limit_days)).reset_index(drop=True)
            return {
                "dates": df[col].dt.strftime("%Y-%m-%d").tolist(),
                "values": df[close_col].values.tolist(),
            }
    raise HTTPException(404, "No VIX data found")


@app.get("/data/cboe/term-structure")
def get_cboe_term_structure():
    """VIX futures term structure from CBOE (used by backtest for xi0 calibration)."""
    try:
        from quant.workflows.backtesting import load_vix_term_structure
        ts = load_vix_term_structure()
        return {"term_structure": {str(k): float(v) for k, v in ts.items()}, "source": "cboe"}
    except Exception as e:
        raise HTTPException(502, f"CBOE term structure failed: {e}")


@app.get("/data/cboe/futures-history")
def get_cboe_futures_history(limit: int = 500):
    """VIX futures history from CBOE for charts / backtest validation."""
    import pandas as pd
    base = PROJECT_ROOT / "data" / "cboe_vix_futures_full"
    try:
        from quant.workflows.backtesting import load_vix_futures_history
        df = load_vix_futures_history()
    except Exception:
        df = pd.DataFrame()
    if df.empty and (base / "vix_futures_front_month.csv").exists():
        df = pd.read_csv(base / "vix_futures_front_month.csv")
        df["trade_date"] = pd.to_datetime(df["Trade Date"], errors="coerce")
        price_col = next((c for c in ["Close", "close", "Settle", "settle"] if c in df.columns), None)
        if price_col:
            df["price"] = pd.to_numeric(df[price_col], errors="coerce")
            df["dte_days"] = 30  # front month approx
            df = df[["trade_date", "dte_days", "price"]].dropna()
            df = df.sort_values("trade_date").tail(limit)
    if df.empty:
        raise HTTPException(404, "No CBOE futures history. Add data/cboe_vix_futures_full/.")
    df = df.tail(limit)
    return {
        "trade_date": df["trade_date"].dt.strftime("%Y-%m-%d").tolist(),
        "dte_days": df["dte_days"].astype(int).tolist(),
        "price": df["price"].values.tolist(),
    }


@app.get("/data/reports")
def get_reports():
    """List available output reports."""
    out_dir = PROJECT_ROOT / "outputs"
    reports = {}
    for f in out_dir.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding='utf-8'))
            reports[f.stem] = data
        except Exception:
            pass
    return reports


# =====================================================================
#  Script runner (SSE stream)
# =====================================================================

ALLOWED_SCRIPTS = {
    "regenerate_data": "bin/regenerate_data.py",
    "fetch_options": "bin/data/fetch_options.py",
    "calibrate": "bin/calibrate.py",
    "train": "bin/train_multi.py",
    "backtest": "bin/backtest.py",
    "usecases": "bin/model_suite.py",
    "dashboard": "bin/dashboard.py",
    "roughness": "bin/verify_roughness.py",
    "hurst": "bin/hurst_diagnostic.py",
    "robustness": "bin/robustness_check.py",
}


@app.get("/scripts/list")
def list_scripts():
    return ALLOWED_SCRIPTS


@app.get("/scripts/run/{script_name}")
async def run_script(script_name: str, args: str = ""):
    if script_name not in ALLOWED_SCRIPTS:
        raise HTTPException(400, f"Unknown script: {script_name}")

    script_path = str(PROJECT_ROOT / ALLOWED_SCRIPTS[script_name])
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args.split())

    async def stream():
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
        )
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            yield f"data: {line.decode('utf-8', errors='replace').rstrip()}\n\n"
        await proc.wait()
        yield f"data: [EXIT {proc.returncode}]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


# =====================================================================
#  Web UI
# =====================================================================

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    return _UI_HTML


_UI_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DeepRoughVol</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
:root{--bg:#0a0e17;--card:#111827;--border:#1e293b;--accent:#3b82f6;--accent2:#8b5cf6;
--green:#10b981;--red:#ef4444;--yellow:#f59e0b;--text:#e2e8f0;--muted:#64748b;--glass:rgba(17,24,39,.85)}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Inter',system-ui,-apple-system,sans-serif;background:var(--bg);color:var(--text);overflow-x:hidden}
.topbar{position:fixed;top:0;left:0;right:0;height:56px;background:var(--glass);backdrop-filter:blur(12px);
border-bottom:1px solid var(--border);display:flex;align-items:center;padding:0 24px;z-index:100}
.topbar h1{font-size:18px;font-weight:700;background:linear-gradient(135deg,var(--accent),var(--accent2));
-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.topbar .ver{font-size:11px;color:var(--muted);margin-left:12px}
nav{position:fixed;top:56px;left:0;bottom:0;width:220px;background:var(--card);border-right:1px solid var(--border);
padding:16px 0;overflow-y:auto;z-index:90}
nav a{display:block;padding:10px 24px;color:var(--muted);text-decoration:none;font-size:13px;
font-weight:500;transition:.15s}
nav a:hover,nav a.active{color:var(--text);background:rgba(59,130,246,.1)}
nav a.active{border-left:3px solid var(--accent);color:var(--accent)}
nav .sep{height:1px;background:var(--border);margin:12px 16px}
main{margin-left:220px;margin-top:56px;padding:24px;min-height:calc(100vh - 56px)}
.section{display:none}
.section.active{display:block}
.grid{display:grid;gap:20px}
.g2{grid-template-columns:1fr 1fr}
.g3{grid-template-columns:1fr 1fr 1fr}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:20px}
.card h3{font-size:14px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.5px;margin-bottom:16px}
.kpi{text-align:center;padding:24px}
.kpi .val{font-size:32px;font-weight:700;margin-bottom:4px}
.kpi .label{font-size:12px;color:var(--muted)}
.kpi.green .val{color:var(--green)}.kpi.red .val{color:var(--red)}
.kpi.blue .val{color:var(--accent)}.kpi.yellow .val{color:var(--yellow)}
.kpi.purple .val{color:var(--accent2)}
input,select{background:#1e293b;border:1px solid var(--border);color:var(--text);padding:8px 12px;
border-radius:8px;font-size:13px;width:100%;margin-bottom:8px}
input:focus,select:focus{outline:none;border-color:var(--accent)}
label{font-size:12px;color:var(--muted);display:block;margin-bottom:4px}
button,.btn{background:linear-gradient(135deg,var(--accent),var(--accent2));color:#fff;border:none;
padding:10px 20px;border-radius:8px;font-size:13px;font-weight:600;cursor:pointer;transition:.2s;width:100%}
button:hover,.btn:hover{opacity:.9;transform:translateY(-1px)}
button:disabled{opacity:.4;cursor:not-allowed;transform:none}
button.secondary{background:var(--border);color:var(--text)}
.result-box{background:#0f172a;border:1px solid var(--border);border-radius:8px;padding:16px;
margin-top:16px;font-family:'Fira Code',monospace;font-size:12px;max-height:400px;overflow-y:auto;
white-space:pre-wrap;line-height:1.6}
.terminal{background:#000;color:#00ff88;border-radius:8px;padding:16px;font-family:'Fira Code',monospace;
font-size:12px;height:400px;overflow-y:auto;white-space:pre-wrap;line-height:1.5}
.terminal .err{color:var(--red)}.terminal .info{color:var(--accent)}
.loader{display:inline-block;width:16px;height:16px;border:2px solid var(--border);border-top-color:var(--accent);
border-radius:50%;animation:spin .6s linear infinite;margin-right:8px;vertical-align:middle}
@keyframes spin{to{transform:rotate(360deg)}}
.tag{display:inline-block;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:600}
.tag.normal{background:rgba(16,185,129,.15);color:var(--green)}
.tag.stressed{background:rgba(245,158,11,.15);color:var(--yellow)}
.tag.crisis{background:rgba(239,68,68,.15);color:var(--red)}
.plot-box{min-height:400px}
.form-row{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.form-row.three{grid-template-columns:1fr 1fr 1fr}
@media(max-width:900px){nav{display:none}main{margin-left:0}.g2,.g3{grid-template-columns:1fr}}
</style>
</head>
<body>

<div class="topbar">
  <h1>DeepRoughVol</h1>
  <span class="ver">v3.2 &mdash; Neural SDE Volatility Engine</span>
</div>

<nav>
  <a href="#" data-sec="overview" class="active">Overview</a>
  <a href="#" data-sec="mc-paths">Monte Carlo Paths</a>
  <a href="#" data-sec="vol-surface">Vol Surface 3D</a>
  <a href="#" data-sec="hedging-anim">Hedging Simulation</a>
  <div class="sep"></div>
  <a href="#" data-sec="pricing">Pricing</a>
  <a href="#" data-sec="risk">Risk (VaR/Stress)</a>
  <a href="#" data-sec="pnl">P&L Attribution</a>
  <a href="#" data-sec="regime">Regime Detection</a>
  <div class="sep"></div>
  <a href="#" data-sec="models">Models Status</a>
  <a href="#" data-sec="scripts">Script Runner</a>
  <a href="#" data-sec="reports">Reports</a>
</nav>

<main>

<!-- ====================== OVERVIEW ====================== -->
<div class="section active" id="sec-overview">
  <div class="grid" style="grid-template-columns:1fr 1fr 1fr 1fr;margin-bottom:20px" id="overview-kpis">
    <div class="card kpi blue"><div class="val" id="kpi-regime">--</div><div class="label">Market Regime</div></div>
    <div class="card kpi green"><div class="val" id="kpi-sofr">--</div><div class="label">SOFR Rate</div></div>
    <div class="card kpi yellow"><div class="val" id="kpi-models">--</div><div class="label">Neural SDE</div></div>
    <div class="card kpi purple"><div class="val" id="kpi-health">--</div><div class="label">API v3.2</div></div>
  </div>
  <div class="grid g2">
    <div class="card"><h3>VIX History</h3><div id="vix-chart" class="plot-box"></div></div>
    <div class="card"><h3>Quick Stats</h3><div id="overview-stats" class="result-box">Loading reports...</div></div>
  </div>
</div>

<!-- ====================== MC PATHS ====================== -->
<div class="section" id="sec-mc-paths">
  <div class="grid g2">
    <div class="card">
      <h3>Monte Carlo Path Generator</h3>
      <div class="form-row">
        <div><label>Spot</label><input id="mc-spot" type="number" value="100" step="1"></div>
        <div><label>Volatility</label><input id="mc-sigma" type="number" value="0.20" step="0.01"></div>
      </div>
      <div class="form-row">
        <div><label>Maturity (years)</label><input id="mc-T" type="number" value="0.25" step="0.05"></div>
        <div><label>Paths</label><input id="mc-n" type="number" value="200" step="50"></div>
      </div>
      <div class="form-row">
        <div><label>Model</label><select id="mc-model"><option value="neural_sde">Neural SDE</option><option value="bergomi">Bergomi</option><option value="bs" selected>Black-Scholes</option></select></div>
        <div><label>Animation frames</label><input id="mc-frames" type="range" min="20" max="120" value="60"></div>
      </div>
      <button onclick="runMC()">Generate & Animate</button>
    </div>
    <div class="card"><h3>Distribution at maturity</h3><div id="mc-hist" class="plot-box"></div></div>
  </div>
  <div class="card" style="margin-top:20px"><div id="mc-plot" class="plot-box" style="height:500px"></div></div>
</div>

<!-- ====================== VOL SURFACE ====================== -->
<div class="section" id="sec-vol-surface">
  <div class="card"><h3>Implied Volatility Surface (from cached SPX options)</h3>
    <button onclick="loadVolSurface()" style="width:auto;margin-bottom:16px;padding:8px 24px">Load Surface</button>
    <div id="vol-surface-plot" class="plot-box" style="height:600px"></div>
  </div>
</div>

<!-- ====================== HEDGING ANIM ====================== -->
<div class="section" id="sec-hedging-anim">
  <div class="grid g2">
    <div class="card">
      <h3>Hedging Simulation</h3>
      <div class="form-row">
        <div><label>Spot</label><input id="hg-spot" type="number" value="100" step="1"></div>
        <div><label>Strike</label><input id="hg-strike" type="number" value="100" step="1"></div>
      </div>
      <div class="form-row">
        <div><label>Volatility</label><input id="hg-sigma" type="number" value="0.20" step="0.01"></div>
        <div><label>Maturity</label><input id="hg-T" type="number" value="0.25" step="0.05"></div>
      </div>
      <div class="form-row three">
        <div><label>Type</label><select id="hg-type"><option value="call">Call</option><option value="put">Put</option></select></div>
        <div><label>MC Paths</label><input id="hg-n" type="number" value="500" step="100"></div>
        <div><label>Model</label><select id="hg-model"><option value="neural_sde" selected>Neural SDE</option><option value="bergomi">Bergomi</option><option value="bs">Black-Scholes</option></select></div>
      </div>
      <button onclick="runHedgeAnim()">Simulate & Animate</button>
      <div id="hedge-stats" class="result-box" style="display:none"></div>
    </div>
    <div class="card"><h3>Hedging P&L Fan Chart</h3><div id="hedge-plot" class="plot-box" style="height:450px"></div></div>
  </div>
</div>

<!-- ====================== PRICING ====================== -->
<div class="section" id="sec-pricing">
  <div class="grid g2">
    <div class="card">
      <h3>Vanilla Option Pricer</h3>
      <div class="form-row three">
        <div><label>Spot</label><input id="pr-spot" type="number" value="684.17" step="1"></div>
        <div><label>Strike</label><input id="pr-strike" type="number" value="680" step="1"></div>
        <div><label>Maturity</label><input id="pr-T" type="number" value="0.08" step="0.01"></div>
      </div>
      <div class="form-row">
        <div><label>Volatility</label><input id="pr-sigma" type="number" value="0.18" step="0.01"></div>
        <div><label>Type</label><select id="pr-type"><option value="call">Call</option><option value="put">Put</option></select></div>
      </div>
      <button onclick="runVanilla()">Price</button>
      <div id="vanilla-result" class="result-box" style="display:none"></div>
    </div>
    <div class="card">
      <h3>Exotic Option Pricer</h3>
      <div class="form-row">
        <div><label>Product</label>
          <select id="ex-product"><option>asian_call</option><option>asian_put</option><option>lookback_call</option>
          <option>lookback_put</option><option>autocallable</option><option>cliquet</option>
          <option>variance_swap</option><option>volatility_swap</option></select>
        </div>
        <div><label>Spot</label><input id="ex-spot" type="number" value="100" step="1"></div>
      </div>
      <div class="form-row three">
        <div><label>Strike</label><input id="ex-strike" type="number" value="100" step="1"></div>
        <div><label>Maturity</label><input id="ex-T" type="number" value="0.5" step="0.05"></div>
        <div><label>MC Paths</label><input id="ex-n" type="number" value="50000" step="10000"></div>
      </div>
      <div class="form-row">
        <div><label>Model</label><select id="ex-model"><option value="neural_sde" selected>Neural SDE</option><option value="bergomi">Bergomi</option><option value="bs">Black-Scholes</option></select></div>
        <div></div>
      </div>
      <button onclick="runExotic()">Price</button>
      <div id="exotic-result" class="result-box" style="display:none"></div>
    </div>
  </div>
</div>

<!-- ====================== RISK ====================== -->
<div class="section" id="sec-risk">
  <div class="grid g2">
    <div class="card">
      <h3>VaR / CVaR</h3>
      <div class="form-row">
        <div><label>Spot</label><input id="var-spot" type="number" value="684.17" step="1"></div>
        <div><label>Horizon (days)</label><input id="var-horizon" type="number" value="1" step="1"></div>
      </div>
      <div class="form-row">
        <div><label>Position (instrument)</label><select id="var-inst"><option value="call">Call</option><option value="put">Put</option></select></div>
        <div><label>Strike</label><input id="var-strike" type="number" value="680" step="1"></div>
      </div>
      <div class="form-row">
        <div><label>Quantity</label><input id="var-qty" type="number" value="10" step="1"></div>
        <div><label>MC Paths</label><input id="var-n" type="number" value="50000" step="10000"></div>
      </div>
      <div class="form-row">
        <div><label>Model</label><select id="var-model"><option value="neural_sde" selected>Neural SDE</option><option value="bergomi">Bergomi</option><option value="bs">Black-Scholes</option></select></div>
        <div></div>
      </div>
      <button onclick="runVaR()">Compute</button>
      <div id="var-result" class="result-box" style="display:none"></div>
    </div>
    <div class="card">
      <h3>Deterministic Stress Test</h3>
      <div class="form-row">
        <div><label>Spot</label><input id="st-spot" type="number" value="684.17" step="1"></div>
        <div><label>Position (instrument)</label><select id="st-inst"><option value="call">Call</option><option value="put">Put</option></select></div>
      </div>
      <div class="form-row">
        <div><label>Strike</label><input id="st-strike" type="number" value="680" step="1"></div>
        <div><label>Quantity</label><input id="st-qty" type="number" value="10" step="1"></div>
      </div>
      <button onclick="runStress()">Run Scenarios</button>
      <div id="stress-result" class="result-box" style="display:none"></div>
      <div id="stress-chart" class="plot-box" style="margin-top:12px"></div>
    </div>
  </div>
  <div class="card" style="margin-top:20px">
    <h3>Neural SDE Stress Test <span class="tag stressed">Model-Driven</span></h3>
    <p style="color:var(--muted);font-size:12px;margin-bottom:16px">Generates MC paths from the Neural SDE conditioned on crisis initial states. Produces full P&amp;L distributions with joint spot-vol dynamics.</p>
    <div class="grid g2">
      <div>
        <div class="form-row">
          <div><label>Spot</label><input id="nst-spot" type="number" value="684.17" step="1"></div>
          <div><label>MC Paths</label><input id="nst-n" type="number" value="5000" step="1000"></div>
        </div>
        <div class="form-row">
          <div><label>Position (instrument)</label><select id="nst-inst"><option value="call">Call</option><option value="put">Put</option></select></div>
          <div><label>Strike</label><input id="nst-strike" type="number" value="680" step="1"></div>
        </div>
        <div class="form-row">
          <div><label>Quantity</label><input id="nst-qty" type="number" value="10" step="1"></div>
          <div></div>
        </div>
        <button onclick="runNeuralStress()">Run Neural Stress Test</button>
      </div>
      <div>
        <div id="nst-result" class="result-box" style="max-height:350px">Results will appear here...</div>
      </div>
    </div>
    <div id="nst-chart" class="plot-box" style="margin-top:12px"></div>
  </div>
</div>

<!-- ====================== P&L ====================== -->
<div class="section" id="sec-pnl">
  <div class="grid g2">
    <div class="card">
      <h3>P&L Attribution</h3>
      <div class="form-row three">
        <div><label>Spot (t)</label><input id="pnl-spot" type="number" value="684.17" step="1"></div>
        <div><label>Strike</label><input id="pnl-strike" type="number" value="680" step="1"></div>
        <div><label>Maturity</label><input id="pnl-T" type="number" value="0.08" step="0.01"></div>
      </div>
      <div class="form-row three">
        <div><label>Vol (t)</label><input id="pnl-sigma" type="number" value="0.18" step="0.01"></div>
        <div><label>Rate</label><input id="pnl-r" type="number" value="0.0373" step="0.001"></div>
        <div><label>Type</label><select id="pnl-type"><option value="call">Call</option><option value="put">Put</option></select></div>
      </div>
      <div class="form-row three">
        <div><label>Spot (t+1)</label><input id="pnl-spot2" type="number" value="680" step="1"></div>
        <div><label>Vol (t+1)</label><input id="pnl-sigma2" type="number" value="0.20" step="0.01"></div>
        <div><label>dt</label><input id="pnl-dt" type="number" value="0.003968" step="0.001"></div>
      </div>
      <button onclick="runPnL()">Decompose</button>
    </div>
    <div class="card"><h3>Greeks Contribution</h3><div id="pnl-chart" class="plot-box" style="height:400px"></div></div>
  </div>
</div>

<!-- ====================== REGIME ====================== -->
<div class="section" id="sec-regime">
  <div class="grid g2">
    <div class="card">
      <h3>Market Regime Detection</h3>
      <button onclick="runRegime()">Detect Current Regime</button>
      <div id="regime-result" class="result-box" style="display:none;margin-top:16px"></div>
    </div>
    <div class="card"><h3>Regime Signals</h3><div id="regime-chart" class="plot-box"></div></div>
  </div>
</div>

<!-- ====================== MODELS ====================== -->
<div class="section" id="sec-models">
  <div class="grid g2">
    <div class="card">
      <h3>Loaded Models</h3>
      <button onclick="loadModelsInfo()" style="margin-bottom:16px">Refresh Status</button>
      <div id="models-status" class="result-box">Click Refresh to check model status...</div>
    </div>
    <div class="card">
      <h3>Model Configuration</h3>
      <div id="models-config" class="result-box">Parameters from config/params.yaml will appear here...</div>
    </div>
  </div>
</div>

<!-- ====================== SCRIPTS ====================== -->
<div class="section" id="sec-scripts">
  <div class="grid g2">
    <div class="card">
      <h3>Pipeline Scripts</h3>
      <div style="display:grid;gap:8px">
        <button onclick="runScript('regenerate_data','--mode yahoo')">1. Regenerate Data (Yahoo)</button>
        <button onclick="runScript('regenerate_data','--mode tradingview')">1b. Regenerate Data (TradingView)</button>
        <button onclick="runScript('fetch_options','')">2. Fetch Options Surface</button>
        <button onclick="runScript('calibrate','')">3. Calibrate Parameters</button>
        <button onclick="runScript('train','')">4. Train P + Q Models</button>
        <button onclick="runScript('backtest','')">5. Run Backtest</button>
        <button onclick="runScript('usecases','--run-usecases')">6. Generate Use-Case Report</button>
        <button onclick="runScript('dashboard','')">7. Build Dashboard HTML</button>
      </div>
      <div class="sep" style="margin:16px 0"></div>
      <h3>Diagnostics</h3>
      <div style="display:grid;gap:8px">
        <button class="secondary" onclick="runScript('roughness','')">Verify Roughness</button>
        <button class="secondary" onclick="runScript('hurst','')">Hurst Diagnostic</button>
        <button class="secondary" onclick="runScript('robustness','')">Robustness Check</button>
      </div>
    </div>
    <div class="card">
      <h3>Console Output</h3>
      <div id="script-terminal" class="terminal">Ready. Select a script to run.</div>
    </div>
  </div>
</div>

<!-- ====================== REPORTS ====================== -->
<div class="section" id="sec-reports">
  <div class="card"><h3>Output Reports</h3><div id="reports-content" class="result-box" style="max-height:700px">Click to load reports...</div>
    <button onclick="loadReports()" style="margin-top:12px;width:auto;padding:8px 24px">Load All Reports</button>
  </div>
</div>

</main>

<script>
const API = '';

// Navigation
document.querySelectorAll('nav a').forEach(a => {
  a.addEventListener('click', e => {
    e.preventDefault();
    document.querySelectorAll('nav a').forEach(x => x.classList.remove('active'));
    a.classList.add('active');
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    document.getElementById('sec-' + a.dataset.sec).classList.add('active');
  });
});

// Helpers
async function api(url, body) {
  const opts = body ? {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)} : {};
  const r = await fetch(API + url, opts);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}
function $(id) { return document.getElementById(id); }
function val(id) { return parseFloat($(id).value); }
function sval(id) { return $(id).value; }
function showResult(id, data) { const el = $(id); el.style.display = 'block'; el.textContent = JSON.stringify(data, null, 2); }

const DARK = {paper_bgcolor:'#111827', plot_bgcolor:'#0f172a',
  font:{color:'#e2e8f0',size:12}, xaxis:{gridcolor:'#1e293b',zerolinecolor:'#1e293b'},
  yaxis:{gridcolor:'#1e293b',zerolinecolor:'#1e293b'}, margin:{l:50,r:20,t:30,b:40}};
const DARK3D = {...DARK, scene:{xaxis:{gridcolor:'#1e293b',backgroundcolor:'#0f172a',title:'Strike'},
  yaxis:{gridcolor:'#1e293b',backgroundcolor:'#0f172a',title:'DTE'},
  zaxis:{gridcolor:'#1e293b',backgroundcolor:'#0f172a',title:'IV (%)'}, bgcolor:'#0f172a'}};

// Overview load
async function loadOverview() {
  try {
    const h = await api('/health'); $('kpi-health').textContent = h.status === 'ok' ? 'Online' : 'Error';
  } catch { $('kpi-health').textContent = 'Offline'; }
  try {
    const s = await api('/rates/sofr'); $('kpi-sofr').textContent = (s.rate * 100).toFixed(2) + '%';
  } catch { $('kpi-sofr').textContent = 'N/A'; }
  try {
    const mi = await api('/models/info');
    $('kpi-models').textContent = mi.neural_sde.loaded ? 'Loaded' : 'Not loaded';
    $('kpi-models').parentElement.className = 'card kpi ' + (mi.neural_sde.loaded ? 'green' : 'red');
  } catch { $('kpi-models').textContent = '?'; }
  try {
    const reg = await api('/regime');
    const regime = reg.regime || reg.label || 'unknown';
    $('kpi-regime').textContent = regime.charAt(0).toUpperCase() + regime.slice(1);
    $('kpi-regime').parentElement.className = 'card kpi ' + (regime === 'normal' ? 'green' : regime === 'crisis' ? 'red' : 'yellow');
  } catch { $('kpi-regime').textContent = '--'; }
  try {
    const vd = await fetch(API + '/data/vix-history?limit_days=400').then(r => r.json());
    const y = vd.values;
    const yMax = Math.max(...y);
    Plotly.newPlot('vix-chart', [{x:vd.dates, y:y, type:'scatter', mode:'lines',
      line:{color:'#3b82f6', width:1.5}, fill:'tozeroy', fillcolor:'rgba(59,130,246,.15)'}],
      {...DARK, height:280, margin:{l:50,r:20,t:20,b:50},
       xaxis:{...DARK.xaxis, type:'date', title:'Date', tickformat:'%Y-%m-%d', dtick:86400000*30, nticks:10},
       yaxis:{...DARK.yaxis, title:'VIX', range:[0, Math.max(25, yMax*1.05)]}}, {responsive:true});
  } catch {}
  try {
    const reps = await api('/data/reports');
    let txt = '';
    if (reps.model_usecases_report) {
      const r = reps.model_usecases_report;
      if (r.profiles && r.profiles.risk && r.profiles.risk.risk_scenarios) {
        const rs = r.profiles.risk.risk_scenarios;
        txt += 'Risk (P-model):\n  VaR 95: ' + (rs.VaR_95*100).toFixed(2) + '%\n  ES 95:  ' + (rs.ES_95*100).toFixed(2) + '%\n  VaR 99: ' + (rs.VaR_99*100).toFixed(2) + '%\n\n';
      }
      if (r.profiles && r.profiles.pricing && r.profiles.pricing.risk_scenarios) {
        const ps = r.profiles.pricing.risk_scenarios;
        txt += 'Pricing (Q-model):\n  Mean vol: ' + (ps.terminal_vol_mean*100).toFixed(1) + '%\n  Vol P95:  ' + (ps.terminal_vol_p95*100).toFixed(1) + '%\n';
      }
    }
    if (reps.backtest_results) {
      txt += '\nBacktest:\n' + JSON.stringify(reps.backtest_results, null, 2).substring(0, 600) + '...';
    }
    $('overview-stats').textContent = txt || 'No reports yet. Run the pipeline first.';
  } catch { $('overview-stats').textContent = 'Run the pipeline to generate reports.'; }
}

// MC Paths animation
async function runMC() {
  const data = await api('/animate/mc-paths', {
    spot: val('mc-spot'), sigma: val('mc-sigma'), T: val('mc-T'),
    n_paths: parseInt($('mc-n').value), n_frames: parseInt($('mc-frames').value),
    model: sval('mc-model')
  });
  const t = data.t;
  const paths = data.paths;
  const n = paths.length;
  const nf = t.length;
  const colors = paths.map((_,i) => `hsla(${(i*360/n)%360},70%,60%,0.3)`);

  // Build frames for animation
  const traces0 = paths.map((p,i) => ({x:[t[0]], y:[p[0]], mode:'lines', line:{color:colors[i],width:0.8},
    showlegend:false, hoverinfo:'skip'}));
  const frames = [];
  for (let f = 1; f < nf; f++) {
    frames.push({name:'f'+f, data: paths.map((p,i) => ({x:t.slice(0,f+1), y:p.slice(0,f+1)}))});
  }
  const layout = {...DARK, xaxis:{...DARK.xaxis,title:'Trading Days',range:[0,t[nf-1]]},
    yaxis:{...DARK.yaxis,title:'Price',range:[data.spot*0.7,data.spot*1.3]},
    updatemenus:[{type:'buttons',showactive:false,x:0,y:1.15,
      buttons:[{label:'Play',method:'animate',args:[null,{frame:{duration:30,redraw:false},fromcurrent:true,transition:{duration:0}}]},
               {label:'Pause',method:'animate',args:[[null],{mode:'immediate',frame:{duration:0}}]}]}]};
  Plotly.newPlot('mc-plot', traces0, layout, {responsive:true}).then(() => {
    Plotly.addFrames('mc-plot', frames);
  });

  // Histogram of terminal values
  const terminal = paths.map(p => p[p.length-1]);
  Plotly.newPlot('mc-hist', [{x:terminal, type:'histogram', nbinsx:50,
    marker:{color:'rgba(139,92,246,0.7)',line:{color:'#8b5cf6',width:1}}}],
    {...DARK, xaxis:{...DARK.xaxis,title:'Terminal Price'},yaxis:{...DARK.yaxis,title:'Count'}}, {responsive:true});
}

// Vol Surface (smoothed surface + raw quotes)
async function loadVolSurface() {
  const d = await api('/animate/vol-surface');
  const traces = [];
  if (d.grid_ivs && d.grid_ivs.length && d.grid_strikes && d.grid_dtes) {
    traces.push({
      type:'surface', x:d.grid_strikes, y:d.grid_dtes, z:d.grid_ivs,
      opacity:0.82, colorscale:'Viridis', name:'Smoothed', showscale:true,
      colorbar:{title:'IV%',titlefont:{color:'#888'},tickfont:{color:'#888'}},
    });
  }
  traces.push({
    x:d.strikes, y:d.dtes, z:d.ivs, type:'scatter3d', mode:'markers', name:'Quotes',
    marker:{size:2.5,color:d.ivs,colorscale:'Plasma',showscale:false},
    text:d.types,
  });
  Plotly.newPlot('vol-surface-plot', traces,
    {...DARK3D, margin:{l:0,r:0,t:0,b:0},
     scene:{...DARK3D.scene,xaxis:{...DARK3D.scene.xaxis,title:'Strike'},
      yaxis:{...DARK3D.scene.yaxis,title:'DTE'},zaxis:{...DARK3D.scene.zaxis,title:'IV (%)'}}},
    {responsive:true});
}

// Hedging animation
async function runHedgeAnim() {
  const d = await api('/animate/hedging', {
    spot:val('hg-spot'), strike:val('hg-strike'), sigma:val('hg-sigma'),
    T:val('hg-T'), opt_type:sval('hg-type'), n_mc_paths:parseInt($('hg-n').value),
    model:sval('hg-model')
  });
  const t = d.t_days;
  const traces = [
    {x:t, y:d.mean_pnl, name:'Mean Hedge P&L', line:{color:'#3b82f6',width:2.5}},
    {x:t, y:d.p95, name:'95th pct', line:{color:'#10b981',width:1,dash:'dash'}, fill:'tonexty', fillcolor:'rgba(16,185,129,0.05)'},
    {x:t, y:d.p5, name:'5th pct', line:{color:'#ef4444',width:1,dash:'dash'}, fill:'tonexty', fillcolor:'rgba(59,130,246,0.1)'},
  ];
  d.sample_paths.forEach((p,i) => {
    traces.push({x:t, y:p, name:'Path '+(i+1), line:{color:`hsla(${i*72},60%,50%,0.4)`,width:0.8}, showlegend:false});
  });
  Plotly.newPlot('hedge-plot', traces,
    {...DARK, xaxis:{...DARK.xaxis,title:'Days'},yaxis:{...DARK.yaxis,title:'Hedge P&L ($)'},
     showlegend:true, legend:{x:0,y:1,bgcolor:'transparent',font:{size:10}}}, {responsive:true});

  $('hedge-stats').style.display = 'block';
  $('hedge-stats').textContent =
    `Option price: $${d.option_price.toFixed(4)}\nFinal mean P&L: $${d.mean_pnl[d.mean_pnl.length-1].toFixed(4)}\n` +
    `5th pct: $${d.p5[d.p5.length-1].toFixed(4)}\n95th pct: $${d.p95[d.p95.length-1].toFixed(4)}`;
}

// Vanilla pricer
async function runVanilla() {
  const d = await api('/price/vanilla', {spot:val('pr-spot'),strike:val('pr-strike'),T:val('pr-T'),
    sigma:val('pr-sigma'),opt_type:sval('pr-type')});
  showResult('vanilla-result', d);
}

// Exotic pricer
async function runExotic() {
  const d = await api('/price/exotic', {product:sval('ex-product'),spot:val('ex-spot'),
    strike:val('ex-strike'),T:val('ex-T'),n_mc_paths:parseInt($('ex-n').value),
    model:sval('ex-model'), extra_params:{sigma:0.20}});
  showResult('exotic-result', d);
}

// VaR
async function runVaR() {
  const d = await api('/risk/var', {spot:val('var-spot'),horizon_days:parseInt($('var-horizon').value),
    n_mc_paths:parseInt($('var-n').value), model:sval('var-model'),
    positions:[{instrument:sval('var-inst'),strike:val('var-strike'),T:0.08,quantity:parseInt($('var-qty').value)}]});
  showResult('var-result', d);
}

// Stress
async function runStress() {
  const d = await api('/risk/stress', {spot:val('st-spot'),
    positions:[{instrument:sval('st-inst'),strike:val('st-strike'),T:0.08,quantity:parseInt($('st-qty').value)}]});
  showResult('stress-result', d);
  const names = Object.keys(d).filter(n => n !== '_info');
  const vals = names.map(n => {
    if (typeof d[n] === 'object' && d[n].pnl !== undefined) return d[n].pnl;
    if (typeof d[n] === 'number') return d[n];
    return 0;
  });
  const colors = vals.map(v => v < 0 ? '#ef4444' : '#10b981');
  Plotly.newPlot('stress-chart',[{x:names,y:vals,type:'bar',marker:{color:colors}}],
    {...DARK,xaxis:{...DARK.xaxis},yaxis:{...DARK.yaxis,title:'P&L'}},{responsive:true});
}

// Neural Stress Test
async function runNeuralStress() {
  const el = $('nst-result');
  el.textContent = 'Running Neural SDE stress test... (may take a few seconds)';
  try {
    const d = await api('/risk/stress/neural', {spot:val('nst-spot'), n_mc_paths:parseInt($('nst-n').value),
      positions:[{instrument:sval('nst-inst'),strike:val('nst-strike'),T:0.25,quantity:parseInt($('nst-qty').value)}]});
    el.textContent = JSON.stringify(d, null, 2);
    // Build grouped bar chart: VaR95, CVaR95 per scenario
    const scenarios = Object.keys(d);
    const var95 = scenarios.map(s => d[s].VaR_95 || 0);
    const cvar95 = scenarios.map(s => d[s].CVaR_95 || 0);
    const meanPnl = scenarios.map(s => d[s].expected_pnl || 0);
    Plotly.newPlot('nst-chart',[
      {x:scenarios, y:var95, name:'VaR 95%', type:'bar', marker:{color:'#ef4444'}},
      {x:scenarios, y:cvar95, name:'CVaR 95%', type:'bar', marker:{color:'#f59e0b'}},
      {x:scenarios, y:meanPnl, name:'E[P&L]', type:'bar', marker:{color:'#3b82f6'}}
    ],{...DARK,barmode:'group',xaxis:{...DARK.xaxis},yaxis:{...DARK.yaxis,title:'$ P&L'},
      showlegend:true,legend:{bgcolor:'transparent',font:{size:11}}},{responsive:true});
  } catch(e) {
    el.textContent = 'Error: ' + e.message + '\n\nMake sure Neural SDE model is trained: python bin/train_multi.py';
  }
}

// Models Info
async function loadModelsInfo() {
  try {
    const d = await api('/models/info');
    let txt = '';
    ['neural_sde','bergomi','bs'].forEach(m => {
      const icon = d[m].loaded ? '\u2705' : '\u274c';
      txt += icon + ' ' + m + ': ' + d[m].status + '\n';
    });
    txt += '\nModel files:\n';
    (d.model_files || []).forEach(f => { txt += '  ' + f + '\n'; });
    $('models-status').textContent = txt;
    if (d.config) {
      $('models-config').textContent = JSON.stringify(d.config, null, 2);
    }
  } catch(e) {
    $('models-status').textContent = 'Error: ' + e.message;
  }
}

// P&L
async function runPnL() {
  const d = await api('/pnl/attribute', {spot:val('pnl-spot'),strike:val('pnl-strike'),T:val('pnl-T'),
    r:val('pnl-r'),sigma:val('pnl-sigma'),opt_type:sval('pnl-type'),
    spot_new:val('pnl-spot2'),sigma_new:val('pnl-sigma2'),dt:val('pnl-dt')});
  const greeks = ['delta_pnl','gamma_pnl','vega_pnl','theta_pnl','vanna_pnl','volga_pnl','rho_pnl','residual'];
  const vals = greeks.map(g => d[g] || 0);
  const colors = vals.map(v => v >= 0 ? '#10b981' : '#ef4444');
  Plotly.newPlot('pnl-chart',[
    {x:greeks.map(g=>g.replace('_pnl','')),y:vals,type:'bar',marker:{color:colors},name:'Contribution'},
    {x:['TOTAL'],y:[d.total_pnl],type:'bar',marker:{color:'#3b82f6'},name:'Total'}
  ],{...DARK,barmode:'group',xaxis:{...DARK.xaxis},yaxis:{...DARK.yaxis,title:'P&L ($)'},
    showlegend:true,legend:{bgcolor:'transparent'}},{responsive:true});
}

// Regime
async function runRegime() {
  const d = await api('/regime');
  showResult('regime-result', d);
  const signals = d.signals || d;
  const keys = Object.keys(signals).filter(k => typeof signals[k] === 'number');
  const vals = keys.map(k => signals[k]);
  Plotly.newPlot('regime-chart',[{type:'scatterpolar',r:vals,theta:keys,fill:'toself',
    fillcolor:'rgba(59,130,246,0.15)',line:{color:'#3b82f6'}}],
    {...DARK,polar:{bgcolor:'#0f172a',radialaxis:{gridcolor:'#1e293b',color:'#64748b'},
    angularaxis:{gridcolor:'#1e293b',color:'#64748b'}}},{responsive:true});
}

// Script runner
async function runScript(name, args) {
  const term = $('script-terminal');
  term.innerHTML = '<span class="info">[Running ' + name + ' ' + args + ']</span>\n';
  const url = API + '/scripts/run/' + name + (args ? '?args=' + encodeURIComponent(args) : '');
  try {
    const response = await fetch(url);
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      const text = decoder.decode(value);
      const lines = text.split('\n');
      lines.forEach(line => {
        if (line.startsWith('data: ')) {
          const msg = line.substring(6);
          if (msg.startsWith('[EXIT')) {
            term.innerHTML += '<span class="info">' + msg + '</span>\n';
          } else if (msg.toLowerCase().includes('error') || msg.toLowerCase().includes('traceback')) {
            term.innerHTML += '<span class="err">' + msg + '</span>\n';
          } else {
            term.innerHTML += msg + '\n';
          }
          term.scrollTop = term.scrollHeight;
        }
      });
    }
  } catch(e) {
    term.innerHTML += '<span class="err">Error: ' + e.message + '</span>\n';
  }
}

// Reports
async function loadReports() {
  const d = await api('/data/reports');
  $('reports-content').textContent = JSON.stringify(d, null, 2);
}

// Init
loadOverview();
</script>
</body>
</html>"""


# =====================================================================
#  Server entry point
# =====================================================================

if __name__ == "__main__":
    import uvicorn
    from utils.config import load_config

    cfg = load_config()
    api_cfg = cfg.get("api", {})
    port = api_cfg.get("port", 8000)

    print("=" * 60)
    print("   DeepRoughVol - Interactive Dashboard")
    print("=" * 60)
    print(f"   UI:      http://localhost:{port}")
    print(f"   Swagger: http://localhost:{port}/docs")
    print("=" * 60)

    uvicorn.run(
      "bin.apps.api_server:app",
        host=api_cfg.get("host", "0.0.0.0"),
        port=port,
        workers=api_cfg.get("workers", 1),
        reload=False,
    )
