"""
DeepRoughVol REST API
=====================
FastAPI server exposing pricing, risk, regime detection, and analytics.

Endpoints:
  GET  /health                     Health check
  GET  /regime                     Current market regime
  POST /price/vanilla              Price European option
  POST /price/exotic               Price exotic option
  POST /risk/var                   Compute VaR/CVaR
  POST /risk/stress                Run stress scenarios
  POST /hedge/simulate             Simulate delta hedging
  POST /pnl/attribute              P&L attribution
  POST /calibrate/eta              Calibrate eta from VVIX
  GET  /rates/sofr                 Current SOFR rate

Launch:
    python bin/api_server.py
    # or: uvicorn bin.api_server:app --host 0.0.0.0 --port 8000
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import numpy as np

app = FastAPI(
    title="DeepRoughVol API",
    description="Neural SDE-based volatility modeling, pricing, and risk management",
    version="2.0.0",
)


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

class ExoticPriceRequest(BaseModel):
    product: str = "asian_call"
    spot: float = 100.0
    strike: float = 100.0
    T: float = 0.25
    r: Optional[float] = None
    n_mc_paths: int = 10000
    model: str = "neural_sde"
    extra_params: Dict = Field(default_factory=dict)

class VaRRequest(BaseModel):
    spot: float = 100.0
    positions: List[Dict] = Field(default_factory=list)
    n_mc_paths: int = 10000
    horizon_days: int = 1
    model: str = "neural_sde"

class StressRequest(BaseModel):
    spot: float = 100.0
    positions: List[Dict] = Field(default_factory=list)
    scenarios: Optional[Dict] = None

class HedgeRequest(BaseModel):
    spot: float = 100.0
    strike: float = 100.0
    T: float = 0.25
    sigma: float = 0.20
    opt_type: str = "call"
    n_mc_paths: int = 5000
    hedge_freq: str = "daily"
    model: str = "neural_sde"

class PnLRequest(BaseModel):
    spot: float = 100.0
    strike: float = 100.0
    T: float = 0.25
    r: float = 0.045
    sigma: float = 0.20
    opt_type: str = "call"
    spot_new: float = 99.0
    sigma_new: Optional[float] = None
    dt: float = 0.003968  # 1/252
    r_new: Optional[float] = None

class EtaCalibRequest(BaseModel):
    H: float = 0.1
    window_days: int = 252


# =====================================================================
#  Singletons (lazy-loaded)
# =====================================================================

_sofr = None
_vvix_cal = None
_regime_detector = None


def _get_sofr():
    global _sofr
    if _sofr is None:
        from utils.sofr_loader import SOFRRateLoader
        _sofr = SOFRRateLoader()
    return _sofr


def _get_vvix():
    global _vvix_cal
    if _vvix_cal is None:
        from utils.vvix_calibrator import VVIXCalibrator
        _vvix_cal = VVIXCalibrator()
    return _vvix_cal


def _get_regime():
    global _regime_detector
    if _regime_detector is None:
        from quant.regime_detector import RegimeDetector
        _regime_detector = RegimeDetector()
    return _regime_detector


def _get_r(r_override=None):
    if r_override is not None:
        return r_override
    sofr = _get_sofr()
    return sofr.get_rate() if sofr.is_available else 0.045


# =====================================================================
#  Endpoints
# =====================================================================

@app.get("/health")
def health():
    return {"status": "ok", "model": "DeepRoughVol", "version": "2.0.0"}


@app.get("/regime")
def get_regime():
    """Detect current market regime from VIX/VVIX/term structure."""
    try:
        detector = _get_regime()
        return detector.detect()
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.get("/rates/sofr")
def get_sofr_rate(as_of_date: Optional[str] = None):
    """Get current SOFR risk-free rate."""
    sofr = _get_sofr()
    return {
        "rate": sofr.get_rate(as_of_date),
        "available": sofr.is_available,
        "as_of_date": as_of_date or "latest",
    }


@app.post("/price/vanilla")
def price_vanilla(req: VanillaPriceRequest):
    """Price a European option using Black-Scholes."""
    from utils.black_scholes import BlackScholes
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
    """Price exotic options via Monte Carlo."""
    from quant.exotic_pricer import ExoticPricer

    r = _get_r(req.r)
    pricer = ExoticPricer(spot=req.spot, r=r, T=req.T)

    # Generate paths (simplified: GBM for API responsiveness)
    n = req.n_mc_paths
    n_steps = 252
    dt_sim = req.T / n_steps
    z = np.random.randn(n, n_steps)
    sigma = req.extra_params.get("sigma", 0.20)
    log_ret = (r - 0.5 * sigma**2) * dt_sim + sigma * np.sqrt(dt_sim) * z
    s_paths = req.spot * np.exp(np.cumsum(log_ret, axis=1))
    s_paths = np.column_stack([np.full(n, req.spot), s_paths])

    product = req.product
    strike = req.strike

    method_map = {
        "asian_call": lambda: pricer.asian_call(s_paths, strike),
        "asian_put": lambda: pricer.asian_put(s_paths, strike),
        "lookback_call": lambda: pricer.lookback_call(s_paths, strike),
        "lookback_put": lambda: pricer.lookback_put(s_paths, strike),
        "autocallable": lambda: pricer.autocallable(s_paths, **req.extra_params),
        "cliquet": lambda: pricer.cliquet(s_paths, **req.extra_params),
        "variance_swap": lambda: pricer.variance_swap(s_paths, sigma**2),
        "volatility_swap": lambda: pricer.volatility_swap(s_paths, sigma),
    }

    if product not in method_map:
        raise HTTPException(400, f"Unknown product: {product}. Available: {list(method_map.keys())}")

    return method_map[product]()


@app.post("/risk/var")
def compute_var(req: VaRRequest):
    """Compute VaR/CVaR for a portfolio."""
    from quant.risk_engine import RiskEngine

    r = _get_r()
    engine = RiskEngine(spot=req.spot, r=r)

    for pos in req.positions:
        p = dict(pos)
        if 'opt_type' in p:
            p['instrument'] = p.pop('opt_type')
        engine.add_position(**p)

    if not engine.positions:
        engine.add_position('call', strike=req.spot, T=0.25, quantity=1)

    n = req.n_mc_paths
    n_steps = max(10, req.horizon_days * 26)
    T_sim = req.horizon_days / 252.0
    dt_sim = T_sim / n_steps
    z = np.random.randn(n, n_steps)
    sigma = 0.20
    log_ret = (r - 0.5 * sigma**2) * dt_sim + sigma * np.sqrt(dt_sim) * z
    s_paths = req.spot * np.exp(np.cumsum(log_ret, axis=1))
    s_paths = np.column_stack([np.full(n, req.spot), s_paths])

    report = engine.compute_var(s_paths)

    return {
        "var_95": report.var_95,
        "var_99": report.var_99,
        "cvar_95": report.cvar_95,
        "cvar_99": report.cvar_99,
        "stressed_var_99": report.stressed_var_99,
        "expected_pnl": report.expected_pnl,
        "pnl_std": report.pnl_std,
        "skew": report.pnl_skew,
        "kurtosis": report.pnl_kurtosis,
        "n_scenarios": report.n_scenarios,
    }


@app.post("/risk/stress")
def run_stress(req: StressRequest):
    """Run stress test scenarios."""
    from quant.risk_engine import RiskEngine

    r = _get_r()
    engine = RiskEngine(spot=req.spot, r=r)

    for pos in req.positions:
        p = dict(pos)
        if 'opt_type' in p:
            p['instrument'] = p.pop('opt_type')
        engine.add_position(**p)

    if not engine.positions:
        engine.add_position('call', strike=req.spot, T=0.25, quantity=1)

    return engine.stress_test(req.scenarios)


@app.post("/pnl/attribute")
def pnl_attribute(req: PnLRequest):
    """P&L attribution via Greeks decomposition."""
    from quant.pnl_attribution import PnLAttributor

    attr = PnLAttributor(req.spot, req.strike, req.T, req.r, req.sigma, req.opt_type)
    result = attr.attribute(req.spot_new, req.sigma_new, req.dt, req.r_new)

    return {
        "total_pnl": result.total_pnl,
        "delta_pnl": result.delta_pnl,
        "gamma_pnl": result.gamma_pnl,
        "vega_pnl": result.vega_pnl,
        "theta_pnl": result.theta_pnl,
        "vanna_pnl": result.vanna_pnl,
        "volga_pnl": result.volga_pnl,
        "rho_pnl": result.rho_pnl,
        "residual": result.residual,
        "explained_pct": result.explained_pct,
    }


@app.post("/hedge/simulate")
def simulate_hedge(req: HedgeRequest):
    """Simulate delta-hedging and compare strategies."""
    from quant.hedging_simulator import HedgingSimulator

    r = _get_r()
    sim = HedgingSimulator(
        spot=req.spot, strike=req.strike, T=req.T,
        r=r, iv=req.sigma, opt_type=req.opt_type,
    )

    n = req.n_mc_paths
    n_steps = 252
    dt_sim = req.T / n_steps
    z = np.random.randn(n, n_steps)
    sigma = req.sigma
    log_ret = (r - 0.5 * sigma**2) * dt_sim + sigma * np.sqrt(dt_sim) * z
    s_paths = req.spot * np.exp(np.cumsum(log_ret, axis=1))
    s_paths = np.column_stack([np.full(n, req.spot), s_paths])

    results = sim.run(s_paths, hedge_freq=req.hedge_freq)

    out = {}
    for name, hr in results.items():
        out[name] = {
            "mean_pnl": float(hr.mean_pnl),
            "std_pnl": float(hr.std_pnl),
            "mean_abs_pnl": float(hr.mean_abs_pnl),
            "max_loss": float(hr.max_loss),
            "sharpe": float(hr.sharpe),
            "tracking_error": float(hr.tracking_error),
            "hedge_cost": float(hr.hedge_cost),
            "n_rebalances": hr.n_rebalances,
        }
    return out


@app.post("/calibrate/eta")
def calibrate_eta(req: EtaCalibRequest):
    """Calibrate vol-of-vol eta from VVIX."""
    cal = _get_vvix()
    return cal.estimate_eta(H=req.H, window_days=req.window_days)


# =====================================================================
#  Server entry point
# =====================================================================

if __name__ == "__main__":
    import uvicorn
    from utils.config import load_config

    cfg = load_config()
    api_cfg = cfg.get("api", {})

    print("=" * 60)
    print("   DeepRoughVol API Server")
    print("=" * 60)
    print(f"   Docs: http://localhost:{api_cfg.get('port', 8000)}/docs")

    uvicorn.run(
        "bin.api_server:app",
        host=api_cfg.get("host", "0.0.0.0"),
        port=api_cfg.get("port", 8000),
        workers=api_cfg.get("workers", 1),
        reload=False,
    )
