"""
Deep Hedging Module
===================
Learns an optimal hedging policy δ_φ via risk-measure minimization.

The policy network maps market features → hedge ratio, trained to minimize
a convex risk measure (entropic / CVaR) of terminal hedging P&L, including
transaction costs. This removes the need for model-specific Greeks.

Features (input to policy, dim=6):
    S_t/S₀, log(V_t), log(S_t/K), τ/T, δ_{t-1}, Δ^BS (hint)

Supported derivatives:
    vanilla (call/put), asian, lookback, cliquet

Baselines (same scan infrastructure for fair comparison):
    BS delta, Bartlett (min-variance), Neural-AD

References:
    Buehler, Gonon, Teichmann & Wood (2019). Deep Hedging. QF 19(8).
    Horvath, Teichmann & Žurič (2021). Deep Hedging under Rough Volatility.
    Carbonneau & Godin (2021). Equal Risk Pricing with Deep Hedging.

Usage:
    from engine.deep_hedger import DeepHedger
    dh = DeepHedger(spot=5500, strike=5500, T=30/365, r=0.045, iv=0.17)
    history = dh.train(spot_paths, var_paths, n_epochs=500)
    result  = dh.evaluate(test_spots, test_vars)
"""

import jax
import jax.numpy as jnp
import jax.nn as jnn
import equinox as eqx
import optax
import numpy as np
import json
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from functools import partial

from quant.models.black_scholes import BlackScholes


# ═══════════════════════════════════════════════════════════════════
#  1.  RISK  MEASURES  (differentiable)
# ═══════════════════════════════════════════════════════════════════

def entropic_risk(pnl: jnp.ndarray, lam: float = 1.0) -> jnp.ndarray:
    """
    Entropic risk measure  ρ_λ(X) = (1/λ) · log E[exp(-λX)].

    Smooth, convex, differentiable everywhere.  Interpolates:
      λ → 0   →  -E[X]   (risk-neutral)
      λ → ∞   →  -ess inf X  (worst case)
    Ref: Buehler et al. (2019) §3.2.
    """
    neg = -lam * pnl
    c = jnp.max(neg)                              # log-sum-exp trick
    return (c + jnp.log(jnp.mean(jnp.exp(neg - c)))) / lam


def cvar_risk(pnl: jnp.ndarray, alpha: float = 0.95) -> jnp.ndarray:
    """CVaR / Expected Shortfall at level α.  Differentiable via sort."""
    losses = -pnl
    k = jnp.int32(jnp.floor(losses.shape[0] * alpha))
    return jnp.mean(jnp.sort(losses)[k:])


def mean_variance_risk(pnl: jnp.ndarray, lam: float = 0.5) -> jnp.ndarray:
    """Mean-variance:  -E[PnL] + λ · Var[PnL]."""
    return -jnp.mean(pnl) + lam * jnp.var(pnl)


# ═══════════════════════════════════════════════════════════════════
#  2.  PAYOFF  FUNCTIONS  (JAX-compatible)
# ═══════════════════════════════════════════════════════════════════

def _vanilla_payoff(path, strike, is_call):
    S_T = path[-1]
    return jnp.where(is_call, jnp.maximum(S_T - strike, 0.0),
                     jnp.maximum(strike - S_T, 0.0))

def _asian_payoff(path, strike, is_call):
    avg = jnp.mean(path[1:])
    return jnp.where(is_call, jnp.maximum(avg - strike, 0.0),
                     jnp.maximum(strike - avg, 0.0))

def _lookback_payoff(path, strike, is_call):
    return jnp.where(is_call,
                     jnp.maximum(jnp.max(path) - strike, 0.0),
                     jnp.maximum(strike - jnp.min(path), 0.0))

def _cliquet_payoff(path, cap, floor, g_floor):
    rets = path[1:] / path[:-1] - 1.0
    return path[0] * jnp.maximum(jnp.sum(jnp.clip(rets, floor, cap)), g_floor)


def _make_payoff_fn(payoff_type: str, strike: float,
                    is_call: bool, **kw) -> Callable:
    """Factory → JAX-traceable payoff closure (constant across paths)."""
    if payoff_type == "vanilla":
        return lambda p: _vanilla_payoff(p, strike, is_call)
    if payoff_type == "asian":
        return lambda p: _asian_payoff(p, strike, is_call)
    if payoff_type == "lookback":
        return lambda p: _lookback_payoff(p, strike, is_call)
    if payoff_type == "cliquet":
        cap   = kw.get("local_cap", 0.05)
        floor = kw.get("local_floor", -0.03)
        gf    = kw.get("global_floor", 0.0)
        return lambda p: _cliquet_payoff(p, cap, floor, gf)
    raise ValueError(f"Unknown payoff type: {payoff_type}")


# ═══════════════════════════════════════════════════════════════════
#  3.  JAX  BS  UTILITIES
# ═══════════════════════════════════════════════════════════════════

def _bs_delta_jax(S, K, tau, r, sigma):
    """BS call delta — JAX-differentiable."""
    tau_s = jnp.maximum(tau, 1e-8)
    d1 = (jnp.log(S / K) + (r + 0.5 * sigma**2) * tau_s) / (sigma * jnp.sqrt(tau_s))
    return jax.scipy.stats.norm.cdf(d1)


# ═══════════════════════════════════════════════════════════════════
#  4.  POLICY  NETWORK
# ═══════════════════════════════════════════════════════════════════

class HedgingPolicy(eqx.Module):
    """
    Neural hedging policy:  features ∈ ℝ⁶  →  δ ∈ [-δ_max, δ_max].

    Architecture: 3-layer MLP (GELU) with tanh output scaling.
    The BS-delta *hint* feature is key for fast convergence
    (Buehler et al. 2019 §4.2).
    """
    net: eqx.nn.MLP
    delta_max: float

    def __init__(self, key, input_dim=6, width=64, depth=3, delta_max=2.0):
        self.net = eqx.nn.MLP(in_size=input_dim, out_size=1,
                               width_size=width, depth=depth,
                               activation=jnn.gelu, key=key)
        self.delta_max = delta_max

    def __call__(self, features):
        return self.delta_max * jnp.tanh(jnp.squeeze(self.net(features)))


# ═══════════════════════════════════════════════════════════════════
#  5.  DELTA-FUNCTION  FACTORIES  (deep, BS, Bartlett)
# ═══════════════════════════════════════════════════════════════════

def _make_deep_delta(policy, S0, strike, T, r, is_call):
    """Closure: policy network → hedge ratio."""
    def fn(S, V, tau, delta_prev):
        sigma = jnp.sqrt(jnp.maximum(V, 1e-8))
        bs_d  = _bs_delta_jax(S, strike, tau, r, sigma)
        bs_d  = jnp.where(is_call, bs_d, bs_d - 1.0)
        feats = jnp.array([
            S / S0,
            jnp.clip(jnp.log(jnp.maximum(V, 1e-10)), -7.0, 2.0),
            jnp.clip(jnp.log(S / strike), -1.0, 1.0),
            tau / T,
            delta_prev,
            bs_d,
        ])
        return policy(feats)
    return fn


def _make_bs_delta(strike, r, is_call):
    """Closure: plain BS delta hedge."""
    def fn(S, V, tau, _delta_prev):
        sigma = jnp.sqrt(jnp.maximum(V, 1e-8))
        d = _bs_delta_jax(S, strike, tau, r, sigma)
        return jnp.where(is_call, d, d - 1.0)
    return fn


def _make_bartlett_delta(strike, r, rho, is_call):
    """Closure: minimum-variance (Bartlett) delta hedge."""
    def fn(S, V, tau, _delta_prev):
        sigma = jnp.sqrt(jnp.maximum(V, 1e-8))
        tau_s = jnp.maximum(tau, 1e-8)
        sqrt_t = jnp.sqrt(tau_s)
        d1 = (jnp.log(S / strike) + (r + 0.5 * sigma**2) * tau_s) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t
        bs_d = jax.scipy.stats.norm.cdf(d1)
        bs_d = jnp.where(is_call, bs_d, bs_d - 1.0)
        vanna = -jax.scipy.stats.norm.pdf(d1) * d2 / sigma
        correction = vanna * rho * sigma / jnp.maximum(S, 1e-4)
        return jnp.clip(bs_d + correction, -2.0, 2.0)
    return fn


# ═══════════════════════════════════════════════════════════════════
#  6.  GENERIC  HEDGING  SCAN  +  BATCH
# ═══════════════════════════════════════════════════════════════════

def _hedge_pnl_single(delta_fn, spot_path, var_path,
                      T, r, tc_rate, premium, payoff_fn):
    """
    Single-path hedging P&L via jax.lax.scan.

    P&L = cash_T + δ_T · S_T  -  payoff(S_{0:T})
    """
    n = spot_path.shape[0] - 1
    dt = T / n
    taus = T - jnp.arange(n, dtype=jnp.float32) * dt

    def step(carry, inp):
        cash, dp = carry
        S, V, tau = inp
        dn = delta_fn(S, V, tau, dp)
        trade = dn - dp
        nc = cash - trade * S - jnp.abs(trade) * S * tc_rate
        nc = nc * jnp.exp(r * dt)
        return (nc, dn), None

    (fc, fd), _ = jax.lax.scan(
        step,
        (jnp.float32(premium), jnp.float32(0.0)),
        (spot_path[:-1], var_path[:-1], taus),
    )
    return fc + fd * spot_path[-1] - payoff_fn(spot_path)


def _hedge_pnl_batch(delta_fn, spot_paths, var_paths,
                     T, r, tc_rate, premium, payoff_fn):
    """Vectorised over paths via jax.vmap."""
    single = lambda s, v: _hedge_pnl_single(
        delta_fn, s, v, T, r, tc_rate, premium, payoff_fn)
    return jax.vmap(single)(spot_paths, var_paths)


# ═══════════════════════════════════════════════════════════════════
#  7.  STATISTICS  HELPER
# ═══════════════════════════════════════════════════════════════════

def _compute_stats(pnl: np.ndarray, premium: float) -> Dict[str, float]:
    """Summary statistics for a P&L distribution."""
    pnl = np.asarray(pnl).ravel()
    s = np.std(pnl)
    return {
        "mean_pnl":       float(np.mean(pnl)),
        "std_pnl":        float(s),
        "sharpe":         float(np.mean(pnl) / s) if s > 1e-10 else 0.0,
        "var_95":         float(-np.percentile(pnl, 5)),
        "cvar_95":        float(-np.mean(pnl[pnl <= np.percentile(pnl, 5)])),
        "var_99":         float(-np.percentile(pnl, 1)),
        "cvar_99":        float(-np.mean(pnl[pnl <= np.percentile(pnl, 1)])),
        "max_loss":       float(np.min(pnl)),
        "tracking_error": float(s / premium) if premium > 1e-10 else 0.0,
    }


# ═══════════════════════════════════════════════════════════════════
#  8.  DEEP  HEDGER  —  MAIN  CLASS
# ═══════════════════════════════════════════════════════════════════

class DeepHedger:
    """
    End-to-end deep hedging trainer + evaluator.

    Trains a neural hedging policy to minimise a convex risk measure
    of terminal hedging P&L.  Provides built-in BS and Bartlett
    baselines on the *same* paths and payoff for fair comparison.

    Example
    -------
    >>> dh = DeepHedger(spot=5500, strike=5500, T=30/365, r=0.045, iv=0.17)
    >>> history = dh.train(spot_paths, var_paths, n_epochs=500)
    >>> result  = dh.evaluate(test_spots, test_vars)
    >>> print(result["summary"])
    """

    def __init__(
        self,
        spot: float,
        strike: float,
        T: float,
        r: float          = 0.045,
        iv: float         = 0.17,
        opt_type: str     = "call",
        tc_bps: float     = 2.0,
        risk_measure: str = "entropic",
        risk_param: float = 1.0,
        payoff_type: str  = "vanilla",
        payoff_kwargs: dict = None,
        rho: float        = -0.7,
        premium: float    = None,
        key               = None,
    ):
        self.spot   = float(spot)
        self.strike = float(strike)
        self.T      = float(T)
        self.r      = float(r)
        self.iv     = float(iv)
        self.is_call = (opt_type == "call")
        self.tc_rate = tc_bps / 10_000.0
        self.rho     = float(rho)
        self.payoff_type = payoff_type
        self.payoff_kwargs = payoff_kwargs or {}
        self.risk_measure_name = risk_measure
        self.risk_param = float(risk_param)

        # Option premium (initial cash from selling the option)
        if premium is not None:
            self.premium = float(premium)
        else:
            self.premium = float(BlackScholes.price(
                spot, strike, T, r, iv, opt_type))

        # Payoff function (constant closure, JAX-compatible)
        self.payoff_fn = _make_payoff_fn(
            payoff_type, self.strike, self.is_call, **self.payoff_kwargs)

        # Risk measure selector
        if risk_measure == "entropic":
            self._risk = lambda pnl: entropic_risk(pnl, self.risk_param)
        elif risk_measure == "cvar":
            self._risk = lambda pnl: cvar_risk(pnl, self.risk_param)
        elif risk_measure == "mean_variance":
            self._risk = lambda pnl: mean_variance_risk(pnl, self.risk_param)
        else:
            raise ValueError(f"Unknown risk measure: {risk_measure}")

        # Initialise policy
        if key is None:
            key = jax.random.PRNGKey(0)
        self.policy = HedgingPolicy(key)
        self.training_history: list = []

    # ──────────────────────────────────────────────────────────────
    #  Training
    # ──────────────────────────────────────────────────────────────
    def train(
        self,
        spot_paths: np.ndarray,
        var_paths: np.ndarray,
        n_epochs: int    = 500,
        batch_size: int  = 4096,
        lr: float        = 1e-3,
        lr_final: float  = 1e-5,
        warmup: int      = 30,
        clip_norm: float = 1.0,
        verbose: bool    = True,
        seed: int        = 42,
    ) -> list:
        """
        Train the hedging policy.

        Parameters
        ----------
        spot_paths : (n_paths, n_steps+1)  absolute spot prices incl. S₀
        var_paths  : (n_paths, n_steps+1)  variance incl. V₀
        n_epochs   : training epochs
        batch_size : mini-batch size (full-batch if ≥ n_paths)
        lr         : peak learning rate
        verbose    : print progress every 50 epochs

        Returns
        -------
        history : list of per-epoch average loss values
        """
        spot_jax = jnp.array(spot_paths)
        var_jax  = jnp.array(var_paths)
        n_paths  = spot_jax.shape[0]
        S0       = float(spot_jax[0, 0])

        # Trim to full batches
        n_complete = max(batch_size, (n_paths // batch_size) * batch_size)
        spot_jax = spot_jax[:n_complete]
        var_jax  = var_jax[:n_complete]
        n_batches = n_complete // batch_size

        # Optimizer (warmup + cosine decay + gradient clipping)
        total_steps = n_epochs * n_batches
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=lr * 0.1, peak_value=lr,
            warmup_steps=min(warmup, total_steps // 5),
            decay_steps=total_steps, end_value=lr_final,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(clip_norm),
            optax.adam(schedule),
        )
        opt_state = optimizer.init(eqx.filter(self.policy, eqx.is_array))

        # Closures for JIT
        payoff_fn = self.payoff_fn
        risk_fn   = self._risk
        T, r, tc  = self.T, self.r, self.tc_rate
        premium   = self.premium
        strike    = self.strike
        is_call   = self.is_call

        @eqx.filter_jit
        def train_step(policy, opt_state, batch_s, batch_v):
            def loss_fn(pol):
                delta_fn = _make_deep_delta(pol, S0, strike, T, r, is_call)
                pnl = _hedge_pnl_batch(
                    delta_fn, batch_s, batch_v, T, r, tc, premium, payoff_fn)
                return risk_fn(pnl)

            loss, grads = eqx.filter_value_and_grad(loss_fn)(policy)
            updates, new_state = optimizer.update(
                grads, opt_state, eqx.filter(policy, eqx.is_array))
            new_policy = eqx.apply_updates(policy, updates)
            return new_policy, new_state, loss

        # Training loop
        key = jax.random.PRNGKey(seed)
        history = []

        if verbose:
            print(f"  Deep Hedger: {n_complete} paths × "
                  f"{spot_jax.shape[1]-1} steps, "
                  f"batch={batch_size}, epochs={n_epochs}")
            print(f"  Compiling JIT graph (first call only)...")

        for epoch in range(n_epochs):
            key, sk = jax.random.split(key)
            perm = jax.random.permutation(sk, n_complete)
            spot_s = spot_jax[perm]
            var_s  = var_jax[perm]

            epoch_loss = 0.0
            for b in range(n_batches):
                i0 = b * batch_size
                i1 = i0 + batch_size
                self.policy, opt_state, loss = train_step(
                    self.policy, opt_state,
                    spot_s[i0:i1], var_s[i0:i1])
                epoch_loss += float(loss)

            epoch_loss /= n_batches
            history.append(epoch_loss)

            if verbose and (epoch % 50 == 0 or epoch == n_epochs - 1):
                print(f"  epoch {epoch:4d}/{n_epochs}  "
                      f"risk={epoch_loss:.6f}")

        self.training_history = history
        return history

    # ──────────────────────────────────────────────────────────────
    #  P&L computation  (trained policy  /  baselines)
    # ──────────────────────────────────────────────────────────────
    def hedge_pnl(self, spot_paths, var_paths) -> np.ndarray:
        """Compute per-path P&L using the *trained* policy."""
        S0 = float(np.asarray(spot_paths)[0, 0])
        delta_fn = _make_deep_delta(
            self.policy, S0, self.strike, self.T, self.r, self.is_call)
        pnl = _hedge_pnl_batch(
            delta_fn, jnp.array(spot_paths), jnp.array(var_paths),
            self.T, self.r, self.tc_rate, self.premium, self.payoff_fn)
        return np.asarray(pnl)

    def bs_hedge_pnl(self, spot_paths, var_paths) -> np.ndarray:
        """BS delta hedge P&L (same payoff, same paths — fair baseline)."""
        delta_fn = _make_bs_delta(self.strike, self.r, self.is_call)
        pnl = _hedge_pnl_batch(
            delta_fn, jnp.array(spot_paths), jnp.array(var_paths),
            self.T, self.r, self.tc_rate, self.premium, self.payoff_fn)
        return np.asarray(pnl)

    def bartlett_hedge_pnl(self, spot_paths, var_paths) -> np.ndarray:
        """Bartlett (min-variance) hedge P&L — fair baseline."""
        delta_fn = _make_bartlett_delta(
            self.strike, self.r, self.rho, self.is_call)
        pnl = _hedge_pnl_batch(
            delta_fn, jnp.array(spot_paths), jnp.array(var_paths),
            self.T, self.r, self.tc_rate, self.premium, self.payoff_fn)
        return np.asarray(pnl)

    # ──────────────────────────────────────────────────────────────
    #  Evaluate  (all three strategies on the same test paths)
    # ──────────────────────────────────────────────────────────────
    def evaluate(self, spot_paths, var_paths) -> Dict[str, Any]:
        """
        Run deep / BS / Bartlett on test paths and return comparison.

        Returns dict with keys: deep, bs, bartlett, summary (table).
        """
        pnl_deep = self.hedge_pnl(spot_paths, var_paths)
        pnl_bs   = self.bs_hedge_pnl(spot_paths, var_paths)
        pnl_bart = self.bartlett_hedge_pnl(spot_paths, var_paths)

        stats_deep = _compute_stats(pnl_deep, self.premium)
        stats_bs   = _compute_stats(pnl_bs, self.premium)
        stats_bart = _compute_stats(pnl_bart, self.premium)

        # Improvement metrics
        def _pct(a, b):
            return (b - a) / abs(b) * 100 if abs(b) > 1e-10 else 0.0

        summary = {
            "cvar95_reduction_vs_bs":   _pct(stats_deep["cvar_95"], stats_bs["cvar_95"]),
            "cvar95_reduction_vs_bart": _pct(stats_deep["cvar_95"], stats_bart["cvar_95"]),
            "std_reduction_vs_bs":      _pct(stats_deep["std_pnl"], stats_bs["std_pnl"]),
            "std_reduction_vs_bart":    _pct(stats_deep["std_pnl"], stats_bart["std_pnl"]),
        }

        return {
            "deep":     {**stats_deep,  "pnl": pnl_deep},
            "bs":       {**stats_bs,    "pnl": pnl_bs},
            "bartlett": {**stats_bart,  "pnl": pnl_bart},
            "summary":  summary,
        }

    # ──────────────────────────────────────────────────────────────
    #  Persistence
    # ──────────────────────────────────────────────────────────────
    def save_policy(self, directory: str = "models/deep_hedger"):
        """Save trained policy weights + config for reproducibility."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        eqx.tree_serialise_leaves(str(d / "policy.eqx"), self.policy)
        meta = {
            "spot": self.spot, "strike": self.strike, "T": self.T,
            "r": self.r, "iv": self.iv,
            "opt_type": "call" if self.is_call else "put",
            "tc_bps": self.tc_rate * 10_000,
            "risk_measure": self.risk_measure_name,
            "risk_param": self.risk_param,
            "payoff_type": self.payoff_type,
            "payoff_kwargs": self.payoff_kwargs,
            "premium": self.premium, "rho": self.rho,
        }
        with open(d / "config.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, directory: str = "models/deep_hedger"):
        """Load a previously saved DeepHedger."""
        d = Path(directory)
        with open(d / "config.json") as f:
            meta = json.load(f)
        hedger = cls(
            spot=meta["spot"], strike=meta["strike"], T=meta["T"],
            r=meta["r"], iv=meta["iv"], opt_type=meta["opt_type"],
            tc_bps=meta["tc_bps"], risk_measure=meta["risk_measure"],
            risk_param=meta["risk_param"], payoff_type=meta["payoff_type"],
            payoff_kwargs=meta.get("payoff_kwargs", {}),
            rho=meta["rho"], premium=meta["premium"],
        )
        hedger.policy = eqx.tree_deserialise_leaves(
            str(d / "policy.eqx"), hedger.policy)
        return hedger
